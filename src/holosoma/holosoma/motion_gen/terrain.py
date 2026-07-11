"""Box-approximated terrain: analytic heightmap + heading-aligned height scans.

OmniRetarget provides each climb terrain as ``multi_boxes_z_scale_{z}.urdf``
referencing per-box ``.obj`` meshes whose vertices are baked in world
coordinates (link origins are identity; the z-scale variant is applied via the
URDF ``scale`` attribute). Every box has a flat top (vertices at z=0 and
z=h) and a possibly yaw-rotated rectangular footprint, so terrain height at a
point is: max over boxes containing the point of the box top height, else 0
(flat ground).

Scan convention (matches the generator's canonical frame): a regular grid in
the *heading-aligned* frame of a query root pose — grid x forward, y left,
centered at the root xy — sampled as absolute terrain heights (meters,
ground = 0). The default grid is forward-biased for locomotion:
x in [-0.3, 1.3], y in [-0.8, 0.8], 0.1 m spacing -> 17x17 = 289 values,
ordered row-major over (x, y). Grid extents are an implementation choice
(the paper does not give the scan resolution).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ScanGrid:
    x_min: float = -0.3
    x_max: float = 1.3
    y_min: float = -0.8
    y_max: float = 0.8
    spacing: float = 0.1

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / self.spacing)) + 1

    @property
    def ny(self) -> int:
        return int(round((self.y_max - self.y_min) / self.spacing)) + 1

    @property
    def dim(self) -> int:
        return self.nx * self.ny

    def offsets(self) -> np.ndarray:
        """(dim, 2) local xy offsets, row-major over (x, y)."""
        xs = self.x_min + self.spacing * np.arange(self.nx)
        ys = self.y_min + self.spacing * np.arange(self.ny)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)

    def to_array(self) -> np.ndarray:
        return np.array([self.x_min, self.x_max, self.y_min, self.y_max, self.spacing])

    @staticmethod
    def from_array(a: np.ndarray) -> "ScanGrid":
        return ScanGrid(*[float(v) for v in np.asarray(a).reshape(-1)])


def _parse_obj_vertices(path: Path) -> np.ndarray:
    verts = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
    return np.asarray(verts)


class BoxTerrain:
    """Analytic heightmap from a multi-box URDF (flat ground elsewhere)."""

    def __init__(self, polygons: list[np.ndarray], top_heights: list[float]):
        # Each polygon: (4, 2) footprint corners in CCW order.
        self.polygons = polygons
        self.top_heights = np.asarray(top_heights)
        # Precompute edge normals for point-in-convex-polygon tests.
        self._edges = []
        for poly in polygons:
            a = poly
            b = np.roll(poly, -1, axis=0)
            self._edges.append((a, b - a))  # origin, direction per edge

    @staticmethod
    def from_urdf(urdf_path: str | Path) -> "BoxTerrain":
        urdf_path = Path(urdf_path)
        text = urdf_path.read_text()
        polygons: list[np.ndarray] = []
        tops: list[float] = []
        seen: set[str] = set()
        for match in re.finditer(r'<mesh filename="([^"]+)" scale="([^"]+)"', text):
            mesh_file, scale_str = match.groups()
            if mesh_file in seen:  # visual + collision reference the same mesh
                continue
            seen.add(mesh_file)
            scale = np.array([float(s) for s in scale_str.split()])
            verts = _parse_obj_vertices(urdf_path.parent / mesh_file) * scale
            if verts.shape[0] != 8:
                raise ValueError(f"{mesh_file}: expected an 8-vertex box, got {verts.shape[0]} vertices")
            top = float(verts[:, 2].max())
            footprint = _order_convex_ccw(np.unique(np.round(verts[:, :2], 6), axis=0))
            polygons.append(footprint)
            tops.append(top)
        if not polygons:
            raise ValueError(f"{urdf_path}: no box meshes found")
        return BoxTerrain(polygons, tops)

    def height_at(self, xy: np.ndarray) -> np.ndarray:
        """Terrain height for query points xy (..., 2); ground = 0."""
        pts: np.ndarray = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        heights = np.zeros(pts.shape[0])
        for (origin, direction), top in zip(self._edges, self.top_heights):
            # inside if the point is left of (or on) every CCW edge
            rel = pts[:, None, :] - origin[None, :, :]  # (N, 4, 2)
            cross = direction[None, :, 0] * rel[..., 1] - direction[None, :, 1] * rel[..., 0]
            inside = (cross >= -1e-9).all(axis=1)
            heights = np.where(inside, np.maximum(heights, top), heights)
        return heights.reshape(np.asarray(xy).shape[:-1])

    def sample_scan(self, root_xy: np.ndarray, root_yaw: float, grid: ScanGrid) -> np.ndarray:
        """(grid.dim,) heading-aligned scan around one root pose."""
        off = grid.offsets()
        c, s = np.cos(root_yaw), np.sin(root_yaw)
        world = np.stack(
            [root_xy[0] + c * off[:, 0] - s * off[:, 1], root_xy[1] + s * off[:, 0] + c * off[:, 1]],
            axis=-1,
        )
        return self.height_at(world)

    def sample_scans(self, root_xy: np.ndarray, root_yaw: np.ndarray, grid: ScanGrid) -> np.ndarray:
        """(T, grid.dim) scans for a trajectory of root poses."""
        return np.stack(
            [self.sample_scan(root_xy[t], float(root_yaw[t]), grid) for t in range(root_xy.shape[0])]
        )


def _order_convex_ccw(points: np.ndarray) -> np.ndarray:
    """Order the (typically 4) footprint corners counter-clockwise."""
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def interpolate_scan_heights(
    scan: torch.Tensor,
    query_xy: torch.Tensor,
    grid: ScanGrid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear terrain height under canonical-frame query points.

    Args:
        scan: (B, grid.dim) heading-aligned scans (anchor frame).
        query_xy: (B, ..., 2) canonical-frame xy positions.
    Returns:
        (heights (B, ...), valid mask (B, ...)); points outside the grid are
        masked invalid (height 0).
    """
    B = scan.shape[0]
    nx, ny = grid.nx, grid.ny
    grid2d = scan.view(B, nx, ny)

    fx = (query_xy[..., 0] - grid.x_min) / grid.spacing
    fy = (query_xy[..., 1] - grid.y_min) / grid.spacing
    valid = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
    fx = fx.clamp(0, nx - 1 - 1e-6)
    fy = fy.clamp(0, ny - 1 - 1e-6)
    x0 = fx.floor().long()
    y0 = fy.floor().long()
    tx = (fx - x0).unsqueeze(-1)
    ty = (fy - y0).unsqueeze(-1)

    def gather(ix, iy):
        flat = (ix * ny + iy).view(B, -1)
        return torch.gather(scan, 1, flat).view(ix.shape)

    h00 = gather(x0, y0)
    h10 = gather(x0 + 1, y0)
    h01 = gather(x0, y0 + 1)
    h11 = gather(x0 + 1, y0 + 1)
    tx, ty = tx.squeeze(-1), ty.squeeze(-1)
    h = (
        h00 * (1 - tx) * (1 - ty)
        + h10 * tx * (1 - ty)
        + h01 * (1 - tx) * ty
        + h11 * tx * ty
    )
    return h * valid.float(), valid
