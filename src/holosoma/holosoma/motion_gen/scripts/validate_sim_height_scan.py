"""Validate the generator scan contract against HoloSoma's GPU mesh ray caster.

The synthetic scene has three spatially separated lanes (box, ascending
stairs, hurdle).  It is intentionally deterministic and is only a coordinate /
ray-cast diagnostic; curriculum terrain generation remains a separate task.

Run from the repository root in the hssim environment::

    python -m holosoma.motion_gen.scripts.validate_sim_height_scan \
        --out logs/motion_gen/stage6/height_scan_debug.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import trimesh
import tyro

from holosoma.motion_gen.terrain import (
    ScanGrid,
    local_scan_world_xy,
    save_height_scan_debug,
)
from holosoma.utils import warp_utils


@dataclass
class Args:
    out: str = "logs/motion_gen/stage6/height_scan_debug.npz"
    device: str = "cuda:0"


def _rect_box(x0: float, x1: float, y0: float, y1: float, height: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(x1 - x0, y1 - y0, height))
    mesh.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, height / 2))
    return mesh


def _ground() -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(6.0, 6.0, 0.02))
    mesh.apply_translation((0.5, 0.0, -0.01))
    return mesh


def _validation_mesh() -> trimesh.Trimesh:
    parts = [
        _ground(),
        _rect_box(0.15, 0.55, -0.75, -0.25, 0.3),
        _rect_box(0.15, 0.45, -0.15, 0.15, 0.1),
        _rect_box(0.45, 0.75, -0.15, 0.15, 0.2),
        _rect_box(0.75, 1.05, -0.15, 0.15, 0.3),
        _rect_box(0.65, 0.95, 0.25, 0.75, 0.6),
    ]
    return trimesh.util.concatenate(parts)


def _expected_height(local_xy: torch.Tensor) -> torch.Tensor:
    x, y = local_xy.unbind(-1)
    height = torch.zeros_like(x)
    box = (x >= 0.15) & (x <= 0.55) & (y >= -0.75) & (y <= -0.25)
    height = torch.where(box, 0.3, height)
    stair_lane = (y >= -0.15) & (y <= 0.15)
    height = torch.where(stair_lane & (x >= 0.15) & (x <= 0.45), 0.1, height)
    height = torch.where(stair_lane & (x >= 0.45) & (x <= 0.75), 0.2, height)
    height = torch.where(stair_lane & (x >= 0.75) & (x <= 1.05), 0.3, height)
    hurdle = (x >= 0.65) & (x <= 0.95) & (y >= 0.25) & (y <= 0.75)
    return torch.where(hurdle, 0.6, height)


def _raycast_scan(
    mesh: trimesh.Trimesh,
    root_xy: torch.Tensor,
    root_yaw: torch.Tensor,
    grid: ScanGrid,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_xy = local_scan_world_xy(root_xy, root_yaw, grid)
    starts = torch.zeros(root_xy.shape[0], grid.dim, 3, device=root_xy.device)
    starts[..., :2] = world_xy
    starts[..., 2] = 100.0
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1.0
    warp_mesh = warp_utils.convert_to_wp_mesh(mesh.vertices, mesh.faces, str(root_xy.device))
    hits = warp_utils.ray_cast(starts, directions, warp_mesh)
    if not torch.isfinite(hits).all():
        raise RuntimeError("Terrain mesh ray cast missed one or more local scan points")
    return hits[..., 2], world_xy


def _rotated_scene(mesh: trimesh.Trimesh, root_xy: torch.Tensor, yaw: float) -> trimesh.Trimesh:
    transform = trimesh.transformations.rotation_matrix(yaw, (0, 0, 1))
    transform[:2, 3] = root_xy.detach().cpu().numpy()
    rotated = mesh.copy()
    rotated.apply_transform(transform)
    return rotated


@torch.no_grad()
def main(args: Args) -> None:
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for the simulator Warp ray-cast validation")

    device = torch.device(args.device)
    grid = ScanGrid()
    root_xy = torch.zeros(1, 2, device=device)
    root_yaw = torch.zeros(1, device=device)

    flat_scan, _ = _raycast_scan(_ground(), root_xy, root_yaw, grid)
    flat_max_abs = float(flat_scan.abs().max())
    raycast_tolerance = 5e-5
    if flat_max_abs >= raycast_tolerance:
        raise AssertionError(
            f"Flat scan max abs {flat_max_abs:.3e} exceeds {raycast_tolerance:.1e} m"
        )

    mesh = _validation_mesh()
    scan, world_xy = _raycast_scan(mesh, root_xy, root_yaw, grid)
    expected = _expected_height(grid.offsets_tensor(device=device)).unsqueeze(0)
    obstacle_max_error = float((scan - expected).abs().max())
    if obstacle_max_error >= raycast_tolerance:
        raise AssertionError(
            f"Obstacle scan max error {obstacle_max_error:.3e} exceeds {raycast_tolerance:.1e} m"
        )

    rotated_root = torch.tensor([[2.0, -3.0]], device=device)
    rotated_yaw = torch.tensor([torch.pi / 2], device=device)
    rotated_mesh = _rotated_scene(mesh, rotated_root[0], float(rotated_yaw[0]))
    rotated_scan, _ = _raycast_scan(rotated_mesh, rotated_root, rotated_yaw, grid)
    yaw_max_error = float((rotated_scan - scan).abs().max())
    if yaw_max_error >= raycast_tolerance:
        raise AssertionError(
            f"Yaw-consistency max error {yaw_max_error:.3e} exceeds {raycast_tolerance:.1e} m"
        )

    output = save_height_scan_debug(
        Path(args.out),
        root_xy=root_xy,
        root_yaw=root_yaw,
        world_xy=world_xy,
        terrain_height=scan,
        grid=grid,
    )
    print(f"flat_max_abs_m={flat_max_abs:.8f}")
    print(f"obstacle_max_error_m={obstacle_max_error:.8f}")
    print(f"yaw_max_error_m={yaw_max_error:.8f}")
    print(f"debug_snapshot={output.resolve()}")


if __name__ == "__main__":
    main(tyro.cli(Args))
