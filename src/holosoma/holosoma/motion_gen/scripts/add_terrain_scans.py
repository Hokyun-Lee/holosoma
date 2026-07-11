"""Attach real terrain height scans to processed climb/scene motions.

For every ``omni_climb_*``/``omni_scene_*`` npz in the processed directory,
samples a heading-aligned height scan around the root at every frame from the
matching OmniRetarget multi-box terrain and rewrites the npz with two new
keys:

    terrain_height  (T, grid.dim) float32, absolute heights (ground = 0)
    terrain_grid    (5,) [x_min, x_max, y_min, y_max, spacing]

Also validates motion-terrain alignment: during foot contacts the foot body
z must sit slightly above the terrain height under the foot (ankle-roll body
origin is ~3.5 cm above the sole). Clips failing this check are reported.

Note: chair-scene scans contain only the stage/platform boxes — the movable
chair object is NOT part of the terrain scan (documented limitation).

Usage (from the repo root, after download_terrain):
    python -m holosoma.motion_gen.scripts.add_terrain_scans
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from holosoma.motion_gen.terrain import BoxTerrain, ScanGrid

_FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]


@dataclass
class Args:
    data_root: str = "data/motion_gen"
    processed_dir: str = "data/motion_gen/processed_paperscale"
    grid: ScanGrid = ScanGrid()
    force: bool = False
    """Recompute scans even if terrain_height already exists."""


def terrain_urdf_for(stem: str, terrain_root: Path) -> Path | None:
    m = re.fullmatch(r"omni_climb_(\d+)_z(\d)_(\d)", stem)
    if m:
        cid, z_hi, z_lo = m.groups()
        return terrain_root / f"climb_{cid}" / f"multi_boxes_z_scale_{z_hi}.{z_lo}.urdf"
    m = re.fullmatch(r"omni_scene_(\d+)", stem)
    if m:
        return terrain_root / f"scene_{m.group(1)}" / "multi_boxes_z_scale_1.0.urdf"
    return None


def quat_yaw_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main(args: Args) -> None:
    terrain_root = Path(args.data_root) / "raw" / "omniretarget_terrain"
    processed = Path(args.processed_dir)
    report: list[tuple[str, float, float]] = []
    skipped, done = 0, 0

    for npz_path in sorted(processed.glob("omni_*.npz")):
        stem = npz_path.stem
        urdf = terrain_urdf_for(stem, terrain_root)
        if urdf is None:
            print(f"[skip] {stem}: no terrain mapping (takes have no terrain model)")
            skipped += 1
            continue
        if not urdf.exists():
            raise FileNotFoundError(f"{urdf} missing — run download_terrain first.")

        data = dict(np.load(npz_path, allow_pickle=True))
        if "terrain_height" in data and not args.force:
            print(f"[skip] {stem}: terrain_height already present")
            done += 1
            continue

        terrain = BoxTerrain.from_urdf(urdf)
        qpos = data["joint_pos"]
        root_xy = qpos[:, :2]
        yaw = quat_yaw_wxyz(qpos[:, 3:7])
        scans: np.ndarray = terrain.sample_scans(root_xy, yaw, args.grid).astype(np.float32)
        data["terrain_height"] = scans
        data["terrain_grid"] = args.grid.to_array()

        # Alignment check: feet vs terrain height directly under them.
        body_names = [str(n) for n in data["body_names"]]
        clearances = []
        for foot in _FOOT_BODIES:
            fp = data["body_pos_w"][:, body_names.index(foot)]  # (T, 3)
            h = terrain.height_at(fp[:, :2])
            clearances.append(fp[:, 2] - h)
        clearance = np.concatenate(clearances)
        min_c, med_c = float(clearance.min()), float(np.median(clearance))
        report.append((stem, min_c, med_c))

        np.savez(npz_path, **data)
        done += 1
        print(f"[ok] {stem}: scans {scans.shape}, foot clearance min={min_c:+.3f} med={med_c:+.3f} m")

    bad = [r for r in report if r[1] < -0.05]
    print(f"\n[done] {done} clips with scans, {skipped} skipped (no terrain)")
    if bad:
        print(f"[WARNING] {len(bad)} clips with feet >5 cm below terrain (alignment suspect):")
        for stem, mn, md in bad:
            print(f"  {stem}: min={mn:+.3f} med={md:+.3f}")
    else:
        print("[ok] alignment check passed: no clip has feet >5 cm below terrain")


if __name__ == "__main__":
    main(tyro.cli(Args))
