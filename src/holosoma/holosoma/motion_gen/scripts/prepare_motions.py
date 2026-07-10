"""Normalize raw motions into HoloSoma-layout qpos npz files + metadata.

Input formats handled (per data_manifest.MANIFEST):
    lafan1        CSV rows = [root_pos xyz, root_quat xyzw, 29 joints], 30 fps
    omniretarget  npz qpos (T, 36|43) = [root_quat wxyz, root_pos, 29 joints,
                  (object pose)], 30 fps; object columns are dropped
    holosoma_demo already-converted 50 fps WBT npz; copied to processed/

Output:
    data/motion_gen/raw_qpos/<stem>.npz  {qpos (T, 36) = [pos, quat wxyz, joints], fps}
    data/motion_gen/metadata/<stem>.json {source, license, flat_terrain, ...}

The final FK conversion to the WBT training format is a separate step (needs
MuJoCo, hsretargeting env):
    src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj_headless.py

Usage (from the repo root):
    python -m holosoma.motion_gen.scripts.prepare_motions [--data-root data/motion_gen]
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from holosoma.motion_gen.data_manifest import MANIFEST, ClipSpec


@dataclass
class Args:
    data_root: str = "data/motion_gen"
    repo_root: str = "."
    force: bool = False


def _sanity_check(stem: str, qpos: np.ndarray) -> None:
    if not np.isfinite(qpos).all():
        raise ValueError(f"{stem}: NaN/Inf in qpos")
    quat_norm = np.linalg.norm(qpos[:, 3:7], axis=-1)
    if not np.allclose(quat_norm, 1.0, atol=0.05):
        raise ValueError(f"{stem}: root quaternion norms deviate from 1 (range {quat_norm.min():.3f}..{quat_norm.max():.3f})")
    z = qpos[:, 2]
    if not (0.1 < z.mean() < 1.5):
        raise ValueError(f"{stem}: implausible mean root height {z.mean():.3f} m — check column layout")


def prepare_lafan1(clip: ClipSpec, raw_dir: Path) -> tuple[np.ndarray, int]:
    csv_path = raw_dir / "lafan1_g1" / clip.origin.rsplit("/", 1)[-1]
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} missing — run download_data first.")
    rows: np.ndarray = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if rows.shape[1] != 36:
        raise ValueError(f"{csv_path}: expected 36 columns, got {rows.shape[1]}")
    if clip.frame_range is not None:
        start, end = clip.frame_range
        if end > rows.shape[0]:
            print(f"[warn] {clip.stem}: frame_range end {end} > {rows.shape[0]} frames, clamping")
            end = rows.shape[0]
        rows = rows[start:end]
    pos = rows[:, 0:3]
    quat_wxyz = rows[:, [6, 3, 4, 5]]  # csv stores xyzw
    joints = rows[:, 7:36]
    return np.concatenate([pos, quat_wxyz, joints], axis=1), clip.input_fps


def prepare_omniretarget(clip: ClipSpec, raw_dir: Path) -> tuple[np.ndarray, int]:
    member = clip.origin.split(":", 1)[1]
    npz_path = raw_dir / "omniretarget" / Path(member).name
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} missing — run download_data first.")
    data = np.load(npz_path)
    qpos_in: np.ndarray = np.asarray(data["qpos"], dtype=np.float64)
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    robot = qpos_in[:, :36]  # object columns (if any) are at the end
    # OmniRetarget layout: [root_quat wxyz (0:4), root_pos (4:7), joints (7:36)]
    pos = robot[:, 4:7]
    quat = robot[:, 0:4]
    joints = robot[:, 7:36]
    if clip.frame_range is not None:
        start, end = clip.frame_range
        pos, quat, joints = pos[start:end], quat[start:end], joints[start:end]
    return np.concatenate([pos, quat, joints], axis=1), fps


def main(args: Args) -> None:
    root = Path(args.data_root)
    raw_dir = root / "raw"
    qpos_dir = root / "raw_qpos"
    processed_dir = root / "processed"
    meta_dir = root / "metadata"
    for d in (qpos_dir, processed_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    for clip in MANIFEST:
        meta = {
            "stem": clip.stem,
            "source": clip.source,
            "license": clip.license,
            "origin": clip.origin,
            "flat_terrain": clip.flat_terrain,
            "description": clip.description,
            "input_fps": clip.input_fps,
            "frame_range": list(clip.frame_range) if clip.frame_range else None,
            "quat_order": "wxyz",
            "up_axis": "z",
        }

        if clip.source == "holosoma_demo":
            src = Path(args.repo_root) / clip.origin
            dst = processed_dir / f"{clip.stem}.npz"
            if dst.exists() and not args.force:
                print(f"[skip] {dst.name} exists")
            else:
                shutil.copy(src, dst)
                print(f"[ok] copied demo motion -> {dst}")
        else:
            dst = qpos_dir / f"{clip.stem}.npz"
            if dst.exists() and not args.force:
                print(f"[skip] {dst.name} exists")
            else:
                if clip.source == "lafan1":
                    qpos, fps = prepare_lafan1(clip, raw_dir)
                elif clip.source == "omniretarget":
                    qpos, fps = prepare_omniretarget(clip, raw_dir)
                else:
                    raise ValueError(f"Unknown source {clip.source}")
                _sanity_check(clip.stem, qpos)
                np.savez(dst, qpos=qpos, fps=np.array(fps))
                print(f"[ok] {clip.stem}: {qpos.shape[0]} frames @ {fps} fps -> {dst}")

        (meta_dir / f"{clip.stem}.json").write_text(json.dumps(meta, indent=2))

    print(
        "\nNext step (hsretargeting env, from src/holosoma_retargeting/holosoma_retargeting):\n"
        "  python data_conversion/convert_data_format_mj_headless.py \\\n"
        f"      --input-dir {qpos_dir.resolve()} \\\n"
        f"      --output-dir {processed_dir.resolve()} \\\n"
        f"      --joint-limits-out {(meta_dir / 'joint_limits.json').resolve()}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
