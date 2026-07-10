"""Synthetic WBT-format motion fixtures for CPU tests (no downloads needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from holosoma.motion_gen.features import FeatureLayout


def make_synthetic_wbt_npz(path: Path, num_frames: int = 120, fps: int = 50, seed: int = 0) -> Path:
    """Write a physically plausible-ish synthetic motion in the WBT schema."""
    rng = np.random.default_rng(seed)
    layout = FeatureLayout()
    t = np.arange(num_frames) / fps

    root_pos = np.stack([0.8 * t, 0.1 * np.sin(t), 0.78 + 0.02 * np.sin(3 * t)], axis=1)
    yaw = 0.1 * np.sin(0.5 * t)
    root_quat = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], axis=1)  # wxyz
    joints = 0.3 * np.sin(2 * np.pi * 1.5 * t[:, None] + rng.uniform(0, 6.28, (1, 29)))

    # 51 bodies like the real G1 model: world + pelvis + 49 others.
    body_names = ["world", *layout.body_names]
    body_names += [f"filler_body_{i}" for i in range(51 - len(body_names))]
    offsets = rng.uniform(-0.3, 0.3, (51, 3))
    offsets[0] = 0.0
    body_pos = root_pos[:, None, :] + offsets[None, :, :]
    body_pos[:, 0, :] = 0.0  # world body
    # feet near the ground
    for foot in ("left_ankle_roll_link", "right_ankle_roll_link"):
        bi = body_names.index(foot)
        body_pos[:, bi, 2] = 0.04 + 0.05 * np.abs(np.sin(2 * np.pi * t))
    body_quat = np.tile(root_quat[:, None, :], (1, 51, 1))

    qpos = np.concatenate([root_pos, root_quat, joints], axis=1)  # (T, 36)
    qvel = np.zeros((num_frames, 35))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        fps=np.array([fps]),
        joint_pos=qpos,
        joint_vel=qvel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros_like(body_pos),
        body_ang_vel_w=np.zeros_like(body_pos),
        joint_names=np.array(layout.joint_names),
        body_names=np.array(body_names),
    )
    return path


def make_synthetic_dataset_dir(root: Path, num_clips: int = 2, num_frames: int = 120) -> dict[str, Path]:
    """Create processed/, metadata/ and splits/ for Trainer tests."""
    processed = root / "processed"
    metadata = root / "metadata"
    splits_dir = root / "splits"
    for d in (processed, metadata, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    stems = []
    for i in range(num_clips):
        stem = f"synthetic_{i}"
        make_synthetic_wbt_npz(processed / f"{stem}.npz", num_frames=num_frames, seed=i)
        (metadata / f"{stem}.json").write_text(
            json.dumps({"source": "synthetic", "flat_terrain": True})
        )
        stems.append(stem)

    layout = FeatureLayout()
    limits = {name: [-3.0, 3.0] for name in layout.joint_names}
    (metadata / "joint_limits.json").write_text(json.dumps(limits))

    splits = {"train": stems[: max(1, num_clips - 1)], "val": stems[-1:]}
    splits_file = splits_dir / "splits.json"
    splits_file.write_text(json.dumps(splits))
    return {"processed": processed, "metadata": metadata, "splits_file": splits_file}
