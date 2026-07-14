"""Dataset pipeline: HoloSoma ``*_mj.npz`` motions -> canonical training windows.

Motions must be in the HoloSoma WBT format produced by
``holosoma_retargeting/data_conversion/convert_data_format_mj*.py``:
    fps, joint_pos (T, 7+J), joint_vel (T, 6+J), body_pos_w (T, NB, 3),
    body_quat_w (T, NB, 4, wxyz), body_lin_vel_w, body_ang_vel_w,
    joint_names (J), body_names (NB)

Windows never cross clip boundaries and contain no padding, so no padding
frames can leak into losses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from holosoma.motion_gen.features import (
    FeatureLayout,
    canonicalize_window,
    compute_target_heading,
    pack_features,
    quat_enforce_continuity,
)
from holosoma.motion_gen.terrain import ScanGrid

# Contact proxy thresholds (implementation choice; no contact labels in data).
_CONTACT_HEIGHT_MARGIN = 0.03  # m above the per-clip minimum foot height
_CONTACT_SPEED_MAX = 0.25  # m/s horizontal foot speed
_SCAN_GRID_ATOL = 1.0e-6


@dataclass
class MotionClip:
    name: str
    fps: float
    features: torch.Tensor  # (T, D) world-frame packed features
    foot_contact: torch.Tensor  # (T, n_feet) bool, contact proxy
    flat_terrain: bool
    source: str = "unknown"
    terrain_scan: torch.Tensor | None = None  # (T, G) heading-aligned heights
    terrain_grid: torch.Tensor | None = None  # (5,) grid definition
    source_path: str | None = None

    @property
    def num_frames(self) -> int:
        return int(self.features.shape[0])


def load_wbt_motion(
    path: str | Path,
    layout: FeatureLayout,
    expected_fps: float = 50.0,
    flat_terrain: bool = False,
    source: str = "unknown",
) -> MotionClip:
    """Load one HoloSoma WBT npz motion and pack it into world-frame features."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    required = {"fps", "joint_pos", "body_pos_w", "joint_names", "body_names"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{path}: missing npz keys {sorted(missing)}; expected HoloSoma WBT format.")

    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if abs(fps - expected_fps) > 1e-6:
        raise ValueError(f"{path}: fps={fps}, expected {expected_fps}. Re-run the data conversion.")

    joint_names = [str(n) for n in data["joint_names"]]
    body_names = [str(n) for n in data["body_names"]]
    for name in layout.joint_names:
        if name not in joint_names:
            raise ValueError(f"{path}: joint '{name}' not found in npz joint_names.")
    for name in layout.body_names:
        if name not in body_names:
            raise ValueError(f"{path}: body '{name}' not found in npz body_names.")
    joint_idx = [joint_names.index(n) for n in layout.joint_names]
    body_idx = [body_names.index(n) for n in layout.body_names]

    qpos = torch.from_numpy(np.asarray(data["joint_pos"], dtype=np.float32))  # (T, 7+J)
    if qpos.shape[1] != 7 + len(joint_names):
        raise ValueError(
            f"{path}: joint_pos has {qpos.shape[1]} columns, expected {7 + len(joint_names)} "
            "(root pos+quat followed by joint positions)."
        )
    root_pos = qpos[:, 0:3]
    root_quat = quat_enforce_continuity(qpos[:, 3:7])
    joint_pos = qpos[:, 7:][:, joint_idx]
    body_pos = torch.from_numpy(np.asarray(data["body_pos_w"], dtype=np.float32))[:, body_idx]

    if not torch.isfinite(qpos).all() or not torch.isfinite(body_pos).all():
        raise ValueError(f"{path}: motion contains NaN/Inf values.")

    features = pack_features(root_pos, root_quat, joint_pos, body_pos)

    terrain_scan = None
    terrain_grid = None
    if "terrain_height" in data.files:
        if "terrain_grid" not in data.files:
            raise ValueError(
                f"{path}: terrain_height is present but terrain_grid is missing; "
                "regenerate the scan metadata with add_terrain_scans."
            )
        terrain_scan_array = np.asarray(data["terrain_height"], dtype=np.float32)
        terrain_grid_array = np.asarray(data["terrain_grid"], dtype=np.float32)
        if terrain_scan_array.ndim != 2:
            raise ValueError(
                f"{path}: terrain_height must have shape (frames, grid.dim), got {terrain_scan_array.shape}."
            )
        if terrain_grid_array.shape != (5,):
            raise ValueError(
                f"{path}: terrain_grid must have shape (5,) "
                "[x_min, x_max, y_min, y_max, spacing], "
                f"got {terrain_grid_array.shape}."
            )
        if not np.isfinite(terrain_scan_array).all() or not np.isfinite(terrain_grid_array).all():
            raise ValueError(f"{path}: terrain_height/terrain_grid contains NaN/Inf values.")
        terrain_scan = torch.from_numpy(terrain_scan_array)
        terrain_grid = torch.from_numpy(terrain_grid_array)
        if terrain_scan.shape[0] != features.shape[0]:
            raise ValueError(
                f"{path}: terrain_height frames {terrain_scan.shape[0]} != motion frames {features.shape[0]}"
            )

    foot_idx = layout.foot_body_indices()
    feet = body_pos[:, foot_idx]  # (T, n_feet, 3)
    z_floor = feet[..., 2].amin(dim=0, keepdim=True)  # per-foot clip minimum
    speed_xy = torch.zeros_like(feet[..., 0])
    speed_xy[1:] = (feet[1:, :, :2] - feet[:-1, :, :2]).norm(dim=-1) * fps
    speed_xy[0] = speed_xy[1]
    contact = (feet[..., 2] < z_floor + _CONTACT_HEIGHT_MARGIN) & (speed_xy < _CONTACT_SPEED_MAX)

    return MotionClip(
        name=path.stem,
        fps=fps,
        features=features,
        foot_contact=contact,
        flat_terrain=flat_terrain,
        source=source,
        terrain_scan=terrain_scan,
        terrain_grid=terrain_grid,
        source_path=str(path),
    )


def load_split_clips(
    processed_dir: str | Path,
    splits_file: str | Path,
    split: str,
    layout: FeatureLayout,
    metadata_dir: str | Path | None = None,
    expected_fps: float = 50.0,
) -> list[MotionClip]:
    """Load all clips of one split (train/val). Splits are motion-level."""
    processed_dir = Path(processed_dir)
    with open(splits_file) as f:
        splits = json.load(f)
    if split not in splits:
        raise KeyError(f"Split '{split}' not in {splits_file} (has {sorted(splits)}).")

    clips = []
    for stem in splits[split]:
        npz_path = processed_dir / f"{stem}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Motion file not found: {npz_path}")
        flat, source = False, "unknown"
        if metadata_dir is not None:
            meta_path = Path(metadata_dir) / f"{stem}.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                flat = bool(meta.get("flat_terrain", False))
                source = str(meta.get("source", "unknown"))
        clips.append(load_wbt_motion(npz_path, layout, expected_fps=expected_fps, flat_terrain=flat, source=source))
    return clips


class MotionWindowDataset(Dataset):
    """Sliding windows of (past + future) frames, canonicalized per window.

    Item keys:
        x        (future, D) canonical future features (diffusion target)
        past     (past, D)   canonical conditioning frames
        heading  (2,)        unit target heading, canonical frame
        terrain  (terrain_dim,) height scan (zeros in Phase A)
        contact  (future, n_feet) GT foot-contact proxy (bool)
        flat     ()           bool, clip is flat-terrain
        anchor_xy (2,), anchor_yaw ()  world-from-canonical transform
        clip_idx (), start ()  provenance
    """

    def __init__(
        self,
        clips: list[MotionClip],
        layout: FeatureLayout,
        past_frames: int = 2,
        future_frames: int = 25,
        stride: int = 1,
        min_heading_disp: float = 0.05,
        terrain_dim: int = 121,
        use_terrain_scan: bool = False,
        scan_grid: ScanGrid | None = None,
    ):
        if past_frames < 1 or future_frames < 1:
            raise ValueError("past_frames and future_frames must be >= 1")
        self.clips = clips
        self.layout = layout
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.window = past_frames + future_frames
        self.min_heading_disp = min_heading_disp
        self.terrain_dim = terrain_dim
        self.use_terrain_scan = use_terrain_scan
        # Implementation contract: every consumed scan must carry the exact
        # configured extents and spacing, not merely the same flattened size.
        # None intentionally resolves to the historical production default.
        self.scan_grid = scan_grid if scan_grid is not None else ScanGrid()
        if use_terrain_scan:
            if terrain_dim != self.scan_grid.dim:
                raise ValueError(
                    f"configured terrain_dim {terrain_dim} != configured scan_grid.dim "
                    f"{self.scan_grid.dim} for {self.scan_grid}."
                )
            for clip in clips:
                if clip.terrain_scan is None:
                    continue  # Legacy/no-scan clips retain their zero condition.
                location = clip.source_path or clip.name
                if clip.terrain_scan.ndim != 2:
                    raise ValueError(
                        f"{location} (clip {clip.name!r}): terrain scan must be 2-D, "
                        f"got shape {tuple(clip.terrain_scan.shape)}."
                    )
                if clip.terrain_scan.shape[1] != terrain_dim:
                    raise ValueError(
                        f"{location} (clip {clip.name!r}): terrain scan dim "
                        f"{clip.terrain_scan.shape[1]} != "
                        f"configured terrain_dim {terrain_dim} (re-run add_terrain_scans "
                        "with a matching grid or fix the config)."
                    )
                if clip.terrain_grid is None:
                    raise ValueError(
                        f"{location} (clip {clip.name!r}): terrain scan is present but "
                        "terrain_grid metadata is missing."
                    )
                actual_grid = clip.terrain_grid.detach().cpu().numpy().astype(np.float64, copy=False)
                if actual_grid.shape != (5,) or not np.isfinite(actual_grid).all():
                    raise ValueError(
                        f"{location} (clip {clip.name!r}): terrain_grid must be five "
                        "finite values [x_min, x_max, y_min, y_max, spacing], "
                        f"got {actual_grid.tolist() if actual_grid.ndim == 1 else actual_grid.shape}."
                    )
                expected_grid = self.scan_grid.to_array().astype(np.float64, copy=False)
                if not np.allclose(actual_grid, expected_grid, rtol=0.0, atol=_SCAN_GRID_ATOL):
                    raise ValueError(
                        f"{location} (clip {clip.name!r}): terrain_grid contract mismatch; "
                        f"stored={actual_grid.tolist()}, configured={expected_grid.tolist()} "
                        f"(absolute tolerance {_SCAN_GRID_ATOL:g})."
                    )

        self._index: list[tuple[int, int]] = []
        for ci, clip in enumerate(clips):
            last_start = clip.num_frames - self.window
            for start in range(0, last_start + 1, stride):
                self._index.append((ci, start))
        if not self._index:
            raise ValueError(
                f"No valid windows: window={self.window} frames, clip lengths={[c.num_frames for c in clips]}."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ci, start = self._index[idx]
        clip = self.clips[ci]
        window = clip.features[start : start + self.window]
        anchor = self.past_frames - 1
        canon, transform = canonicalize_window(window, self.layout, anchor_index=anchor)
        heading = compute_target_heading(canon, self.layout, anchor_index=anchor, min_disp=self.min_heading_disp)
        contact = clip.foot_contact[start + self.past_frames : start + self.window]
        terrain = torch.zeros(self.terrain_dim)
        has_scan = False
        if self.use_terrain_scan and clip.terrain_scan is not None:
            terrain = clip.terrain_scan[start + anchor]
            has_scan = True
        return {
            "x": canon[self.past_frames :],
            "past": canon[: self.past_frames],
            "heading": heading,
            "terrain": terrain,
            "has_scan": torch.tensor(has_scan),
            "contact": contact,
            "flat": torch.tensor(clip.flat_terrain),
            "anchor_xy": transform.anchor_xy,
            "anchor_yaw": transform.anchor_yaw,
            "clip_idx": torch.tensor(ci),
            "start": torch.tensor(start),
        }
