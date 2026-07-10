"""Export generated motions to ``.npz``.

Two formats:

1. ``*_gen_raw.npz`` (generator-native): exactly what the model produced —
   root state, joint positions and the 14 tracked body positions. Used for
   plots, metrics and debugging.

2. ``*_gen_qpos.npz``: ``{qpos (T, 36), fps}`` in the HoloSoma retargeting
   output layout ``[root_pos(3), root_quat wxyz(4), joints(29)]``. This is the
   input format of the official conversion step
   (``holosoma_retargeting/data_conversion/convert_data_format_mj_headless.py``)
   which runs MuJoCo FK to produce a *full* WBT-schema npz (all 51 bodies,
   velocities) that the HoloSoma motion tracker can load. Body quantities the
   generator does not predict (body orientations, the other 37 bodies) are
   reconstructed there by FK, which is why this conversion is a separate step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from holosoma.motion_gen.features import FeatureLayout, quat_normalize, unpack_features


def export_generated_raw_npz(
    features: torch.Tensor,
    layout: FeatureLayout,
    fps: float,
    path: str | Path,
    gt_features: torch.Tensor | None = None,
    heading: torch.Tensor | None = None,
) -> Path:
    """Save one generated window (T, D) in generator-native format."""
    path = Path(path)
    parts = unpack_features(features, layout)
    payload = {
        "fps": np.array([fps]),
        "root_pos": parts["root_pos"].numpy(),
        "root_quat_wxyz": quat_normalize(parts["root_quat"]).numpy(),
        "root_quat_raw": parts["root_quat"].numpy(),
        "joint_pos": parts["joint_pos"].numpy(),
        "body_pos": parts["body_pos"].numpy(),
        "joint_names": np.array(layout.joint_names),
        "body_names": np.array(layout.body_names),
        "frame": np.array("canonical_or_world_see_metadata"),
    }
    if gt_features is not None:
        gt = unpack_features(gt_features, layout)
        payload["gt_root_pos"] = gt["root_pos"].numpy()
        payload["gt_root_quat_wxyz"] = gt["root_quat"].numpy()
        payload["gt_joint_pos"] = gt["joint_pos"].numpy()
        payload["gt_body_pos"] = gt["body_pos"].numpy()
    if heading is not None:
        payload["target_heading"] = heading.numpy()
    np.savez(path, **payload)
    return path


def export_generated_qpos_npz(
    features: torch.Tensor,
    layout: FeatureLayout,
    fps: float,
    path: str | Path,
) -> Path:
    """Save a generated motion (T, D) as a HoloSoma-layout qpos npz.

    The features should be in the world frame (de-canonicalized) if the file
    is meant for replay/tracking. The predicted quaternion is re-normalized.
    """
    path = Path(path)
    parts = unpack_features(features, layout)
    qpos = torch.cat(
        [parts["root_pos"], quat_normalize(parts["root_quat"]), parts["joint_pos"]], dim=-1
    )
    np.savez(path, qpos=qpos.numpy().astype(np.float64), fps=np.array(int(round(fps))))
    return path
