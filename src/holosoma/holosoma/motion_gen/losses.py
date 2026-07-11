"""Training losses.

The paper states the generator is trained with a reconstruction loss against
the ground-truth future sequence plus geometric losses ("velocity, joint
consistency, terrain penetration", similar to PARC). Loss weights are not
published; all defaults below are implementation choices, configurable via
``LossWeights``.

All losses are computed in physical (denormalized, canonical-frame) units:
meters for positions, radians for joints. Inputs are the model's x0
prediction (converted from epsilon if needed) and the ground-truth window.

Implemented:
    root_pos / root_quat / joint_pos / body_pos  -- reconstruction split
    quat_norm      -- unit-norm regularizer on the predicted quaternion
    velocity       -- MSE of temporal finite differences (all feature dims)
    bone_length    -- distances between kinematically adjacent tracked bodies
                      must match GT (surrogate for FK joint-consistency; no
                      differentiable FK is used in this stage)
    foot_slide     -- xy displacement of feet during GT contact frames
    terrain_penetration -- relu(-z) of predicted body positions; applied only
                      to flat-terrain clips (ground plane z=0); zero otherwise
Not implemented (no data): contact-consistency loss (no contact labels).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from holosoma.motion_gen.features import FeatureLayout, quat_normalize, unpack_features
from holosoma.motion_gen.terrain import ScanGrid, interpolate_scan_heights


@dataclass
class LossWeights:
    root_pos: float = 1.0
    root_quat: float = 1.0
    joint_pos: float = 1.0
    body_pos: float = 1.0
    quat_norm: float = 0.1
    velocity: float = 1.0
    bone_length: float = 0.5
    foot_slide: float = 0.3
    terrain_penetration: float = 0.5


def compute_losses(
    pred_x0: torch.Tensor,
    gt_x0: torch.Tensor,
    layout: FeatureLayout,
    weights: LossWeights,
    contact: torch.Tensor | None = None,
    flat: torch.Tensor | None = None,
    seq_mask: torch.Tensor | None = None,
    terrain_scan: torch.Tensor | None = None,
    has_scan: torch.Tensor | None = None,
    scan_grid: ScanGrid | None = None,
) -> dict[str, torch.Tensor]:
    """Compute all loss terms.

    Args:
        pred_x0: (B, H, D) predicted clean future window, physical units.
        gt_x0: (B, H, D) ground truth.
        contact: (B, H, n_feet) bool GT foot-contact proxy (for foot_slide).
        flat: (B,) bool, clip is flat terrain (gates terrain_penetration).
        seq_mask: (B, H) bool, True = valid frame. Invalid frames contribute
            zero to every loss term.
        terrain_scan: (B, G) anchor-frame height scans; with ``has_scan``
            (B,) bool and ``scan_grid``, enables the scan-based penetration
            loss (bodies below the interpolated terrain surface). Bodies
            outside the scan grid are excluded.
    Returns:
        dict with each unweighted term and the weighted "total".
    """
    B, H, _ = pred_x0.shape
    if seq_mask is None:
        seq_mask = torch.ones(B, H, dtype=torch.bool, device=pred_x0.device)
    m = seq_mask.float().unsqueeze(-1)  # (B, H, 1)
    denom = m.sum().clamp_min(1.0)

    def masked_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # mean over valid frames, mean over feature dims
        return (((a - b) ** 2).mean(dim=-1, keepdim=True) * m).sum() / denom

    pred = unpack_features(pred_x0, layout)
    gt = unpack_features(gt_x0, layout)

    losses: dict[str, torch.Tensor] = {}
    losses["root_pos"] = masked_mse(pred["root_pos"], gt["root_pos"])
    losses["joint_pos"] = masked_mse(pred["joint_pos"], gt["joint_pos"])
    losses["body_pos"] = masked_mse(
        pred["body_pos"].flatten(-2), gt["body_pos"].flatten(-2)
    )

    # Quaternion: sign-invariant distance on the normalized prediction plus a
    # separate unit-norm regularizer (element-wise MSE alone is sign-ambiguous).
    pred_q = pred["root_quat"]
    gt_q = quat_normalize(gt["root_quat"])
    dot = (quat_normalize(pred_q) * gt_q).sum(dim=-1, keepdim=True)  # (B, H, 1)
    losses["root_quat"] = ((1.0 - dot**2) * m).sum() / denom
    losses["quat_norm"] = (((pred_q.norm(dim=-1, keepdim=True) - 1.0) ** 2) * m).sum() / denom

    # Velocity: finite differences along time over all feature dims.
    vm = (seq_mask[:, 1:] & seq_mask[:, :-1]).float().unsqueeze(-1)
    vdenom = vm.sum().clamp_min(1.0)
    dv = (pred_x0[:, 1:] - pred_x0[:, :-1]) - (gt_x0[:, 1:] - gt_x0[:, :-1])
    losses["velocity"] = ((dv**2).mean(dim=-1, keepdim=True) * vm).sum() / vdenom

    # Bone lengths between adjacent tracked bodies (skeleton consistency).
    pairs = layout.bone_pair_indices()
    ia = torch.tensor([p[0] for p in pairs], device=pred_x0.device)
    ib = torch.tensor([p[1] for p in pairs], device=pred_x0.device)
    pred_len = (pred["body_pos"][:, :, ia] - pred["body_pos"][:, :, ib]).norm(dim=-1)
    gt_len = (gt["body_pos"][:, :, ia] - gt["body_pos"][:, :, ib]).norm(dim=-1)
    losses["bone_length"] = (((pred_len - gt_len) ** 2).mean(dim=-1, keepdim=True) * m).sum() / denom

    # Foot sliding during GT contact.
    foot_idx = layout.foot_body_indices()
    if contact is not None and foot_idx:
        feet = pred["body_pos"][:, :, foot_idx, :2]  # (B, H, F, 2)
        disp = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)  # (B, H-1, F)
        both = (contact[:, 1:] & contact[:, :-1]).float() * vm
        losses["foot_slide"] = (disp**2 * both).sum() / both.sum().clamp_min(1.0)
    else:
        losses["foot_slide"] = torch.zeros((), device=pred_x0.device)

    # Terrain penetration: ground plane z=0 for flat clips; interpolated
    # multi-box terrain height for clips with a real scan (Phase B).
    pen_total = pred_x0.new_zeros(())
    z = pred["body_pos"][..., 2]  # (B, H, num_bodies)
    if flat is not None:
        pen = torch.relu(-z) * flat.view(B, 1, 1).float() * m
        pen_total = pen_total + (pen**2).sum() / denom
    if terrain_scan is not None and has_scan is not None and scan_grid is not None and has_scan.any():
        h, valid = interpolate_scan_heights(terrain_scan, pred["body_pos"][..., :2], scan_grid)
        gate = has_scan.view(B, 1, 1).float() * valid.float() * m
        pen = torch.relu(h - z) * gate
        pen_total = pen_total + (pen**2).sum() / gate.sum().clamp_min(1.0)
    losses["terrain_penetration"] = pen_total

    total = pred_x0.new_zeros(())
    for name, value in losses.items():
        total = total + getattr(weights, name) * value
    losses["total"] = total
    return losses
