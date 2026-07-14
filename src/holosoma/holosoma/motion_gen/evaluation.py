"""Quantitative metrics for generated vs. ground-truth future windows.

All metrics are computed in physical units (m, rad) in the canonical frame.
Train and validation metrics must be reported separately by the caller;
single-motion overfit numbers are not generalization numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from holosoma.motion_gen.features import FeatureLayout, quat_angle, unpack_features
from holosoma.motion_gen.terrain import interpolate_scan_heights


def load_joint_limits(path: str | Path, layout: FeatureLayout) -> torch.Tensor | None:
    """Load joint limits json {joint_name: [lo, hi]} -> (2, J) tensor."""
    path = Path(path)
    if not path.exists():
        return None
    limits = json.loads(path.read_text())
    lo, hi = [], []
    for name in layout.joint_names:
        if name not in limits:
            return None
        lo.append(float(limits[name][0]))
        hi.append(float(limits[name][1]))
    return torch.tensor([lo, hi])


@torch.no_grad()
def compute_metrics(
    pred_x0: torch.Tensor,
    gt_x0: torch.Tensor,
    layout: FeatureLayout,
    fps: float = 50.0,
    joint_limits: torch.Tensor | None = None,
    contact: torch.Tensor | None = None,
    flat: torch.Tensor | None = None,
    terrain_scan: torch.Tensor | None = None,
    has_scan: torch.Tensor | None = None,
    scan_grid=None,
    fk_model: torch.nn.Module | None = None,
) -> dict[str, float]:
    """Metrics over a batch of windows (B, H, D), physical canonical units."""
    pred = unpack_features(pred_x0, layout)
    gt = unpack_features(gt_x0, layout)

    metrics: dict[str, float] = {}
    metrics["total_recon_mse"] = ((pred_x0 - gt_x0) ** 2).mean().item()
    metrics["root_pos_err_m"] = (pred["root_pos"] - gt["root_pos"]).norm(dim=-1).mean().item()
    metrics["root_quat_err_rad"] = quat_angle(pred["root_quat"], gt["root_quat"]).mean().item()
    metrics["joint_pos_err_rad"] = (pred["joint_pos"] - gt["joint_pos"]).abs().mean().item()
    metrics["body_mpjpe_m"] = (pred["body_pos"] - gt["body_pos"]).norm(dim=-1).mean().item()

    dv = (pred_x0[:, 1:] - pred_x0[:, :-1]) - (gt_x0[:, 1:] - gt_x0[:, :-1])
    metrics["velocity_err"] = (dv**2).mean().item() * fps * fps  # (units/s)^2

    metrics["quat_norm_dev"] = (pred["root_quat"].norm(dim=-1) - 1.0).abs().mean().item()

    if joint_limits is not None:
        lo = joint_limits[0].to(pred_x0.device)
        hi = joint_limits[1].to(pred_x0.device)
        lower = torch.relu(lo - pred["joint_pos"])
        upper = torch.relu(pred["joint_pos"] - hi)
        depth = torch.maximum(lower, upper)
        viol = depth > 0.0
        metrics["joint_limit_violation_rate"] = viol.float().mean().item()
        metrics["joint_limit_frame_rate"] = viol.any(dim=-1).float().mean().item()
        metrics["joint_limit_max_violation_rad"] = depth.max().item()

    foot_idx = layout.foot_body_indices()
    if contact is not None and foot_idx:
        feet = pred["body_pos"][:, :, foot_idx, :2]
        disp = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)  # (B, H-1, F) m/frame
        both = (contact[:, 1:] & contact[:, :-1]).float()
        denom = both.sum().clamp_min(1.0)
        metrics["foot_slide_m_per_s"] = ((disp * both).sum() / denom).item() * fps

    if flat is not None and flat.any():
        z = pred["body_pos"][..., 2]
        pen = torch.relu(-z) * flat.view(-1, 1, 1).float()
        metrics["terrain_penetration_m"] = (
            pen.sum() / flat.float().sum().clamp_min(1.0) / z.shape[1] / z.shape[2]
        ).item()

    if terrain_scan is not None and has_scan is not None and scan_grid is not None and has_scan.any():
        z = pred["body_pos"][..., 2]
        h, valid = interpolate_scan_heights(terrain_scan, pred["body_pos"][..., :2], scan_grid)
        gate = has_scan.view(-1, 1, 1).float() * valid.float()
        pen = torch.relu(h - z) * gate
        metrics["scan_penetration_m"] = (pen.sum() / gate.sum().clamp_min(1.0)).item()

    if fk_model is not None:
        fk_body, fk_quat = fk_model.tracked_body_transforms(pred["root_pos"], pred["root_quat"], pred["joint_pos"])
        fk_error = (pred["body_pos"] - fk_body).norm(dim=-1)
        metrics["fk_body_error_mean_m"] = fk_error.mean().item()
        metrics["fk_body_error_p95_m"] = torch.quantile(fk_error.float(), 0.95).item()
        metrics["fk_body_error_max_m"] = fk_error.max().item()

        centers, radii = fk_model.lower_body_collision_spheres_from_tracked_transforms(fk_body, fk_quat)
        bottom_z = centers[..., 2] - radii.view(1, 1, -1)
        proxy_depth = torch.zeros_like(bottom_z)
        proxy_gate = torch.zeros_like(bottom_z, dtype=torch.bool)
        if flat is not None:
            gate = flat.view(-1, 1, 1).expand_as(bottom_z)
            proxy_depth = torch.where(gate, torch.relu(-bottom_z), proxy_depth)
            proxy_gate |= gate
        if terrain_scan is not None and has_scan is not None and scan_grid is not None and has_scan.any():
            height, valid = interpolate_scan_heights(terrain_scan, centers[..., :2], scan_grid)
            gate = has_scan.view(-1, 1, 1) & valid
            depth = torch.relu(height - bottom_z)
            proxy_depth = torch.where(gate, depth, proxy_depth)
            proxy_gate |= gate
        if proxy_gate.any():
            gated_depth = torch.where(proxy_gate, proxy_depth, torch.zeros_like(proxy_depth))
            metrics["lower_body_proxy_penetration_mean_m"] = (gated_depth.sum() / proxy_gate.sum().clamp_min(1)).item()
            metrics["lower_body_proxy_penetration_max_m"] = gated_depth.max().item()
            metrics["lower_body_proxy_penetration_value_rate_5mm"] = (
                ((gated_depth > 0.005) & proxy_gate).sum() / proxy_gate.sum().clamp_min(1)
            ).item()
            valid_frames = proxy_gate.any(dim=-1)
            violating_frames = ((gated_depth > 0.005) & proxy_gate).any(dim=-1)
            metrics["lower_body_proxy_penetration_frame_rate_5mm"] = (
                violating_frames.sum() / valid_frames.sum().clamp_min(1)
            ).item()

    return metrics
