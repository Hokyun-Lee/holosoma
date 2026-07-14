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
                      must match GT (lightweight geometric regularizer)
    fk_consistency -- predicted body positions must match differentiable FK
                      of predicted root pose and joint positions (optional)
    joint_limit    -- squared hinge outside configurable MJCF joint margins
    foot_slide     -- xy displacement of feet during GT contact frames
    terrain_penetration -- relu(-z) of predicted body positions; applied only
                      to flat-terrain clips (ground plane z=0); zero otherwise
    lower_body_terrain_penetration -- exact pinned-MJCF 12-sphere lower-leg /
                      sole proxy against flat ground or the terrain scan
Not implemented (no data): contact-consistency loss (no contact labels).
"""

from __future__ import annotations

import math
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
    fk_consistency: float = 0.0
    """Differentiable-FK consistency weight (zero preserves Stage 1--7)."""
    joint_limit: float = 0.0
    """Soft MJCF joint-limit hinge weight (zero preserves earlier stages)."""
    lower_body_terrain_penetration: float = 0.0
    """Pinned-MJCF lower-leg/sole collision-proxy weight."""
    lower_body_terrain_tail: float = 0.0
    """Top-tail lower-body collision loss for rare deep penetrations."""


def validate_terrain_condition_masks(
    flat: torch.Tensor | None,
    has_scan: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    """Validate the mutually-exclusive terrain membership contract.

    ``flat`` means the legacy analytic ground-plane condition, while
    ``has_scan`` selects an explicit sampled height field.  A sample cannot
    belong to both because counting both surfaces changes loss/metric scale
    and makes terrain depth ambiguous.
    """

    for name, mask in (("flat", flat), ("has_scan", has_scan)):
        if mask is None:
            continue
        if tuple(mask.shape) != (batch_size,):
            raise ValueError(f"{name} must have shape ({batch_size},), got {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must be a boolean tensor, got {mask.dtype}")
        if mask.device != device:
            raise ValueError(f"{name} must be on {device}, got {mask.device}")

    if flat is not None and has_scan is not None:
        overlap = flat & has_scan
        if bool(overlap.any()):
            indices = torch.nonzero(overlap, as_tuple=False).flatten().detach().cpu().tolist()
            raise ValueError(
                f"flat and has_scan must be mutually exclusive per sample; overlap at batch indices {indices}"
            )


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
    fk_model: torch.nn.Module | None = None,
    joint_limits: torch.Tensor | None = None,
    joint_limit_margin_rad: float = 0.0,
    terrain_penetration_tolerance_m: float = 0.0,
    terrain_penetration_tail_fraction: float = 0.01,
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
        fk_model: Optional differentiable module mapping predicted root pose
            and joints to tracked body positions. Required when the FK weight
            or lower-body collision-proxy weight is non-zero.
        joint_limits: ``(2, J)`` lower/upper MJCF hinge limits in radians.
        joint_limit_margin_rad: Optional inward margin used by the soft hinge.
        terrain_penetration_tolerance_m: Allowed collision-proxy surface depth
            before the lower-body terrain loss activates.
        terrain_penetration_tail_fraction: Fraction of valid proxy values used
            by the optional top-tail mean-squared penetration term.
    Returns:
        dict with each unweighted term and the weighted "total".
    """
    B, H, _ = pred_x0.shape
    validate_terrain_condition_masks(
        flat,
        has_scan,
        batch_size=B,
        device=pred_x0.device,
    )
    if seq_mask is None:
        seq_mask = torch.ones(B, H, dtype=torch.bool, device=pred_x0.device)
    m = seq_mask.float().unsqueeze(-1)  # (B, H, 1)
    denom = m.sum().clamp_min(1.0)

    def masked_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # mean over valid frames, mean over feature dims
        return (((a - b) ** 2).mean(dim=-1, keepdim=True) * m).sum() / denom

    pred = unpack_features(pred_x0, layout)
    gt = unpack_features(gt_x0, layout)

    if not torch.isfinite(torch.tensor(joint_limit_margin_rad)) or joint_limit_margin_rad < 0.0:
        raise ValueError("joint_limit_margin_rad must be finite and >= 0")
    if not torch.isfinite(torch.tensor(terrain_penetration_tolerance_m)) or terrain_penetration_tolerance_m < 0.0:
        raise ValueError("terrain_penetration_tolerance_m must be finite and >= 0")
    if (
        not torch.isfinite(torch.tensor(terrain_penetration_tail_fraction))
        or not 0.0 < terrain_penetration_tail_fraction <= 1.0
    ):
        raise ValueError("terrain_penetration_tail_fraction must be finite and in (0, 1]")

    losses: dict[str, torch.Tensor] = {}
    losses["root_pos"] = masked_mse(pred["root_pos"], gt["root_pos"])
    losses["joint_pos"] = masked_mse(pred["joint_pos"], gt["joint_pos"])
    losses["body_pos"] = masked_mse(pred["body_pos"].flatten(-2), gt["body_pos"].flatten(-2))

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

    # Couple the independently predicted joint/root and body-position heads.
    # This remains opt-in so all Stage 1--7 presets retain their exact loss.
    fk_body_pos = None
    fk_body_quat = None
    if fk_model is not None:
        if hasattr(fk_model, "tracked_body_transforms"):
            fk_body_pos, fk_body_quat = fk_model.tracked_body_transforms(
                pred["root_pos"], pred["root_quat"], pred["joint_pos"]
            )
        else:
            fk_body_pos = fk_model(pred["root_pos"], pred["root_quat"], pred["joint_pos"])
        if fk_body_pos.shape != pred["body_pos"].shape:
            raise ValueError(
                "FK body output shape "
                f"{tuple(fk_body_pos.shape)} != predicted body shape {tuple(pred['body_pos'].shape)}"
            )
        fk_delta = pred["body_pos"] - fk_body_pos
        losses["fk_consistency"] = (fk_delta.square().mean(dim=(-1, -2)).unsqueeze(-1) * m).sum() / denom
        losses["fk_body_error_m"] = (fk_delta.norm(dim=-1).mean(dim=-1, keepdim=True) * m).sum() / denom
    elif (
        weights.fk_consistency != 0.0
        or weights.lower_body_terrain_penetration != 0.0
        or weights.lower_body_terrain_tail != 0.0
    ):
        raise ValueError(
            "A differentiable fk_model is required when fk_consistency or a lower-body terrain weight is non-zero"
        )

    # Soft joint-limit hinge in radians.  Sum over joints and mean over valid
    # frames so a small number of offending hinges is not diluted by 29.
    if joint_limits is not None:
        if tuple(joint_limits.shape) != (2, layout.num_joints):
            raise ValueError(f"joint_limits must have shape (2, {layout.num_joints}), got {tuple(joint_limits.shape)}")
        limits = joint_limits.to(device=pred_x0.device, dtype=pred_x0.dtype)
        lo = limits[0] + joint_limit_margin_rad
        hi = limits[1] - joint_limit_margin_rad
        if bool((lo >= hi).any()):
            raise ValueError("joint_limit_margin_rad collapses at least one joint range")
        lower_violation = torch.relu(lo - pred["joint_pos"])
        upper_violation = torch.relu(pred["joint_pos"] - hi)
        margin_violation = lower_violation + upper_violation
        losses["joint_limit"] = (margin_violation.square().sum(dim=-1, keepdim=True) * m).sum() / denom
        strict_lo = limits[0]
        strict_hi = limits[1]
        strict_violation = torch.maximum(
            torch.relu(strict_lo - pred["joint_pos"]),
            torch.relu(pred["joint_pos"] - strict_hi),
        )
        losses["joint_limit_max_violation_rad"] = torch.where(
            m.bool(), strict_violation, torch.zeros_like(strict_violation)
        ).max()
        losses["joint_limit_frame_rate"] = (
            strict_violation.gt(0.0).any(dim=-1, keepdim=True).float() * m
        ).sum() / denom
    elif weights.joint_limit != 0.0:
        raise ValueError("joint_limits are required when joint_limit weight is non-zero")

    # Foot sliding during GT contact.
    foot_idx = layout.foot_body_indices()
    if contact is not None and foot_idx:
        feet = pred["body_pos"][:, :, foot_idx, :2]  # (B, H, F, 2)
        disp = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)  # (B, H-1, F)
        both = (contact[:, 1:] & contact[:, :-1]).float() * vm
        losses["foot_slide"] = (disp**2 * both).sum() / both.sum().clamp_min(1.0)
    else:
        losses["foot_slide"] = torch.zeros((), device=pred_x0.device)

    # Terrain penetration implementation contract: ground plane z=0 for flat
    # clips and interpolated height for scanned clips are disjoint surfaces.
    # Average once over every eligible (sample, frame, body) value so the
    # scale is independent of flat/scanned batch composition.  In particular,
    # no-surface samples and seq-masked frames are absent from the denominator.
    pen_sumsq = pred_x0.new_zeros(())
    pen_count = pred_x0.new_zeros(())
    z = pred["body_pos"][..., 2]  # (B, H, num_bodies)
    if flat is not None:
        gate = (flat.view(B, 1, 1) & seq_mask.unsqueeze(-1)).expand_as(z)
        pen = torch.relu(-z)
        pen_sumsq = pen_sumsq + (pen.square() * gate).sum()
        pen_count = pen_count + gate.sum()
    if terrain_scan is not None and has_scan is not None and scan_grid is not None and has_scan.any():
        h, valid = interpolate_scan_heights(terrain_scan, pred["body_pos"][..., :2], scan_grid)
        gate = has_scan.view(B, 1, 1) & valid & seq_mask.unsqueeze(-1)
        pen = torch.relu(h - z)
        pen_sumsq = pen_sumsq + (pen.square() * gate).sum()
        pen_count = pen_count + gate.sum()
    losses["terrain_penetration"] = pen_sumsq / pen_count.clamp_min(1.0)

    # The tracked body origins above miss foot surfaces and the lower part of
    # each shin.  Use the pinned G1 MJCF's 12 collision-marker spheres so this
    # term acts directly on root/joint FK instead of the independent body head.
    if weights.lower_body_terrain_penetration != 0.0 or weights.lower_body_terrain_tail != 0.0:
        if fk_model is None or fk_body_pos is None or fk_body_quat is None:
            raise ValueError("lower_body_terrain_penetration requires an FK model with tracked_body_transforms()")
        centers, radii = fk_model.lower_body_collision_spheres_from_tracked_transforms(fk_body_pos, fk_body_quat)
        bottom_z = centers[..., 2] - radii.view(1, 1, -1)
        proxy_sumsq = pred_x0.new_zeros(())
        proxy_count = pred_x0.new_zeros(())
        raw_depth_values: list[torch.Tensor] = []
        raw_gate_values: list[torch.Tensor] = []

        if flat is not None:
            flat_gate = flat.view(B, 1, 1).float() * m
            flat_gate = flat_gate.expand_as(bottom_z)
            raw_depth = torch.relu(-bottom_z)
            active_depth = torch.relu(raw_depth - terrain_penetration_tolerance_m)
            proxy_sumsq = proxy_sumsq + (active_depth.square() * flat_gate).sum()
            proxy_count = proxy_count + flat_gate.sum()
            raw_depth_values.append(raw_depth)
            raw_gate_values.append(flat_gate)

        if terrain_scan is not None and has_scan is not None and scan_grid is not None and has_scan.any():
            terrain_h, valid = interpolate_scan_heights(terrain_scan, centers[..., :2], scan_grid)
            scan_gate = has_scan.view(B, 1, 1).float() * valid.float() * m
            raw_depth = torch.relu(terrain_h - bottom_z)
            active_depth = torch.relu(raw_depth - terrain_penetration_tolerance_m)
            proxy_sumsq = proxy_sumsq + (active_depth.square() * scan_gate).sum()
            proxy_count = proxy_count + scan_gate.sum()
            raw_depth_values.append(raw_depth)
            raw_gate_values.append(scan_gate)

        losses["lower_body_terrain_penetration"] = proxy_sumsq / proxy_count.clamp_min(1.0)
        if raw_depth_values:
            max_depths = [
                torch.where(gate.bool(), depth, torch.zeros_like(depth)).max()
                for depth, gate in zip(raw_depth_values, raw_gate_values)
            ]
            violation_count = sum(
                ((depth > 0.005).float() * gate).sum() for depth, gate in zip(raw_depth_values, raw_gate_values)
            )
            losses["lower_body_max_penetration_m"] = torch.stack(max_depths).max()
            losses["lower_body_penetration_value_rate_5mm"] = violation_count / proxy_count.clamp_min(1.0)
            active_values = torch.cat(
                [
                    torch.relu(depth - terrain_penetration_tolerance_m)[gate.bool()]
                    for depth, gate in zip(raw_depth_values, raw_gate_values, strict=True)
                ]
            )
            if active_values.numel():
                tail_count = max(
                    1,
                    math.ceil(terrain_penetration_tail_fraction * active_values.numel()),
                )
                tail_values = torch.topk(active_values, tail_count, sorted=False).values
                losses["lower_body_terrain_tail"] = tail_values.square().mean()
                losses["lower_body_terrain_tail_count"] = pred_x0.new_tensor(tail_count)
            else:
                losses["lower_body_terrain_tail"] = pred_x0.new_zeros(())
                losses["lower_body_terrain_tail_count"] = pred_x0.new_zeros(())
        else:
            losses["lower_body_max_penetration_m"] = pred_x0.new_zeros(())
            losses["lower_body_penetration_value_rate_5mm"] = pred_x0.new_zeros(())
            losses["lower_body_terrain_tail"] = pred_x0.new_zeros(())
            losses["lower_body_terrain_tail_count"] = pred_x0.new_zeros(())

    total = pred_x0.new_zeros(())
    for name in weights.__dataclass_fields__:
        weight = getattr(weights, name)
        if name not in losses:
            if weight != 0.0:
                raise ValueError(f"Loss term {name!r} has weight {weight} but was not computed")
            continue
        total = total + weight * losses[name]
    losses["total"] = total
    return losses
