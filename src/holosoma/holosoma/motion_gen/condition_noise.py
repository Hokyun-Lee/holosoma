"""Structured physical-unit noise for motion-generator conditions only."""

from __future__ import annotations

import math
from dataclasses import fields

import torch
from torch.nn import functional as F

from holosoma.motion_gen.configs import ConditionNoiseCfg
from holosoma.motion_gen.features import (
    FeatureLayout,
    pack_features,
    quat_mul,
    quat_normalize,
    unpack_features,
)
from holosoma.motion_gen.terrain import ScanGrid

_TERRAIN_NOISE_FIELDS = (
    "terrain_height_std_m",
    "terrain_point_dropout_prob",
    "terrain_height_bias_std_m",
    "terrain_xy_std_m",
    "terrain_yaw_std_rad",
)


def validate_condition_noise_config(
    cfg: ConditionNoiseCfg,
    *,
    use_terrain_scan: bool,
    terrain_dim: int,
    scan_grid: ScanGrid,
) -> None:
    """Validate magnitudes and the terrain-scan contract once at setup."""
    for item in fields(cfg):
        value = getattr(cfg, item.name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"condition_noise.{item.name} must be a finite non-negative float")
        if not math.isfinite(float(value)) or value < 0.0:
            raise ValueError(f"condition_noise.{item.name} must be finite and >= 0, got {value}")
    if cfg.terrain_point_dropout_prob > 1.0:
        raise ValueError("condition_noise.terrain_point_dropout_prob must be <= 1")

    terrain_noise_enabled = any(getattr(cfg, name) > 0.0 for name in _TERRAIN_NOISE_FIELDS)
    if terrain_noise_enabled and not use_terrain_scan:
        raise ValueError("Terrain condition noise requires data.use_terrain_scan=True")
    if terrain_noise_enabled and terrain_dim != scan_grid.dim:
        raise ValueError(
            f"Terrain condition noise requires terrain_dim={scan_grid.dim} for the configured "
            f"scan grid, got {terrain_dim}"
        )


def axis_angle_to_quat_wxyz(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Convert unit axes and angles to normalized wxyz quaternions."""
    if axis.shape[-1] != 3 or angle.shape != axis.shape[:-1]:
        raise ValueError(
            f"axis/angle shapes must be (..., 3) and (...), got {axis.shape} and {angle.shape}"
        )
    axis_norm = axis.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(axis)
    fallback[..., 0] = 1.0
    unit_axis = torch.where(axis_norm > 1.0e-8, axis / axis_norm.clamp_min(1.0e-8), fallback)
    half = 0.5 * angle
    return quat_normalize(torch.cat([torch.cos(half).unsqueeze(-1), unit_axis * torch.sin(half).unsqueeze(-1)], dim=-1))


def _preserve_quaternion_sign_continuity(
    quaternion: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Keep the reference's first sign branch and temporal sign continuity."""
    result = quat_normalize(quaternion).clone()
    first_dot = (result[..., 0, :] * reference[..., 0, :]).sum(dim=-1, keepdim=True)
    result[..., 0, :] = torch.where(first_dot < 0.0, -result[..., 0, :], result[..., 0, :])
    for frame in range(1, result.shape[-2]):
        dot = (result[..., frame, :] * result[..., frame - 1, :]).sum(dim=-1, keepdim=True)
        result[..., frame, :] = torch.where(
            dot < 0.0,
            -result[..., frame, :],
            result[..., frame, :],
        )
    return result


def warp_terrain_scan(
    terrain: torch.Tensor,
    grid: ScanGrid,
    translation_xy: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """Bilinearly resample scans under local xy/yaw sensor-frame error.

    For each output grid point ``p``, the clean scan is queried at
    ``R(yaw) p + translation_xy``. Values outside the scan use absolute
    ground height zero. The x-major, y-fastest flatten order is preserved.
    """
    if terrain.ndim != 2 or terrain.shape[1] != grid.dim:
        raise ValueError(f"terrain must have shape (B, {grid.dim}), got {tuple(terrain.shape)}")
    batch_size = terrain.shape[0]
    if translation_xy.shape != (batch_size, 2):
        raise ValueError(
            f"translation_xy must have shape ({batch_size}, 2), got {tuple(translation_xy.shape)}"
        )
    if yaw.shape != (batch_size,):
        raise ValueError(f"yaw must have shape ({batch_size},), got {tuple(yaw.shape)}")
    if translation_xy.device != terrain.device or yaw.device != terrain.device:
        raise ValueError("terrain, translation_xy, and yaw must share a device")

    offsets = grid.offsets_tensor(device=terrain.device, dtype=terrain.dtype)
    px = offsets[:, 0].unsqueeze(0)
    py = offsets[:, 1].unsqueeze(0)
    cosine = torch.cos(yaw).unsqueeze(1)
    sine = torch.sin(yaw).unsqueeze(1)
    query_x = cosine * px - sine * py + translation_xy[:, :1]
    query_y = sine * px + cosine * py + translation_xy[:, 1:]

    norm_x = 2.0 * (query_x - grid.x_min) / (grid.x_max - grid.x_min) - 1.0
    norm_y = 2.0 * (query_y - grid.y_min) / (grid.y_max - grid.y_min) - 1.0
    # grid_sample coordinates are (width, height); our image axes are
    # (scan-x, scan-y), so width receives local y and height receives local x.
    sample_grid = torch.stack([norm_y, norm_x], dim=-1).reshape(
        batch_size, grid.nx, grid.ny, 2
    )
    scan_image = terrain.reshape(batch_size, 1, grid.nx, grid.ny)
    warped = F.grid_sample(
        scan_image,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped.reshape(batch_size, grid.dim)


def apply_condition_noise(
    past: torch.Tensor,
    terrain: torch.Tensor,
    layout: FeatureLayout,
    cfg: ConditionNoiseCfg,
    scan_grid: ScanGrid,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return noisy condition tensors without mutating clean batch tensors.

    ``past`` remains in physical canonical units here. Feature normalization is
    deliberately left to the caller after this function returns.
    """
    if past.ndim != 3 or past.shape[-1] != layout.dim:
        raise ValueError(f"past must have shape (B, P, {layout.dim}), got {tuple(past.shape)}")
    if terrain.ndim != 2 or terrain.shape[0] != past.shape[0]:
        raise ValueError(
            f"terrain must have shape (B, G) with B={past.shape[0]}, got {tuple(terrain.shape)}"
        )
    if not past.is_floating_point() or not terrain.is_floating_point():
        raise TypeError("past and terrain conditions must be floating-point tensors")
    if past.device != terrain.device:
        raise ValueError("past and terrain conditions must share a device")
    if not cfg.is_enabled():
        return past, terrain

    parts = unpack_features(past, layout)
    root_pos = parts["root_pos"].clone()
    root_quat = parts["root_quat"].clone()
    joint_pos = parts["joint_pos"].clone()
    body_pos = parts["body_pos"].clone()

    def randn(shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(shape, device=past.device, dtype=dtype, generator=generator)

    if cfg.root_position_std_m > 0.0:
        root_pos += cfg.root_position_std_m * randn(tuple(root_pos.shape), dtype=root_pos.dtype)
    if cfg.root_orientation_std_rad > 0.0:
        axis = randn((*root_quat.shape[:-1], 3), dtype=root_quat.dtype)
        angle = cfg.root_orientation_std_rad * randn(root_quat.shape[:-1], dtype=root_quat.dtype)
        delta = axis_angle_to_quat_wxyz(axis, angle)
        # Left composition makes the sampled axis live in the canonical/world
        # frame. This distribution is an implementation choice, not a value or
        # convention specified by the paper.
        root_quat = quat_mul(delta, quat_normalize(root_quat))
        root_quat = _preserve_quaternion_sign_continuity(root_quat, parts["root_quat"])
    if cfg.joint_position_std_rad > 0.0:
        joint_pos += cfg.joint_position_std_rad * randn(tuple(joint_pos.shape), dtype=joint_pos.dtype)
    if cfg.body_position_std_m > 0.0:
        body_pos += cfg.body_position_std_m * randn(tuple(body_pos.shape), dtype=body_pos.dtype)
    noisy_past = pack_features(root_pos, root_quat, joint_pos, body_pos)

    noisy_terrain = terrain.clone()
    batch_size = terrain.shape[0]
    if cfg.terrain_xy_std_m > 0.0 or cfg.terrain_yaw_std_rad > 0.0:
        translation = cfg.terrain_xy_std_m * torch.randn(
            (batch_size, 2),
            device=terrain.device,
            dtype=terrain.dtype,
            generator=generator,
        )
        yaw = cfg.terrain_yaw_std_rad * torch.randn(
            (batch_size,),
            device=terrain.device,
            dtype=terrain.dtype,
            generator=generator,
        )
        noisy_terrain = warp_terrain_scan(noisy_terrain, scan_grid, translation, yaw)
    if cfg.terrain_height_bias_std_m > 0.0:
        bias = cfg.terrain_height_bias_std_m * torch.randn(
            (batch_size, 1),
            device=terrain.device,
            dtype=terrain.dtype,
            generator=generator,
        )
        noisy_terrain += bias
    if cfg.terrain_height_std_m > 0.0:
        noisy_terrain += cfg.terrain_height_std_m * torch.randn(
            noisy_terrain.shape,
            device=terrain.device,
            dtype=terrain.dtype,
            generator=generator,
        )
    if cfg.terrain_point_dropout_prob > 0.0:
        dropped = torch.rand(
            noisy_terrain.shape,
            device=terrain.device,
            generator=generator,
        ) < cfg.terrain_point_dropout_prob
        noisy_terrain = noisy_terrain.masked_fill(dropped, 0.0)
    return noisy_past, noisy_terrain
