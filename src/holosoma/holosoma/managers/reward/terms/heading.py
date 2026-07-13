"""Shared tensor math for world-frame heading rewards and metrics."""

from __future__ import annotations

import math

import torch


def _validate_heading_inputs(velocity_xy: torch.Tensor, direction_xy: torch.Tensor) -> None:
    if velocity_xy.shape != direction_xy.shape or velocity_xy.ndim < 1 or velocity_xy.shape[-1] != 2:
        raise ValueError(
            "velocity_xy and direction_xy must have the same (..., 2) shape, "
            f"got {tuple(velocity_xy.shape)} and {tuple(direction_xy.shape)}"
        )


def velocity_heading_reward(
    velocity_xy: torch.Tensor,
    direction_xy: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Return ``dot(v_xy, d_xy) / (norm(v_xy) + epsilon)``.

    ``direction_xy`` is expected to be a unit world-frame direction.  It is
    intentionally not normalized here so this function implements the
    requested reward equation exactly.
    """
    _validate_heading_inputs(velocity_xy, direction_xy)
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    dot = torch.sum(velocity_xy * direction_xy, dim=-1)
    speed = torch.linalg.vector_norm(velocity_xy, dim=-1)
    return dot / (speed + epsilon)


def velocity_heading_error_rad(
    velocity_xy: torch.Tensor,
    direction_xy: torch.Tensor,
    *,
    speed_threshold: float,
) -> torch.Tensor:
    """Unsigned velocity-direction error in [0, pi].

    Direction is undefined below ``speed_threshold``; those samples are
    reported as pi/2 so a stopped robot is neutral rather than aligned.
    """
    _validate_heading_inputs(velocity_xy, direction_xy)
    if speed_threshold < 0.0:
        raise ValueError(f"speed_threshold must be non-negative, got {speed_threshold}")
    dot = torch.sum(velocity_xy * direction_xy, dim=-1)
    cross = velocity_xy[..., 0] * direction_xy[..., 1] - velocity_xy[..., 1] * direction_xy[..., 0]
    error = torch.atan2(cross.abs(), dot)
    speed = torch.linalg.vector_norm(velocity_xy, dim=-1)
    return torch.where(speed >= speed_threshold, error, torch.full_like(error, math.pi / 2.0))
