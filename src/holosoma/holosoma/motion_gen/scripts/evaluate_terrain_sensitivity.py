"""Evaluate generator sensitivity to flat, 0.30 m, and 0.60 m obstacles.

All conditions reuse the same motion history, world heading, and deterministic
DDIM seed.  By default the world heading is derived from the final history
frame's anchor yaw, so it points along the same anchor-local +x axis as the
obstacle.  Each condition is sampled in a separate size-one call so resetting
the generator to the same seed produces byte-identical initial diffusion noise
without changing the public inference API.

The default run records observations and exits successfully even when motion
does not change meaningfully or the link-origin penetration proxy becomes
worse.  Assertions are strictly opt-in through the ``require_*`` flags.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from holosoma.motion_gen.dataset import load_wbt_motion
from holosoma.motion_gen.features import (
    DEFAULT_BODY_NAMES,
    DEFAULT_JOINT_NAMES,
    quat_normalize,
    quat_yaw,
    unpack_features,
)
from holosoma.motion_gen.sampling import MotionGenerator, MotionGeneratorInput
from holosoma.motion_gen.terrain import ScanGrid

_LEFT_FOOT = "left_ankle_roll_link"
_RIGHT_FOOT = "right_ankle_roll_link"
_LEFT_KNEE = "left_knee_joint"
_RIGHT_KNEE = "right_knee_joint"

_EXPECTED_MONOTONIC_DIRECTIONS = {
    "root_z_max": "nondecreasing",
    "root_z_mean": "nondecreasing",
    "root_z_end": "nondecreasing",
    "left_foot_z_max": "nondecreasing",
    "right_foot_z_max": "nondecreasing",
    "left_foot_clearance_max": "nondecreasing",
    "right_foot_clearance_max": "nondecreasing",
    "left_knee_flexion_max": "nondecreasing",
    "left_knee_flexion_mean": "nondecreasing",
    "right_knee_flexion_max": "nondecreasing",
    "right_knee_flexion_mean": "nondecreasing",
    "body_origin_penetration_proxy_max": "nonincreasing",
    "body_origin_penetration_proxy_mean": "nonincreasing",
    "body_origin_penetration_proxy_rate": "nonincreasing",
    "foot_origin_penetration_proxy_max": "nonincreasing",
    "foot_origin_penetration_proxy_mean": "nonincreasing",
    "foot_origin_penetration_proxy_rate": "nonincreasing",
}

_MEANINGFUL_METRIC_GROUPS = {
    "root_height_m": ("root_z_max", "root_z_mean", "root_z_end"),
    "foot_height_m": ("left_foot_z_max", "right_foot_z_max"),
    "knee_flexion_rad": (
        "left_knee_flexion_max",
        "left_knee_flexion_mean",
        "right_knee_flexion_max",
        "right_knee_flexion_mean",
    ),
    "forward_displacement_m": ("forward_displacement",),
}


@dataclass(frozen=True)
class Args:
    checkpoint: str = "logs/motion_gen/terrain_4090/checkpoints/final.pt"
    clip: str = "data/motion_gen/processed_paperscale/lafan1_walk4_subject1.npz"
    start: int = 5750
    seed: int = 123
    device: str = "cuda:0"
    num_steps: int = 2
    heading_x: float | None = None
    heading_y: float | None = None
    obstacle_heights: tuple[float, float, float] = (0.0, 0.30, 0.60)
    obstacle_x_min: float = 0.30
    obstacle_x_max: float = 0.90
    obstacle_y_min: float = -0.40
    obstacle_y_max: float = 0.40
    origin_penetration_proxy_threshold: float = 0.005
    meaningful_root_height_delta_m: float = 0.01
    meaningful_foot_height_delta_m: float = 0.01
    meaningful_knee_flexion_delta_rad: float = 0.05
    meaningful_forward_displacement_delta_m: float = 0.02
    monotonic_tolerance: float = 1.0e-6
    guidance_scale: float | None = None
    output_dir: str = "logs/motion_gen/terrain_sensitivity"
    require_meaningful: bool = False
    require_nonincreasing_origin_penetration_proxy: bool = False


def build_rectangular_obstacle_scans(
    grid: ScanGrid,
    heights: SequenceFloat,
    bounds: tuple[float, float, float, float],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(conditions, grid.dim)`` scans and the rectangular point mask."""
    height_values = tuple(float(value) for value in heights)
    if not height_values or any(value < 0.0 for value in height_values):
        raise ValueError("Obstacle heights must be a non-empty sequence of non-negative values.")
    x_min, x_max, y_min, y_max = (float(value) for value in bounds)
    if x_min > x_max or y_min > y_max:
        raise ValueError(f"Invalid obstacle bounds {bounds}.")
    if x_min < grid.x_min or x_max > grid.x_max or y_min < grid.y_min or y_max > grid.y_max:
        raise ValueError(f"Obstacle bounds {bounds} must lie inside the scan grid extents.")

    offsets = grid.offsets_tensor(device=device, dtype=dtype)
    comparison_tolerance = max(grid.spacing * 1.0e-6, torch.finfo(dtype).eps * 8.0)
    obstacle_mask = (
        (offsets[:, 0] >= x_min - comparison_tolerance)
        & (offsets[:, 0] <= x_max + comparison_tolerance)
        & (offsets[:, 1] >= y_min - comparison_tolerance)
        & (offsets[:, 1] <= y_max + comparison_tolerance)
    )
    if not bool(torch.any(obstacle_mask)):
        raise ValueError("Obstacle bounds contain no scan samples.")
    scans = torch.zeros(len(height_values), grid.dim, device=device, dtype=dtype)
    scans[:, obstacle_mask] = torch.tensor(height_values, device=device, dtype=dtype).unsqueeze(-1)
    return scans, obstacle_mask


def bilinear_sample_height(
    terrain_scan: torch.Tensor,
    local_xy: torch.Tensor,
    grid: ScanGrid,
) -> torch.Tensor:
    """Sample one x-major/y-fastest scan at arbitrary local XY coordinates.

    Points outside the finite scan use flat ground height zero.
    """
    if terrain_scan.ndim != 1 or terrain_scan.shape[0] != grid.dim:
        raise ValueError(f"terrain_scan must have shape ({grid.dim},), got {tuple(terrain_scan.shape)}.")
    if local_xy.shape[-1:] != (2,):
        raise ValueError(f"local_xy must have shape (..., 2), got {tuple(local_xy.shape)}.")
    if terrain_scan.device != local_xy.device or terrain_scan.dtype != local_xy.dtype:
        raise TypeError("terrain_scan and local_xy must share floating dtype and device.")
    if not terrain_scan.is_floating_point():
        raise TypeError("terrain_scan and local_xy must be floating point.")

    fx = (local_xy[..., 0] - grid.x_min) / grid.spacing
    fy = (local_xy[..., 1] - grid.y_min) / grid.spacing
    inside = (fx >= 0.0) & (fx <= grid.nx - 1) & (fy >= 0.0) & (fy <= grid.ny - 1)
    ix0 = torch.floor(fx).to(torch.long).clamp(0, grid.nx - 1)
    iy0 = torch.floor(fy).to(torch.long).clamp(0, grid.ny - 1)
    ix1 = (ix0 + 1).clamp(max=grid.nx - 1)
    iy1 = (iy0 + 1).clamp(max=grid.ny - 1)
    wx = (fx - ix0.to(fx.dtype)).clamp(0.0, 1.0)
    wy = (fy - iy0.to(fy.dtype)).clamp(0.0, 1.0)

    z00 = terrain_scan[ix0 * grid.ny + iy0]
    z01 = terrain_scan[ix0 * grid.ny + iy1]
    z10 = terrain_scan[ix1 * grid.ny + iy0]
    z11 = terrain_scan[ix1 * grid.ny + iy1]
    interpolated = z00 * (1.0 - wx) * (1.0 - wy) + z01 * (1.0 - wx) * wy + z10 * wx * (1.0 - wy) + z11 * wx * wy
    return torch.where(inside, interpolated, torch.zeros_like(interpolated))


def world_xy_to_scan_local(
    world_xy: torch.Tensor,
    anchor_position: torch.Tensor,
    anchor_quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Convert world XY points to the anchor-yaw-aligned scan coordinates."""
    if world_xy.shape[-1:] != (2,) or anchor_position.shape != (3,) or anchor_quaternion_wxyz.shape != (4,):
        raise ValueError("Expected world_xy (...,2), anchor_position (3,), and anchor quaternion (4,).")
    if world_xy.device != anchor_position.device or world_xy.device != anchor_quaternion_wxyz.device:
        raise TypeError("World points and anchor pose must share a device.")
    if world_xy.dtype != anchor_position.dtype or world_xy.dtype != anchor_quaternion_wxyz.dtype:
        raise TypeError("World points and anchor pose must share a dtype.")
    yaw = quat_yaw(quat_normalize(anchor_quaternion_wxyz))
    delta = world_xy - anchor_position[:2]
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        [c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]],
        dim=-1,
    )


def resolve_world_heading(
    anchor_quaternion_wxyz: torch.Tensor,
    heading_x: float | None,
    heading_y: float | None,
) -> tuple[torch.Tensor, str]:
    """Resolve a world heading, defaulting to the scan anchor's local +x axis."""
    if anchor_quaternion_wxyz.shape != (4,):
        raise ValueError("anchor_quaternion_wxyz must have shape (4,).")
    if not anchor_quaternion_wxyz.is_floating_point():
        raise TypeError("anchor_quaternion_wxyz must be floating point.")
    if (heading_x is None) != (heading_y is None):
        raise ValueError("heading_x and heading_y must either both be omitted or both be specified.")
    if heading_x is None:
        yaw = quat_yaw(quat_normalize(anchor_quaternion_wxyz))
        heading = torch.stack((torch.cos(yaw), torch.sin(yaw)))
        source = "anchor_yaw_forward"
    else:
        heading = anchor_quaternion_wxyz.new_tensor([heading_x, heading_y])
        source = "explicit_world_xy"
    norm = torch.linalg.vector_norm(heading)
    if not bool(norm > 0.0):
        raise ValueError("World heading must be non-zero.")
    return heading / norm, source


def compute_motion_metrics(
    *,
    root_position: torch.Tensor,
    joint_position: torch.Tensor,
    body_position: torch.Tensor,
    terrain_scan: torch.Tensor,
    grid: ScanGrid,
    anchor_position: torch.Tensor,
    anchor_quaternion_wxyz: torch.Tensor,
    world_heading: torch.Tensor,
    joint_names: tuple[str, ...] = tuple(DEFAULT_JOINT_NAMES),
    body_names: tuple[str, ...] = tuple(DEFAULT_BODY_NAMES),
    origin_penetration_proxy_threshold: float = 0.005,
) -> dict[str, float]:
    """Compute root, foot, knee, link-origin penetration proxy, and travel metrics."""
    if joint_names != tuple(DEFAULT_JOINT_NAMES):
        raise ValueError("Metrics require the exact G1 29-DoF dataset joint order.")
    if body_names != tuple(DEFAULT_BODY_NAMES):
        raise ValueError("Metrics require the exact 14-body motion-generator order.")
    if root_position.ndim != 2 or root_position.shape[-1] != 3:
        raise ValueError(f"root_position must be (T,3), got {tuple(root_position.shape)}.")
    frames = root_position.shape[0]
    if joint_position.shape != (frames, len(joint_names)):
        raise ValueError(f"joint_position must be ({frames},{len(joint_names)}).")
    if body_position.shape != (frames, len(body_names), 3):
        raise ValueError(f"body_position must be ({frames},{len(body_names)},3).")
    if frames < 1:
        raise ValueError("At least one generated frame is required.")
    if origin_penetration_proxy_threshold < 0.0:
        raise ValueError("origin_penetration_proxy_threshold must be non-negative.")
    tensors = (root_position, joint_position, body_position, terrain_scan, anchor_position, anchor_quaternion_wxyz)
    if any(value.device != root_position.device or value.dtype != root_position.dtype for value in tensors):
        raise TypeError("All metric tensors must share dtype and device.")
    if world_heading.shape != (2,) or world_heading.device != root_position.device:
        raise ValueError("world_heading must have shape (2,) on the motion device.")
    heading = world_heading.to(dtype=root_position.dtype)
    heading_norm = torch.linalg.vector_norm(heading)
    if not bool(heading_norm > 0.0):
        raise ValueError("world_heading must be non-zero.")
    heading = heading / heading_norm

    local_body_xy = world_xy_to_scan_local(body_position[..., :2], anchor_position, anchor_quaternion_wxyz)
    terrain_at_body = bilinear_sample_height(terrain_scan, local_body_xy, grid)
    body_origin_penetration_proxy = (terrain_at_body - body_position[..., 2]).clamp_min(0.0)

    left_foot_index = body_names.index(_LEFT_FOOT)
    right_foot_index = body_names.index(_RIGHT_FOOT)
    foot_indices = (left_foot_index, right_foot_index)
    feet = body_position[:, foot_indices]
    terrain_at_feet = terrain_at_body[:, foot_indices]
    foot_clearance = feet[..., 2] - terrain_at_feet
    foot_origin_penetration_proxy = body_origin_penetration_proxy[:, foot_indices]

    def origin_penetration_proxy_metrics(prefix: str, values: torch.Tensor) -> dict[str, float]:
        return {
            f"{prefix}_origin_penetration_proxy_max": float(values.max()),
            f"{prefix}_origin_penetration_proxy_mean": float(values.mean()),
            f"{prefix}_origin_penetration_proxy_rate": float(
                (values > origin_penetration_proxy_threshold).to(values.dtype).mean()
            ),
        }

    metrics = {
        "root_z_max": float(root_position[:, 2].max()),
        "root_z_mean": float(root_position[:, 2].mean()),
        "root_z_end": float(root_position[-1, 2]),
        "forward_displacement": float(torch.dot(root_position[-1, :2] - anchor_position[:2], heading)),
    }
    knee_indices = (joint_names.index(_LEFT_KNEE), joint_names.index(_RIGHT_KNEE))
    for side, foot_index, knee_index in zip(("left", "right"), range(2), knee_indices):
        metrics.update(
            {
                f"{side}_foot_z_max": float(feet[:, foot_index, 2].max()),
                f"{side}_foot_clearance_max": float(foot_clearance[:, foot_index].max()),
                f"{side}_foot_clearance_mean": float(foot_clearance[:, foot_index].mean()),
                f"{side}_foot_clearance_min": float(foot_clearance[:, foot_index].min()),
                f"{side}_knee_flexion_max": float(joint_position[:, knee_index].max()),
                f"{side}_knee_flexion_mean": float(joint_position[:, knee_index].mean()),
            }
        )
        metrics.update(
            origin_penetration_proxy_metrics(
                f"{side}_foot",
                foot_origin_penetration_proxy[:, foot_index],
            )
        )
    metrics.update(origin_penetration_proxy_metrics("body", body_origin_penetration_proxy))
    metrics.update(origin_penetration_proxy_metrics("foot", foot_origin_penetration_proxy))
    return metrics


def summarize_sensitivity(
    heights: SequenceFloat,
    metrics_by_height: list[dict[str, float]],
    *,
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Report deltas plus both observed and expected monotonicity for each metric."""
    height_values = [float(value) for value in heights]
    if len(height_values) != len(metrics_by_height) or not height_values:
        raise ValueError("heights and metrics_by_height must have the same non-zero length.")
    if any(next_height <= height for height, next_height in zip(height_values, height_values[1:])):
        raise ValueError("Obstacle heights must be strictly increasing.")
    keys = tuple(metrics_by_height[0])
    if any(tuple(metrics) != keys for metrics in metrics_by_height):
        raise ValueError("Every condition must contain metrics in the same order.")
    if tolerance < 0.0:
        raise ValueError("Monotonic tolerance must be non-negative.")

    metric_report: dict[str, Any] = {}
    expected_results = []
    for key in keys:
        values = [float(metrics[key]) for metrics in metrics_by_height]
        adjacent = [next_value - value for value, next_value in zip(values, values[1:])]
        nondecreasing = all(delta >= -tolerance for delta in adjacent)
        nonincreasing = all(delta <= tolerance for delta in adjacent)
        expected = _EXPECTED_MONOTONIC_DIRECTIONS.get(key)
        expected_monotonic = None
        if expected is not None:
            expected_monotonic = nondecreasing if expected == "nondecreasing" else nonincreasing
            expected_results.append(expected_monotonic)
        metric_report[key] = {
            "values": values,
            "delta_vs_flat": [value - values[0] for value in values],
            "adjacent_deltas": adjacent,
            "nondecreasing": nondecreasing,
            "nonincreasing": nonincreasing,
            "expected_direction": expected,
            "expected_monotonic": expected_monotonic,
        }

    penetration_keys = [key for key in keys if "penetration" in key]
    penetration_increased = any(
        metrics_by_height[index][key] > metrics_by_height[0][key] + tolerance
        for key in penetration_keys
        for index in range(1, len(metrics_by_height))
    )
    return {
        "heights": height_values,
        "metrics": metric_report,
        "all_expected_monotonic": all(expected_results),
        "origin_penetration_proxy_increased_vs_flat": penetration_increased,
    }


def evaluate_meaningful_physical_response(
    metrics_by_height: list[dict[str, float]],
    *,
    root_height_delta_m: float,
    foot_height_delta_m: float,
    knee_flexion_delta_rad: float,
    forward_displacement_delta_m: float,
) -> dict[str, Any]:
    """Evaluate response using interpretable physical deltas, never packed features."""
    if len(metrics_by_height) < 2:
        raise ValueError("At least flat and one obstacle metric set are required.")
    thresholds = {
        "root_height_m": float(root_height_delta_m),
        "foot_height_m": float(foot_height_delta_m),
        "knee_flexion_rad": float(knee_flexion_delta_rad),
        "forward_displacement_m": float(forward_displacement_delta_m),
    }
    if any(value < 0.0 for value in thresholds.values()):
        raise ValueError("Meaningful-response thresholds must be non-negative.")

    flat = metrics_by_height[0]
    group_reports: dict[str, Any] = {}
    for group, metric_names in _MEANINGFUL_METRIC_GROUPS.items():
        missing = [name for name in metric_names if any(name not in metrics for metrics in metrics_by_height)]
        if missing:
            raise ValueError(f"Metrics required by {group} are missing: {missing}.")
        deltas_by_metric = {
            name: [float(metrics[name] - flat[name]) for metrics in metrics_by_height[1:]] for name in metric_names
        }
        max_abs_delta = max(abs(delta) for deltas in deltas_by_metric.values() for delta in deltas)
        threshold = thresholds[group]
        group_reports[group] = {
            "metric_names": list(metric_names),
            "delta_vs_flat": deltas_by_metric,
            "max_abs_delta": max_abs_delta,
            "threshold": threshold,
            "exceeded": max_abs_delta > threshold,
        }

    return {
        "criterion": "any physical metric group max absolute delta vs flat exceeds its configured threshold",
        "meaningful": any(report["exceeded"] for report in group_reports.values()),
        "groups": group_reports,
    }


SequenceFloat = tuple[float, ...] | list[float]


@torch.no_grad()
def main(args: Args) -> None:
    if args.start < 0:
        raise ValueError("start must be non-negative.")
    if args.num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if len(args.obstacle_heights) < 2 or args.obstacle_heights[0] != 0.0:
        raise ValueError("obstacle_heights must start with flat height 0.0 and include an obstacle.")
    if any(next_height <= height for height, next_height in zip(args.obstacle_heights, args.obstacle_heights[1:])):
        raise ValueError("obstacle_heights must be strictly increasing.")
    generator = MotionGenerator.from_checkpoint(args.checkpoint, device=args.device)
    if tuple(generator.layout.joint_names) != tuple(DEFAULT_JOINT_NAMES):
        raise ValueError("Checkpoint joint mapping does not match the G1 29-DoF dataset order.")
    if tuple(generator.layout.body_names) != tuple(DEFAULT_BODY_NAMES):
        raise ValueError("Checkpoint body mapping does not match the 14-body generator order.")
    if not generator.cfg.data.use_terrain_scan:
        raise ValueError("Checkpoint was not trained with terrain scans.")
    grid = generator.cfg.data.scan_grid
    if generator.cfg.data.terrain_dim != grid.dim:
        raise ValueError(f"Checkpoint terrain_dim={generator.cfg.data.terrain_dim} but scan grid dim={grid.dim}.")

    clip = load_wbt_motion(args.clip, generator.layout, expected_fps=generator.cfg.data.fps)
    past_frames = generator.cfg.data.past_frames
    if args.start + past_frames > clip.num_frames:
        raise ValueError(
            f"Clip has {clip.num_frames} frames but start={args.start} requires {past_frames} history frames."
        )
    past = clip.features[args.start : args.start + past_frames].unsqueeze(0).to(generator.device)
    past_parts = unpack_features(past[0], generator.layout)
    anchor_position = past_parts["root_pos"][-1]
    anchor_quaternion = past_parts["root_quat"][-1]
    heading, heading_source = resolve_world_heading(
        anchor_quaternion,
        args.heading_x,
        args.heading_y,
    )

    bounds = (args.obstacle_x_min, args.obstacle_x_max, args.obstacle_y_min, args.obstacle_y_max)
    terrain_scans, obstacle_mask = build_rectangular_obstacle_scans(
        grid,
        args.obstacle_heights,
        bounds,
        device=generator.device,
        dtype=past.dtype,
    )
    outputs = [
        generator.generate(
            MotionGeneratorInput(
                past_motion=past,
                target_heading=heading.unsqueeze(0),
                terrain_height=terrain_scan.unsqueeze(0),
            ),
            num_steps=args.num_steps,
            deterministic=True,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
        )
        for terrain_scan in terrain_scans
    ]

    metrics_by_height = [
        compute_motion_metrics(
            root_position=output.root_pos[0],
            joint_position=output.joint_pos[0],
            body_position=output.body_pos[0],
            terrain_scan=terrain_scans[index],
            grid=grid,
            anchor_position=anchor_position,
            anchor_quaternion_wxyz=anchor_quaternion,
            world_heading=heading,
            joint_names=tuple(generator.layout.joint_names),
            body_names=tuple(generator.layout.body_names),
            origin_penetration_proxy_threshold=args.origin_penetration_proxy_threshold,
        )
        for index, output in enumerate(outputs)
    ]
    sensitivity = summarize_sensitivity(
        args.obstacle_heights,
        metrics_by_height,
        tolerance=args.monotonic_tolerance,
    )

    flat_features = outputs[0].features[0]
    feature_deltas = []
    for output in outputs:
        delta = (output.features[0] - flat_features).abs()
        feature_deltas.append(
            {
                "mean_abs_delta_vs_flat": float(delta.mean()),
                "max_abs_delta_vs_flat": float(delta.max()),
            }
        )
    meaningful_response = evaluate_meaningful_physical_response(
        metrics_by_height,
        root_height_delta_m=args.meaningful_root_height_delta_m,
        foot_height_delta_m=args.meaningful_foot_height_delta_m,
        knee_flexion_delta_rad=args.meaningful_knee_flexion_delta_rad,
        forward_displacement_delta_m=args.meaningful_forward_displacement_delta_m,
    )

    labels = ["flat" if height == 0.0 else f"obstacle_{height:.2f}m" for height in args.obstacle_heights]
    conditions = [
        {
            "label": label,
            "height": float(height),
            "metrics": metrics,
            "packed_feature_delta_diagnostic": feature_delta,
        }
        for label, height, metrics, feature_delta in zip(
            labels,
            args.obstacle_heights,
            metrics_by_height,
            feature_deltas,
        )
    ]
    report = {
        "config": asdict(args),
        "checkpoint_step": generator.checkpoint_step,
        "fps": generator.cfg.data.fps,
        "future_frames": generator.cfg.data.future_frames,
        "layout": generator.layout.to_metadata(),
        "scan": {
            "grid": grid.to_array().tolist(),
            "flatten_order": "x-major,y-fastest",
            "height_units": "absolute-world-z-metres",
            "obstacle_bounds_local_xy": list(bounds),
            "obstacle_point_count": int(obstacle_mask.sum()),
            "obstacle_axis": "anchor-local +x",
        },
        "conditioning": {
            "world_heading": heading.tolist(),
            "heading_source": heading_source,
            "heading_alignment": "anchor-local +x" if heading_source == "anchor_yaw_forward" else "explicit world XY",
        },
        "sampling": {
            "num_steps": args.num_steps,
            "deterministic": True,
            "seed": args.seed,
            "shared_initial_noise": True,
            "method": "separate_equal-shape calls with identically reset seed",
        },
        "conditions": conditions,
        "sensitivity": sensitivity,
        "meaningful_physical_response": meaningful_response,
        "metric_semantics": {
            "origin_penetration_proxy": (
                "max(terrain height at a predicted link origin - predicted link-origin world z, 0); "
                "this is not collision-shape or mesh penetration"
            ),
            "origin_penetration_proxy_threshold_m": args.origin_penetration_proxy_threshold,
            "packed_feature_delta_diagnostic": "diagnostic only; never used for meaningful-response classification",
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"terrain_sensitivity_s{args.start}_seed{args.seed}"
    json_path = output_dir / f"{stem}.json"
    npz_path = output_dir / f"{stem}.npz"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    np.savez_compressed(
        npz_path,
        obstacle_heights=np.asarray(args.obstacle_heights, dtype=np.float32),
        obstacle_bounds=np.asarray(bounds, dtype=np.float32),
        obstacle_mask=obstacle_mask.cpu().numpy(),
        scan_local_xy=grid.offsets(),
        terrain_scans=terrain_scans.cpu().numpy(),
        past_motion=past.cpu().numpy(),
        world_heading=heading.cpu().numpy(),
        features=torch.stack([output.features[0] for output in outputs]).cpu().numpy(),
        root_position=torch.stack([output.root_pos[0] for output in outputs]).cpu().numpy(),
        joint_position=torch.stack([output.joint_pos[0] for output in outputs]).cpu().numpy(),
        body_position=torch.stack([output.body_pos[0] for output in outputs]).cpu().numpy(),
        metrics_json=np.asarray(json.dumps(report)),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"JSON: {json_path}")
    print(f"NPZ: {npz_path}")

    if args.require_meaningful and not meaningful_response["meaningful"]:
        raise AssertionError("No physical response metric exceeded its configured opt-in threshold.")
    if (
        args.require_nonincreasing_origin_penetration_proxy
        and sensitivity["origin_penetration_proxy_increased_vs_flat"]
    ):
        raise AssertionError("Link-origin penetration proxy increased relative to flat under an opt-in strict check.")


if __name__ == "__main__":
    main(tyro.cli(Args))
