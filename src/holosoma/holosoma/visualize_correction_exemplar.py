"""Validate and visualize a Stage-10 tracker-correction exemplar.

The terrain evaluator writes ``<prefix>_first_correction_exemplar.npz`` when
the robot has a smaller body-origin terrain-height proxy than its reference.
This module deliberately treats that value as a point-origin diagnostic, not
as collision geometry or evidence of policy intent.

Example:
    python -m holosoma.visualize_correction_exemplar \
        logs/eval/d_first_correction_exemplar.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ATOL = 1.0e-5
_RTOL = 1.0e-5
_PROXY_WATERMARK = "BODY-ORIGIN PROXY, NOT COLLISION GEOMETRY"

_REQUIRED_KEYS = frozenset(
    {
        "action",
        "action_semantics",
        "body_names",
        "body_position_frame_units",
        "correction_proxy_case",
        "env_id",
        "episode_step",
        "evaluation_step",
        "local_scan_configured",
        "local_scan_flatten_order",
        "local_scan_grid",
        "local_scan_height_units",
        "local_scan_height_w",
        "local_scan_local_xy",
        "local_scan_root_xy_w",
        "local_scan_root_yaw_w",
        "local_scan_valid",
        "local_scan_world_xy",
        "minimum_improvement_threshold_m",
        "proxy_limitation",
        "reference_body_origin_penetration_m",
        "reference_body_pos_w",
        "reference_body_terrain_height_w",
        "reference_max_body_origin_penetration_m",
        "reference_minus_robot_max_body_origin_penetration_m",
        "reference_penetration_threshold_m",
        "robot_body_origin_penetration_m",
        "robot_body_pos_w",
        "robot_body_terrain_height_w",
        "robot_max_body_origin_penetration_m",
        "root_state_layout",
        "root_state_w",
        "target_heading_w",
        "terrain_level",
        "terrain_type",
    }
)


class CorrectionExemplarError(ValueError):
    """Raised when an exemplar is unsafe to interpret or internally inconsistent."""


@dataclass(frozen=True)
class ValidatedCorrectionExemplar:
    """Validated, pickle-free exemplar arrays and derived scan dimensions."""

    arrays: dict[str, np.ndarray]
    body_names: tuple[str, ...]
    scan_shape: tuple[int, int] | None

    @property
    def body_count(self) -> int:
        return len(self.body_names)

    @property
    def scan_configured(self) -> bool:
        return _scalar_bool(self.arrays, "local_scan_configured")

    @property
    def scan_valid(self) -> bool:
        return _scalar_bool(self.arrays, "local_scan_valid")


def _require_shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if array.shape != shape:
        raise CorrectionExemplarError(f"{name} must have shape {shape}, got {array.shape}")


def _require_numeric(array: np.ndarray, name: str, *, finite: bool = True) -> np.ndarray:
    if array.dtype.kind not in "fiu":
        raise CorrectionExemplarError(f"{name} must be a real numeric array, got dtype {array.dtype}")
    converted = np.asarray(array, dtype=np.float64)
    if finite and not np.isfinite(converted).all():
        raise CorrectionExemplarError(f"{name} contains NaN or infinity")
    return converted


def _scalar_string(arrays: dict[str, np.ndarray], name: str) -> str:
    array = arrays[name]
    _require_shape(array, (), name)
    if array.dtype.kind not in "US":
        raise CorrectionExemplarError(f"{name} must be a scalar string, got dtype {array.dtype}")
    value = array.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _scalar_bool(arrays: dict[str, np.ndarray], name: str) -> bool:
    array = arrays[name]
    _require_shape(array, (), name)
    if array.dtype.kind != "b":
        raise CorrectionExemplarError(f"{name} must be a scalar boolean, got dtype {array.dtype}")
    return bool(array.item())


def _scalar_int(arrays: dict[str, np.ndarray], name: str) -> int:
    array = arrays[name]
    _require_shape(array, (), name)
    if array.dtype.kind not in "iu":
        raise CorrectionExemplarError(f"{name} must be a scalar integer, got dtype {array.dtype}")
    return int(array.item())


def _scalar_float(arrays: dict[str, np.ndarray], name: str) -> float:
    array = arrays[name]
    _require_shape(array, (), name)
    value = float(_require_numeric(array, name).item())
    if not math.isfinite(value):
        raise CorrectionExemplarError(f"{name} must be finite")
    return value


def _assert_close(actual: np.ndarray | float, expected: np.ndarray | float, name: str) -> None:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if not np.allclose(actual_array, expected_array, atol=_ATOL, rtol=_RTOL):
        max_error = float(np.max(np.abs(actual_array - expected_array)))
        raise CorrectionExemplarError(f"{name} is inconsistent (maximum absolute error {max_error:.6g})")


def _decode_body_names(array: np.ndarray) -> tuple[str, ...]:
    if array.ndim != 1 or array.dtype.kind not in "US":
        raise CorrectionExemplarError(
            f"body_names must be a one-dimensional string array, got shape {array.shape}, dtype {array.dtype}"
        )
    names = tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in array.tolist()
    )
    if not names or any(not name.strip() for name in names):
        raise CorrectionExemplarError("body_names must contain at least one non-empty name")
    if len(set(names)) != len(names):
        raise CorrectionExemplarError("body_names must be unique")
    return names


def _validate_scan(arrays: dict[str, np.ndarray]) -> tuple[int, int] | None:
    configured = _scalar_bool(arrays, "local_scan_configured")
    valid = _scalar_bool(arrays, "local_scan_valid")
    flatten_order = _scalar_string(arrays, "local_scan_flatten_order")
    height_units = _scalar_string(arrays, "local_scan_height_units")
    if flatten_order != "x-major,y-fastest":
        raise CorrectionExemplarError(
            "local_scan_flatten_order must be 'x-major,y-fastest', "
            f"got {flatten_order!r}"
        )
    if height_units != "absolute-world-z-metres":
        raise CorrectionExemplarError(
            "local_scan_height_units must be 'absolute-world-z-metres', "
            f"got {height_units!r}"
        )

    if not configured:
        if valid:
            raise CorrectionExemplarError("local_scan_valid cannot be true when local_scan_configured is false")
        expected_empty_shapes = {
            "local_scan_height_w": (0,),
            "local_scan_root_xy_w": (0,),
            "local_scan_local_xy": (0, 2),
            "local_scan_world_xy": (0, 2),
            "local_scan_grid": (0,),
        }
        for name, shape in expected_empty_shapes.items():
            numeric = _require_numeric(arrays[name], name)
            _require_shape(numeric, shape, name)
        root_yaw = _require_numeric(
            arrays["local_scan_root_yaw_w"],
            "local_scan_root_yaw_w",
            finite=False,
        )
        _require_shape(root_yaw, (), "local_scan_root_yaw_w")
        if not np.isnan(root_yaw.item()):
            raise CorrectionExemplarError(
                "local_scan_root_yaw_w must be NaN when local_scan_configured is false"
            )
        return None

    grid_array = _require_numeric(arrays["local_scan_grid"], "local_scan_grid")
    _require_shape(grid_array, (5,), "local_scan_grid")
    x_min, x_max, y_min, y_max, spacing = (float(value) for value in grid_array)
    if spacing <= 0.0 or x_max < x_min or y_max < y_min:
        raise CorrectionExemplarError(
            "local_scan_grid must have increasing extents and positive spacing, "
            f"got {grid_array.tolist()}"
        )
    nx_steps = (x_max - x_min) / spacing
    ny_steps = (y_max - y_min) / spacing
    if not math.isclose(nx_steps, round(nx_steps), abs_tol=_ATOL) or not math.isclose(
        ny_steps, round(ny_steps), abs_tol=_ATOL
    ):
        raise CorrectionExemplarError("local_scan_grid extents must be integer multiples of spacing")
    nx = round(nx_steps) + 1
    ny = round(ny_steps) + 1
    scan_dim = nx * ny

    scan_height = _require_numeric(arrays["local_scan_height_w"], "local_scan_height_w")
    root_xy = _require_numeric(arrays["local_scan_root_xy_w"], "local_scan_root_xy_w")
    root_yaw = _require_numeric(arrays["local_scan_root_yaw_w"], "local_scan_root_yaw_w")
    local_xy = _require_numeric(arrays["local_scan_local_xy"], "local_scan_local_xy")
    world_xy = _require_numeric(arrays["local_scan_world_xy"], "local_scan_world_xy")
    _require_shape(scan_height, (scan_dim,), "local_scan_height_w")
    _require_shape(root_xy, (2,), "local_scan_root_xy_w")
    _require_shape(root_yaw, (), "local_scan_root_yaw_w")
    _require_shape(local_xy, (scan_dim, 2), "local_scan_local_xy")
    _require_shape(world_xy, (scan_dim, 2), "local_scan_world_xy")

    xs = x_min + spacing * np.arange(nx, dtype=np.float64)
    ys = y_min + spacing * np.arange(ny, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    expected_local = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)
    _assert_close(local_xy, expected_local, "local_scan_local_xy flatten order")
    yaw = float(root_yaw.item())
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    expected_world = np.empty_like(expected_local)
    expected_world[:, 0] = root_xy[0] + cosine * expected_local[:, 0] - sine * expected_local[:, 1]
    expected_world[:, 1] = root_xy[1] + sine * expected_local[:, 0] + cosine * expected_local[:, 1]
    _assert_close(world_xy, expected_world, "local_scan_world_xy transform")
    return nx, ny


def validate_correction_exemplar(arrays: dict[str, np.ndarray]) -> ValidatedCorrectionExemplar:
    """Validate array shapes, proxy arithmetic, thresholds, units, and scan order."""
    missing = sorted(_REQUIRED_KEYS.difference(arrays))
    if missing:
        raise CorrectionExemplarError(f"Correction exemplar is missing required keys: {', '.join(missing)}")

    body_names = _decode_body_names(arrays["body_names"])
    body_count = len(body_names)
    position_shape = (body_count, 3)
    vector_shape = (body_count,)
    numeric: dict[str, np.ndarray] = {}
    for name in ("robot_body_pos_w", "reference_body_pos_w"):
        numeric[name] = _require_numeric(arrays[name], name)
        _require_shape(numeric[name], position_shape, name)
    for name in (
        "robot_body_terrain_height_w",
        "reference_body_terrain_height_w",
        "robot_body_origin_penetration_m",
        "reference_body_origin_penetration_m",
    ):
        numeric[name] = _require_numeric(arrays[name], name)
        _require_shape(numeric[name], vector_shape, name)

    target_heading = _require_numeric(arrays["target_heading_w"], "target_heading_w")
    action = _require_numeric(arrays["action"], "action")
    root_state = _require_numeric(arrays["root_state_w"], "root_state_w")
    _require_shape(target_heading, (2,), "target_heading_w")
    if action.ndim != 1 or action.size == 0:
        raise CorrectionExemplarError(f"action must be a non-empty vector, got shape {action.shape}")
    _require_shape(root_state, (13,), "root_state_w")
    _assert_close(np.linalg.norm(target_heading), 1.0, "target_heading_w norm")

    for name in ("env_id", "terrain_level", "evaluation_step", "episode_step"):
        _scalar_int(arrays, name)
    if _scalar_int(arrays, "env_id") < 0 or _scalar_int(arrays, "terrain_level") < 0:
        raise CorrectionExemplarError("env_id and terrain_level must be non-negative")
    if _scalar_int(arrays, "evaluation_step") < 0 or _scalar_int(arrays, "episode_step") < -1:
        raise CorrectionExemplarError("evaluation_step must be non-negative and episode_step must be at least -1")
    if not _scalar_string(arrays, "terrain_type").strip():
        raise CorrectionExemplarError("terrain_type must not be empty")
    if _scalar_string(arrays, "body_position_frame_units") != "world_xyz_metres":
        raise CorrectionExemplarError("body_position_frame_units must be 'world_xyz_metres'")
    if _scalar_string(arrays, "root_state_layout") != (
        "position_xyz,quaternion_xyzw,linear_velocity_xyz,angular_velocity_xyz"
    ):
        raise CorrectionExemplarError("root_state_layout does not match the expected 13-value simulator layout")
    if _scalar_string(arrays, "action_semantics") != "raw_policy_action_passed_to_environment_step":
        raise CorrectionExemplarError("action_semantics does not identify the raw policy action")
    limitation = _scalar_string(arrays, "proxy_limitation").lower()
    if "body-origin" not in limitation or "not collision-shape" not in limitation:
        raise CorrectionExemplarError("proxy_limitation must state that this is not collision-shape penetration")

    robot_expected = np.maximum(
        numeric["robot_body_terrain_height_w"] - numeric["robot_body_pos_w"][:, 2],
        0.0,
    )
    reference_expected = np.maximum(
        numeric["reference_body_terrain_height_w"] - numeric["reference_body_pos_w"][:, 2],
        0.0,
    )
    robot_proxy = numeric["robot_body_origin_penetration_m"]
    reference_proxy = numeric["reference_body_origin_penetration_m"]
    if np.any(robot_proxy < -_ATOL) or np.any(reference_proxy < -_ATOL):
        raise CorrectionExemplarError("body-origin penetration proxies must be non-negative")
    _assert_close(robot_proxy, robot_expected, "robot_body_origin_penetration_m")
    _assert_close(reference_proxy, reference_expected, "reference_body_origin_penetration_m")

    robot_max = _scalar_float(arrays, "robot_max_body_origin_penetration_m")
    reference_max = _scalar_float(arrays, "reference_max_body_origin_penetration_m")
    signed_improvement = _scalar_float(
        arrays,
        "reference_minus_robot_max_body_origin_penetration_m",
    )
    _assert_close(robot_max, float(robot_proxy.max()), "robot_max_body_origin_penetration_m")
    _assert_close(reference_max, float(reference_proxy.max()), "reference_max_body_origin_penetration_m")
    _assert_close(
        signed_improvement,
        reference_max - robot_max,
        "reference_minus_robot_max_body_origin_penetration_m",
    )
    reference_threshold = _scalar_float(arrays, "reference_penetration_threshold_m")
    improvement_threshold = _scalar_float(arrays, "minimum_improvement_threshold_m")
    if reference_threshold < 0.0 or improvement_threshold < 0.0:
        raise CorrectionExemplarError("proxy thresholds must be non-negative")
    if not _scalar_bool(arrays, "correction_proxy_case"):
        raise CorrectionExemplarError("correction_proxy_case must be true for a saved correction exemplar")
    if reference_max + _ATOL < reference_threshold:
        raise CorrectionExemplarError(
            "reference maximum body-origin proxy does not meet reference_penetration_threshold_m"
        )
    if signed_improvement + _ATOL < improvement_threshold:
        raise CorrectionExemplarError(
            "signed max-proxy improvement does not meet minimum_improvement_threshold_m"
        )

    scan_shape = _validate_scan(arrays)
    return ValidatedCorrectionExemplar(arrays=arrays, body_names=body_names, scan_shape=scan_shape)


def load_correction_exemplar(path: str | Path) -> ValidatedCorrectionExemplar:
    """Load an evaluator NPZ with pickle disabled, copy it, and validate it."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Correction exemplar does not exist: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except ValueError as error:
        raise CorrectionExemplarError(
            f"Could not load {source} without pickle; object arrays are not accepted"
        ) from error
    return validate_correction_exemplar(arrays)


def _body_summary(
    exemplar: ValidatedCorrectionExemplar,
    index: int,
) -> dict[str, Any]:
    arrays = exemplar.arrays
    robot_position = np.asarray(arrays["robot_body_pos_w"], dtype=np.float64)[index]
    reference_position = np.asarray(arrays["reference_body_pos_w"], dtype=np.float64)[index]
    robot_height = float(arrays["robot_body_terrain_height_w"][index])
    reference_height = float(arrays["reference_body_terrain_height_w"][index])
    robot_proxy = float(arrays["robot_body_origin_penetration_m"][index])
    reference_proxy = float(arrays["reference_body_origin_penetration_m"][index])
    return {
        "index": index,
        "name": exemplar.body_names[index],
        "reference_origin_z_m": float(reference_position[2]),
        "reference_queried_terrain_z_m": reference_height,
        "reference_signed_clearance_m": float(reference_position[2] - reference_height),
        "reference_body_origin_penetration_proxy_m": reference_proxy,
        "robot_origin_z_m": float(robot_position[2]),
        "robot_queried_terrain_z_m": robot_height,
        "robot_signed_clearance_m": float(robot_position[2] - robot_height),
        "robot_body_origin_penetration_proxy_m": robot_proxy,
        "reference_minus_robot_same_body_proxy_m": reference_proxy - robot_proxy,
    }


def summarize_correction_exemplar(exemplar: ValidatedCorrectionExemplar) -> dict[str, Any]:
    """Build a compact JSON-serializable audit summary."""
    arrays = exemplar.arrays
    reference_proxy = np.asarray(arrays["reference_body_origin_penetration_m"], dtype=np.float64)
    robot_proxy = np.asarray(arrays["robot_body_origin_penetration_m"], dtype=np.float64)
    per_body_improvement = reference_proxy - robot_proxy
    reference_worst_index = int(np.argmax(reference_proxy))
    robot_worst_index = int(np.argmax(robot_proxy))
    largest_reduction_index = int(np.argmax(per_body_improvement))
    body_summaries = [_body_summary(exemplar, index) for index in range(exemplar.body_count)]
    scan_summary: dict[str, Any] = {
        "configured": exemplar.scan_configured,
        "valid": exemplar.scan_valid,
        "flatten_order": _scalar_string(arrays, "local_scan_flatten_order"),
        "height_units": _scalar_string(arrays, "local_scan_height_units"),
    }
    if exemplar.scan_shape is not None:
        grid = np.asarray(arrays["local_scan_grid"], dtype=np.float64)
        heights = np.asarray(arrays["local_scan_height_w"], dtype=np.float64)
        scan_summary.update(
            {
                "shape_nx_ny": list(exemplar.scan_shape),
                "sample_count": int(heights.size),
                "grid": {
                    "x_min_m": float(grid[0]),
                    "x_max_m": float(grid[1]),
                    "y_min_m": float(grid[2]),
                    "y_max_m": float(grid[3]),
                    "spacing_m": float(grid[4]),
                },
                "height_min_m": float(heights.min()),
                "height_max_m": float(heights.max()),
                "anchor_root_xy_w_m": np.asarray(arrays["local_scan_root_xy_w"], dtype=float).tolist(),
                "anchor_root_yaw_w_rad": float(arrays["local_scan_root_yaw_w"]),
            }
        )

    return {
        "schema_version": 1,
        "interpretation": (
            "Body-origin terrain-height proxy only; not collision geometry, mesh penetration, "
            "or proof of intentional policy correction."
        ),
        "frame_and_units": _scalar_string(arrays, "body_position_frame_units"),
        "selection": {
            "env_id": _scalar_int(arrays, "env_id"),
            "terrain_type": _scalar_string(arrays, "terrain_type"),
            "terrain_level": _scalar_int(arrays, "terrain_level"),
            "evaluation_step": _scalar_int(arrays, "evaluation_step"),
            "episode_step": _scalar_int(arrays, "episode_step"),
            "target_heading_w": np.asarray(arrays["target_heading_w"], dtype=float).tolist(),
        },
        "thresholds_m": {
            "reference_maximum_at_least": _scalar_float(arrays, "reference_penetration_threshold_m"),
            "signed_maximum_improvement_at_least": _scalar_float(
                arrays,
                "minimum_improvement_threshold_m",
            ),
        },
        "selection_gate_passed": True,
        "aggregate_m": {
            "reference_maximum_body_origin_penetration_proxy": _scalar_float(
                arrays,
                "reference_max_body_origin_penetration_m",
            ),
            "robot_maximum_body_origin_penetration_proxy": _scalar_float(
                arrays,
                "robot_max_body_origin_penetration_m",
            ),
            "reference_minus_robot_maximum_signed_improvement": _scalar_float(
                arrays,
                "reference_minus_robot_max_body_origin_penetration_m",
            ),
        },
        "worst_corrected_body": {
            "selection_semantics": (
                "Body with the largest reference proxy; robot values are for the same named body. "
                "This label does not imply intentional policy correction."
            ),
            **body_summaries[reference_worst_index],
        },
        "worst_reference_body": body_summaries[reference_worst_index],
        "worst_robot_body": body_summaries[robot_worst_index],
        "largest_per_body_proxy_reduction": body_summaries[largest_reduction_index],
        "bodies": body_summaries,
        "local_scan": scan_summary,
    }


def _world_to_scan_local(points_w: np.ndarray, anchor_xy: np.ndarray, yaw: float) -> np.ndarray:
    delta = np.asarray(points_w, dtype=np.float64) - np.asarray(anchor_xy, dtype=np.float64)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.stack(
        [cosine * delta[:, 0] + sine * delta[:, 1], -sine * delta[:, 0] + cosine * delta[:, 1]],
        axis=-1,
    )


def render_correction_exemplar(
    exemplar: ValidatedCorrectionExemplar,
    output_path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """Write a non-interactive PNG or SVG audit figure."""
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    destination = Path(output_path).expanduser()
    if destination.suffix.lower() not in {".png", ".svg"}:
        raise ValueError("Figure output must end in .png or .svg")

    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    arrays = exemplar.arrays
    body_count = exemplar.body_count
    y_positions = np.arange(body_count)
    reference_position = np.asarray(arrays["reference_body_pos_w"], dtype=np.float64)
    robot_position = np.asarray(arrays["robot_body_pos_w"], dtype=np.float64)
    reference_height = np.asarray(arrays["reference_body_terrain_height_w"], dtype=np.float64)
    robot_height = np.asarray(arrays["robot_body_terrain_height_w"], dtype=np.float64)
    reference_clearance = reference_position[:, 2] - reference_height
    robot_clearance = robot_position[:, 2] - robot_height
    reference_proxy = np.asarray(arrays["reference_body_origin_penetration_m"], dtype=np.float64)
    robot_proxy = np.asarray(arrays["robot_body_origin_penetration_m"], dtype=np.float64)
    per_body_reduction = reference_proxy - robot_proxy
    reference_worst_index = int(np.argmax(reference_proxy))
    largest_reduction_index = int(np.argmax(per_body_reduction))
    global_improvement = _scalar_float(
        arrays,
        "reference_minus_robot_max_body_origin_penetration_m",
    )

    show_scan = exemplar.scan_configured and exemplar.scan_valid
    row_count = 2 if show_scan else 1
    figure_height = max(7.0, 0.43 * body_count + (5.3 if show_scan else 2.0))
    figure = plt.figure(figsize=(16.5, figure_height), constrained_layout=True)
    grid_spec = figure.add_gridspec(
        row_count,
        2,
        height_ratios=[max(4.5, 0.36 * body_count), 4.5] if show_scan else None,
        width_ratios=[1.25, 1.0],
    )
    clearance_axis = figure.add_subplot(grid_spec[0, 0])
    proxy_axis = figure.add_subplot(grid_spec[0, 1], sharey=clearance_axis)
    bar_height = 0.36
    reference_color = "#d95f02"
    robot_color = "#1b75bb"

    clearance_axis.barh(
        y_positions - bar_height / 2.0,
        reference_clearance,
        height=bar_height,
        color=reference_color,
        label="reference signed clearance",
    )
    clearance_axis.barh(
        y_positions + bar_height / 2.0,
        robot_clearance,
        height=bar_height,
        color=robot_color,
        label="robot signed clearance",
    )
    clearance_axis.axvline(0.0, color="black", linewidth=1.1)
    clearance_min = min(float(reference_clearance.min()), float(robot_clearance.min()), 0.0)
    if clearance_min < 0.0:
        clearance_axis.axvspan(clearance_min * 1.08, 0.0, color="#ef5350", alpha=0.10)
    clearance_axis.set_yticks(y_positions, labels=exemplar.body_names, fontsize=8)
    clearance_axis.invert_yaxis()
    clearance_axis.set_xlabel("origin z - queried terrain z [m]  (negative = proxy penetration)")
    clearance_axis.set_title("Per-body signed clearance at each body's queried XY")
    clearance_axis.grid(axis="x", alpha=0.25)
    clearance_axis.legend(fontsize=8, loc="best")

    proxy_axis.barh(
        y_positions - bar_height / 2.0,
        reference_proxy,
        height=bar_height,
        color=reference_color,
        label="reference proxy",
    )
    proxy_axis.barh(
        y_positions + bar_height / 2.0,
        robot_proxy,
        height=bar_height,
        color=robot_color,
        label="robot proxy",
    )
    reference_threshold = _scalar_float(arrays, "reference_penetration_threshold_m")
    proxy_axis.axvline(
        reference_threshold,
        color="#7b1fa2",
        linestyle="--",
        linewidth=1.0,
        label=f"reference gate {reference_threshold:.3f} m",
    )
    proxy_axis.scatter(
        [reference_proxy[reference_worst_index]],
        [reference_worst_index - bar_height / 2.0],
        marker="*",
        s=120,
        color="black",
        zorder=5,
        label="worst reference body",
    )
    proxy_axis.scatter(
        [max(reference_proxy[largest_reduction_index], robot_proxy[largest_reduction_index])],
        [largest_reduction_index],
        marker="D",
        s=40,
        facecolors="none",
        edgecolors="#2e7d32",
        linewidths=1.5,
        zorder=5,
        label="largest same-body reduction",
    )
    proxy_axis.set_xlabel("max(queried terrain z - origin z, 0) [m]")
    proxy_axis.set_title("Per-body positive body-origin proxy")
    proxy_axis.grid(axis="x", alpha=0.25)
    proxy_axis.tick_params(axis="y", labelleft=False)
    proxy_axis.legend(fontsize=7, loc="best")

    if show_scan:
        scan_axis = figure.add_subplot(grid_spec[1, 0])
        info_axis = figure.add_subplot(grid_spec[1, 1])
        assert exemplar.scan_shape is not None
        nx, ny = exemplar.scan_shape
        local_xy = np.asarray(arrays["local_scan_local_xy"], dtype=np.float64)
        scan_height = np.asarray(arrays["local_scan_height_w"], dtype=np.float64).reshape(nx, ny)
        grid_x = local_xy[:, 0].reshape(nx, ny)
        grid_y = local_xy[:, 1].reshape(nx, ny)
        heatmap = scan_axis.pcolormesh(grid_x, grid_y, scan_height, shading="nearest", cmap="terrain")
        anchor_xy = np.asarray(arrays["local_scan_root_xy_w"], dtype=np.float64)
        anchor_yaw = float(arrays["local_scan_root_yaw_w"])
        reference_local = _world_to_scan_local(reference_position[:, :2], anchor_xy, anchor_yaw)
        robot_local = _world_to_scan_local(robot_position[:, :2], anchor_xy, anchor_yaw)
        scan_axis.scatter(
            reference_local[:, 0],
            reference_local[:, 1],
            color=reference_color,
            marker="o",
            facecolors="none",
            linewidths=1.2,
            label="reference body XY",
        )
        scan_axis.scatter(
            robot_local[:, 0],
            robot_local[:, 1],
            color=robot_color,
            marker="+",
            linewidths=1.2,
            label="robot body XY",
        )
        scan_axis.scatter([0.0], [0.0], marker="x", s=60, color="black", label="scan anchor")
        heading_w = np.asarray(arrays["target_heading_w"], dtype=np.float64).reshape(1, 2)
        heading_local = _world_to_scan_local(heading_w + anchor_xy, anchor_xy, anchor_yaw)[0]
        scan_axis.arrow(
            0.0,
            0.0,
            0.35 * heading_local[0],
            0.35 * heading_local[1],
            width=0.008,
            color="#4a148c",
            length_includes_head=True,
        )
        for index in {reference_worst_index, largest_reduction_index}:
            scan_axis.annotate(
                exemplar.body_names[index],
                reference_local[index],
                fontsize=7,
                color=reference_color,
                xytext=(4, 4),
                textcoords="offset points",
            )
        scan_axis.set_xlabel("scan-local x forward [m]")
        scan_axis.set_ylabel("scan-local y left [m]")
        scan_axis.set_title("Cached local height scan (x-major, y-fastest) with body-origin XY")
        scan_axis.set_aspect("equal", adjustable="box")
        scan_axis.legend(fontsize=7, loc="best")
        figure.colorbar(heatmap, ax=scan_axis, label="absolute world terrain z [m]")

        info_axis.axis("off")
        info_lines = [
            "AUDIT FRAME",
            f"terrain: {_scalar_string(arrays, 'terrain_type')} / level {_scalar_int(arrays, 'terrain_level')}",
            f"env/eval/episode step: {_scalar_int(arrays, 'env_id')} / "
            f"{_scalar_int(arrays, 'evaluation_step')} / {_scalar_int(arrays, 'episode_step')}",
            "",
            f"worst reference body: {exemplar.body_names[reference_worst_index]}",
            f"reference max proxy: {reference_proxy.max():.4f} m",
            f"robot max proxy: {robot_proxy.max():.4f} m",
            f"signed max-proxy improvement: {global_improvement:+.4f} m",
            "",
            f"largest same-body reduction: {exemplar.body_names[largest_reduction_index]}",
            f"signed reduction: {per_body_reduction[largest_reduction_index]:+.4f} m",
            "",
            "Correlation exemplar only; no policy-intent claim.",
        ]
        info_axis.text(
            0.02,
            0.98,
            "\n".join(info_lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )

    terrain_label = _scalar_string(arrays, "terrain_type")
    figure.suptitle(
        f"{_PROXY_WATERMARK}\n"
        f"{terrain_label} level {_scalar_int(arrays, 'terrain_level')} — "
        f"worst reference: {exemplar.body_names[reference_worst_index]} — "
        f"signed max-proxy improvement {global_improvement:+.4f} m",
        color="#8b0000",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.5,
        _PROXY_WATERMARK,
        ha="center",
        va="center",
        rotation=25,
        fontsize=30,
        color="#8b0000",
        alpha=0.045,
        fontweight="bold",
    )
    figure.supxlabel(
        "Each terrain height is queried at that reference/robot body's own world XY. "
        "This is not collision-shape or mesh penetration.",
        fontsize=8,
        color="#8b0000",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visualize_correction_exemplar(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    dpi: int = 150,
) -> dict[str, Path]:
    """Load, validate, summarize, and render one evaluator exemplar."""
    source = Path(input_path).expanduser().resolve()
    figure_path = source.with_suffix(".png") if output_path is None else Path(output_path).expanduser()
    json_path = source.with_suffix(".json") if summary_path is None else Path(summary_path).expanduser()
    exemplar = load_correction_exemplar(source)
    summary = summarize_correction_exemplar(exemplar)
    summary["source"] = {
        "npz_path": str(source),
        "npz_sha256": _sha256(source),
    }
    resolved_figure = render_correction_exemplar(exemplar, figure_path, dpi=dpi).resolve()
    summary["artifacts"] = {"figure_path": str(resolved_figure)}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"figure": resolved_figure, "summary": json_path.resolve()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and render an evaluator *_first_correction_exemplar.npz as an auditable "
            "body-origin proxy figure and JSON summary."
        )
    )
    parser.add_argument("input_path", type=Path, help="Evaluator correction exemplar NPZ")
    parser.add_argument("--output-path", type=Path, default=None, help="Output .png or .svg path")
    parser.add_argument("--summary-path", type=Path, default=None, help="Output JSON summary path")
    parser.add_argument("--dpi", type=int, default=150, help="Raster DPI (default: 150)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = visualize_correction_exemplar(
        args.input_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        dpi=args.dpi,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
