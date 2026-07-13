from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from holosoma.visualize_correction_exemplar import (
    CorrectionExemplarError,
    load_correction_exemplar,
    summarize_correction_exemplar,
    validate_correction_exemplar,
    visualize_correction_exemplar,
)


def _configured_exemplar() -> dict[str, np.ndarray]:
    body_names = np.asarray(["pelvis", "left_foot", "right_foot"])
    reference_pos = np.asarray(
        [[0.02, 0.02, 0.45], [0.05, 0.08, 0.28], [0.09, 0.01, 0.25]],
        dtype=np.float32,
    )
    robot_pos = np.asarray(
        [[0.03, 0.02, 0.49], [0.05, 0.09, 0.31], [0.08, 0.01, 0.22]],
        dtype=np.float32,
    )
    reference_height = np.asarray([0.50, 0.30, 0.20], dtype=np.float32)
    robot_height = np.asarray([0.50, 0.30, 0.20], dtype=np.float32)
    reference_proxy = np.maximum(reference_height - reference_pos[:, 2], 0.0)
    robot_proxy = np.maximum(robot_height - robot_pos[:, 2], 0.0)

    grid = np.asarray([0.0, 0.1, 0.0, 0.1, 0.1], dtype=np.float32)
    local_xy = np.asarray([[0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [0.1, 0.1]], dtype=np.float32)
    anchor = np.asarray([0.01, -0.02], dtype=np.float32)
    yaw = np.float32(0.3)
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    world_xy = np.empty_like(local_xy)
    world_xy[:, 0] = anchor[0] + cosine * local_xy[:, 0] - sine * local_xy[:, 1]
    world_xy[:, 1] = anchor[1] + sine * local_xy[:, 0] + cosine * local_xy[:, 1]

    root_state = np.zeros(13, dtype=np.float32)
    root_state[2] = 0.8
    root_state[6] = 1.0
    return {
        "env_id": np.asarray(2, dtype=np.int64),
        "terrain_type": np.asarray("box"),
        "terrain_level": np.asarray(3, dtype=np.int64),
        "evaluation_step": np.asarray(17, dtype=np.int64),
        "episode_step": np.asarray(8, dtype=np.int64),
        "target_heading_w": np.asarray([1.0, 0.0], dtype=np.float32),
        "action": np.linspace(-0.2, 0.2, 29, dtype=np.float32),
        "root_state_w": root_state,
        "body_names": body_names,
        "robot_body_pos_w": robot_pos,
        "reference_body_pos_w": reference_pos,
        "robot_body_terrain_height_w": robot_height,
        "reference_body_terrain_height_w": reference_height,
        "robot_body_origin_penetration_m": robot_proxy,
        "reference_body_origin_penetration_m": reference_proxy,
        "robot_max_body_origin_penetration_m": np.asarray(robot_proxy.max()),
        "reference_max_body_origin_penetration_m": np.asarray(reference_proxy.max()),
        "reference_minus_robot_max_body_origin_penetration_m": np.asarray(
            reference_proxy.max() - robot_proxy.max()
        ),
        "correction_proxy_case": np.asarray(True),
        "reference_penetration_threshold_m": np.asarray(0.02, dtype=np.float64),
        "minimum_improvement_threshold_m": np.asarray(0.01, dtype=np.float64),
        "proxy_limitation": np.asarray(
            "Body-origin terrain-height proxy only; not collision-shape penetration or proof of policy intent."
        ),
        "root_state_layout": np.asarray(
            "position_xyz,quaternion_xyzw,linear_velocity_xyz,angular_velocity_xyz"
        ),
        "body_position_frame_units": np.asarray("world_xyz_metres"),
        "action_semantics": np.asarray("raw_policy_action_passed_to_environment_step"),
        "local_scan_configured": np.asarray(True),
        "local_scan_valid": np.asarray(True),
        "local_scan_height_w": np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32),
        "local_scan_root_xy_w": anchor,
        "local_scan_root_yaw_w": np.asarray(yaw),
        "local_scan_local_xy": local_xy,
        "local_scan_world_xy": world_xy,
        "local_scan_grid": grid,
        "local_scan_flatten_order": np.asarray("x-major,y-fastest"),
        "local_scan_height_units": np.asarray("absolute-world-z-metres"),
    }


def _unconfigured_exemplar() -> dict[str, np.ndarray]:
    arrays = _configured_exemplar()
    arrays.update(
        {
            "local_scan_configured": np.asarray(False),
            "local_scan_valid": np.asarray(False),
            "local_scan_height_w": np.empty((0,), dtype=np.float32),
            "local_scan_root_xy_w": np.empty((0,), dtype=np.float32),
            "local_scan_root_yaw_w": np.asarray(np.nan, dtype=np.float32),
            "local_scan_local_xy": np.empty((0, 2), dtype=np.float32),
            "local_scan_world_xy": np.empty((0, 2), dtype=np.float32),
            "local_scan_grid": np.empty((0,), dtype=np.float32),
        }
    )
    return arrays


def test_validation_and_summary_preserve_body_and_scan_semantics() -> None:
    exemplar = validate_correction_exemplar(_configured_exemplar())
    summary = summarize_correction_exemplar(exemplar)

    assert exemplar.body_names == ("pelvis", "left_foot", "right_foot")
    assert exemplar.scan_shape == (2, 2)
    assert summary["selection"]["terrain_type"] == "box"
    assert summary["aggregate_m"][
        "reference_minus_robot_maximum_signed_improvement"
    ] == pytest.approx(0.04)
    assert summary["worst_reference_body"]["name"] == "pelvis"
    assert summary["worst_corrected_body"]["name"] == "pelvis"
    assert summary["worst_reference_body"]["reference_signed_clearance_m"] == pytest.approx(-0.05)
    assert summary["largest_per_body_proxy_reduction"]["name"] == "pelvis"
    assert summary["largest_per_body_proxy_reduction"][
        "reference_minus_robot_same_body_proxy_m"
    ] == pytest.approx(0.04)
    assert summary["local_scan"]["shape_nx_ny"] == [2, 2]
    assert summary["local_scan"]["flatten_order"] == "x-major,y-fastest"
    assert "not collision geometry" in summary["interpretation"]


def test_unconfigured_scan_is_explicit_and_does_not_require_scan_arrays() -> None:
    exemplar = validate_correction_exemplar(_unconfigured_exemplar())
    summary = summarize_correction_exemplar(exemplar)

    assert exemplar.scan_shape is None
    assert summary["local_scan"] == {
        "configured": False,
        "valid": False,
        "flatten_order": "x-major,y-fastest",
        "height_units": "absolute-world-z-metres",
    }


def test_validation_rejects_proxy_arithmetic_and_threshold_mismatch() -> None:
    arrays = _configured_exemplar()
    arrays["robot_body_origin_penetration_m"] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(CorrectionExemplarError, match="robot_body_origin_penetration_m is inconsistent"):
        validate_correction_exemplar(arrays)

    arrays = _configured_exemplar()
    arrays["minimum_improvement_threshold_m"] = np.asarray(0.05)
    with pytest.raises(CorrectionExemplarError, match="does not meet minimum_improvement_threshold_m"):
        validate_correction_exemplar(arrays)


def test_validation_rejects_wrong_scan_flatten_order() -> None:
    arrays = _configured_exemplar()
    arrays["local_scan_local_xy"] = arrays["local_scan_local_xy"][[0, 2, 1, 3]]
    with pytest.raises(CorrectionExemplarError, match="flatten order"):
        validate_correction_exemplar(arrays)


def test_pickle_free_loader_rejects_object_arrays(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.npz"
    arrays = _configured_exemplar()
    arrays["body_names"] = np.asarray(["pelvis", "left_foot", "right_foot"], dtype=object)
    np.savez_compressed(path, **arrays)

    with pytest.raises(CorrectionExemplarError, match="without pickle"):
        load_correction_exemplar(path)


def test_visualization_writes_noninteractive_png_and_json(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    input_path = tmp_path / "eval_first_correction_exemplar.npz"
    np.savez_compressed(input_path, **_configured_exemplar())

    outputs = visualize_correction_exemplar(input_path, dpi=72)

    assert outputs["figure"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    payload = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert payload["source"]["npz_path"] == str(input_path.resolve())
    assert len(payload["source"]["npz_sha256"]) == 64
    assert payload["artifacts"]["figure_path"] == str(outputs["figure"])
    assert payload["local_scan"]["height_units"] == "absolute-world-z-metres"
