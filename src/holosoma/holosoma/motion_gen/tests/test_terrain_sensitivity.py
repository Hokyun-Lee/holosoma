"""Synthetic CPU tests for terrain-sensitivity metrics."""

from __future__ import annotations

import math

import pytest
import torch

from holosoma.motion_gen.features import DEFAULT_BODY_NAMES, DEFAULT_JOINT_NAMES
from holosoma.motion_gen.scripts.evaluate_terrain_sensitivity import (
    bilinear_sample_height,
    build_rectangular_obstacle_scans,
    compute_motion_metrics,
    evaluate_meaningful_physical_response,
    resolve_world_heading,
    summarize_sensitivity,
    world_xy_to_scan_local,
)
from holosoma.motion_gen.terrain import ScanGrid


def _small_grid() -> ScanGrid:
    return ScanGrid(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, spacing=1.0)


def test_rectangular_scans_preserve_x_major_y_fastest_order() -> None:
    grid = _small_grid()
    scans, mask = build_rectangular_obstacle_scans(
        grid,
        [0.0, 0.3, 0.6],
        (0.0, 1.0, 0.0, 0.0),
    )

    assert scans.shape == (3, 9)
    assert torch.equal(torch.nonzero(mask).flatten(), torch.tensor([4, 7]))
    torch.testing.assert_close(scans[0], torch.zeros(9))
    torch.testing.assert_close(scans[1, mask], torch.full((2,), 0.3))
    torch.testing.assert_close(scans[2, mask], torch.full((2,), 0.6))

    production_grid = ScanGrid()
    _, production_mask = build_rectangular_obstacle_scans(
        production_grid,
        [0.0, 0.3, 0.6],
        (0.3, 0.9, -0.4, 0.4),
    )
    assert int(production_mask.sum()) == 7 * 9


def test_bilinear_scan_sampling_matches_linear_height_field_and_outside_ground() -> None:
    grid = _small_grid()
    offsets = grid.offsets_tensor(device="cpu")
    scan = 2.0 * offsets[:, 0] + 3.0 * offsets[:, 1] + 5.0
    points = torch.tensor([[0.25, -0.50], [-0.75, 0.50], [1.20, 0.00]])

    sampled = bilinear_sample_height(scan, points, grid)

    torch.testing.assert_close(sampled[:2], torch.tensor([4.0, 5.0]))
    assert sampled[2] == 0.0


def test_world_to_scan_local_uses_anchor_position_and_yaw() -> None:
    anchor_position = torch.tensor([10.0, -2.0, 0.8])
    yaw = torch.tensor(math.pi / 2.0)
    anchor_quaternion = torch.tensor([torch.cos(yaw / 2.0), 0.0, 0.0, torch.sin(yaw / 2.0)])
    world_xy = torch.tensor([[10.0, -1.0], [9.0, -2.0]])

    local = world_xy_to_scan_local(world_xy, anchor_position, anchor_quaternion)

    torch.testing.assert_close(local, torch.tensor([[1.0, 0.0], [0.0, 1.0]]), atol=1.0e-6, rtol=0.0)


def test_world_heading_defaults_to_anchor_yaw_forward_and_supports_explicit_override() -> None:
    yaw = torch.tensor(math.pi / 2.0)
    anchor_quaternion = torch.tensor([torch.cos(yaw / 2.0), 0.0, 0.0, torch.sin(yaw / 2.0)])

    default_heading, default_source = resolve_world_heading(anchor_quaternion, None, None)
    explicit_heading, explicit_source = resolve_world_heading(anchor_quaternion, 1.0, 1.0)

    torch.testing.assert_close(default_heading, torch.tensor([0.0, 1.0]), atol=1.0e-6, rtol=0.0)
    assert default_source == "anchor_yaw_forward"
    torch.testing.assert_close(
        explicit_heading,
        torch.full((2,), math.sqrt(0.5)),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert explicit_source == "explicit_world_xy"

    with pytest.raises(ValueError, match="both be omitted"):
        resolve_world_heading(anchor_quaternion, 1.0, None)


def test_motion_metrics_root_feet_knees_origin_penetration_proxy_and_displacement() -> None:
    grid = _small_grid()
    frames = 2
    root = torch.tensor([[0.2, 0.0, 0.8], [0.7, 0.0, 1.0]])
    joints = torch.zeros(frames, 29)
    joints[:, DEFAULT_JOINT_NAMES.index("left_knee_joint")] = torch.tensor([0.2, 0.6])
    joints[:, DEFAULT_JOINT_NAMES.index("right_knee_joint")] = torch.tensor([0.4, 0.8])
    bodies = torch.zeros(frames, 14, 3)
    bodies[..., 0] = 0.5
    bodies[..., 2] = 1.0
    left_foot = DEFAULT_BODY_NAMES.index("left_ankle_roll_link")
    right_foot = DEFAULT_BODY_NAMES.index("right_ankle_roll_link")
    bodies[:, left_foot, 2] = torch.tensor([0.2, 0.4])
    bodies[:, right_foot, 2] = torch.tensor([0.1, 0.3])
    scan = torch.full((grid.dim,), 0.3)

    metrics = compute_motion_metrics(
        root_position=root,
        joint_position=joints,
        body_position=bodies,
        terrain_scan=scan,
        grid=grid,
        anchor_position=torch.tensor([0.0, 0.0, 0.8]),
        anchor_quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        world_heading=torch.tensor([1.0, 0.0]),
        origin_penetration_proxy_threshold=0.05,
    )

    assert metrics["root_z_max"] == pytest.approx(1.0)
    assert metrics["root_z_mean"] == pytest.approx(0.9)
    assert metrics["root_z_end"] == pytest.approx(1.0)
    assert metrics["forward_displacement"] == pytest.approx(0.7)
    assert metrics["left_foot_z_max"] == pytest.approx(0.4)
    assert metrics["left_foot_clearance_max"] == pytest.approx(0.1)
    assert metrics["left_foot_clearance_min"] == pytest.approx(-0.1)
    assert metrics["right_foot_z_max"] == pytest.approx(0.3)
    assert metrics["left_knee_flexion_max"] == pytest.approx(0.6)
    assert metrics["left_knee_flexion_mean"] == pytest.approx(0.4)
    assert metrics["right_knee_flexion_max"] == pytest.approx(0.8)
    assert metrics["foot_origin_penetration_proxy_max"] == pytest.approx(0.2)
    assert metrics["foot_origin_penetration_proxy_mean"] == pytest.approx(0.075)
    assert metrics["foot_origin_penetration_proxy_rate"] == pytest.approx(0.5)
    assert metrics["body_origin_penetration_proxy_mean"] == pytest.approx(0.3 / 28.0)
    assert metrics["body_origin_penetration_proxy_rate"] == pytest.approx(2.0 / 28.0)


def test_motion_metrics_reject_layout_reordering() -> None:
    root = torch.zeros(1, 3)
    joints = torch.zeros(1, 29)
    bodies = torch.zeros(1, 14, 3)
    wrong_joints = list(DEFAULT_JOINT_NAMES)
    wrong_joints[0], wrong_joints[1] = wrong_joints[1], wrong_joints[0]

    with pytest.raises(ValueError, match="exact G1 29-DoF"):
        compute_motion_metrics(
            root_position=root,
            joint_position=joints,
            body_position=bodies,
            terrain_scan=torch.zeros(_small_grid().dim),
            grid=_small_grid(),
            anchor_position=torch.zeros(3),
            anchor_quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            world_heading=torch.tensor([1.0, 0.0]),
            joint_names=tuple(wrong_joints),
        )


def test_sensitivity_summary_records_monotonicity_and_origin_proxy_regression() -> None:
    monotonic = summarize_sensitivity(
        [0.0, 0.3, 0.6],
        [
            {"root_z_max": 0.8, "body_origin_penetration_proxy_max": 0.2, "forward_displacement": 1.0},
            {"root_z_max": 0.9, "body_origin_penetration_proxy_max": 0.1, "forward_displacement": 0.8},
            {"root_z_max": 1.0, "body_origin_penetration_proxy_max": 0.0, "forward_displacement": 0.6},
        ],
    )
    assert monotonic["all_expected_monotonic"]
    assert not monotonic["origin_penetration_proxy_increased_vs_flat"]
    assert monotonic["metrics"]["root_z_max"]["delta_vs_flat"] == pytest.approx([0.0, 0.1, 0.2])

    regression = summarize_sensitivity(
        [0.0, 0.3, 0.6],
        [
            {"root_z_max": 0.8, "body_origin_penetration_proxy_max": 0.0},
            {"root_z_max": 0.7, "body_origin_penetration_proxy_max": 0.1},
            {"root_z_max": 0.9, "body_origin_penetration_proxy_max": 0.2},
        ],
    )
    assert not regression["all_expected_monotonic"]
    assert regression["origin_penetration_proxy_increased_vs_flat"]
    assert not regression["metrics"]["root_z_max"]["nondecreasing"]


def test_meaningful_response_uses_only_configured_physical_metric_thresholds() -> None:
    flat = {
        "root_z_max": 0.8,
        "root_z_mean": 0.75,
        "root_z_end": 0.8,
        "left_foot_z_max": 0.1,
        "right_foot_z_max": 0.1,
        "left_knee_flexion_max": 0.5,
        "left_knee_flexion_mean": 0.3,
        "right_knee_flexion_max": 0.5,
        "right_knee_flexion_mean": 0.3,
        "forward_displacement": 0.4,
    }
    obstacle = dict(flat)
    obstacle["right_knee_flexion_max"] += 0.06
    obstacle["forward_displacement"] -= 0.01

    report = evaluate_meaningful_physical_response(
        [flat, obstacle],
        root_height_delta_m=0.01,
        foot_height_delta_m=0.01,
        knee_flexion_delta_rad=0.05,
        forward_displacement_delta_m=0.02,
    )

    assert report["meaningful"]
    assert report["groups"]["knee_flexion_rad"]["exceeded"]
    assert report["groups"]["knee_flexion_rad"]["max_abs_delta"] == pytest.approx(0.06)
    assert not report["groups"]["forward_displacement_m"]["exceeded"]


def test_meaningful_response_rejects_subthreshold_changes_and_negative_thresholds() -> None:
    metrics = {
        "root_z_max": 0.8,
        "root_z_mean": 0.75,
        "root_z_end": 0.8,
        "left_foot_z_max": 0.1,
        "right_foot_z_max": 0.1,
        "left_knee_flexion_max": 0.5,
        "left_knee_flexion_mean": 0.3,
        "right_knee_flexion_max": 0.5,
        "right_knee_flexion_mean": 0.3,
        "forward_displacement": 0.4,
    }
    report = evaluate_meaningful_physical_response(
        [metrics, dict(metrics)],
        root_height_delta_m=0.01,
        foot_height_delta_m=0.01,
        knee_flexion_delta_rad=0.05,
        forward_displacement_delta_m=0.02,
    )
    assert not report["meaningful"]

    with pytest.raises(ValueError, match="non-negative"):
        evaluate_meaningful_physical_response(
            [metrics, dict(metrics)],
            root_height_delta_m=-0.01,
            foot_height_delta_m=0.01,
            knee_flexion_delta_rad=0.05,
            forward_displacement_delta_m=0.02,
        )
