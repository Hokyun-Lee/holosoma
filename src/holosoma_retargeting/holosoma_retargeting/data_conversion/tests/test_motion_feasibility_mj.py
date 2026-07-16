from __future__ import annotations

import importlib
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

DATA_CONVERSION_DIR = Path(__file__).resolve().parents[1]
RETARGETING_DIR = DATA_CONVERSION_DIR.parent
sys.path.insert(0, str(DATA_CONVERSION_DIR))

feasibility = importlib.import_module("evaluate_motion_feasibility_mj")
viewer = importlib.import_module("view_motion_mj")


def _stationary_qpos(frames: int) -> np.ndarray:
    qpos = np.zeros((frames, 36), dtype=np.float64)
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1.0
    return qpos


def test_periodic_transition_phase_respects_slice_offset() -> None:
    report = feasibility._motion_derivatives(
        _stationary_qpos(20),
        np.arange(7, 36),
        fps=50.0,
        history_frames=2,
        replan_stride=12,
        sequence_frame_offset=10,
        transition_kind="generated_replan_boundaries",
    )

    transitions = report["periodic_transitions"]
    assert transitions["local_frames"] == [4, 16]
    assert transitions["sequence_frames"] == [14, 26]


def test_required_terrain_and_raw_inputs_cannot_silently_pass() -> None:
    robot_xml = RETARGETING_DIR / "models/g1/g1_29dof.xml"
    model, _ = viewer.build_model(viewer.Args(motion="unused.npz", robot_xml=str(robot_xml)))
    args = feasibility.Args(motion="unused.npz", robot_xml=str(robot_xml))

    report = feasibility.analyze_motion(
        model=model,
        qpos=_stationary_qpos(3),
        fps=50.0,
        args=args,
        enforce_raw_checks=True,
    )

    assert report["checks"]["terrain_geometry_supplied"] is False
    assert report["checks"]["raw_motion_supplied"] is False
    assert report["checks"]["raw_qpos_joint_mapping"] is False
    assert report["checks"]["generator_body_head_fk"] is False
    assert report["kinematic_gate_pass"] is False


def test_discontinuous_motion_cannot_silently_pass() -> None:
    robot_xml = RETARGETING_DIR / "models/g1/g1_29dof.xml"
    model, _ = viewer.build_model(viewer.Args(motion="unused.npz", robot_xml=str(robot_xml)))
    args = feasibility.Args(
        motion="unused.npz",
        robot_xml=str(robot_xml),
        require_terrain_urdf_for_gate=False,
        require_raw_motion_for_gate=False,
    )
    qpos = _stationary_qpos(3)
    qpos[1, 0] = 1.0

    report = feasibility.analyze_motion(
        model=model,
        qpos=qpos,
        fps=50.0,
        args=args,
        enforce_raw_checks=True,
    )

    assert report["checks"]["motion_continuity"] is False
    assert report["motion_continuity"]["gate"]["root_linear_speed"]["pass"] is False
    assert report["kinematic_gate_pass"] is False


def test_ground_penetration_uses_five_millimetre_gate() -> None:
    robot_xml = RETARGETING_DIR / "models/g1/g1_29dof.xml"
    model, _ = viewer.build_model(viewer.Args(motion="unused.npz", robot_xml=str(robot_xml)))
    args = feasibility.Args(
        motion="unused.npz",
        robot_xml=str(robot_xml),
        require_terrain_urdf_for_gate=False,
        require_raw_motion_for_gate=False,
    )
    qpos = _stationary_qpos(1)
    qpos[:, 2] = 0.0

    report = feasibility.analyze_motion(
        model=model,
        qpos=qpos,
        fps=50.0,
        args=args,
        enforce_raw_checks=True,
    )

    collision = report["environment_collision"]
    assert collision["surface_tolerance_m"] == pytest.approx(0.005)
    assert collision["max_ground_depth_m"] > collision["surface_tolerance_m"]
    assert collision["robot_body_breakdown"]
    assert collision["robot_body_breakdown"][0]["max_depth_m"] == pytest.approx(collision["max_depth_m"])
    assert report["checks"]["environment_penetration_within_tolerance"] is False


def test_robot_body_collision_breakdown_is_per_frame_and_deterministic() -> None:
    breakdown = feasibility._robot_body_collision_breakdown(
        {
            "zeta_link": np.array([0.0, 0.01, 0.01, 0.0]),
            "alpha_link": np.array([0.01, 0.0, 0.0, 0.0]),
            "left_ankle_roll_link": np.array([0.0, 0.06, 0.0, 0.0]),
        },
        {
            "zeta_link": False,
            "alpha_link": False,
            "left_ankle_roll_link": True,
        },
        penetration_tolerance_m=0.005,
        deep_penetration_threshold_m=0.05,
    )

    assert [row["robot_body"] for row in breakdown] == [
        "left_ankle_roll_link",
        "alpha_link",
        "zeta_link",
    ]
    assert breakdown[0] == {
        "robot_body": "left_ankle_roll_link",
        "is_foot": True,
        "max_depth_m": 0.06,
        "contact_frame_rate": 0.25,
        "over_tolerance_frame_rate": 0.25,
        "deep_penetration_frame_rate": 0.25,
    }
    assert breakdown[2]["contact_frame_rate"] == pytest.approx(0.5)
    assert breakdown[2]["over_tolerance_frame_rate"] == pytest.approx(0.5)
    assert breakdown[2]["deep_penetration_frame_rate"] == 0.0


def test_reference_comparison_includes_surface_and_nonfoot_rate_deltas() -> None:
    def motion(over_tolerance: float, nonfoot_over_tolerance: float) -> dict:
        return {
            "environment_collision": {
                "max_depth_m": 0.02,
                "deep_penetration_frame_rate": 0.0,
                "over_tolerance_frame_rate": over_tolerance,
                "nonfoot_over_tolerance_frame_rate": nonfoot_over_tolerance,
            },
            "joint_limits": {"max_violation_rad": 0.0},
            "motion_continuity": {"periodic_transitions": {"joint_l2_step_rad": {"max": 0.1}}},
        }

    comparison = feasibility._compare_to_reference(
        motion(over_tolerance=0.75, nonfoot_over_tolerance=0.5),
        motion(over_tolerance=0.25, nonfoot_over_tolerance=0.125),
    )

    assert comparison["over_tolerance_frame_rate_delta"] == pytest.approx(0.5)
    assert comparison["nonfoot_over_tolerance_frame_rate_delta"] == pytest.approx(0.375)


def test_nonzero_terrain_urdf_origin_is_rejected(tmp_path: Path) -> None:
    urdf = tmp_path / "terrain.urdf"
    urdf.write_text(
        '<robot name="bad"><link name="box"><visual><origin xyz="1 0 0" rpy="0 0 0"/></visual></link></robot>'
    )
    spec = mujoco.MjSpec.from_file(str(RETARGETING_DIR / "models/g1/g1_29dof.xml"))

    with pytest.raises(NotImplementedError, match="zero URDF origins"):
        viewer._add_terrain_boxes(spec, urdf)


def test_empty_terrain_urdf_is_rejected(tmp_path: Path) -> None:
    urdf = tmp_path / "terrain.urdf"
    urdf.write_text('<robot name="empty"/>')
    spec = mujoco.MjSpec.from_file(str(RETARGETING_DIR / "models/g1/g1_29dof.xml"))

    with pytest.raises(ValueError, match="at least one OmniRetarget mesh box"):
        viewer._add_terrain_boxes(spec, urdf)


def test_raw_motion_requires_exact_generator_body_order(tmp_path: Path) -> None:
    raw = tmp_path / "raw.npz"
    np.savez(
        raw,
        root_pos=np.zeros((1, 3)),
        root_quat_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        joint_pos=np.zeros((1, 29)),
        joint_names=np.asarray([f"joint_{index}" for index in range(29)]),
        body_pos=np.zeros((1, 1, 3)),
        body_names=np.asarray(["pelvis"]),
    )

    with pytest.raises(ValueError, match="exact 14-body generator order"):
        feasibility._load_raw_motion(str(raw), start=0, frames=1)


def test_nondivisible_video_fps_uses_actual_sample_rate() -> None:
    stride, encoded_fps = viewer._video_stride_and_fps(50.0, 30, 0.25)

    assert stride == 2
    assert encoded_fps == pytest.approx(6.25)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("penetration_tolerance_m", np.nan), ("max_joint_speed_rad_s", np.inf)],
)
def test_nonfinite_gate_threshold_is_rejected(field: str, bad_value: float) -> None:
    args = feasibility.Args(motion="unused.npz")
    setattr(args, field, bad_value)

    with pytest.raises(ValueError, match="finite and non-negative"):
        feasibility.main(args)
