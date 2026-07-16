"""Evaluate generated G1 qpos against terrain with MuJoCo collision geometry.

This is a *kinematic* feasibility audit.  Every qpos frame is placed directly
in MuJoCo and passed through ``mj_forward``; no controller, actuator torque, or
time integration is used.  The report therefore catches invalid quaternions,
joint-limit violations, large re-plan discontinuities, FK disagreement, and
robot/terrain collision-shape penetration, but it cannot prove balance or
dynamic trackability.

Run from ``src/holosoma_retargeting/holosoma_retargeting``::

    python data_conversion/evaluate_motion_feasibility_mj.py \
      --motion <generated>_gen_qpos.npz \
      --raw-motion <generated>_gen_raw.npz \
      --terrain-urdf <terrain>/multi_boxes_z_scale_1.0.urdf \
      --reference ../../../data/motion_gen/processed_paperscale/<clip>.npz \
      --reference-start-frame 100 \
      --output <run>/feasibility.json
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco  # type: ignore[import-not-found]
import numpy as np
import tyro
from view_motion_mj import Args as ViewerArgs
from view_motion_mj import build_model, load_qpos

_TERRAIN_GEOM_PREFIX = "terrain_box_"
_FOOT_BODY_PREFIXES = ("left_ankle_roll", "right_ankle_roll")
_GENERATOR_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


@dataclass
class Args:
    motion: str
    """Generated qpos/WBT npz to audit."""
    terrain_urdf: str | None = None
    """OmniRetarget multi-box terrain used when generating the motion."""
    raw_motion: str | None = None
    """Optional generator ``*_gen_raw.npz`` for body-head versus MuJoCo FK."""
    reference: str | None = None
    """Optional source WBT/qpos clip audited with the identical collision scene."""
    reference_start_frame: int = 0
    """Reference frame aligned with generated frame zero, before start_frame slicing."""
    start_frame: int = 0
    num_frames: int | None = None
    robot_xml: str = "models/g1/g1_29dof.xml"
    output: str = "feasibility.json"
    history_frames: int = 2
    """History prefix retained by motion_gen receding-horizon rollout."""
    replan_stride: int = 12
    penetration_tolerance_m: float = 0.005
    """Contact depths at or below this value are treated as surface tolerance."""
    deep_penetration_threshold_m: float = 0.05
    """Diagnostic severity threshold; the gate itself uses penetration_tolerance_m."""
    joint_limit_tolerance_rad: float = 0.001
    quaternion_norm_tolerance: float = 0.001
    fk_body_error_threshold_m: float = 0.02
    max_root_linear_speed_m_s: float = 5.0
    max_root_angular_speed_rad_s: float = 20.0
    max_joint_speed_rad_s: float = 20.0
    max_joint_acceleration_rad_s2: float = 500.0
    max_frame_joint_l2_step_rad: float = 0.5
    """Continuity gates; implementation choices rather than paper-published limits."""
    worst_contacts: int = 20
    preview: str | None = None
    """Optional GIF/MP4 to include in the W&B artifact."""
    generator_checkpoint: str | None = None
    """Generator checkpoint used upstream; hashed but not uploaded to W&B."""
    generator_clip: str | None = None
    generator_seed: int | None = None
    generator_num_steps: int | None = None
    generator_num_cycles: int | None = None
    artifact_files: tuple[str, ...] = ()
    """Additional small diagnostics to include in the W&B artifact."""
    wandb_mode: str = "disabled"
    """disabled, offline, or online. Requires an environment containing wandb."""
    wandb_entity: str | None = None
    wandb_project: str = "HoloSomaMotionGenerator"
    wandb_group: str = "terrain_feasibility"
    wandb_name: str | None = None
    require_terrain_urdf_for_gate: bool = True
    """Prevent a terrain-feasibility PASS when only the ground plane was checked."""
    require_raw_motion_for_gate: bool = True
    """Require raw body/joint heads and their qpos/FK consistency for a PASS."""
    require_kinematic_gate: bool = False
    """Exit non-zero after writing/logging the report when the gate fails."""


def _slice_qpos(
    qpos: np.ndarray,
    *,
    start: int,
    count: int | None,
    expected_nq: int,
    label: str,
) -> np.ndarray:
    if qpos.ndim != 2 or qpos.shape[1] != expected_nq:
        raise ValueError(f"{label}: expected qpos (T,{expected_nq}), got {qpos.shape}")
    if start < 0 or start >= qpos.shape[0]:
        raise ValueError(f"{label}: start frame {start} outside [0,{qpos.shape[0]})")
    if count is not None and count < 1:
        raise ValueError("num_frames must be positive when provided")
    stop = qpos.shape[0] if count is None else min(start + count, qpos.shape[0])
    result = np.asarray(qpos[start:stop], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{label}: qpos contains non-finite values")
    return result


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or f"unnamed_{object_id}"


def _environment_geom_ids(model: mujoco.MjModel) -> tuple[set[int], dict[int, str]]:
    environment: dict[int, str] = {}
    for geom_id in range(model.ngeom):
        name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name == "ground" or name.startswith(_TERRAIN_GEOM_PREFIX):
            environment[geom_id] = name
    if not environment:
        raise RuntimeError("Compiled scene contains neither ground nor named terrain geoms")
    return set(environment), environment


def _robot_joint_columns(
    model: mujoco.MjModel,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names: list[str] = []
    columns: list[int] = []
    ranges: list[np.ndarray] = []
    limited: list[bool] = []
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        names.append(_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        columns.append(int(model.jnt_qposadr[joint_id]))
        ranges.append(np.asarray(model.jnt_range[joint_id], dtype=np.float64))
        limited.append(bool(model.jnt_limited[joint_id]))
    return names, np.asarray(columns), np.stack(ranges), np.asarray(limited)


def _summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"max": 0.0, "mean": 0.0, "p95": 0.0}
    return {
        "max": float(values.max()),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
    }


def _robot_body_collision_breakdown(
    body_frame_max_depth: dict[str, np.ndarray],
    body_is_foot: dict[str, bool],
    *,
    penetration_tolerance_m: float,
    deep_penetration_threshold_m: float,
) -> list[dict[str, Any]]:
    """Summarize penetrating environment contacts once per body and frame."""
    breakdown = []
    for robot_body, frame_depths in body_frame_max_depth.items():
        frame_max_depth = np.asarray(frame_depths, dtype=np.float64)
        breakdown.append(
            {
                "robot_body": robot_body,
                "is_foot": body_is_foot[robot_body],
                "max_depth_m": float(frame_max_depth.max()),
                "contact_frame_rate": float(np.mean(frame_max_depth > 0.0)),
                "over_tolerance_frame_rate": float(np.mean(frame_max_depth > penetration_tolerance_m)),
                "deep_penetration_frame_rate": float(np.mean(frame_max_depth > deep_penetration_threshold_m)),
            }
        )
    return sorted(
        breakdown,
        key=lambda record: (-record["max_depth_m"], record["robot_body"]),
    )


def _motion_derivatives(
    qpos: np.ndarray,
    joint_columns: np.ndarray,
    fps: float,
    history_frames: int,
    replan_stride: int,
    sequence_frame_offset: int,
    transition_kind: str,
) -> dict[str, Any]:
    joints = qpos[:, joint_columns]
    joint_step = np.linalg.norm(np.diff(joints, axis=0), axis=1)
    joint_speed = np.abs(np.diff(joints, axis=0)) * fps
    joint_acceleration = np.abs(np.diff(joints, n=2, axis=0)) * fps**2
    root_step = np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1)
    root_speed = root_step * fps

    quat = qpos[:, 3:7]
    quat_norm = np.linalg.norm(quat, axis=1)
    normalized = quat / np.maximum(quat_norm[:, None], np.finfo(np.float64).eps)
    dots = np.abs(np.sum(normalized[1:] * normalized[:-1], axis=1)).clip(0.0, 1.0)
    root_angular_speed = 2.0 * np.arccos(dots) * fps

    first_boundary = history_frames + replan_stride
    sequence_stop = sequence_frame_offset + len(qpos)
    sequence_boundary_frames = np.arange(first_boundary, sequence_stop, replan_stride)
    sequence_boundary_frames = sequence_boundary_frames[sequence_boundary_frames > sequence_frame_offset]
    boundary_frames = sequence_boundary_frames - sequence_frame_offset
    boundary_steps = boundary_frames - 1
    boundary_steps = boundary_steps[(boundary_steps >= 0) & (boundary_steps < len(joint_step))]
    nonboundary_mask = np.ones(len(joint_step), dtype=bool)
    nonboundary_mask[boundary_steps] = False
    boundary_joint = joint_step[boundary_steps]
    nonboundary_joint = joint_step[nonboundary_mask]
    boundary_root = root_step[boundary_steps]
    nonboundary_root = root_step[nonboundary_mask]

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
        if numerator.size == 0:
            return 0.0
        baseline = float(np.quantile(denominator, 0.95)) if denominator.size else 0.0
        return float(numerator.max() / max(baseline, np.finfo(np.float64).eps))

    return {
        "root_linear_speed_m_s": _summarize(root_speed),
        "root_angular_speed_rad_s": _summarize(root_angular_speed),
        "joint_speed_abs_rad_s": _summarize(joint_speed),
        "joint_acceleration_abs_rad_s2": _summarize(joint_acceleration),
        "frame_joint_l2_step_rad": _summarize(joint_step),
        "periodic_transitions": {
            "kind": transition_kind,
            "history_frames": history_frames,
            "stride_frames": replan_stride,
            "local_frames": boundary_frames.tolist(),
            "sequence_frames": sequence_boundary_frames.tolist(),
            "joint_l2_step_rad": _summarize(boundary_joint),
            "root_translation_step_m": _summarize(boundary_root),
            "joint_max_vs_nonboundary_p95_ratio": ratio(boundary_joint, nonboundary_joint),
            "root_max_vs_nonboundary_p95_ratio": ratio(boundary_root, nonboundary_root),
        },
        "quaternion_norm": {
            **_summarize(quat_norm),
            "max_abs_error_from_one": float(np.max(np.abs(quat_norm - 1.0))),
        },
    }


def _load_raw_motion(
    path: str,
    *,
    start: int,
    frames: int,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, tuple[str, ...]]:
    raw = np.load(path, allow_pickle=False)
    required = {
        "root_pos",
        "root_quat_wxyz",
        "joint_pos",
        "joint_names",
        "body_pos",
        "body_names",
    }
    if not required.issubset(raw.files):
        raise ValueError(f"{path}: expected keys {sorted(required)}, got {list(raw.files)}")
    body_pos = np.asarray(raw["body_pos"], dtype=np.float64)
    body_names = tuple(str(name) for name in raw["body_names"].tolist())
    if body_names != _GENERATOR_BODY_NAMES:
        raise ValueError(
            f"{path}: expected the exact 14-body generator order {_GENERATOR_BODY_NAMES}, got {body_names}"
        )
    root_pos = np.asarray(raw["root_pos"], dtype=np.float64)
    root_quat = np.asarray(raw["root_quat_wxyz"], dtype=np.float64)
    joint_pos = np.asarray(raw["joint_pos"], dtype=np.float64)
    joint_names = tuple(str(name) for name in raw["joint_names"].tolist())
    if body_pos.ndim != 3 or body_pos.shape[1:] != (len(body_names), 3):
        raise ValueError(f"{path}: inconsistent body_pos/body_names shapes")
    if root_pos.shape != (len(body_pos), 3) or root_quat.shape != (len(body_pos), 4):
        raise ValueError(f"{path}: inconsistent root pose shapes")
    if joint_pos.shape != (len(body_pos), len(joint_names)):
        raise ValueError(f"{path}: inconsistent joint_pos/joint_names shapes")
    stop = start + frames
    if start < 0 or stop > body_pos.shape[0]:
        raise ValueError(f"{path}: raw body range [{start},{stop}) outside {body_pos.shape[0]} frames")
    raw_qpos = np.concatenate((root_pos, root_quat, joint_pos), axis=1)
    return body_pos[start:stop], body_names, raw_qpos[start:stop], joint_names


def analyze_motion(
    *,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fps: float,
    args: Args,
    raw_body_positions: np.ndarray | None = None,
    raw_body_names: tuple[str, ...] = (),
    input_consistency: dict[str, Any] | None = None,
    enforce_raw_checks: bool = False,
    sequence_frame_offset: int = 0,
    file_frame_offset: int = 0,
    transition_kind: str = "generated_replan_boundaries",
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    environment_ids, environment_names = _environment_geom_ids(model)
    joint_names, joint_columns, joint_ranges, joint_limited = _robot_joint_columns(model)
    joints = qpos[:, joint_columns]

    lower_violation = np.maximum(joint_ranges[:, 0] - joints, 0.0)
    upper_violation = np.maximum(joints - joint_ranges[:, 1], 0.0)
    joint_violation = np.maximum(lower_violation, upper_violation)
    joint_violation[:, ~joint_limited] = 0.0
    violating = np.argwhere(joint_violation > args.joint_limit_tolerance_rad)

    frame_max_depth = np.zeros(len(qpos), dtype=np.float64)
    frame_max_terrain_depth = np.zeros(len(qpos), dtype=np.float64)
    frame_max_ground_depth = np.zeros(len(qpos), dtype=np.float64)
    frame_max_nonfoot_depth = np.zeros(len(qpos), dtype=np.float64)
    frame_environment_contact = np.zeros(len(qpos), dtype=bool)
    body_frame_max_depth: dict[str, np.ndarray] = {}
    body_is_foot: dict[str, bool] = {}
    contact_records_by_key: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    fk_errors: list[np.ndarray] = []

    raw_body_ids: np.ndarray | None = None
    if raw_body_positions is not None:
        if len(raw_body_names) != raw_body_positions.shape[1]:
            raise ValueError("raw body names do not match body-position columns")
        raw_body_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in raw_body_names])
        missing = [name for name, body_id in zip(raw_body_names, raw_body_ids) if body_id < 0]
        if missing:
            raise ValueError(f"Raw motion body names absent from MJCF: {missing}")

    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        if raw_body_ids is not None and raw_body_positions is not None:
            fk_errors.append(np.linalg.norm(data.xpos[raw_body_ids] - raw_body_positions[frame], axis=1))
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            geom1_environment = geom1 in environment_ids
            geom2_environment = geom2 in environment_ids
            if geom1_environment == geom2_environment:
                continue
            environment_geom = geom1 if geom1_environment else geom2
            robot_geom = geom2 if geom1_environment else geom1
            depth = max(-float(contact.dist), 0.0)
            if depth <= 0.0:
                continue
            robot_body_id = int(model.geom_bodyid[robot_geom])
            robot_body = _name(model, mujoco.mjtObj.mjOBJ_BODY, robot_body_id)
            environment = environment_names[environment_geom]
            is_foot = robot_body.startswith(_FOOT_BODY_PREFIXES)
            if robot_body not in body_frame_max_depth:
                body_frame_max_depth[robot_body] = np.zeros(len(qpos), dtype=np.float64)
                body_is_foot[robot_body] = is_foot
            body_frame_max_depth[robot_body][frame] = max(body_frame_max_depth[robot_body][frame], depth)
            frame_environment_contact[frame] = True
            frame_max_depth[frame] = max(frame_max_depth[frame], depth)
            if environment == "ground":
                frame_max_ground_depth[frame] = max(frame_max_ground_depth[frame], depth)
            else:
                frame_max_terrain_depth[frame] = max(frame_max_terrain_depth[frame], depth)
            if not is_foot:
                frame_max_nonfoot_depth[frame] = max(frame_max_nonfoot_depth[frame], depth)
            robot_geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom)
            key = (frame, robot_body, robot_geom_name, environment)
            existing = contact_records_by_key.get(key)
            if existing is None or depth > existing["depth_m"]:
                contact_records_by_key[key] = {
                    "frame": frame,
                    "file_frame": file_frame_offset + frame,
                    "time_s": frame / fps,
                    "depth_m": depth,
                    "robot_body": robot_body,
                    "robot_geom": robot_geom_name,
                    "environment": environment,
                    "is_foot": is_foot,
                }

    contact_records = sorted(
        contact_records_by_key.values(),
        key=lambda record: record["depth_m"],
        reverse=True,
    )
    over_tolerance = frame_max_depth > args.penetration_tolerance_m
    deep_penetration = frame_max_depth > args.deep_penetration_threshold_m
    nonfoot_over_tolerance = frame_max_nonfoot_depth > args.penetration_tolerance_m
    robot_body_breakdown = _robot_body_collision_breakdown(
        body_frame_max_depth,
        body_is_foot,
        penetration_tolerance_m=args.penetration_tolerance_m,
        deep_penetration_threshold_m=args.deep_penetration_threshold_m,
    )
    derivatives = _motion_derivatives(
        qpos,
        joint_columns,
        fps,
        args.history_frames,
        args.replan_stride,
        sequence_frame_offset,
        transition_kind,
    )

    fk_report: dict[str, Any] | None = None
    if fk_errors:
        flattened = np.concatenate(fk_errors)
        fk_report = {
            "body_origin_error_m": _summarize(flattened),
            "threshold_m": args.fk_body_error_threshold_m,
            "pass": bool(flattened.max() <= args.fk_body_error_threshold_m),
            "semantics": "generator-predicted 14 body origins versus MuJoCo FK from exported qpos",
        }

    quaternion_pass = derivatives["quaternion_norm"]["max_abs_error_from_one"] <= args.quaternion_norm_tolerance
    joint_limit_pass = len(violating) == 0
    collision_pass = not bool(np.any(over_tolerance))
    terrain_geom_count = sum(name.startswith(_TERRAIN_GEOM_PREFIX) for name in environment_names.values())
    terrain_coverage_pass = (
        args.terrain_urdf is not None and terrain_geom_count > 0
    ) or not args.require_terrain_urdf_for_gate
    continuity_gate = {
        "root_linear_speed": {
            "value": derivatives["root_linear_speed_m_s"]["max"],
            "limit": args.max_root_linear_speed_m_s,
        },
        "root_angular_speed": {
            "value": derivatives["root_angular_speed_rad_s"]["max"],
            "limit": args.max_root_angular_speed_rad_s,
        },
        "joint_speed": {
            "value": derivatives["joint_speed_abs_rad_s"]["max"],
            "limit": args.max_joint_speed_rad_s,
        },
        "joint_acceleration": {
            "value": derivatives["joint_acceleration_abs_rad_s2"]["max"],
            "limit": args.max_joint_acceleration_rad_s2,
        },
        "frame_joint_l2_step": {
            "value": derivatives["frame_joint_l2_step_rad"]["max"],
            "limit": args.max_frame_joint_l2_step_rad,
        },
    }
    for gate in continuity_gate.values():
        gate["pass"] = gate["value"] <= gate["limit"]
    continuity_pass = all(gate["pass"] for gate in continuity_gate.values())
    checks = {
        "finite_qpos": True,
        "quaternion_norm": quaternion_pass,
        "joint_position_limits": joint_limit_pass,
        "motion_continuity": continuity_pass,
        "environment_penetration_within_tolerance": collision_pass,
        "terrain_geometry_supplied": terrain_coverage_pass,
    }
    if enforce_raw_checks:
        raw_available = fk_report is not None and input_consistency is not None
        checks["raw_motion_supplied"] = raw_available or not args.require_raw_motion_for_gate
        checks["raw_qpos_joint_mapping"] = (
            bool(input_consistency and input_consistency["pass"])
            if raw_available
            else not args.require_raw_motion_for_gate
        )
        checks["generator_body_head_fk"] = (
            bool(fk_report and fk_report["pass"]) if raw_available else not args.require_raw_motion_for_gate
        )

    violating_joints = sorted({joint_names[joint_index] for _, joint_index in violating})
    return {
        "frames": len(qpos),
        "duration_s": len(qpos) / fps,
        "fps": fps,
        "checks": checks,
        "kinematic_gate_pass": all(checks.values()),
        "joint_limits": {
            "tolerance_rad": args.joint_limit_tolerance_rad,
            "max_violation_rad": float(joint_violation.max()),
            "violating_value_count": len(violating),
            "violating_frame_rate": float(np.mean(np.any(joint_violation > args.joint_limit_tolerance_rad, axis=1))),
            "violating_value_rate": float(np.mean(joint_violation > args.joint_limit_tolerance_rad)),
            "violating_joints": violating_joints,
        },
        "motion_continuity": {**derivatives, "gate": continuity_gate},
        "environment_collision": {
            "surface_tolerance_m": args.penetration_tolerance_m,
            "deep_penetration_threshold_m": args.deep_penetration_threshold_m,
            "max_depth_m": float(frame_max_depth.max()),
            "max_terrain_depth_m": float(frame_max_terrain_depth.max()),
            "max_ground_depth_m": float(frame_max_ground_depth.max()),
            "frame_max_depth_m": _summarize(frame_max_depth),
            "contact_frame_rate": float(frame_environment_contact.mean()),
            "over_tolerance_frame_rate": float(over_tolerance.mean()),
            "deep_penetration_frame_rate": float(deep_penetration.mean()),
            "nonfoot_over_tolerance_frame_rate": float(nonfoot_over_tolerance.mean()),
            "robot_body_breakdown": robot_body_breakdown,
            "robot_body_breakdown_semantics": (
                "contacting robot bodies only; rates use each body's maximum penetrating environment "
                "contact depth once per frame"
            ),
            "worst_contacts": contact_records[: args.worst_contacts],
            "semantics": (
                "negative MuJoCo pairwise contact signed gap after direct qpos placement; "
                "this is collision-algorithm output, not a general mesh signed-distance field; "
                "robot self-collision and environment geoms other than ground/named terrain boxes are excluded"
            ),
        },
        "fk_consistency": fk_report,
        "input_consistency": input_consistency,
    }


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flat.update(_flatten_scalars(child, child_prefix))
    elif isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
        flat[prefix] = float(value)
    return flat


def _compare_to_reference(generated: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    generated_collision = generated["environment_collision"]
    reference_collision = reference["environment_collision"]
    reference_max_depth = reference_collision["max_depth_m"]
    return {
        "max_environment_depth_ratio": generated_collision["max_depth_m"]
        / max(reference_max_depth, np.finfo(np.float64).eps),
        "max_environment_depth_delta_m": generated_collision["max_depth_m"] - reference_max_depth,
        "deep_penetration_frame_rate_delta": generated_collision["deep_penetration_frame_rate"]
        - reference_collision["deep_penetration_frame_rate"],
        "over_tolerance_frame_rate_delta": generated_collision["over_tolerance_frame_rate"]
        - reference_collision["over_tolerance_frame_rate"],
        "nonfoot_over_tolerance_frame_rate_delta": generated_collision["nonfoot_over_tolerance_frame_rate"]
        - reference_collision["nonfoot_over_tolerance_frame_rate"],
        "joint_limit_max_violation_delta_rad": generated["joint_limits"]["max_violation_rad"]
        - reference["joint_limits"]["max_violation_rad"],
        "periodic_transition_joint_step_max_ratio": generated["motion_continuity"]["periodic_transitions"][
            "joint_l2_step_rad"
        ]["max"]
        / max(
            reference["motion_continuity"]["periodic_transitions"]["joint_l2_step_rad"]["max"],
            np.finfo(np.float64).eps,
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: str | Path, role: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role}: {resolved}")
    return {
        "role": role,
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _terrain_asset_paths(urdf_path: str | Path) -> list[Path]:
    urdf = Path(urdf_path).resolve()
    tree = ET.parse(urdf)  # noqa: S314 - local, generated OmniRetarget asset
    mesh_files = [mesh.attrib.get("filename") for mesh in tree.iterfind(".//mesh")]
    if any(mesh_file is None for mesh_file in mesh_files):
        raise ValueError(f"{urdf}: terrain mesh is missing a filename")
    paths = {(urdf.parent / mesh_file).resolve() for mesh_file in mesh_files if mesh_file is not None}
    if not paths:
        raise ValueError(f"{urdf}: expected at least one terrain mesh")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Terrain URDF references missing assets: {missing}")
    return sorted(paths)


def _build_provenance(args: Args) -> dict[str, Any]:
    records = [
        _file_record(args.motion, "generated_qpos"),
        _file_record(args.robot_xml, "robot_mjcf"),
        _file_record(Path(__file__), "feasibility_evaluator_source"),
        _file_record(Path(__file__).with_name("view_motion_mj.py"), "mujoco_viewer_source"),
    ]
    optional_paths = (
        (args.raw_motion, "generated_raw"),
        (args.reference, "reference_motion_hash_only"),
        (args.terrain_urdf, "terrain_urdf"),
        (args.generator_checkpoint, "generator_checkpoint_hash_only"),
        (args.preview, "preview"),
    )
    records.extend(_file_record(path, role) for path, role in optional_paths if path is not None)
    if args.terrain_urdf is not None:
        records.extend(_file_record(path, "terrain_asset") for path in _terrain_asset_paths(args.terrain_urdf))
    records.extend(
        _file_record(path, f"additional_diagnostic_{index}") for index, path in enumerate(args.artifact_files)
    )
    generation = {
        "checkpoint": args.generator_checkpoint,
        "clip": args.generator_clip,
        "seed": args.generator_seed,
        "num_steps": args.generator_num_steps,
        "num_cycles": args.generator_num_cycles,
        "history_frames": args.history_frames,
        "replan_stride": args.replan_stride,
        "reference_start_frame": args.reference_start_frame,
    }
    complete = all(
        value is not None
        for value in (
            args.generator_checkpoint,
            args.generator_clip,
            args.generator_seed,
            args.generator_num_steps,
            args.generator_num_cycles,
            args.raw_motion,
            args.reference,
            args.terrain_urdf,
        )
    )
    return {
        "generation": generation,
        "complete_generation_metadata": complete,
        "files": records,
        "upload_policy": (
            "checkpoint and reference are hash-only; qpos/raw/report/preview, scene definition, "
            "source scripts, terrain assets, and requested diagnostics are uploaded"
        ),
    }


def _log_wandb(args: Args, report: dict[str, Any], output_path: Path) -> str | None:
    if args.wandb_mode == "disabled":
        return None
    if args.wandb_mode not in {"offline", "online"}:
        raise ValueError("wandb_mode must be disabled, offline, or online")
    try:
        import wandb  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "wandb logging requested but this environment has no wandb; use hsmujoco or disable logging"
        ) from error

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name or output_path.stem,
        mode=args.wandb_mode,
        config=asdict(args),
        job_type="terrain-feasibility-eval",
    )
    url = run.url
    report["wandb_url"] = url
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    run.log(
        _flatten_scalars(
            {
                "generated": report["generated"],
                "reference": report["reference"],
                "comparison": report["comparison_to_reference"],
            }
        )
    )
    artifact = wandb.Artifact(f"{run.name}-report", type="terrain-feasibility-report")
    artifact.add_file(str(output_path), name=f"report/{output_path.name}")
    artifact.add_file(args.motion, name=f"generated/{Path(args.motion).name}")
    if args.raw_motion is not None:
        artifact.add_file(args.raw_motion, name=f"generated/{Path(args.raw_motion).name}")
    if args.preview is not None:
        artifact.add_file(args.preview, name=f"generated/{Path(args.preview).name}")
    artifact.add_file(args.robot_xml, name=f"scene/{Path(args.robot_xml).name}")
    artifact.add_file(str(Path(__file__)), name=f"source/{Path(__file__).name}")
    viewer_source = Path(__file__).with_name("view_motion_mj.py")
    artifact.add_file(str(viewer_source), name=f"source/{viewer_source.name}")
    if args.terrain_urdf is not None:
        terrain_urdf = Path(args.terrain_urdf)
        artifact.add_file(str(terrain_urdf), name=f"scene/terrain/{terrain_urdf.name}")
        for asset in _terrain_asset_paths(terrain_urdf):
            relative = asset.relative_to(terrain_urdf.resolve().parent)
            artifact.add_file(str(asset), name=f"scene/terrain/{relative}")
    for index, path in enumerate(args.artifact_files):
        artifact.add_file(path, name=f"diagnostics/{index:02d}_{Path(path).name}")
    run.log_artifact(artifact)
    run.finish()
    return url


def main(args: Args) -> None:
    if args.history_frames < 1 or args.replan_stride < 1:
        raise ValueError("history_frames and replan_stride must be positive")
    if args.worst_contacts < 0:
        raise ValueError("worst_contacts must be non-negative")
    for name, value in {
        "penetration_tolerance_m": args.penetration_tolerance_m,
        "deep_penetration_threshold_m": args.deep_penetration_threshold_m,
        "joint_limit_tolerance_rad": args.joint_limit_tolerance_rad,
        "quaternion_norm_tolerance": args.quaternion_norm_tolerance,
        "fk_body_error_threshold_m": args.fk_body_error_threshold_m,
        "max_root_linear_speed_m_s": args.max_root_linear_speed_m_s,
        "max_root_angular_speed_rad_s": args.max_root_angular_speed_rad_s,
        "max_joint_speed_rad_s": args.max_joint_speed_rad_s,
        "max_joint_acceleration_rad_s2": args.max_joint_acceleration_rad_s2,
        "max_frame_joint_l2_step_rad": args.max_frame_joint_l2_step_rad,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    viewer_args = ViewerArgs(
        motion=args.motion,
        terrain_urdf=args.terrain_urdf,
        robot_xml=args.robot_xml,
    )
    model, _ = build_model(viewer_args)
    motion_qpos, motion_fps = load_qpos(args.motion)
    motion_qpos = _slice_qpos(
        motion_qpos,
        start=args.start_frame,
        count=args.num_frames,
        expected_nq=model.nq,
        label="motion",
    )
    raw_body_positions = None
    raw_body_names: tuple[str, ...] = ()
    input_consistency = None
    if args.raw_motion is not None:
        raw_body_positions, raw_body_names, raw_qpos, raw_joint_names = _load_raw_motion(
            args.raw_motion,
            start=args.start_frame,
            frames=len(motion_qpos),
        )
        model_joint_names = tuple(_robot_joint_columns(model)[0])
        names_match = raw_joint_names == model_joint_names
        max_qpos_error = float(np.max(np.abs(raw_qpos - motion_qpos)))
        input_consistency = {
            "raw_joint_names_match_mjcf": names_match,
            "raw_qpos_max_abs_error": max_qpos_error,
            "tolerance": 1.0e-6,
            "pass": names_match and max_qpos_error <= 1.0e-6,
            "semantics": "raw root/quaternion/joints versus exported qpos and MJCF hinge order",
        }

    generated = analyze_motion(
        model=model,
        qpos=motion_qpos,
        fps=motion_fps,
        args=args,
        raw_body_positions=raw_body_positions,
        raw_body_names=raw_body_names,
        input_consistency=input_consistency,
        enforce_raw_checks=True,
        sequence_frame_offset=args.start_frame,
        file_frame_offset=args.start_frame,
        transition_kind="generated_replan_boundaries",
    )
    reference = None
    if args.reference is not None:
        reference_qpos, reference_fps = load_qpos(args.reference)
        if abs(reference_fps - motion_fps) > 1.0e-6:
            raise ValueError(f"fps mismatch: generated={motion_fps}, reference={reference_fps}")
        reference_qpos = _slice_qpos(
            reference_qpos,
            start=args.reference_start_frame + args.start_frame,
            count=len(motion_qpos),
            expected_nq=model.nq,
            label="reference",
        )
        if len(reference_qpos) != len(motion_qpos):
            raise ValueError("reference does not contain enough frames for an aligned comparison")
        reference = analyze_motion(
            model=model,
            qpos=reference_qpos,
            fps=reference_fps,
            args=args,
            sequence_frame_offset=args.start_frame,
            file_frame_offset=args.reference_start_frame + args.start_frame,
            transition_kind="aligned_reference_periodic_samples",
        )

    comparison = _compare_to_reference(generated, reference) if reference is not None else None
    provenance = _build_provenance(args)
    report: dict[str, Any] = {
        "schema_version": 4,
        "kind": "kinematic-terrain-feasibility-audit",
        "config": asdict(args),
        "scene": {
            "robot_xml": str(Path(args.robot_xml).resolve()),
            "terrain_urdf": str(Path(args.terrain_urdf).resolve()) if args.terrain_urdf else None,
            "mujoco_version": mujoco.__version__,
        },
        "generated": generated,
        "reference": reference,
        "comparison_to_reference": comparison,
        "provenance": provenance,
        "verdict": {
            "kinematic_gate_pass": generated["kinematic_gate_pass"],
            "dynamic_feasibility": "not_evaluated",
            "scope": (
                "Direct qpos placement + mj_forward only. Passing does not establish balance, torque "
                "limits, contact stability, or tracker executability. The collision gate covers only "
                "ground and named terrain geoms, not robot self-collision."
            ),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    wandb_url = _log_wandb(args, report, output_path)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"JSON: {output_path}")
    if wandb_url is not None:
        print(f"W&B: {wandb_url}")
    if args.require_kinematic_gate and not generated["kinematic_gate_pass"]:
        raise AssertionError("Generated motion failed the configured kinematic feasibility gate")


if __name__ == "__main__":
    main(tyro.cli(Args))
