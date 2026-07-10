"""Headless batch version of ``convert_data_format_mj.py`` (no viewer, no sleep).

Converts retargeted qpos motions into the HoloSoma WBT npz format by running
MuJoCo forward kinematics frame by frame. Input npz layout (HoloSoma):

    qpos: (T, 7 + DOF) = [root_pos(3), root_quat wxyz(4), joints(DOF)]
    fps:  scalar frames-per-second (e.g. 30)

Output npz (same schema as convert_data_format_mj.py):
    fps [output_fps], joint_pos (T,7+DOF), joint_vel (T,6+DOF),
    body_pos_w/quat_w/lin_vel_w/ang_vel_w (T, nbody, ...),
    joint_names (DOF), body_names (nbody)

Notes:
    - Joint columns in the input must follow DataConversionConfig.JOINT_NAMES
      order; they are remapped to the MuJoCo model's dof order internally.
    - The interactive script's MotionLoader reads the npz "fps" key as
      seconds-per-frame (``round(1/fps)``), which breaks for files storing
      frames-per-second; this script reads frames-per-second directly and is
      the supported path for the motion-generator data pipeline.
    - Run from ``src/holosoma_retargeting/holosoma_retargeting/`` so the
      relative MuJoCo model paths (models/g1/...) resolve, e.g.:

      python data_conversion/convert_data_format_mj_headless.py \\
          --input-dir ../../../data/motion_gen/raw_qpos \\
          --output-dir ../../../data/motion_gen/processed \\
          --joint-limits-out ../../../data/motion_gen/metadata/joint_limits.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import numpy as np
import torch
import tyro

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from convert_data_format_mj import quat_conjugate, quat_mul, quat_to_rotvec, world_body_velocities  # noqa: E402
from holosoma_retargeting.config_types.data_conversion import _ROBOT_JOINT_NAMES_DEFAULT  # noqa: E402


@dataclass
class HeadlessConversionConfig:
    input_file: str | None = None
    """Single input qpos npz (mutually exclusive with input_dir)."""
    input_dir: str | None = None
    """Directory of input qpos npz files (all *.npz are converted)."""
    output_name: str | None = None
    """Output path for single-file mode."""
    output_dir: str | None = None
    """Output directory for directory mode (files keep their stem)."""
    robot: str = "g1"
    robot_xml: str = "models/g1/g1_29dof.xml"
    output_fps: int = 50
    joint_limits_out: str | None = None
    """Optional path to dump {joint_name: [lo, hi]} limits from the model."""
    overwrite: bool = False


def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    if t.ndim == q0.ndim - 1:
        t = t.unsqueeze(-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    close = sin_theta.abs() < eps
    s0 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    s1 = torch.sin(t * theta) / (sin_theta + eps)
    out = torch.where(close, (1.0 - t) * q0 + t * q1, s0 * q0 + s1 * q1)
    return torch.nn.functional.normalize(out, dim=-1)


def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = quat_to_rotvec(q_rel) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def load_and_resample(path: Path, output_fps: int) -> dict[str, torch.Tensor]:
    data = np.load(path)
    if "qpos" not in data or "fps" not in data:
        raise ValueError(f"{path}: expected keys 'qpos' and 'fps', got {list(data.files)}")
    qpos = torch.from_numpy(np.asarray(data["qpos"], dtype=np.float32))
    input_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if input_fps <= 0 or input_fps > 1000:
        raise ValueError(f"{path}: implausible fps={input_fps}")

    base_pos, base_quat, dof_pos = qpos[:, 0:3], qpos[:, 3:7], qpos[:, 7:]
    frames = qpos.shape[0]
    duration = (frames - 1) / input_fps
    times = torch.arange(0, duration, 1.0 / output_fps, dtype=torch.float32)
    phase = times / duration
    idx0 = (phase * (frames - 1)).floor().long()
    idx1 = torch.minimum(idx0 + 1, torch.tensor(frames - 1))
    blend = (phase * (frames - 1) - idx0).unsqueeze(-1)

    out_pos = base_pos[idx0] * (1 - blend) + base_pos[idx1] * blend
    out_quat = _slerp(base_quat[idx0], base_quat[idx1], blend.squeeze(-1))
    out_dof = dof_pos[idx0] * (1 - blend) + dof_pos[idx1] * blend

    dt = 1.0 / output_fps
    return {
        "base_pos": out_pos,
        "base_quat": out_quat,
        "dof_pos": out_dof,
        "base_lin_vel": torch.gradient(out_pos, spacing=dt, dim=0)[0],
        "base_ang_vel": _so3_derivative(out_quat, dt),
        "dof_vel": torch.gradient(out_dof, spacing=dt, dim=0)[0],
    }


def convert_file(
    path: Path,
    out_path: Path,
    model: "mujoco.MjModel",
    joint_names: list[str],
    output_fps: int,
) -> None:
    motion = load_and_resample(path, output_fps)
    n_frames = motion["base_pos"].shape[0]
    data = mujoco.MjData(model)

    model_dof_names = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        model_dof_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
    dof_index = [joint_names.index(n) for n in model_dof_names]

    log: dict[str, list] = {k: [] for k in (
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"
    )}
    for f in range(n_frames):
        data.qpos[:3] = motion["base_pos"][f].numpy()
        data.qpos[3:7] = motion["base_quat"][f].numpy()
        data.qpos[7:] = motion["dof_pos"][f].numpy()[dof_index]
        data.qvel[:3] = motion["base_lin_vel"][f].numpy()
        data.qvel[3:6] = motion["base_ang_vel"][f].numpy()
        data.qvel[6:] = motion["dof_vel"][f].numpy()[dof_index]
        mujoco.mj_forward(model, data)
        lin_w, ang_w = world_body_velocities(model, data)
        log["joint_pos"].append(data.qpos.copy())
        log["joint_vel"].append(data.qvel.copy())
        log["body_pos_w"].append(data.xpos.copy())
        log["body_quat_w"].append(data.xquat.copy())
        log["body_lin_vel_w"].append(lin_w.copy())
        log["body_ang_vel_w"].append(ang_w.copy())

    out: dict[str, np.ndarray] = {k: np.stack(v, axis=0) for k, v in log.items()}
    out["fps"] = np.array([output_fps])
    all_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    out["joint_names"] = np.array(all_joint_names[1:])  # drop root free joint
    out["body_names"] = np.array(
        [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"[OK] {path.name}: {n_frames} frames @ {output_fps} fps -> {out_path}")


def dump_joint_limits(model: "mujoco.MjModel", path: Path) -> None:
    limits = {}
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        lo, hi = model.jnt_range[i]
        limits[name] = [float(lo), float(hi)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(limits, indent=2))
    print(f"[OK] joint limits -> {path}")


def main(cfg: HeadlessConversionConfig) -> None:
    if cfg.robot not in _ROBOT_JOINT_NAMES_DEFAULT:
        raise ValueError(f"No joint names for robot '{cfg.robot}'")
    joint_names = _ROBOT_JOINT_NAMES_DEFAULT[cfg.robot]

    model = mujoco.MjModel.from_xml_path(cfg.robot_xml)
    if cfg.joint_limits_out:
        dump_joint_limits(model, Path(cfg.joint_limits_out))

    jobs: list[tuple[Path, Path]] = []
    if cfg.input_file:
        if not cfg.output_name:
            raise ValueError("--output-name is required with --input-file")
        jobs.append((Path(cfg.input_file), Path(cfg.output_name)))
    elif cfg.input_dir:
        out_dir = Path(cfg.output_dir or cfg.input_dir)
        for p in sorted(Path(cfg.input_dir).glob("*.npz")):
            jobs.append((p, out_dir / p.name))
    elif not cfg.joint_limits_out:
        raise ValueError("Provide --input-file or --input-dir (or only --joint-limits-out)")

    for src, dst in jobs:
        if dst.exists() and not cfg.overwrite:
            print(f"[skip] {dst} exists (use --overwrite to redo)")
            continue
        convert_file(src, dst, model, joint_names, cfg.output_fps)


if __name__ == "__main__":
    main(tyro.cli(HeadlessConversionConfig))
