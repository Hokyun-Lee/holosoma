"""Replay a motion npz on the G1 model in MuJoCo (kinematic, no physics),
optionally overlaid with a ground-truth motion as a translucent second robot.

Accepts either format for both --motion and --gt:
    - qpos npz:  {qpos (T, 36) = [root pos, root quat wxyz, 29 joints], fps}
      (retargeting output or generated ``*_gen_qpos.npz`` from the motion
      generator's ``sample.py``)
    - WBT npz:   {joint_pos (T, 36), fps, ...}  (``*_mj.npz`` / processed files)

Modes:
    default          interactive mujoco.viewer window, loops the motion
    --video out.gif  offscreen render (no window needed); .mp4 requires an
                     ffmpeg backend, .gif works out of the box

GT overlay: pass the original clip and the frame where generation started
(``sample.py --start``); the GT robot is drawn translucent blue. Use
``--offset-y`` to place it side by side instead of overlaid.

Run from ``src/holosoma_retargeting/holosoma_retargeting/`` so the model path
resolves, e.g.:

    python data_conversion/view_motion_mj.py \\
        --motion <run>/samples/manual/xxx_gen_qpos.npz \\
        --gt ../../../data/motion_gen/processed/lafan1_walk4_subject1.npz \\
        --gt-start 0
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import numpy as np
import tyro


@dataclass
class Args:
    motion: str
    gt: str | None = None
    """Optional ground-truth motion npz drawn as a translucent second robot."""
    gt_start: int = 0
    """GT frame aligned with the first frame of --motion (= sample.py --start)."""
    offset_y: float = 0.0
    """Shift the GT robot in +y to view side by side instead of overlaid."""
    terrain_urdf: str | None = None
    """Optional OmniRetarget multi-box terrain URDF drawn as static boxes."""
    robot_xml: str = "models/g1/g1_29dof.xml"
    video: str | None = None
    """Output video path (.gif recommended; .mp4 needs imageio-ffmpeg)."""
    video_fps: int = 25
    """Playback fps of the exported video (frames are subsampled to this)."""
    width: int = 640
    height: int = 480
    loop: bool = True
    """Loop the motion in the interactive viewer."""


def load_qpos(path: str) -> tuple[np.ndarray, float]:
    data = np.load(path, allow_pickle=True)
    if "qpos" in data.files:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
    elif "joint_pos" in data.files:
        qpos = np.asarray(data["joint_pos"], dtype=np.float64)
    else:
        raise ValueError(f"{path}: expected a 'qpos' or 'joint_pos' key, got {list(data.files)}")
    if "fps" not in data.files:
        raise ValueError(f"{path}: missing 'fps' key")
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if not (0 < fps <= 1000):
        raise ValueError(f"{path}: implausible fps={fps}")
    return qpos, fps


import re


def _add_terrain_boxes(spec: "mujoco.MjSpec", urdf_path: Path) -> None:
    """Add the multi-box terrain as static geoms (parsed from the URDF objs)."""
    text = urdf_path.read_text()
    seen: set[str] = set()
    for match in re.finditer(r'<mesh filename="([^"]+)" scale="([^"]+)"', text):
        mesh_file, scale_str = match.groups()
        if mesh_file in seen:
            continue
        seen.add(mesh_file)
        scale = np.array([float(s) for s in scale_str.split()])
        verts = []
        for line in (urdf_path.parent / mesh_file).read_text().splitlines():
            if line.startswith("v "):
                verts.append([float(v) for v in line.split()[1:4]])
        v = np.asarray(verts) * scale
        # oriented box from the (yaw-rotated) footprint + flat top
        xy = np.unique(np.round(v[:, :2], 6), axis=0)
        center_xy = xy.mean(axis=0)
        rel = xy - center_xy
        order = np.argsort(np.arctan2(rel[:, 1], rel[:, 0]))
        corners = xy[order]
        e1 = corners[1] - corners[0]
        e2 = corners[2] - corners[1]
        yaw = float(np.arctan2(e1[1], e1[0]))
        half_a = float(np.linalg.norm(e1)) / 2
        half_b = float(np.linalg.norm(e2)) / 2
        z0, z1 = float(v[:, 2].min()), float(v[:, 2].max())
        body = spec.worldbody.add_body()
        body.pos = [float(center_xy[0]), float(center_xy[1]), (z0 + z1) / 2]
        body.quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [half_a, half_b, (z1 - z0) / 2]
        geom.rgba = [0.55, 0.45, 0.3, 1.0]


def build_model(args: Args) -> tuple["mujoco.MjModel", int | None]:
    """Compile the scene; returns (model, qpos offset of the GT robot or None)."""
    spec = mujoco.MjSpec.from_file(args.robot_xml)
    gt_offset = None
    if args.gt is not None:
        gt_spec = mujoco.MjSpec.from_file(args.robot_xml)
        for g in gt_spec.geoms:
            g.rgba = [0.3, 0.5, 1.0, 0.35]  # translucent blue ground truth
        spec.worldbody.add_frame().attach_body(gt_spec.bodies[1], "gt_", "")
    if args.terrain_urdf is not None:
        _add_terrain_boxes(spec, Path(args.terrain_urdf))
    model = spec.compile()
    if args.gt is not None:
        free_addrs = [
            int(model.jnt_qposadr[j])
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        assert len(free_addrs) == 2, "expected two free joints (generated + GT robots)"
        gt_offset = max(free_addrs)  # attached GT copy comes second
    return model, gt_offset


def set_frame(
    data: "mujoco.MjData",
    qpos: np.ndarray,
    gt_qpos: np.ndarray | None,
    frame: int,
    gt_offset: int | None,
    args: Args,
) -> None:
    n = qpos.shape[1]
    data.qpos[:n] = qpos[min(frame, qpos.shape[0] - 1)]
    if gt_qpos is not None and gt_offset is not None:
        g = min(args.gt_start + frame, gt_qpos.shape[0] - 1)
        data.qpos[gt_offset : gt_offset + n] = gt_qpos[g]
        data.qpos[gt_offset + 1] += args.offset_y


def track_pelvis(cam: "mujoco.MjvCamera", qpos_frame: np.ndarray) -> None:
    cam.lookat[:] = qpos_frame[:3]
    cam.distance = 3.0
    cam.elevation = -15.0
    cam.azimuth = 135.0


def run_viewer(model, data, qpos, gt_qpos, gt_offset, fps: float, args: Args) -> None:
    import mujoco.viewer as mjv  # type: ignore[import-not-found]

    dt = 1.0 / fps
    with mjv.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        frame = 0
        while viewer.is_running():
            start = time.perf_counter()
            set_frame(data, qpos, gt_qpos, frame, gt_offset, args)
            mujoco.mj_forward(model, data)
            viewer.sync()
            frame += 1
            if frame >= qpos.shape[0]:
                if not args.loop:
                    break
                frame = 0
            time.sleep(max(0.0, dt - (time.perf_counter() - start)))


def render_video(model, data, qpos, gt_qpos, gt_offset, fps: float, args: Args) -> None:
    import imageio.v2 as imageio

    out = Path(args.video)  # type: ignore[arg-type]
    stride = max(1, round(fps / args.video_fps))
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    frames = []
    for i in range(0, qpos.shape[0], stride):
        set_frame(data, qpos, gt_qpos, i, gt_offset, args)
        mujoco.mj_forward(model, data)
        track_pelvis(cam, qpos[i])
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    if out.suffix.lower() == ".gif":
        imageio.mimsave(out, frames, duration=1.0 / args.video_fps, loop=0)
    else:
        imageio.mimsave(out, frames, fps=args.video_fps)  # needs imageio-ffmpeg for mp4
    print(f"[OK] {len(frames)} frames -> {out}")


def main(args: Args) -> None:
    qpos, fps = load_qpos(args.motion)
    gt_qpos = None
    if args.gt is not None:
        gt_qpos, gt_fps = load_qpos(args.gt)
        if abs(gt_fps - fps) > 1e-6:
            raise ValueError(f"fps mismatch: motion {fps} vs gt {gt_fps}")
    model, gt_offset = build_model(args)
    # Kinematic replay only: disable contacts/constraints so overlapping
    # robots (generated + GT) cannot blow up the constraint solver.
    model.opt.disableflags |= (
        mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    )
    data = mujoco.MjData(model)
    if model.nq != qpos.shape[1] * (2 if gt_qpos is not None else 1):
        raise ValueError(f"{args.motion}: qpos has {qpos.shape[1]} columns, model expects nq={model.nq}")
    print(
        f"Replaying {args.motion}: {qpos.shape[0]} frames @ {fps:g} fps"
        + (f" | GT overlay: {args.gt} from frame {args.gt_start}" if args.gt else "")
    )
    if args.video:
        render_video(model, data, qpos, gt_qpos, gt_offset, fps, args)
    else:
        run_viewer(model, data, qpos, gt_qpos, gt_offset, fps, args)


if __name__ == "__main__":
    main(tyro.cli(Args))
