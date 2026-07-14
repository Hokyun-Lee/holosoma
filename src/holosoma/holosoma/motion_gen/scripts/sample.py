"""Generate motion from a trained checkpoint.

Modes:
    window   generate one 0.5 s future window conditioned on a GT clip window
             and compare against ground truth (plots + npz).
    rollout  receding-horizon generation: re-plan every ``replan_stride``
             frames, stitch a long motion, export a HoloSoma-layout qpos npz
             that ``convert_data_format_mj_headless.py`` can turn into a full
             WBT-schema motion for replay/tracking.

Usage (from the repo root, hssim env):
    python -m holosoma.motion_gen.scripts.sample \\
        --ckpt logs/motion_gen/debug/checkpoints/latest.pt \\
        --clip lafan1_walk4_subject1 --mode window --start 100
    python -m holosoma.motion_gen.scripts.sample \\
        --ckpt ... --clip lafan1_walk4_subject1 --mode rollout --num-cycles 16
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from loguru import logger

from holosoma.motion_gen.dataset import load_wbt_motion
from holosoma.motion_gen.export import export_generated_qpos_npz, export_generated_raw_npz
from holosoma.motion_gen.features import quat_yaw
from holosoma.motion_gen.sampling import MotionGenerator, MotionGeneratorInput
from holosoma.motion_gen.terrain import BoxTerrain
from holosoma.motion_gen.visualization import plot_long_rollout, plot_window_comparison


@dataclass
class Args:
    ckpt: str
    clip: str
    """Processed clip stem (e.g. lafan1_walk4_subject1) or a path to an npz."""
    mode: str = "window"  # "window" | "rollout"
    start: int = 0
    """Start frame in the clip for conditioning."""
    num_steps: int | None = 50
    """DDIM steps; None = full-step DDPM; 2 = paper deployment (experimental)."""
    deterministic: bool = True
    seed: int = 0
    guidance_scale: float | None = None
    num_cycles: int = 16
    replan_stride: int | None = None
    """Frames consumed per generator query. None uses the full checkpoint horizon;
    shorter explicit values are generator-only diagnostics."""
    heading: tuple[float, float] | None = None
    """Fixed world-frame target heading. In rollout mode, None captures the
    initial anchor facing once; window mode uses the GT heading when None."""
    terrain_urdf: str | None = None
    """Phase B: multi-box terrain URDF for scan conditioning. Window mode of a
    scan-enabled checkpoint uses the clip's stored scans automatically; this
    flag is for rollout mode, where scans must follow the generated root."""
    out_dir: str | None = None


def _resolve_replan_stride(requested: int | None, future_frames: int) -> int:
    """Resolve the CLI default before naming or writing rollout artifacts."""
    if future_frames < 1:
        raise ValueError(f"future_frames must be positive, got {future_frames}")
    resolved = future_frames if requested is None else requested
    if resolved < 1 or resolved > future_frames:
        raise ValueError(f"replan_stride must be in [1, {future_frames}] or None, got {requested}")
    return resolved


def _rollout_stem(
    clip_name: str,
    *,
    start: int,
    num_cycles: int,
    resolved_stride: int,
) -> str:
    return f"{clip_name}_s{start}_rollout{num_cycles}x{resolved_stride}"


@torch.no_grad()
def main(args: Args) -> None:
    gen = MotionGenerator.from_checkpoint(args.ckpt)
    cfg = gen.cfg
    clip_path = Path(args.clip) if args.clip.endswith(".npz") else Path(cfg.data.processed_dir) / f"{args.clip}.npz"
    clip = load_wbt_motion(clip_path, gen.layout, expected_fps=cfg.data.fps)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent.parent / "samples" / "manual"
    out_dir.mkdir(parents=True, exist_ok=True)

    P, H = cfg.data.past_frames, cfg.data.future_frames
    if args.start + P + H > clip.num_frames:
        raise ValueError(f"start {args.start} too late: clip has {clip.num_frames} frames, need {P + H}")

    past = clip.features[args.start : args.start + P].unsqueeze(0).to(gen.device)
    heading = torch.tensor([list(args.heading)], device=gen.device) if args.heading else None

    use_scan = cfg.data.use_terrain_scan
    if args.mode == "window":
        gt_future = clip.features[args.start + P : args.start + P + H]
        if heading is None:
            disp = gt_future[-1, :2] - clip.features[args.start + P - 1, :2]
            if disp.norm() > cfg.data.min_heading_disp:
                heading = (disp / disp.norm()).unsqueeze(0).to(gen.device)
        terrain = None
        if use_scan and clip.terrain_scan is not None:
            terrain = clip.terrain_scan[args.start + P - 1].unsqueeze(0).to(gen.device)
            logger.info("Conditioning on the clip's stored terrain scan (anchor frame)")
        out = gen.generate(
            MotionGeneratorInput(past_motion=past, target_heading=heading, terrain_height=terrain),
            num_steps=args.num_steps,
            deterministic=args.deterministic,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
        )
        stem = f"{clip.name}_s{args.start}_win"
        plot_window_comparison(out.features[0].cpu(), gt_future, gen.layout, out_dir / f"{stem}.png")
        export_generated_raw_npz(
            out.features[0].cpu(),
            gen.layout,
            cfg.data.fps,
            out_dir / f"{stem}_gen_raw.npz",
            gt_features=gt_future,
        )
        export_generated_qpos_npz(out.features[0].cpu(), gen.layout, cfg.data.fps, out_dir / f"{stem}_gen_qpos.npz")
    elif args.mode == "rollout":
        resolved_stride = _resolve_replan_stride(args.replan_stride, H)
        rollout_heading = heading
        heading_mode = "explicit_fixed_world"
        if rollout_heading is None:
            initial_yaw = quat_yaw(past[:, -1, gen.layout.root_quat_slice])
            rollout_heading = torch.stack(
                (torch.cos(initial_yaw), torch.sin(initial_yaw)),
                dim=-1,
            )
            heading_mode = "initial_facing_fixed_world"
        terrain_fn = None
        if args.terrain_urdf is not None:
            if not use_scan:
                raise ValueError("--terrain-urdf given but the checkpoint was trained without terrain scans")
            terrain = BoxTerrain.from_urdf(args.terrain_urdf)
            grid = cfg.data.scan_grid

            def terrain_fn(past_win: torch.Tensor) -> torch.Tensor:
                anchor = past_win[:, -1]
                scans = [
                    terrain.sample_scan(anchor[b, :2].cpu().numpy(), float(quat_yaw(anchor[b, 3:7])), grid)
                    for b in range(anchor.shape[0])
                ]
                return torch.from_numpy(np.stack(scans)).float().to(gen.device)

        traj = gen.receding_horizon(
            past[0],
            num_cycles=args.num_cycles,
            replan_stride=resolved_stride,
            target_heading=rollout_heading[0],
            num_steps=args.num_steps,
            deterministic=args.deterministic,
            seed=args.seed,
            terrain_fn=terrain_fn,
        ).cpu()
        stem = _rollout_stem(
            clip.name,
            start=args.start,
            num_cycles=args.num_cycles,
            resolved_stride=resolved_stride,
        )
        plot_long_rollout(traj, gen.layout, out_dir / f"{stem}.png")
        raw_path = export_generated_raw_npz(
            traj,
            gen.layout,
            cfg.data.fps,
            out_dir / f"{stem}_gen_raw.npz",
        )
        qpos_path = export_generated_qpos_npz(
            traj,
            gen.layout,
            cfg.data.fps,
            out_dir / f"{stem}_gen_qpos.npz",
        )
        metadata_path = out_dir / f"{stem}_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "motion-generator-receding-horizon-sample",
                    "checkpoint": str(Path(args.ckpt).expanduser().resolve()),
                    "clip": str(clip_path.expanduser().resolve()),
                    "start_frame": args.start,
                    "past_frames": P,
                    "future_frames": H,
                    "fps": cfg.data.fps,
                    "num_cycles": args.num_cycles,
                    "requested_replan_stride": args.replan_stride,
                    "resolved_replan_stride": resolved_stride,
                    "full_horizon_stride": resolved_stride == H,
                    "trajectory_frames": int(traj.shape[0]),
                    "num_steps": args.num_steps,
                    "deterministic": args.deterministic,
                    "seed": args.seed,
                    "guidance_scale": args.guidance_scale,
                    "requested_target_heading_world_xy": (list(args.heading) if args.heading is not None else None),
                    "resolved_target_heading_world_xy": [float(value) for value in rollout_heading[0].detach().cpu()],
                    "heading_mode": heading_mode,
                    "terrain_urdf": args.terrain_urdf,
                    "files": {
                        "raw": str(raw_path.resolve()),
                        "qpos": str(qpos_path.resolve()),
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        logger.info(
            "To build a full WBT-schema npz (all bodies + velocities) run, from "
            "src/holosoma_retargeting/holosoma_retargeting (hsretargeting env):\n"
            f"  python data_conversion/convert_data_format_mj_headless.py "
            f"--input-file {(out_dir / (stem + '_gen_qpos.npz')).resolve()} "
            f"--output-name {(out_dir / (stem + '_gen_mj.npz')).resolve()} --output-fps {int(cfg.data.fps)}"
        )
        logger.info(f"Resolved replan stride: {resolved_stride}/{H} frames; metadata: {metadata_path}")
    else:
        raise ValueError(f"Unknown mode {args.mode}")
    logger.info(f"Outputs in {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
