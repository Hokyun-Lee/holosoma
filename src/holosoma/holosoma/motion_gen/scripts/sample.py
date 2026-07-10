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

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from loguru import logger

from holosoma.motion_gen.dataset import load_wbt_motion
from holosoma.motion_gen.export import export_generated_qpos_npz, export_generated_raw_npz
from holosoma.motion_gen.sampling import MotionGenerator, MotionGeneratorInput
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
    replan_stride: int = 12
    heading: tuple[float, float] | None = None
    """Fixed world-frame target heading; None keeps the current direction
    (window mode uses the GT heading when None)."""
    out_dir: str | None = None


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

    if args.mode == "window":
        gt_future = clip.features[args.start + P : args.start + P + H]
        if heading is None:
            disp = gt_future[-1, :2] - clip.features[args.start + P - 1, :2]
            if disp.norm() > cfg.data.min_heading_disp:
                heading = (disp / disp.norm()).unsqueeze(0).to(gen.device)
        out = gen.generate(
            MotionGeneratorInput(past_motion=past, target_heading=heading),
            num_steps=args.num_steps, deterministic=args.deterministic,
            seed=args.seed, guidance_scale=args.guidance_scale,
        )
        stem = f"{clip.name}_s{args.start}_win"
        plot_window_comparison(out.features[0].cpu(), gt_future, gen.layout, out_dir / f"{stem}.png")
        export_generated_raw_npz(
            out.features[0].cpu(), gen.layout, cfg.data.fps, out_dir / f"{stem}_gen_raw.npz",
            gt_features=gt_future,
        )
        export_generated_qpos_npz(out.features[0].cpu(), gen.layout, cfg.data.fps, out_dir / f"{stem}_gen_qpos.npz")
    elif args.mode == "rollout":
        traj = gen.receding_horizon(
            past[0], num_cycles=args.num_cycles, replan_stride=args.replan_stride,
            target_heading=heading[0] if heading is not None else None,
            num_steps=args.num_steps, deterministic=args.deterministic, seed=args.seed,
        ).cpu()
        stem = f"{clip.name}_s{args.start}_rollout{args.num_cycles}x{args.replan_stride}"
        plot_long_rollout(traj, gen.layout, out_dir / f"{stem}.png")
        export_generated_raw_npz(traj, gen.layout, cfg.data.fps, out_dir / f"{stem}_gen_raw.npz")
        export_generated_qpos_npz(traj, gen.layout, cfg.data.fps, out_dir / f"{stem}_gen_qpos.npz")
        logger.info(
            "To build a full WBT-schema npz (all bodies + velocities) run, from "
            "src/holosoma_retargeting/holosoma_retargeting (hsretargeting env):\n"
            f"  python data_conversion/convert_data_format_mj_headless.py "
            f"--input-file {(out_dir / (stem + '_gen_qpos.npz')).resolve()} "
            f"--output-name {(out_dir / (stem + '_gen_mj.npz')).resolve()} --output-fps {int(cfg.data.fps)}"
        )
    else:
        raise ValueError(f"Unknown mode {args.mode}")
    logger.info(f"Outputs in {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
