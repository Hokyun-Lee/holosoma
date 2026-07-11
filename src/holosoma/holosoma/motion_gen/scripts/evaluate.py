"""Evaluate a trained generator on a split: sampling metrics vs. ground truth.

Usage (from the repo root, hssim env):
    python -m holosoma.motion_gen.scripts.evaluate \\
        --ckpt logs/motion_gen/debug/checkpoints/latest.pt --split val
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from loguru import logger
from torch.utils.data import DataLoader

from holosoma.motion_gen.dataset import MotionWindowDataset, load_split_clips
from holosoma.motion_gen.evaluation import compute_metrics, load_joint_limits
from holosoma.motion_gen.sampling import MotionGenerator, MotionGeneratorInput
from holosoma.motion_gen.visualization import plot_window_comparison


@dataclass
class Args:
    ckpt: str
    split: str = "val"  # "val" or "train"
    num_batches: int = 16
    batch_size: int = 64
    num_steps: int = 50
    """DDIM denoising steps (2 = paper deployment setting, experimental)."""
    seed: int = 123
    use_ema: bool = True
    out_dir: str | None = None
    """Defaults to <run_dir>/eval_<split>."""
    num_plots: int = 4


@torch.no_grad()
def main(args: Args) -> None:
    gen = MotionGenerator.from_checkpoint(args.ckpt, use_ema=args.use_ema)
    cfg = gen.cfg
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent.parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = load_split_clips(
        cfg.data.processed_dir, cfg.data.splits_file, args.split, gen.layout,
        cfg.data.metadata_dir, cfg.data.fps,
    )
    dataset = MotionWindowDataset(
        clips, gen.layout,
        past_frames=cfg.data.past_frames, future_frames=cfg.data.future_frames,
        stride=cfg.data.val_stride, min_heading_disp=cfg.data.min_heading_disp,
        terrain_dim=cfg.data.terrain_dim, use_terrain_scan=cfg.data.use_terrain_scan,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    joint_limits = load_joint_limits(Path(cfg.data.metadata_dir) / "joint_limits.json", gen.layout)

    torch.manual_seed(args.seed)
    agg: dict[str, float] = {}
    counts: dict[str, int] = {}
    plotted = 0
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        batch = {k: v.to(gen.device) for k, v in batch.items()}
        # Dataset windows are canonical; the anchor is at origin/yaw=0, so
        # generate() applies an identity canonicalization.
        out = gen.generate(
            MotionGeneratorInput(
                past_motion=batch["past"],
                target_heading=batch["heading"],
                terrain_height=batch["terrain"],
            ),
            num_steps=args.num_steps,
            deterministic=True,
            seed=args.seed + i,
        )
        metrics = compute_metrics(
            out.features, batch["x"], gen.layout, cfg.data.fps,
            joint_limits=joint_limits, contact=batch["contact"], flat=batch["flat"],
            terrain_scan=batch["terrain"] if cfg.data.use_terrain_scan else None,
            has_scan=batch.get("has_scan"),
            scan_grid=cfg.data.scan_grid if cfg.data.use_terrain_scan else None,
        )
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
        if plotted < args.num_plots:
            plot_window_comparison(
                out.features[0].cpu(), batch["x"][0].cpu(), gen.layout,
                out_dir / f"eval_window_{i:02d}.png",
            )
            plotted += 1

    result = {
        "checkpoint": args.ckpt,
        "checkpoint_step": gen.checkpoint_step,
        "split": args.split,
        "num_steps": args.num_steps,
        "num_batches": min(args.num_batches, len(loader)),
        "metrics": {k: agg[k] / counts[k] for k in agg},
    }
    (out_dir / "metrics_eval.json").write_text(json.dumps(result, indent=2))
    logger.info(json.dumps(result["metrics"], indent=2))
    logger.info(f"Wrote {out_dir / 'metrics_eval.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
