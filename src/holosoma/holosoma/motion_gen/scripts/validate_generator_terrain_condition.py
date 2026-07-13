"""Check that a scan-trained generator responds to terrain conditioning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import tyro

from holosoma.motion_gen.dataset import load_wbt_motion
from holosoma.motion_gen.sampling import MotionGenerator, MotionGeneratorInput


@dataclass
class Args:
    checkpoint: str = "logs/motion_gen/terrain_4090/checkpoints/final.pt"
    clip: str = "data/motion_gen/processed_paperscale/lafan1_walk4_subject1.npz"
    start: int = 100
    device: str = "cuda:0"
    seed: int = 123
    obstacle_height: float = 0.3


@torch.no_grad()
def main(args: Args) -> None:
    generator = MotionGenerator.from_checkpoint(args.checkpoint, device=args.device)
    if not generator.cfg.data.use_terrain_scan:
        raise ValueError("Checkpoint was not trained with terrain scans")
    clip = load_wbt_motion(args.clip, generator.layout, expected_fps=generator.cfg.data.fps)
    past_frames = generator.cfg.data.past_frames
    past = clip.features[args.start : args.start + past_frames].unsqueeze(0).to(generator.device)
    if past.shape[1] != past_frames:
        raise ValueError(f"Clip is too short for start={args.start} and past_frames={past_frames}")

    heading = torch.tensor([[1.0, 0.0]], device=generator.device)
    grid = generator.cfg.data.scan_grid
    offsets = grid.offsets_tensor(device=generator.device)
    flat = torch.zeros(1, grid.dim, device=generator.device)
    obstacle = flat.clone()
    obstacle_mask = (
        (offsets[:, 0] >= 0.3)
        & (offsets[:, 0] <= 0.9)
        & (offsets[:, 1].abs() <= 0.4)
    )
    obstacle[:, obstacle_mask] = args.obstacle_height

    sample_kwargs = {"num_steps": 2, "deterministic": True, "seed": args.seed}
    flat_output = generator.generate(
        MotionGeneratorInput(past_motion=past, target_heading=heading, terrain_height=flat),
        **sample_kwargs,
    )
    obstacle_output = generator.generate(
        MotionGeneratorInput(past_motion=past, target_heading=heading, terrain_height=obstacle),
        **sample_kwargs,
    )

    deltas = {
        "feature": (flat_output.features - obstacle_output.features).abs(),
        "root": (flat_output.root_pos - obstacle_output.root_pos).abs(),
        "joint": (flat_output.joint_pos - obstacle_output.joint_pos).abs(),
        "body": (flat_output.body_pos - obstacle_output.body_pos).abs(),
    }
    for name, delta in deltas.items():
        print(f"{name}_mean_abs_delta={float(delta.mean()):.8f}")
        print(f"{name}_max_abs_delta={float(delta.max()):.8f}")
    if float(deltas["feature"].max()) <= 1e-4:
        raise AssertionError("Generator output did not respond meaningfully to the terrain scan")


if __name__ == "__main__":
    main(tyro.cli(Args))
