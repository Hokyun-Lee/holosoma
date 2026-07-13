"""Inference API for the trained motion generator.

Designed as the interface the future RL motion tracker will call:

    gen = MotionGenerator.from_checkpoint("logs/motion_gen/<run>/checkpoints/latest.pt")
    out = gen.generate(MotionGeneratorInput(past_motion=..., target_heading=...))

Inputs/outputs are world-frame; canonicalization (heading-normalized anchor
frame) happens inside. Receding-horizon generation follows the paper's
deployment scheme (re-plan every 0.25 s = 12/13 frames at 50 Hz; here the
stride is configurable), feeding generated frames back as the next past.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from holosoma.motion_gen.configs import (
    ConditionNoiseCfg,
    DataCfg,
    DiffusionCfg,
    ModelCfg,
    TrainConfig,
)
from holosoma.motion_gen.diffusion import GaussianDiffusion
from holosoma.motion_gen.features import (
    FeatureLayout,
    canonicalize_window,
    decanonicalize_window,
    pack_features,
    quat_normalize,
    unpack_features,
    yaw_rotate_xy,
)
from holosoma.motion_gen.losses import LossWeights
from holosoma.motion_gen.model import MotionDiffusionTransformer
from holosoma.motion_gen.normalization import FeatureNormalizer
from holosoma.motion_gen.terrain import ScanGrid
from holosoma.motion_gen.training import CKPT_FORMAT_VERSION


@dataclass
class MotionGeneratorInput:
    past_motion: torch.Tensor
    """(B, P, D) world-frame packed features of the past conditioning frames."""
    target_heading: torch.Tensor | None = None
    """(B, 2) desired world-frame xy heading (unit vector). None = keep the
    current facing direction."""
    terrain_height: torch.Tensor | None = None
    """(B, terrain_dim) height scan. None = flat zeros (Phase A)."""
    mask: torch.Tensor | None = None
    """(B, H) optional future-frame validity mask."""


@dataclass
class MotionGeneratorOutput:
    root_pos: torch.Tensor  # (B, H, 3) world
    root_quat: torch.Tensor  # (B, H, 4) wxyz, world, normalized
    joint_pos: torch.Tensor  # (B, H, J)
    body_pos: torch.Tensor  # (B, H, B, 3) world
    features: torch.Tensor  # (B, H, D) world-frame packed features
    metadata: dict = field(default_factory=dict)


class MotionGenerator:
    def __init__(
        self,
        model: MotionDiffusionTransformer,
        diffusion: GaussianDiffusion,
        normalizer: FeatureNormalizer,
        layout: FeatureLayout,
        cfg: TrainConfig,
        device: torch.device,
        checkpoint_step: int = -1,
    ):
        # Inference wrappers are also used inside PPO.  ``torch.no_grad`` on
        # generate() avoids graph construction, while requires_grad_(False)
        # makes the frozen-generator contract explicit and auditable.
        self.model = model.eval().to(device).requires_grad_(False)
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.layout = layout
        self.cfg = cfg
        self.device = device
        self.checkpoint_step = checkpoint_step

    @staticmethod
    def from_checkpoint(
        path: str,
        device: str = "auto",
        use_ema: bool = True,
    ) -> MotionGenerator:
        dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        if ckpt.get("format_version") != CKPT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format {ckpt.get('format_version')} (expected {CKPT_FORMAT_VERSION})."
            )
        layout = FeatureLayout.from_metadata(ckpt["layout"])
        cfg = _config_from_dict(ckpt["config"])

        model = MotionDiffusionTransformer(
            feature_dim=layout.dim,
            past_frames=cfg.data.past_frames,
            future_frames=cfg.data.future_frames,
            terrain_dim=cfg.data.terrain_dim,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            d_ff=cfg.model.d_ff,
            dropout=cfg.model.dropout,
        )
        state = ckpt["ema"] if (use_ema and ckpt.get("ema") is not None) else ckpt["model"]
        model.load_state_dict(state)
        diffusion = GaussianDiffusion(
            timesteps=cfg.diffusion.timesteps, schedule=cfg.diffusion.schedule, param=cfg.diffusion.param
        )
        normalizer = FeatureNormalizer.from_state_dict(ckpt["normalizer"])
        if normalizer.dim != layout.dim:
            raise ValueError("Checkpoint normalizer dim does not match feature layout.")
        return MotionGenerator(model, diffusion, normalizer, layout, cfg, dev, int(ckpt.get("step", -1)))

    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        inp: MotionGeneratorInput,
        num_steps: int | None = 50,
        deterministic: bool = False,
        seed: int = 0,
        guidance_scale: float | None = None,
    ) -> MotionGeneratorOutput:
        """Generate H future frames (world frame) for a batch of conditions.

        num_steps: DDIM steps; None = full-step ancestral DDPM. Few-step
        settings (e.g. the paper's deployment value of 2) are experimental.
        deterministic: fixed-seed initial noise + DDIM eta=0.
        guidance_scale: classifier-free guidance (needs cond_dropout > 0 at
        training time); combined linearly in the model's prediction space.
        """
        past_world = inp.past_motion.to(self.device)
        bsz, P, D = past_world.shape
        if self.layout.dim != D:
            raise ValueError(f"past_motion feature dim {D} != layout dim {self.layout.dim}")
        H = self.cfg.data.future_frames

        past_canon, transform = canonicalize_window(past_world, self.layout, anchor_index=P - 1)
        if inp.target_heading is not None:
            hw = inp.target_heading.to(self.device)
            h3 = torch.cat([hw, torch.zeros_like(hw[..., :1])], dim=-1)
            heading = yaw_rotate_xy(-transform.anchor_yaw, h3)[..., :2]
            heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            heading = torch.zeros(bsz, 2, device=self.device)
            heading[:, 0] = 1.0  # keep current facing direction
        terrain = (
            inp.terrain_height.to(self.device)
            if inp.terrain_height is not None
            else torch.zeros(bsz, self.cfg.data.terrain_dim, device=self.device)
        )

        past_norm = self.normalizer.normalize(past_canon)
        seq_mask = inp.mask.to(self.device) if inp.mask is not None else None

        def model_fn(x_t, t, **_):
            pred = self.model(x_t, t, past_norm, heading, terrain, seq_mask=seq_mask)
            if guidance_scale is not None and guidance_scale != 1.0:
                ones = torch.ones(bsz, dtype=torch.bool, device=self.device)
                uncond = self.model(
                    x_t, t, past_norm, heading, terrain,
                    drop_past=ones, drop_heading=ones, drop_terrain=ones, seq_mask=seq_mask,
                )
                pred = uncond + guidance_scale * (pred - uncond)
            return pred

        generator = None
        if deterministic:
            generator = torch.Generator(device=self.device.type).manual_seed(seed)
        shape = (bsz, H, self.layout.dim)
        if num_steps is None:
            x0_norm = self.diffusion.ddpm_sample(model_fn, shape, self.device, generator=generator)
        else:
            x0_norm = self.diffusion.ddim_sample(
                model_fn, shape, self.device, num_steps=num_steps, eta=0.0, generator=generator
            )

        canon = self.normalizer.denormalize(x0_norm)
        world = decanonicalize_window(canon, self.layout, transform)
        parts = unpack_features(world, self.layout)
        root_quat = quat_normalize(parts["root_quat"])
        # Repack with the normalized quaternion so downstream consumers
        # (receding-horizon feedback, export) never see off-manifold quats.
        world = pack_features(parts["root_pos"], root_quat, parts["joint_pos"], parts["body_pos"])
        return MotionGeneratorOutput(
            root_pos=parts["root_pos"],
            root_quat=root_quat,
            joint_pos=parts["joint_pos"],
            body_pos=parts["body_pos"],
            features=world,
            metadata={
                "num_steps": num_steps,
                "deterministic": deterministic,
                "guidance_scale": guidance_scale,
                "checkpoint_step": self.checkpoint_step,
                "fps": self.cfg.data.fps,
            },
        )

    @torch.no_grad()
    def receding_horizon(
        self,
        past_motion: torch.Tensor,
        num_cycles: int = 8,
        replan_stride: int = 12,
        target_heading: torch.Tensor | Callable[[int], torch.Tensor] | None = None,
        num_steps: int | None = 50,
        deterministic: bool = False,
        seed: int = 0,
        terrain_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Stitch a long motion by re-planning every ``replan_stride`` frames.

        The paper re-plans every 0.25 s (12.5 frames at 50 Hz) at deployment;
        default 12. Returns (P + num_cycles * replan_stride, D) world features
        for a single sequence (unbatched input (P, D) is accepted).

        terrain_fn: optional callback mapping the current past window
        (B, P, D world features) to a (B, terrain_dim) height scan around the
        anchor pose (Phase B terrain conditioning during rollout).
        """
        single = past_motion.ndim == 2
        input_device = past_motion.device
        past = (past_motion.unsqueeze(0) if single else past_motion).to(self.device)
        P = past.shape[1]
        frames = [past.clone()]
        for cycle in range(num_cycles):
            if callable(target_heading):
                heading = target_heading(cycle)
                heading = heading.unsqueeze(0) if heading.ndim == 1 else heading
            elif target_heading is not None:
                heading = target_heading.unsqueeze(0) if target_heading.ndim == 1 else target_heading
            else:
                heading = None
            terrain = terrain_fn(past) if terrain_fn is not None else None
            out = self.generate(
                MotionGeneratorInput(past_motion=past, target_heading=heading, terrain_height=terrain),
                num_steps=num_steps,
                deterministic=deterministic,
                seed=seed + cycle,
            )
            new_frames = out.features[:, :replan_stride]
            frames.append(new_frames)
            past = torch.cat([past, new_frames], dim=1)[:, -P:]
        result = torch.cat(frames, dim=1).to(input_device)
        return result[0] if single else result


def _config_from_dict(d: dict) -> TrainConfig:
    """Reconstruct TrainConfig from a checkpoint dict without extra deps."""
    data = dict(d["data"])
    if isinstance(data.get("scan_grid"), dict):
        data["scan_grid"] = ScanGrid(**data["scan_grid"])
    return TrainConfig(
        **{
            **{
                k: v
                for k, v in d.items()
                if k not in ("data", "model", "diffusion", "loss", "condition_noise")
            },
            "data": DataCfg(**data),
            "model": ModelCfg(**d["model"]),
            "diffusion": DiffusionCfg(**d["diffusion"]),
            "loss": LossWeights(**d["loss"]),
            # Stage 1--7 checkpoints predate condition noise and therefore
            # intentionally reconstruct with the all-zero defaults.
            "condition_noise": ConditionNoiseCfg(**d.get("condition_noise", {})),
        }
    )
