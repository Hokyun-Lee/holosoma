"""Training loop: AMP, gradient accumulation, EMA, checkpointing, TensorBoard.

Run outputs (under ``<out_root>/<run_name>/``):
    config.yaml               resolved config
    normalization_stats.npz   train-split feature statistics
    checkpoints/ckpt_*.pt, checkpoints/latest.pt
    metrics.json              train/val metric history
    samples/step_*/           generated npz exports
    plots/step_*/             comparison plots
    logs/                     TensorBoard events
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from holosoma.motion_gen.configs import TrainConfig
from holosoma.motion_gen.dataset import MotionWindowDataset, load_split_clips
from holosoma.motion_gen.diffusion import GaussianDiffusion
from holosoma.motion_gen.evaluation import compute_metrics, load_joint_limits
from holosoma.motion_gen.export import export_generated_raw_npz
from holosoma.motion_gen.features import FeatureLayout
from holosoma.motion_gen.losses import compute_losses
from holosoma.motion_gen.model import MotionDiffusionTransformer
from holosoma.motion_gen.normalization import FeatureNormalizer, compute_normalizer
from holosoma.motion_gen.visualization import plot_window_comparison

CKPT_FORMAT_VERSION = 1


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_checkpoint(
    step: int,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: TrainConfig,
    normalizer: FeatureNormalizer,
    layout: FeatureLayout,
) -> dict:
    return {
        "format_version": CKPT_FORMAT_VERSION,
        "step": step,
        "model": model.state_dict(),
        "ema": ema_model.state_dict() if ema_model is not None else None,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": dataclasses.asdict(cfg),
        "normalizer": normalizer.state_dict(),
        "layout": layout.to_metadata(),
        "torch_version": torch.__version__,
    }


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        seed_everything(cfg.seed)

        self.out_dir = Path(cfg.out_root) / cfg.run_name
        for sub in ("checkpoints", "samples", "plots", "logs"):
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)
        (self.out_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))

        self.layout = FeatureLayout()
        self._build_data()
        self._build_model()

        self.writer = SummaryWriter(log_dir=str(self.out_dir / "logs"))
        self.metrics_history: list[dict] = []
        self.step = 0
        self._nonfinite_streak = 0

        if cfg.resume:
            self.load_checkpoint(cfg.resume)

    # -- setup ----------------------------------------------------------------

    def _build_data(self) -> None:
        cfg = self.cfg
        d = cfg.data
        train_clips = load_split_clips(
            d.processed_dir, d.splits_file, "train", self.layout, d.metadata_dir, d.fps
        )
        if d.max_train_clips:
            train_clips = train_clips[: d.max_train_clips]
        if d.overfit:
            val_clips = train_clips
        else:
            val_clips = load_split_clips(
                d.processed_dir, d.splits_file, "val", self.layout, d.metadata_dir, d.fps
            )
            if d.max_val_clips:
                val_clips = val_clips[: d.max_val_clips]
        logger.info(
            f"Train clips: {[c.name for c in train_clips]} | "
            f"val clips ({'train/overfit' if d.overfit else 'val'}): {[c.name for c in val_clips]}"
        )

        if d.use_terrain_scan and d.terrain_dim != d.scan_grid.dim:
            raise ValueError(
                f"terrain_dim ({d.terrain_dim}) != scan_grid.dim ({d.scan_grid.dim}); "
                "set data.terrain_dim to match the scan grid."
            )

        def make_dataset(clips, stride: int) -> MotionWindowDataset:
            return MotionWindowDataset(
                clips,
                layout=self.layout,
                past_frames=d.past_frames,
                future_frames=d.future_frames,
                stride=stride,
                min_heading_disp=d.min_heading_disp,
                terrain_dim=d.terrain_dim,
                use_terrain_scan=d.use_terrain_scan,
            )

        self.train_dataset = make_dataset(train_clips, d.train_stride)
        self.val_dataset = make_dataset(val_clips, d.val_stride)
        logger.info(f"Train windows: {len(self.train_dataset)}, val windows: {len(self.val_dataset)}")

        norm_path = self.out_dir / "normalization_stats.npz"
        if cfg.resume:
            self.normalizer = None  # loaded from checkpoint
        elif norm_path.exists():
            self.normalizer = FeatureNormalizer.load(norm_path)
            logger.info(f"Loaded existing normalization stats from {norm_path}")
        else:
            logger.info("Computing normalization statistics on the train split...")
            self.normalizer = compute_normalizer(self.train_dataset, cfg.norm_max_windows, cfg.seed)
            self.normalizer.save(norm_path)

        loader_kwargs = dict(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
        )
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

        self.joint_limits = load_joint_limits(
            Path(d.metadata_dir) / "joint_limits.json", self.layout
        )
        if self.joint_limits is None:
            logger.warning("No joint_limits.json found; joint-limit violation metric disabled.")

    def _build_model(self) -> None:
        cfg = self.cfg
        self.model = MotionDiffusionTransformer(
            feature_dim=self.layout.dim,
            past_frames=cfg.data.past_frames,
            future_frames=cfg.data.future_frames,
            terrain_dim=cfg.data.terrain_dim,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            d_ff=cfg.model.d_ff,
            dropout=cfg.model.dropout,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params / 1e6:.2f} M")

        self.diffusion = GaussianDiffusion(
            timesteps=cfg.diffusion.timesteps,
            schedule=cfg.diffusion.schedule,
            param=cfg.diffusion.param,
        )
        self.ema_model = None
        if cfg.use_ema:
            self.ema_model = copy.deepcopy(self.model).eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        def lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return (step + 1) / max(1, cfg.warmup_steps)
            progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = torch.amp.GradScaler(enabled=cfg.amp and self.device.type == "cuda")

    # -- EMA -------------------------------------------------------------------

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self.ema_model is None:
            return
        d = self.cfg.ema_decay
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.lerp_(p, 1.0 - d)
        for ema_b, b in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_b.copy_(b)

    # -- checkpointing -----------------------------------------------------------

    def save_checkpoint(self, name: str | None = None) -> Path:
        assert self.normalizer is not None
        ckpt = build_checkpoint(
            self.step, self.model, self.ema_model, self.optimizer, self.scaler,
            self.cfg, self.normalizer, self.layout,
        )
        path = self.out_dir / "checkpoints" / (name or f"ckpt_{self.step:08d}.pt")
        torch.save(ckpt, path)
        torch.save(ckpt, self.out_dir / "checkpoints" / "latest.pt")
        logger.info(f"Saved checkpoint: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("format_version") != CKPT_FORMAT_VERSION:
            raise ValueError(f"Checkpoint format {ckpt.get('format_version')} != {CKPT_FORMAT_VERSION}")
        if ckpt["layout"]["dim"] != self.layout.dim:
            raise ValueError(
                f"Checkpoint feature dim {ckpt['layout']['dim']} != current layout {self.layout.dim}"
            )
        self.model.load_state_dict(ckpt["model"])
        if self.ema_model is not None and ckpt.get("ema") is not None:
            self.ema_model.load_state_dict(ckpt["ema"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.normalizer = FeatureNormalizer.from_state_dict(ckpt["normalizer"])
        self.step = int(ckpt["step"])
        for _ in range(self.step):
            self.scheduler.step()
        logger.info(f"Resumed from {path} at step {self.step}")

    # -- core steps ---------------------------------------------------------------

    def _batch_to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _forward_losses(self, batch: dict, generator: torch.Generator | None = None) -> dict:
        cfg = self.cfg
        assert self.normalizer is not None
        gt_phys = batch["x"]  # (B, H, D), physical canonical units
        x0 = self.normalizer.normalize(gt_phys)
        past = self.normalizer.normalize(batch["past"])
        bsz = x0.shape[0]

        t = self.diffusion.sample_timesteps(bsz, self.device, generator)
        noise = torch.randn(x0.shape, device=self.device, generator=generator)
        x_t = self.diffusion.q_sample(x0, t, noise)

        drop = None
        if self.model.training and cfg.cond_dropout > 0:
            # Joint dropout of all conditions (MDM-style) so the fully
            # unconditional branch used by classifier-free guidance is trained.
            drop = torch.rand(bsz, device=self.device, generator=generator) < cfg.cond_dropout
        pred = self.model(
            x_t, t, past, batch["heading"], batch["terrain"],
            drop_past=drop, drop_heading=drop, drop_terrain=drop,
        )
        x0_hat = self.diffusion.pred_to_x0(pred, x_t, t)
        x0_hat_phys = self.normalizer.denormalize(x0_hat)
        return compute_losses(
            x0_hat_phys, gt_phys, self.layout, cfg.loss,
            contact=batch["contact"], flat=batch["flat"],
            terrain_scan=batch["terrain"] if cfg.data.use_terrain_scan else None,
            has_scan=batch.get("has_scan"),
            scan_grid=cfg.data.scan_grid if cfg.data.use_terrain_scan else None,
        )

    def train(self) -> None:
        cfg = self.cfg
        self.model.train()
        train_iter = iter(self.train_loader)
        amp_enabled = cfg.amp and self.device.type == "cuda"
        logger.info(f"Training on {self.device} (AMP={amp_enabled}) for {cfg.max_steps} steps")

        while self.step < cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            last_losses = None
            for _ in range(cfg.grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)
                batch = self._batch_to_device(batch)
                with torch.autocast(self.device.type, enabled=amp_enabled):
                    losses = self._forward_losses(batch)
                    loss = losses["total"] / cfg.grad_accum
                if not torch.isfinite(loss):
                    self._nonfinite_streak += 1
                    logger.warning(f"Non-finite loss at step {self.step} (streak {self._nonfinite_streak})")
                    if self._nonfinite_streak >= 5:
                        raise RuntimeError("5 consecutive non-finite losses; aborting. Check data/LR.")
                    continue
                self._nonfinite_streak = 0
                self.scaler.scale(loss).backward()
                last_losses = losses

            if last_losses is None:
                continue  # step/scheduler stay in sync when a step is skipped
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self._update_ema()
            self.step += 1

            if self.step % cfg.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in last_losses.items())
                logger.info(f"step {self.step}/{cfg.max_steps} lr={lr:.2e} {msg}")
                for k, v in last_losses.items():
                    self.writer.add_scalar(f"train/{k}", v.item(), self.step)
                self.writer.add_scalar("train/lr", lr, self.step)
                if self.device.type == "cuda":
                    self.writer.add_scalar(
                        "train/max_vram_gb", torch.cuda.max_memory_allocated() / 2**30, self.step
                    )

            if self.step % cfg.val_interval == 0:
                self.validate()
                self.model.train()
            if self.step % cfg.sample_interval == 0:
                self.sample_and_export()
                self.model.train()
            if self.step % cfg.ckpt_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint("final.pt")
        self.validate()
        self.sample_and_export()
        self.writer.flush()
        logger.info(f"Training done. Outputs: {self.out_dir}")

    # -- validation / sampling ------------------------------------------------------

    def _sampling_model(self) -> torch.nn.Module:
        return self.ema_model if self.ema_model is not None else self.model

    @torch.no_grad()
    def validate(self) -> dict:
        cfg = self.cfg
        self.model.eval()
        gen = torch.Generator(device=self.device.type).manual_seed(cfg.val_seed)

        # 1) Deterministic diffusion-loss evaluation on the val split.
        agg: dict[str, float] = {}
        n = 0
        for i, batch in enumerate(self.val_loader):
            if i >= cfg.val_batches:
                break
            batch = self._batch_to_device(batch)
            losses = self._forward_losses(batch, generator=gen)
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + v.item()
            n += 1
        val_losses = {k: v / max(n, 1) for k, v in agg.items()}

        # 2) Sampling metrics on one fixed val batch (DDIM, fixed seed).
        batch = self._batch_to_device(next(iter(self.val_loader)))
        sample_phys = self._generate_for_batch(batch, gen)
        metrics = compute_metrics(
            sample_phys, batch["x"], self.layout, cfg.data.fps,
            joint_limits=self.joint_limits, contact=batch["contact"], flat=batch["flat"],
            terrain_scan=batch["terrain"] if cfg.data.use_terrain_scan else None,
            has_scan=batch.get("has_scan"),
            scan_grid=cfg.data.scan_grid if cfg.data.use_terrain_scan else None,
        )

        for k, v in val_losses.items():
            self.writer.add_scalar(f"val/loss_{k}", v, self.step)
        for k, v in metrics.items():
            self.writer.add_scalar(f"val/{k}", v, self.step)
        entry = {"step": self.step, "val_loss": val_losses, "val_sample_metrics": metrics}
        self.metrics_history.append(entry)
        (self.out_dir / "metrics.json").write_text(json.dumps({"history": self.metrics_history}, indent=2))
        logger.info(f"[val @ {self.step}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return entry

    @torch.no_grad()
    def _generate_for_batch(self, batch: dict, generator: torch.Generator) -> torch.Tensor:
        """DDIM-sample futures for a batch's conditions; returns physical units."""
        assert self.normalizer is not None
        cfg = self.cfg
        model = self._sampling_model()
        past = self.normalizer.normalize(batch["past"])

        def model_fn(x_t, t, **_):
            return model(x_t, t, past, batch["heading"], batch["terrain"])

        shape = batch["x"].shape
        x0_norm = self.diffusion.ddim_sample(
            model_fn, tuple(shape), self.device,
            num_steps=cfg.val_sample_steps, generator=generator,
        )
        return self.normalizer.denormalize(x0_norm)

    @torch.no_grad()
    def sample_and_export(self, num_items: int = 4) -> None:
        """Generate a few val windows; export plots and a raw npz."""
        cfg = self.cfg
        self.model.eval()
        gen = torch.Generator(device=self.device.type).manual_seed(cfg.val_seed)
        batch = self._batch_to_device(next(iter(self.val_loader)))
        sample_phys = self._generate_for_batch(batch, gen)

        plot_dir = self.out_dir / "plots" / f"step_{self.step:08d}"
        sample_dir = self.out_dir / "samples" / f"step_{self.step:08d}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        for i in range(min(num_items, sample_phys.shape[0])):
            plot_window_comparison(
                sample_phys[i].cpu(), batch["x"][i].cpu(), self.layout,
                plot_dir / f"window_{i:02d}.png",
            )
            export_generated_raw_npz(
                sample_phys[i].cpu(), self.layout, cfg.data.fps,
                sample_dir / f"window_{i:02d}_gen_raw.npz",
                gt_features=batch["x"][i].cpu(),
                heading=batch["heading"][i].cpu(),
            )
        logger.info(f"Exported samples to {sample_dir} and plots to {plot_dir}")
