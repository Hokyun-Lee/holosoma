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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from holosoma.motion_gen.condition_noise import (
    apply_condition_noise,
    validate_condition_noise_config,
)
from holosoma.motion_gen.configs import TrainConfig
from holosoma.motion_gen.dataset import MotionWindowDataset, load_split_clips
from holosoma.motion_gen.diffusion import GaussianDiffusion
from holosoma.motion_gen.evaluation import compute_metrics, load_joint_limits
from holosoma.motion_gen.export import export_generated_raw_npz
from holosoma.motion_gen.features import FeatureLayout, unpack_features
from holosoma.motion_gen.kinematics import G1ForwardKinematics
from holosoma.motion_gen.losses import compute_losses
from holosoma.motion_gen.model import MotionDiffusionTransformer
from holosoma.motion_gen.normalization import FeatureNormalizer, compute_normalizer
from holosoma.motion_gen.visualization import plot_window_comparison
from holosoma.motion_gen.wandb_logging import MotionGenWandbLogger

CKPT_FORMAT_VERSION = 1


def terrain_stratum_indices(dataset: MotionWindowDataset) -> dict[str, list[int]]:
    """Partition dataset windows by whether their source clip has a scan.

    The ``terrain`` stratum is the condition the generator can actually see:
    a real height scan.  ``flat`` is retained as the concise logging name for
    the complementary no-scan stratum; its exact counts are recorded so the
    distinction remains auditable.
    """
    strata: dict[str, list[int]] = {"flat": [], "terrain": []}
    for window_index, (clip_index, _) in enumerate(dataset._index):
        clip = dataset.clips[clip_index]
        name = "terrain" if dataset.use_terrain_scan and clip.terrain_scan is not None else "flat"
        strata[name].append(window_index)
    return strata


def terrain_balanced_sample_weights(
    dataset: MotionWindowDataset,
    terrain_fraction: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Return per-window weights assigning total mass to each scan stratum."""
    if not math.isfinite(terrain_fraction) or not 0.0 < terrain_fraction < 1.0:
        raise ValueError("terrain_train_fraction must be finite and strictly between 0 and 1")
    strata = terrain_stratum_indices(dataset)
    counts = {name: len(indices) for name, indices in strata.items()}
    if not counts["flat"] or not counts["terrain"]:
        raise ValueError(f"terrain-balanced sampling requires both scanned-terrain and no-scan windows; got {counts}")
    weights = torch.empty(len(dataset), dtype=torch.double)
    weights[strata["terrain"]] = terrain_fraction / counts["terrain"]
    weights[strata["flat"]] = (1.0 - terrain_fraction) / counts["flat"]
    return weights, counts


class ValidationLossAccumulator:
    """Term-aware aggregation of scalar batch losses.

    Motion windows have a fixed, unpadded horizon, so weighting ordinary
    frame-mean losses by batch size is exactly equivalent to weighting by valid
    frames.  Contact and top-tail terms expose usable counts in the current
    batch and are weighted by those counts.  Max diagnostics are reduced by
    max, and count diagnostics by sum.  Terms whose true denominator depends on
    predicted scan-grid validity remain explicitly marked as approximations.
    """

    _MAX_TERMS = {
        "joint_limit_max_violation_rad",
        "lower_body_max_penetration_m",
    }
    _SUM_TERMS = {"lower_body_terrain_tail_count"}
    _VARIABLE_DENOMINATOR_TERMS = {
        "terrain_penetration",
        "lower_body_terrain_penetration",
        "lower_body_penetration_value_rate_5mm",
    }

    def __init__(self) -> None:
        self.weighted_sums: dict[str, float] = {}
        self.weights: dict[str, float] = {}
        self.maxima: dict[str, float] = {}
        self.sums: dict[str, float] = {}
        self.seen_terms: set[str] = set()
        self.num_batches = 0
        self.num_samples = 0

    def update(self, losses: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> None:
        batch_size = int(batch["x"].shape[0])
        self.num_batches += 1
        self.num_samples += batch_size
        for name, tensor in losses.items():
            self.seen_terms.add(name)
            if tensor.numel() != 1:
                raise ValueError(f"Validation loss {name!r} must be scalar, got shape {tuple(tensor.shape)}")
            value = float(tensor.item())
            if name in self._MAX_TERMS:
                self.maxima[name] = max(self.maxima.get(name, -math.inf), value)
                continue
            if name in self._SUM_TERMS:
                self.sums[name] = self.sums.get(name, 0.0) + value
                continue
            weight = self._term_weight(name, losses, batch, batch_size)
            if weight <= 0.0:
                continue
            self.weighted_sums[name] = self.weighted_sums.get(name, 0.0) + value * weight
            self.weights[name] = self.weights.get(name, 0.0) + weight

    @staticmethod
    def _term_weight(
        name: str,
        losses: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
        batch_size: int,
    ) -> float:
        if name == "foot_slide":
            contact = batch.get("contact")
            if contact is None or contact.shape[1] < 2:
                return 0.0
            return float((contact[:, 1:] & contact[:, :-1]).sum().item())
        if name == "lower_body_terrain_tail":
            count = losses.get("lower_body_terrain_tail_count")
            return float(count.item()) if count is not None else float(batch_size)
        return float(batch_size)

    def finalize(self, loss_weights: Any | None = None) -> tuple[dict[str, float], dict[str, Any]]:
        if self.num_batches == 0:
            raise RuntimeError("validation loader produced no batches")
        values = {name: total / self.weights[name] for name, total in self.weighted_sums.items()}
        values.update(self.maxima)
        values.update(self.sums)
        for name in self.seen_terms:
            values.setdefault(name, 0.0)
        total_recomputed = loss_weights is not None
        if loss_weights is not None:
            values["total"] = sum(
                float(getattr(loss_weights, name)) * values.get(name, 0.0) for name in loss_weights.__dataclass_fields__
            )
        approximated = sorted(self._VARIABLE_DENOMINATOR_TERMS & values.keys())
        metadata = {
            "method": "term_aware_weighted_batch_reduction",
            "num_batches": self.num_batches,
            "num_samples": self.num_samples,
            "default_weight": "batch_samples_equivalent_to_valid_frames_for_fixed_unpadded_horizon",
            "count_weighted_terms": ["foot_slide", "lower_body_terrain_tail"],
            "max_reduced_terms": sorted(self._MAX_TERMS & values.keys()),
            "sum_reduced_terms": sorted(self._SUM_TERMS & values.keys()),
            "sample_weight_approximations": approximated,
            "total_recomputed_from_aggregated_terms": total_recomputed,
        }
        return values, metadata


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
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    cfg: TrainConfig,
    normalizer: FeatureNormalizer,
    layout: FeatureLayout,
    *,
    train_sampler_generator: torch.Generator | None = None,
    metrics_history: list[dict] | None = None,
) -> dict:
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
        ),
    }
    return {
        "format_version": CKPT_FORMAT_VERSION,
        "step": step,
        "model": model.state_dict(),
        "ema": ema_model.state_dict() if ema_model is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": rng_state,
        "train_sampler_generator_state": (
            train_sampler_generator.get_state() if train_sampler_generator is not None else None
        ),
        # WeightedRandomSampler materializes an epoch draw when its iterator is
        # created.  Its generator state improves epoch-boundary continuation,
        # but does not encode the cursor of an already materialized draw.
        "sampler_resume_semantics": "generator_state_only_no_inflight_cursor",
        "metrics_history": copy.deepcopy(metrics_history or []),
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
        self._guard_output_directory()
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

        wandb_cfg = copy.deepcopy(cfg.wandb)
        if wandb_cfg.name is None:
            wandb_cfg.name = cfg.run_name
        self.wandb = MotionGenWandbLogger(wandb_cfg).init(
            resolved_config=dataclasses.asdict(cfg),
            run_dir=self.out_dir,
        )
        if self.wandb.enabled:
            run_info = {
                "run_id": self.wandb.run_id,
                "run_url": self.wandb.run_url,
                "mode": wandb_cfg.mode,
            }
            (self.out_dir / "wandb_run.json").write_text(json.dumps(run_info, indent=2))
            self.wandb.summary(
                {
                    "run": {
                        "output_directory": str(self.out_dir.resolve()),
                        "model_parameters": self.num_model_parameters,
                        "train_windows": len(self.train_dataset),
                        "val_windows": len(self.val_dataset),
                    },
                    "data": {"val_stratum_counts": self.val_stratum_counts},
                }
            )
            logger.info(f"W&B run: {self.wandb.run_url or self.wandb.run_id}")

    def _guard_output_directory(self) -> None:
        """Reject accidental scratch reuse before writing anything."""
        if self.cfg.resume is not None or not self.out_dir.exists():
            return
        entries = sorted(self.out_dir.iterdir(), key=lambda path: path.name)
        if not entries:
            return
        if self.cfg.allow_existing_output:
            logger.warning(f"Scratch run is reusing a non-empty output directory by explicit opt-in: {self.out_dir}")
            return
        preview = ", ".join(path.name for path in entries[:5])
        if len(entries) > 5:
            preview += f", ... ({len(entries)} entries)"
        raise FileExistsError(
            "Refusing to start a scratch run in non-empty output directory "
            f"{self.out_dir} ({preview}). Choose a new run_name, pass --resume, "
            "or explicitly set allow_existing_output=True."
        )

    # -- setup ----------------------------------------------------------------

    def _build_data(self) -> None:
        cfg = self.cfg
        d = cfg.data
        train_clips = load_split_clips(d.processed_dir, d.splits_file, "train", self.layout, d.metadata_dir, d.fps)
        if d.max_train_clips:
            train_clips = train_clips[: d.max_train_clips]
        if d.overfit:
            val_clips = train_clips
        else:
            val_clips = load_split_clips(d.processed_dir, d.splits_file, "val", self.layout, d.metadata_dir, d.fps)
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
        validate_condition_noise_config(
            cfg.condition_noise,
            use_terrain_scan=d.use_terrain_scan,
            terrain_dim=d.terrain_dim,
            scan_grid=d.scan_grid,
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
                scan_grid=d.scan_grid,
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

        common_loader_kwargs = {
            "batch_size": cfg.batch_size,
            "pin_memory": self.device.type == "cuda",
        }
        train_loader_kwargs = {
            **common_loader_kwargs,
            "num_workers": cfg.num_workers,
            "persistent_workers": cfg.num_workers > 0,
        }
        val_loader_kwargs = {
            **common_loader_kwargs,
            "num_workers": cfg.val_num_workers,
            # There can be all/flat/terrain validation loaders.  Keeping their
            # pools alive multiplied worker count and memory for the full run.
            "persistent_workers": False,
        }
        self.train_sampler_generator: torch.Generator | None = None
        if d.terrain_train_fraction is None:
            self.train_loader = DataLoader(
                self.train_dataset,
                shuffle=True,
                drop_last=True,
                **train_loader_kwargs,
            )
        else:
            weights, counts = terrain_balanced_sample_weights(
                self.train_dataset,
                d.terrain_train_fraction,
            )
            self.train_sampler_generator = torch.Generator().manual_seed(cfg.seed)
            sampler = WeightedRandomSampler(
                weights,
                num_samples=len(self.train_dataset),
                replacement=True,
                generator=self.train_sampler_generator,
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                sampler=sampler,
                shuffle=False,
                drop_last=True,
                **train_loader_kwargs,
            )
            logger.info(
                "Terrain-balanced train sampler enabled "
                f"(target terrain={d.terrain_train_fraction:.3f}, windows={counts})"
            )
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, drop_last=False, **val_loader_kwargs)
        self.val_stratum_loaders: dict[str, DataLoader] = {}
        self.val_stratum_counts: dict[str, int] = {}
        if d.stratified_validation:
            strata = terrain_stratum_indices(self.val_dataset)
            self.val_stratum_counts = {name: len(indices) for name, indices in strata.items()}
            missing = [name for name, indices in strata.items() if not indices]
            if missing:
                raise ValueError(
                    f"stratified validation requires both strata; missing {missing}, counts={self.val_stratum_counts}"
                )
            self.val_stratum_loaders = {
                name: DataLoader(
                    Subset(self.val_dataset, indices),
                    shuffle=False,
                    drop_last=False,
                    **val_loader_kwargs,
                )
                for name, indices in strata.items()
            }
            logger.info(f"Stratified validation windows: {self.val_stratum_counts}")

        self.joint_limits = load_joint_limits(Path(d.metadata_dir) / "joint_limits.json", self.layout)
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
        self.num_model_parameters = n_params
        logger.info(f"Model parameters: {n_params / 1e6:.2f} M")

        self.diffusion = GaussianDiffusion(
            timesteps=cfg.diffusion.timesteps,
            schedule=cfg.diffusion.schedule,
            param=cfg.diffusion.param,
        )
        for loss_name in cfg.loss.__dataclass_fields__:
            loss_weight = getattr(cfg.loss, loss_name)
            if not math.isfinite(loss_weight) or loss_weight < 0.0:
                raise ValueError(f"loss.{loss_name} must be finite and >= 0")
        if not math.isfinite(cfg.joint_limit_margin_rad) or cfg.joint_limit_margin_rad < 0.0:
            raise ValueError("joint_limit_margin_rad must be finite and >= 0")
        if not math.isfinite(cfg.terrain_penetration_tolerance_m) or cfg.terrain_penetration_tolerance_m < 0.0:
            raise ValueError("terrain_penetration_tolerance_m must be finite and >= 0")
        if (
            not math.isfinite(cfg.terrain_penetration_tail_fraction)
            or not 0.0 < cfg.terrain_penetration_tail_fraction <= 1.0
        ):
            raise ValueError("terrain_penetration_tail_fraction must be finite and in (0, 1]")
        if cfg.loss.joint_limit != 0.0 and self.joint_limits is None:
            raise ValueError("loss.joint_limit is non-zero but joint_limits.json is unavailable")
        self.joint_limits_device = self.joint_limits.to(self.device) if self.joint_limits is not None else None

        self.fk_model: G1ForwardKinematics | None = None
        needs_fk = (
            cfg.loss.fk_consistency != 0.0
            or cfg.loss.lower_body_terrain_penetration != 0.0
            or cfg.loss.lower_body_terrain_tail != 0.0
        )
        if needs_fk:
            source_mjcf = (
                Path(__file__).resolve().parents[4]
                / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
            )
            self.fk_model = G1ForwardKinematics(
                joint_names=self.layout.joint_names,
                body_names=self.layout.body_names,
                device=self.device,
                dtype=torch.float32,
                source_mjcf_path=source_mjcf,
            ).eval()
            logger.info(
                "Differentiable FK geometry enabled "
                f"(consistency_weight={cfg.loss.fk_consistency:g}, "
                "lower_body_terrain_weight="
                f"{cfg.loss.lower_body_terrain_penetration:g}, tail_weight="
                f"{cfg.loss.lower_body_terrain_tail:g}, source={source_mjcf})"
            )
            self._validate_fk_dataset_calibration()
        self.ema_model = None
        if cfg.use_ema:
            self.ema_model = copy.deepcopy(self.model).eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        def lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return (step + 1) / max(1, cfg.warmup_steps)
            progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = torch.amp.GradScaler(enabled=cfg.amp and self.device.type == "cuda")

    @torch.no_grad()
    def _validate_fk_dataset_calibration(self, num_windows: int = 8) -> None:
        """Fail fast if dataset body positions do not match the pinned MJCF."""
        if self.fk_model is None:
            return
        tolerance = self.cfg.fk_calibration_tolerance_m
        if tolerance is None:
            logger.warning("FK dataset calibration check disabled by config")
            return
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("fk_calibration_tolerance_m must be finite and > 0, or None")
        count = min(num_windows, len(self.train_dataset))
        indices = torch.linspace(0, len(self.train_dataset) - 1, count).round().long().tolist()
        gt = torch.stack([self.train_dataset[index]["x"] for index in indices]).to(self.device)
        parts = unpack_features(gt, self.layout)
        fk_body = self.fk_model(
            parts["root_pos"],
            parts["root_quat"],
            parts["joint_pos"],
        )
        error = (fk_body - parts["body_pos"]).norm(dim=-1)
        max_error = float(error.max())
        mean_error = float(error.mean())
        if max_error > tolerance:
            raise ValueError(
                "Motion dataset is inconsistent with the pinned G1 FK model: "
                f"max body error {max_error:.6g} m exceeds {tolerance:.6g} m"
            )
        logger.info(f"FK dataset calibration passed ({count} windows, mean={mean_error:.3e} m, max={max_error:.3e} m)")

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
            self.step,
            self.model,
            self.ema_model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.cfg,
            self.normalizer,
            self.layout,
            train_sampler_generator=self.train_sampler_generator,
            metrics_history=self.metrics_history,
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
            raise ValueError(f"Checkpoint feature dim {ckpt['layout']['dim']} != current layout {self.layout.dim}")
        self.model.load_state_dict(ckpt["model"])
        if self.ema_model is not None:
            if ckpt.get("ema") is not None:
                self.ema_model.load_state_dict(ckpt["ema"])
            else:
                # A no-EMA checkpoint must not leave the freshly randomized
                # EMA copy as the model used by validation and inference.
                self.ema_model.load_state_dict(ckpt["model"])
                logger.warning("Checkpoint has no EMA state; initialized EMA from the loaded model weights")
        self.normalizer = FeatureNormalizer.from_state_dict(ckpt["normalizer"])
        if self.cfg.resume_weights_only:
            self.step = 0
            logger.info(f"Loaded weights/EMA/normalizer from {path}; optimizer, scheduler, and step restart at zero")
        else:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.step = int(ckpt["step"])
            scheduler_state = ckpt.get("scheduler")
            if scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
            else:
                self._position_scheduler_for_legacy_checkpoint()
                logger.warning("Legacy checkpoint has no scheduler state; reconstructed its position from step")

            history = ckpt.get("metrics_history", [])
            if not isinstance(history, list):
                raise ValueError("Checkpoint metrics_history must be a list")
            self.metrics_history = copy.deepcopy(history)

            sampler_state = ckpt.get("train_sampler_generator_state")
            if self.train_sampler_generator is not None and sampler_state is not None:
                self.train_sampler_generator.set_state(sampler_state.detach().cpu())
                logger.info("Restored terrain sampler generator state; an in-flight epoch cursor is not checkpointed")
            elif self.train_sampler_generator is not None and sampler_state is None:
                logger.warning("Checkpoint has no terrain sampler generator state; sampler restarts from config seed")
            elif self.train_sampler_generator is None and sampler_state is not None:
                logger.warning("Checkpoint sampler state ignored because the current config has no weighted sampler")

            rng_state = ckpt.get("rng_state")
            if rng_state is not None:
                self._restore_rng_state(rng_state)
            else:
                logger.warning("Legacy checkpoint has no RNG state; random streams restart from config seed")
            logger.info(f"Resumed from {path} at step {self.step}")

    def _position_scheduler_for_legacy_checkpoint(self) -> None:
        """Position LambdaLR without replaying ``step()`` before optimizer steps."""
        self.scheduler.last_epoch = self.step
        self.scheduler._step_count = self.step + 1
        self.scheduler._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _restore_rng_state(self, state: Mapping[str, Any]) -> None:
        """Restore process RNGs saved by :func:`build_checkpoint`."""
        if not isinstance(state, Mapping):
            raise ValueError("Checkpoint rng_state must be a mapping")
        required = {"python", "numpy", "torch_cpu"}
        missing = required - set(state)
        if missing:
            raise ValueError(f"Checkpoint rng_state is missing {sorted(missing)}")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].detach().cpu())

        cuda_states = state.get("torch_cuda")
        if cuda_states is None or not torch.cuda.is_available():
            return
        device_count = torch.cuda.device_count()
        for device_index, cuda_state in enumerate(cuda_states[:device_count]):
            torch.cuda.set_rng_state(cuda_state.detach().cpu(), device=device_index)
        if len(cuda_states) != device_count:
            logger.warning(
                "Checkpoint CUDA RNG device count differs from this host "
                f"({len(cuda_states)} saved, {device_count} available); restored the overlap"
            )

    # -- core steps ---------------------------------------------------------------

    def _batch_to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _prepare_conditions(
        self,
        batch: dict,
        *,
        apply_noise: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare physical conditions, optionally noisy, before normalization."""
        if not apply_noise:
            return batch["past"], batch["terrain"]
        return apply_condition_noise(
            batch["past"],
            batch["terrain"],
            self.layout,
            self.cfg.condition_noise,
            self.cfg.data.scan_grid,
            generator=generator,
        )

    def _forward_losses(
        self,
        batch: dict,
        generator: torch.Generator | None = None,
        *,
        apply_condition_noise_input: bool | None = None,
        condition_generator: torch.Generator | None = None,
    ) -> dict:
        cfg = self.cfg
        assert self.normalizer is not None
        gt_phys = batch["x"]  # (B, H, D), physical canonical units
        x0 = self.normalizer.normalize(gt_phys)
        if apply_condition_noise_input is None:
            apply_condition_noise_input = self.model.training and cfg.condition_noise.is_enabled()
        past_phys, terrain_condition = self._prepare_conditions(
            batch,
            apply_noise=apply_condition_noise_input,
            generator=condition_generator if condition_generator is not None else generator,
        )
        past = self.normalizer.normalize(past_phys)
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
            x_t,
            t,
            past,
            batch["heading"],
            terrain_condition,
            drop_past=drop,
            drop_heading=drop,
            drop_terrain=drop,
        )
        x0_hat = self.diffusion.pred_to_x0(pred, x_t, t)
        x0_hat_phys = self.normalizer.denormalize(x0_hat)
        return compute_losses(
            x0_hat_phys,
            gt_phys,
            self.layout,
            cfg.loss,
            contact=batch["contact"],
            flat=batch["flat"],
            terrain_scan=batch["terrain"] if cfg.data.use_terrain_scan else None,
            has_scan=batch.get("has_scan"),
            scan_grid=cfg.data.scan_grid if cfg.data.use_terrain_scan else None,
            fk_model=getattr(self, "fk_model", None),
            joint_limits=getattr(self, "joint_limits_device", None),
            joint_limit_margin_rad=cfg.joint_limit_margin_rad,
            terrain_penetration_tolerance_m=cfg.terrain_penetration_tolerance_m,
            terrain_penetration_tail_fraction=cfg.terrain_penetration_tail_fraction,
        )

    def train(self) -> None:
        """Run training and close TensorBoard/W&B cleanly on every exit."""
        try:
            self._train_impl()
        except BaseException:
            self.writer.flush()
            self.writer.close()
            self.wandb.finish(exit_code=1)
            raise

    def _train_impl(self) -> None:
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
                train_log: dict[str, Any] = {
                    "train": {key: value.item() for key, value in last_losses.items()},
                    "optimizer": {"lr": lr},
                }
                if self.device.type == "cuda":
                    max_vram_gb = torch.cuda.max_memory_allocated() / 2**30
                    self.writer.add_scalar("train/max_vram_gb", max_vram_gb, self.step)
                    train_log["system"] = {"max_vram_gb": max_vram_gb}
                self.wandb.log(train_log, step=self.step)

            if self.step % cfg.val_interval == 0:
                self.validate()
                self.model.train()
            if self.step % cfg.sample_interval == 0:
                self.sample_and_export()
                self.model.train()
            if self.step % cfg.ckpt_interval == 0:
                self.save_checkpoint()

        final_checkpoint = self.save_checkpoint("final.pt")
        self.validate()
        self.sample_and_export()
        self.writer.flush()
        self.wandb.summary(
            {
                "run": {"final_step": self.step, "completed": True},
                "data": {"val_stratum_counts": self.val_stratum_counts},
            }
        )
        self.wandb.artifact(
            final_checkpoint,
            metadata={
                "step": self.step,
                "preset_run_name": self.cfg.run_name,
                "training_from_scratch": self.cfg.resume is None,
            },
        )
        self.wandb.finish(exit_code=0)
        self.writer.close()
        logger.info(f"Training done. Outputs: {self.out_dir}")

    # -- validation / sampling ------------------------------------------------------

    def _sampling_model(self) -> torch.nn.Module:
        return self.ema_model if self.ema_model is not None else self.model

    @torch.no_grad()
    def _validate_loader(self, loader: DataLoader, seed: int) -> dict[str, Any]:
        """Evaluate deterministic clean/noisy loss and sampling on one stratum."""
        cfg = self.cfg
        clean_diffusion_gen = torch.Generator(device=self.device.type).manual_seed(seed)
        noise_enabled = cfg.condition_noise.is_enabled()
        noisy_diffusion_gen = torch.Generator(device=self.device.type).manual_seed(seed)
        condition_gen = torch.Generator(device=self.device.type).manual_seed(seed + 1)

        clean_agg = ValidationLossAccumulator()
        noisy_agg = ValidationLossAccumulator()
        for i, cpu_batch in enumerate(loader):
            if i >= cfg.val_batches:
                break
            batch = self._batch_to_device(cpu_batch)
            clean_losses = self._forward_losses(
                batch,
                generator=clean_diffusion_gen,
                apply_condition_noise_input=False,
            )
            clean_agg.update(clean_losses, batch)
            if noise_enabled:
                noisy_losses = self._forward_losses(
                    batch,
                    generator=noisy_diffusion_gen,
                    apply_condition_noise_input=True,
                    condition_generator=condition_gen,
                )
                noisy_agg.update(noisy_losses, batch)
        clean_losses, clean_aggregation = clean_agg.finalize(cfg.loss)
        if noise_enabled:
            noisy_losses, noisy_aggregation = noisy_agg.finalize(cfg.loss)
        else:
            noisy_losses = dict(clean_losses)
            noisy_aggregation = copy.deepcopy(clean_aggregation)

        batch = self._batch_to_device(next(iter(loader)))
        if noise_enabled:
            sample_phys, noisy_sample_phys = self._generate_validation_pair(
                batch,
                clean_diffusion_gen,
                condition_gen,
            )
        else:
            sample_phys = self._generate_for_batch(batch, clean_diffusion_gen)
            noisy_sample_phys = sample_phys

        metric_kwargs = {
            "joint_limits": self.joint_limits_device,
            "contact": batch["contact"],
            "flat": batch["flat"],
            "terrain_scan": batch["terrain"] if cfg.data.use_terrain_scan else None,
            "has_scan": batch.get("has_scan"),
            "scan_grid": cfg.data.scan_grid if cfg.data.use_terrain_scan else None,
            "fk_model": self.fk_model,
        }
        clean_metrics = compute_metrics(
            sample_phys,
            batch["x"],
            self.layout,
            cfg.data.fps,
            **metric_kwargs,
        )
        noisy_metrics = compute_metrics(
            noisy_sample_phys,
            batch["x"],
            self.layout,
            cfg.data.fps,
            **metric_kwargs,
        )
        robustness = {
            f"{key}_delta": noisy_metrics[key] - clean_metrics[key]
            for key in sorted(clean_metrics.keys() & noisy_metrics.keys())
        }
        robustness["sample_condition_response_mean_abs_mixed_units"] = (
            (noisy_sample_phys - sample_phys).abs().mean().item()
        )

        return {
            "loss_clean": clean_losses,
            "loss_noisy": noisy_losses,
            "sample_metrics_clean": clean_metrics,
            "sample_metrics_noisy": noisy_metrics,
            "condition_noise_delta": robustness,
            "loss_aggregation_clean": clean_aggregation,
            "loss_aggregation_noisy": noisy_aggregation,
            "loss_aggregation_scope": ("full_loader" if clean_agg.num_batches == len(loader) else "first_n_batches"),
            "sample_scope": "first_batch",
            "sample_num_items": int(batch["x"].shape[0]),
            "num_loss_batches": clean_agg.num_batches,
            "num_loss_samples": clean_agg.num_samples,
            "seed": seed,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        cfg = self.cfg
        self.model.eval()
        noise_enabled = cfg.condition_noise.is_enabled()
        loaders: dict[str, DataLoader] = {"all": self.val_loader}
        loaders.update(self.val_stratum_loaders)
        seed_offsets = {"all": 0, "flat": 10_000, "terrain": 20_000}
        results = {
            name: self._validate_loader(loader, cfg.val_seed + seed_offsets.get(name, 30_000))
            for name, loader in loaders.items()
        }

        wandb_payload: dict[str, Any] = {}
        for name, result in results.items():
            prefix = "val" if name == "all" else f"val_{name}"
            for key, value in result["loss_clean"].items():
                self.writer.add_scalar(f"{prefix}/loss_{key}", value, self.step)
            for key, value in result["sample_metrics_clean"].items():
                self.writer.add_scalar(f"{prefix}/{key}", value, self.step)
            if noise_enabled:
                for key, value in result["loss_noisy"].items():
                    self.writer.add_scalar(f"{prefix}_noisy/loss_{key}", value, self.step)
                for key, value in result["sample_metrics_noisy"].items():
                    self.writer.add_scalar(f"{prefix}_noisy/{key}", value, self.step)
                for key, value in result["condition_noise_delta"].items():
                    self.writer.add_scalar(
                        f"{prefix}_condition_noise/{key}",
                        value,
                        self.step,
                    )
            wandb_payload[prefix] = {
                "loss_clean": result["loss_clean"],
                "sample_clean": result["sample_metrics_clean"],
                "loss_noisy": result["loss_noisy"],
                "sample_noisy": result["sample_metrics_noisy"],
                "condition_noise_delta": result["condition_noise_delta"],
                "loss_aggregation_scope": result["loss_aggregation_scope"],
                "sample_scope": result["sample_scope"],
                "num_loss_samples": result["num_loss_samples"],
                "sample_num_items": result["sample_num_items"],
            }
            logger.info(
                f"[{prefix} first-batch sample clean @ {self.step}] "
                + " ".join(f"{key}={value:.4f}" for key, value in result["sample_metrics_clean"].items())
            )
        self.wandb.log(wandb_payload, step=self.step)

        primary = results["all"]
        entry = {
            "step": self.step,
            # Keep the Stage 1--7 keys as clean-metric aliases.
            "val_loss": primary["loss_clean"],
            "val_sample_metrics": primary["sample_metrics_clean"],
            "val_loss_clean": primary["loss_clean"],
            "val_loss_noisy": primary["loss_noisy"],
            "val_sample_metrics_clean": primary["sample_metrics_clean"],
            "val_sample_metrics_noisy": primary["sample_metrics_noisy"],
            "val_condition_noise_delta": primary["condition_noise_delta"],
            "val_loss_aggregation": primary["loss_aggregation_clean"],
            "val_loss_aggregation_scope": primary["loss_aggregation_scope"],
            "val_sample_scope": primary["sample_scope"],
            "val_sample_num_items": primary["sample_num_items"],
            "val_condition_noise": {
                "enabled": noise_enabled,
                "diffusion_seed": cfg.val_seed,
                "condition_seed": cfg.val_seed + 1,
                "shared_ddim_initial_noise": True,
            },
            "val_strata": {name: result for name, result in results.items() if name != "all"},
            "val_stratum_counts": self.val_stratum_counts,
        }
        self.metrics_history.append(entry)
        (self.out_dir / "metrics.json").write_text(json.dumps({"history": self.metrics_history}, indent=2))
        return entry

    @torch.no_grad()
    def _generate_validation_pair(
        self,
        batch: dict,
        diffusion_generator: torch.Generator,
        condition_generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate clean/noisy validation samples from identical DDIM noise."""
        init_noise = torch.randn(
            batch["x"].shape,
            device=self.device,
            generator=diffusion_generator,
        )
        clean = self._generate_for_batch(
            batch,
            diffusion_generator,
            apply_condition_noise_input=False,
            init_noise=init_noise,
        )
        noisy = self._generate_for_batch(
            batch,
            diffusion_generator,
            apply_condition_noise_input=True,
            condition_generator=condition_generator,
            init_noise=init_noise.clone(),
        )
        return clean, noisy

    @torch.no_grad()
    def _generate_for_batch(
        self,
        batch: dict,
        generator: torch.Generator,
        *,
        apply_condition_noise_input: bool = False,
        condition_generator: torch.Generator | None = None,
        init_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DDIM-sample futures for a batch's conditions; returns physical units."""
        assert self.normalizer is not None
        cfg = self.cfg
        model = self._sampling_model()
        past_phys, terrain_condition = self._prepare_conditions(
            batch,
            apply_noise=apply_condition_noise_input,
            generator=condition_generator,
        )
        past = self.normalizer.normalize(past_phys)

        def model_fn(x_t, t, **_):
            return model(x_t, t, past, batch["heading"], terrain_condition)

        shape = batch["x"].shape
        x0_norm = self.diffusion.ddim_sample(
            model_fn,
            tuple(shape),
            self.device,
            num_steps=cfg.val_sample_steps,
            generator=generator,
            init_noise=init_noise,
        )
        return self.normalizer.denormalize(x0_norm)

    @torch.no_grad()
    def sample_and_export(self, num_items: int = 4) -> None:
        """Generate a few val windows; export plots and a raw npz."""
        cfg = self.cfg
        self.model.eval()
        loaders = self.val_stratum_loaders or {"all": self.val_loader}
        seed_offsets = {"all": 0, "flat": 10_000, "terrain": 20_000}
        for name, loader in loaders.items():
            gen = torch.Generator(device=self.device.type).manual_seed(cfg.val_seed + seed_offsets.get(name, 30_000))
            batch = self._batch_to_device(next(iter(loader)))
            sample_phys = self._generate_for_batch(batch, gen)

            plot_dir = self.out_dir / "plots" / f"step_{self.step:08d}"
            sample_dir = self.out_dir / "samples" / f"step_{self.step:08d}"
            if name != "all":
                plot_dir = plot_dir / name
                sample_dir = sample_dir / name
            plot_dir.mkdir(parents=True, exist_ok=True)
            sample_dir.mkdir(parents=True, exist_ok=True)
            for i in range(min(num_items, sample_phys.shape[0])):
                plot_window_comparison(
                    sample_phys[i].cpu(),
                    batch["x"][i].cpu(),
                    self.layout,
                    plot_dir / f"window_{i:02d}.png",
                )
                export_generated_raw_npz(
                    sample_phys[i].cpu(),
                    self.layout,
                    cfg.data.fps,
                    sample_dir / f"window_{i:02d}_gen_raw.npz",
                    gt_features=batch["x"][i].cpu(),
                    heading=batch["heading"][i].cpu(),
                )
            logger.info(f"Exported {name} samples to {sample_dir} and plots to {plot_dir}")
