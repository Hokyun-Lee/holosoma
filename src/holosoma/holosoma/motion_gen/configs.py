"""Configuration dataclasses and presets (smoke / debug / baseline_4090).

Values that come from the paper (arXiv:2604.17335): horizon 25 frames = 0.5 s
at 50 Hz, 2 past conditioning frames. Architecture defaults (d_model=512,
8 layers, 4 heads, ff=1024, dropout 0.1) follow the official MDM
implementation since the paper does not publish them. Optimizer, batch size,
learning rate, EMA and loss weights are implementation choices sized for a
single RTX 4090.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from holosoma.motion_gen.losses import LossWeights


@dataclass
class DataCfg:
    processed_dir: str = "data/motion_gen/processed"
    metadata_dir: str = "data/motion_gen/metadata"
    splits_file: str = "data/motion_gen/splits/splits.json"
    past_frames: int = 2  # paper
    future_frames: int = 25  # paper: 0.5 s horizon at 50 Hz
    fps: float = 50.0
    train_stride: int = 1
    val_stride: int = 5
    min_heading_disp: float = 0.05
    terrain_dim: int = 121
    """Terrain scan interface size (Phase A: zeros). 11x11 grid at 0.1 m
    spacing is an implementation choice; the paper gives no scan resolution."""
    max_train_clips: int | None = None
    """Optional cap on the number of train clips (debug presets)."""
    max_val_clips: int | None = None
    """Optional cap on the number of val clips (debug presets)."""
    overfit: bool = False
    """If True, validate on the train split (single-motion overfit checks)."""


@dataclass
class ModelCfg:
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1


@dataclass
class DiffusionCfg:
    timesteps: int = 1000
    schedule: str = "cosine"  # or "linear"
    param: str = "x0"  # or "eps"


@dataclass
class TrainConfig:
    run_name: str = "motion_gen"
    out_root: str = "logs/motion_gen"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    seed: int = 42

    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    diffusion: DiffusionCfg = field(default_factory=DiffusionCfg)
    loss: LossWeights = field(default_factory=LossWeights)

    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    max_steps: int = 200_000
    grad_accum: int = 1
    grad_clip: float = 1.0
    amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9999
    cond_dropout: float = 0.1
    """Probability of jointly masking all conditions (past + heading + terrain,
    MDM-style) during training; enables classifier-free guidance."""

    num_workers: int = 4
    log_interval: int = 50
    ckpt_interval: int = 10_000
    val_interval: int = 2_000
    sample_interval: int = 20_000
    val_batches: int = 8
    val_sample_steps: int = 50  # DDIM steps used for validation sampling
    val_seed: int = 123
    resume: str | None = None
    norm_max_windows: int = 2000


def smoke() -> TrainConfig:
    """Pipeline check: loader, forward, backward, checkpoint, sampling. CPU-ok."""
    return TrainConfig(
        run_name="smoke",
        model=ModelCfg(d_model=64, n_layers=2, n_heads=2, d_ff=128, dropout=0.0),
        diffusion=DiffusionCfg(timesteps=100),
        batch_size=8,
        max_steps=30,
        warmup_steps=5,
        amp=False,
        use_ema=True,
        ema_decay=0.9,  # short runs need a fast EMA to reflect training at all
        num_workers=0,
        log_interval=5,
        ckpt_interval=30,
        val_interval=15,
        sample_interval=30,
        val_batches=1,
        val_sample_steps=5,
        data=DataCfg(max_train_clips=1, max_val_clips=1, train_stride=10, val_stride=50, overfit=True),
    )


def debug() -> TrainConfig:
    """Small-data overfit: loss must go down, reconstruction must look right."""
    return TrainConfig(
        run_name="debug",
        model=ModelCfg(d_model=128, n_layers=4, n_heads=4, d_ff=256),
        batch_size=64,
        max_steps=3000,
        warmup_steps=100,
        ema_decay=0.995,
        num_workers=2,
        log_interval=25,
        ckpt_interval=1000,
        val_interval=500,
        sample_interval=1000,
        val_batches=2,
        val_sample_steps=50,
        data=DataCfg(max_train_clips=1, max_val_clips=1, val_stride=25, overfit=True),
    )


def baseline_4090() -> TrainConfig:
    """~10 motions on one RTX 4090 (24 GB).

    Measured on the 11-clip dataset (2026-07-10): ~52 steps/s, 1.3 GB torch-
    allocated VRAM. A 200k-step run overfits badly past ~10k steps (train
    MPJPE 0.005 m vs val 0.237 m at 200k; best val ~0.166 m around 10k), so
    the default stops at 50k with frequent validation — pick the checkpoint
    with the best val metrics. Scale max_steps up with the dataset."""
    return TrainConfig(
        run_name="baseline_4090",
        model=ModelCfg(),  # MDM defaults: 512 / 8 layers / 4 heads / ff 1024
        batch_size=256,
        max_steps=50_000,
        warmup_steps=1000,
        num_workers=4,
        ckpt_interval=5_000,
        val_interval=1_000,
        sample_interval=10_000,
        data=DataCfg(),
    )


PRESETS = {
    "smoke": smoke,
    "debug": debug,
    "baseline_4090": baseline_4090,
}
