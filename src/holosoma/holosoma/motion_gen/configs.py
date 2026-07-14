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
from holosoma.motion_gen.terrain import ScanGrid
from holosoma.motion_gen.wandb_logging import WandbLoggingConfig


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
    use_terrain_scan: bool = False
    """Phase B: feed real heading-aligned height scans (requires
    add_terrain_scans; clips without scans still get zeros)."""
    scan_grid: ScanGrid = field(default_factory=ScanGrid)
    """Grid definition for Phase B scans; terrain_dim must equal scan_grid.dim
    when use_terrain_scan is enabled."""
    max_train_clips: int | None = None
    """Optional cap on the number of train clips (debug presets)."""
    max_val_clips: int | None = None
    """Optional cap on the number of val clips (debug presets)."""
    overfit: bool = False
    """If True, validate on the train split (single-motion overfit checks)."""
    terrain_train_fraction: float | None = None
    """Optional total sampling probability assigned to scanned-terrain windows.

    ``None`` preserves the original window-uniform loader.  Non-``None``
    values use a weighted sampler that is uniform inside the scanned-terrain
    and no-scan strata.  The paper does not publish a balancing ratio.
    """
    stratified_validation: bool = False
    """Report deterministic flat/no-scan and scanned-terrain validation separately."""


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
class ConditionNoiseCfg:
    """Physical-unit noise applied only to generator conditioning inputs.

    Every default is zero, preserving Stage 1--7 training and checkpoint
    behavior.  The paper does not publish these magnitudes; non-zero values in
    robustness presets are explicit implementation choices.
    """

    root_position_std_m: float = 0.0
    root_orientation_std_rad: float = 0.0
    joint_position_std_rad: float = 0.0
    body_position_std_m: float = 0.0
    terrain_height_std_m: float = 0.0
    terrain_point_dropout_prob: float = 0.0
    terrain_height_bias_std_m: float = 0.0
    terrain_xy_std_m: float = 0.0
    terrain_yaw_std_rad: float = 0.0

    def is_enabled(self) -> bool:
        """Return whether any conditioning perturbation is configured."""
        return any(
            value > 0.0
            for value in (
                self.root_position_std_m,
                self.root_orientation_std_rad,
                self.joint_position_std_rad,
                self.body_position_std_m,
                self.terrain_height_std_m,
                self.terrain_point_dropout_prob,
                self.terrain_height_bias_std_m,
                self.terrain_xy_std_m,
                self.terrain_yaw_std_rad,
            )
        )


@dataclass
class TrainConfig:
    run_name: str = "motion_gen"
    out_root: str = "logs/motion_gen"
    allow_existing_output: bool = False
    """Allow a scratch run to reuse a non-empty output directory.

    The safe default prevents silent config/metric/checkpoint overwrites.  This
    opt-in retains the earlier normalization-stat reuse workflow when it is
    explicitly intended.  Resume runs are always allowed to use their target
    output directory.
    """
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    seed: int = 42

    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    diffusion: DiffusionCfg = field(default_factory=DiffusionCfg)
    loss: LossWeights = field(default_factory=LossWeights)
    condition_noise: ConditionNoiseCfg = field(default_factory=ConditionNoiseCfg)
    wandb: WandbLoggingConfig = field(default_factory=WandbLoggingConfig)
    fk_calibration_tolerance_m: float | None = 1.0e-3
    """Maximum GT-vs-FK body error accepted when FK loss is enabled.

    This fail-fast threshold is an implementation choice. ``None`` disables
    the dataset/MJCF calibration check (intended only for synthetic tests).
    """
    joint_limit_margin_rad: float = 0.0
    """Soft joint-limit safety margin; an implementation choice in radians."""
    terrain_penetration_tolerance_m: float = 0.0
    """Allowed lower-body proxy contact depth before loss, in metres."""
    terrain_penetration_tail_fraction: float = 0.01
    """Valid lower-body proxy fraction optimized by the optional tail loss."""

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
    val_num_workers: int = 0
    """Validation loader workers; zero avoids one persistent pool per stratum."""
    log_interval: int = 50
    ckpt_interval: int = 10_000
    val_interval: int = 2_000
    sample_interval: int = 20_000
    val_batches: int = 8
    val_sample_steps: int = 50  # DDIM steps used for validation sampling
    val_seed: int = 123
    resume: str | None = None
    resume_weights_only: bool = False
    """Load model/EMA/normalizer but restart optimizer, scheduler, and step."""
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


def paperscale_4090() -> TrainConfig:
    """Paper-scale data (~195 clips, ~2.7 h) on one RTX 4090.

    Same architecture as baseline_4090; more data allows longer training.
    200k steps measured ~64 min at batch 256 on the small profile; expect
    similar (estimate). Pick the best-val checkpoint via metrics.json."""
    return TrainConfig(
        run_name="paperscale_4090",
        model=ModelCfg(),
        batch_size=256,
        max_steps=200_000,
        warmup_steps=1000,
        num_workers=4,
        ckpt_interval=10_000,
        val_interval=2_000,
        sample_interval=20_000,
        norm_max_windows=4000,
        data=DataCfg(
            processed_dir="data/motion_gen/processed_paperscale",
            metadata_dir="data/motion_gen/metadata_paperscale",
            splits_file="data/motion_gen/splits/splits_paperscale.json",
        ),
    )


def terrain_4090() -> TrainConfig:
    """Phase B: paperscale data + real terrain height scans (climb/scene
    clips; flat clips keep zero scans). Experimental until validated."""
    cfg = paperscale_4090()
    cfg.run_name = "terrain_4090"
    cfg.data.use_terrain_scan = True
    cfg.data.scan_grid = ScanGrid()
    cfg.data.terrain_dim = ScanGrid().dim  # 17x17 forward-biased grid = 289
    return cfg


def terrain_robust_4090() -> TrainConfig:
    """Terrain generator with structured condition noise on one RTX 4090.

    All noise values below are implementation choices: arXiv:2604.17335 does
    not publish condition-noise magnitudes or distributions.
    """
    cfg = terrain_4090()
    cfg.run_name = "terrain_robust_4090"
    cfg.resume_weights_only = True
    cfg.condition_noise = ConditionNoiseCfg(
        root_position_std_m=0.01,
        root_orientation_std_rad=0.02,
        joint_position_std_rad=0.01,
        body_position_std_m=0.01,
        terrain_height_std_m=0.01,
        terrain_point_dropout_prob=0.05,
        terrain_height_bias_std_m=0.01,
        terrain_xy_std_m=0.02,
        terrain_yaw_std_rad=0.02,
    )
    return cfg


def terrain_robust_fk_4090() -> TrainConfig:
    """Structured condition noise plus differentiable FK consistency.

    The FK weight is an implementation choice because arXiv:2604.17335 does
    not publish its geometric-loss weights.  Keeping a separate preset leaves
    the condition-noise-only variant available for ablation.
    """
    cfg = terrain_robust_4090()
    cfg.run_name = "terrain_robust_fk_4090"
    cfg.loss.fk_consistency = 0.1
    return cfg


def terrain_feasibility_4090() -> TrainConfig:
    """Full-scratch terrain-feasibility training on one RTX 4090.

    This preset fixes the terrain-blind validation path and adds explicit
    joint-limit and pinned-MJCF lower-body collision-proxy losses.  Every loss
    weight, margin, tolerance, and the 50/50 sampler are implementation choices
    because arXiv:2604.17335 does not publish them.  The name describes the
    training objective; feasibility still has to pass the MuJoCo gate.
    """
    cfg = terrain_robust_fk_4090()
    cfg.run_name = "terrain_feasibility_4090"
    cfg.resume = None
    cfg.resume_weights_only = False
    cfg.max_steps = 200_000
    cfg.ckpt_interval = 25_000
    cfg.val_interval = 2_000
    cfg.sample_interval = 10_000
    cfg.val_batches = 32
    cfg.val_sample_steps = 2
    cfg.data.terrain_train_fraction = 0.5
    cfg.data.stratified_validation = True
    cfg.loss.joint_limit = 10.0
    cfg.loss.lower_body_terrain_penetration = 1.0
    # Implementation choice: the raw 1% tail term is roughly two orders of
    # magnitude larger than the mean collision term on the legacy checkpoint.
    # A 0.1 weight keeps it influential without dominating reconstruction.
    cfg.loss.lower_body_terrain_tail = 0.1
    cfg.loss.fk_consistency = 10.0
    cfg.joint_limit_margin_rad = 0.005
    cfg.terrain_penetration_tolerance_m = 0.005
    cfg.terrain_penetration_tail_fraction = 0.01
    cfg.wandb = WandbLoggingConfig(
        mode="online",
        entity="hkleetony-dyros",
        project="HoloSomaMotionGenerator",
        group="terrain_feasibility_retrain",
        tags=["generator", "terrain", "feasibility", "scratch"],
        log_final_checkpoint_artifact=True,
    )
    return cfg


PRESETS = {
    "smoke": smoke,
    "debug": debug,
    "baseline_4090": baseline_4090,
    "paperscale_4090": paperscale_4090,
    "terrain_4090": terrain_4090,
    "terrain_robust_4090": terrain_robust_4090,
    "terrain_robust_fk_4090": terrain_robust_fk_4090,
    "terrain_feasibility_4090": terrain_feasibility_4090,
}
