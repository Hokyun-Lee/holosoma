from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from holosoma.motion_gen.configs import (
    PRESETS,
    DataCfg,
    DiffusionCfg,
    ModelCfg,
    TrainConfig,
    baseline_4090,
    debug,
    paperscale_4090,
    smoke,
    terrain_4090,
    terrain_feasibility_4090,
    terrain_robust_4090,
    terrain_robust_fk_4090,
)
from holosoma.motion_gen.dataset import MotionClip, MotionWindowDataset
from holosoma.motion_gen.features import FeatureLayout
from holosoma.motion_gen.terrain import ScanGrid
from holosoma.motion_gen.tests.synthetic import make_synthetic_wbt_npz
from holosoma.motion_gen.training import (
    Trainer,
    terrain_balanced_sample_weights,
    terrain_stratum_indices,
)


def _memory_clip(
    name: str,
    *,
    num_frames: int,
    terrain_dim: int,
    scanned: bool,
) -> MotionClip:
    layout = FeatureLayout()
    features = torch.zeros(num_frames, layout.dim)
    features[:, 0] = torch.arange(num_frames) * 0.01
    features[:, layout.root_quat_slice.start] = 1.0
    terrain_scan = torch.full((num_frames, terrain_dim), 0.25) if scanned else None
    return MotionClip(
        name=name,
        fps=50.0,
        features=features,
        foot_contact=torch.zeros(num_frames, 2, dtype=torch.bool),
        flat_terrain=not scanned,
        source="synthetic",
        terrain_scan=terrain_scan,
        terrain_grid=None,
    )


def _memory_dataset(*, include_flat: bool = True, include_terrain: bool = True) -> MotionWindowDataset:
    terrain_dim = 9
    clips = []
    if include_flat:
        clips.append(
            _memory_clip(
                "flat",
                num_frames=10,
                terrain_dim=terrain_dim,
                scanned=False,
            )
        )
    if include_terrain:
        clips.append(
            _memory_clip(
                "terrain",
                num_frames=8,
                terrain_dim=terrain_dim,
                scanned=True,
            )
        )
    return MotionWindowDataset(
        clips,
        FeatureLayout(),
        past_frames=2,
        future_frames=3,
        stride=1,
        terrain_dim=terrain_dim,
        use_terrain_scan=True,
    )


def test_terrain_stratum_indices_follow_source_clip_scan_availability():
    dataset = _memory_dataset()

    strata = terrain_stratum_indices(dataset)

    assert strata == {
        "flat": list(range(6)),
        "terrain": list(range(6, 10)),
    }
    assert not set(strata["flat"]) & set(strata["terrain"])
    assert sorted(strata["flat"] + strata["terrain"]) == list(range(len(dataset)))


@pytest.mark.parametrize("terrain_fraction", [0.2, 0.5, 0.8])
def test_balanced_weights_assign_requested_total_probability_mass(terrain_fraction: float):
    dataset = _memory_dataset()
    strata = terrain_stratum_indices(dataset)

    weights, counts = terrain_balanced_sample_weights(dataset, terrain_fraction)

    assert weights.dtype == torch.double
    assert counts == {"flat": 6, "terrain": 4}
    assert weights.sum().item() == pytest.approx(1.0)
    assert weights[strata["terrain"]].sum().item() == pytest.approx(terrain_fraction)
    assert weights[strata["flat"]].sum().item() == pytest.approx(1.0 - terrain_fraction)
    assert torch.unique(weights[strata["terrain"]]).numel() == 1
    assert torch.unique(weights[strata["flat"]]).numel() == 1


def test_weighted_sampler_is_deterministic_for_a_fixed_generator_seed():
    weights, _ = terrain_balanced_sample_weights(_memory_dataset(), 0.5)

    def draw(seed: int) -> list[int]:
        sampler = WeightedRandomSampler(
            weights,
            num_samples=128,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        return list(sampler)

    first = draw(123)
    second = draw(123)

    assert first == second
    assert first != draw(124)
    assert any(index < 6 for index in first)
    assert any(index >= 6 for index in first)


@pytest.mark.parametrize(
    "terrain_fraction",
    [-1.0, 0.0, 1.0, 2.0, float("nan"), float("inf"), -float("inf")],
)
def test_balanced_weights_reject_invalid_fraction(terrain_fraction: float):
    with pytest.raises(ValueError, match="finite and strictly between"):
        terrain_balanced_sample_weights(_memory_dataset(), terrain_fraction)


@pytest.mark.parametrize(
    ("include_flat", "include_terrain", "expected_counts"),
    [
        (True, False, "'flat': 6, 'terrain': 0"),
        (False, True, "'flat': 0, 'terrain': 4"),
    ],
)
def test_balanced_weights_reject_missing_stratum(
    include_flat: bool,
    include_terrain: bool,
    expected_counts: str,
):
    dataset = _memory_dataset(
        include_flat=include_flat,
        include_terrain=include_terrain,
    )
    with pytest.raises(ValueError, match="requires both") as exc_info:
        terrain_balanced_sample_weights(dataset, 0.5)
    assert expected_counts in str(exc_info.value)


def test_new_preset_is_scratch_balanced_and_old_presets_remain_opt_out():
    old_presets = [
        smoke(),
        debug(),
        baseline_4090(),
        paperscale_4090(),
        terrain_4090(),
        terrain_robust_4090(),
        terrain_robust_fk_4090(),
    ]
    for cfg in old_presets:
        assert cfg.data.terrain_train_fraction is None
        assert not cfg.data.stratified_validation
        assert cfg.wandb.mode == "disabled"
        assert cfg.loss.joint_limit == 0.0
        assert cfg.loss.lower_body_terrain_penetration == 0.0
        assert cfg.joint_limit_margin_rad == 0.0
        assert cfg.terrain_penetration_tolerance_m == 0.0

    robust_fk = terrain_robust_fk_4090()
    assert robust_fk.resume_weights_only
    assert robust_fk.loss.fk_consistency == pytest.approx(0.1)

    cfg = terrain_feasibility_4090()
    assert PRESETS["terrain_feasibility_4090"] is terrain_feasibility_4090
    assert cfg.run_name == "terrain_feasibility_4090"
    assert cfg.resume is None
    assert not cfg.resume_weights_only
    assert cfg.max_steps == 200_000
    assert cfg.ckpt_interval == 25_000
    assert cfg.val_sample_steps == 2
    assert cfg.data.terrain_train_fraction == pytest.approx(0.5)
    assert cfg.data.stratified_validation
    assert cfg.loss.joint_limit == pytest.approx(10.0)
    assert cfg.loss.lower_body_terrain_penetration == pytest.approx(1.0)
    assert cfg.loss.fk_consistency == pytest.approx(10.0)
    assert cfg.joint_limit_margin_rad == pytest.approx(0.005)
    assert cfg.terrain_penetration_tolerance_m == pytest.approx(0.005)
    assert cfg.wandb.mode == "online"
    assert cfg.wandb.group == "terrain_feasibility_retrain"
    assert cfg.wandb.log_final_checkpoint_artifact


def _write_trainer_dataset(root: Path, *, num_frames: int = 50) -> dict[str, Path]:
    processed = root / "processed"
    metadata = root / "metadata"
    splits_dir = root / "splits"
    for directory in (processed, metadata, splits_dir):
        directory.mkdir(parents=True, exist_ok=True)

    stems = ["flat_train", "terrain_train", "flat_val", "terrain_val"]
    grid = ScanGrid()
    for seed, stem in enumerate(stems):
        path = make_synthetic_wbt_npz(
            processed / f"{stem}.npz",
            num_frames=num_frames,
            seed=seed,
        )
        scanned = stem.startswith("terrain")
        if scanned:
            with np.load(path, allow_pickle=True) as data:
                payload = {key: data[key] for key in data.files}
            payload["terrain_height"] = np.full(
                (num_frames, grid.dim),
                0.25,
                dtype=np.float32,
            )
            payload["terrain_grid"] = grid.to_array()
            np.savez(path, **payload)
        (metadata / f"{stem}.json").write_text(
            json.dumps(
                {
                    "source": "synthetic",
                    "flat_terrain": not scanned,
                }
            )
        )

    layout = FeatureLayout()
    (metadata / "joint_limits.json").write_text(json.dumps({name: [-3.0, 3.0] for name in layout.joint_names}))
    splits_file = splits_dir / "splits.json"
    splits_file.write_text(
        json.dumps(
            {
                "train": ["flat_train", "terrain_train"],
                "val": ["flat_val", "terrain_val"],
            }
        )
    )
    return {
        "processed": processed,
        "metadata": metadata,
        "splits_file": splits_file,
    }


def test_trainer_stratified_loader_yields_real_terrain_batch(tmp_path: Path):
    paths = _write_trainer_dataset(tmp_path / "data")
    cfg = TrainConfig(
        run_name="stratified_loader_test",
        out_root=str(tmp_path / "logs"),
        device="cpu",
        seed=123,
        data=DataCfg(
            processed_dir=str(paths["processed"]),
            metadata_dir=str(paths["metadata"]),
            splits_file=str(paths["splits_file"]),
            future_frames=5,
            train_stride=5,
            val_stride=5,
            terrain_dim=ScanGrid().dim,
            use_terrain_scan=True,
            terrain_train_fraction=0.5,
            stratified_validation=True,
        ),
        model=ModelCfg(d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0),
        diffusion=DiffusionCfg(timesteps=20),
        batch_size=4,
        max_steps=1,
        warmup_steps=1,
        amp=False,
        use_ema=False,
        num_workers=0,
        val_batches=1,
        val_sample_steps=2,
        norm_max_windows=20,
    )

    trainer = Trainer(cfg)
    try:
        assert isinstance(trainer.train_loader.sampler, WeightedRandomSampler)
        assert trainer.train_sampler_generator is not None
        generator_state = trainer.train_sampler_generator.get_state()
        first_epoch_indices = list(trainer.train_loader.sampler)
        trainer.train_sampler_generator.set_state(generator_state)
        assert list(trainer.train_loader.sampler) == first_epoch_indices
        train_strata = terrain_stratum_indices(trainer.train_dataset)
        assert set(first_epoch_indices) & set(train_strata["flat"])
        assert set(first_epoch_indices) & set(train_strata["terrain"])

        assert trainer.val_stratum_counts == {"flat": 9, "terrain": 9}
        assert set(trainer.val_stratum_loaders) == {"flat", "terrain"}

        terrain_batch = next(iter(trainer.val_stratum_loaders["terrain"]))
        assert bool(terrain_batch["has_scan"].all())
        assert torch.allclose(
            terrain_batch["terrain"],
            torch.full_like(terrain_batch["terrain"], 0.25),
        )

        flat_batch = next(iter(trainer.val_stratum_loaders["flat"]))
        assert not bool(flat_batch["has_scan"].any())
        assert torch.count_nonzero(flat_batch["terrain"]) == 0
    finally:
        trainer.writer.close()
        trainer.wandb.finish()
