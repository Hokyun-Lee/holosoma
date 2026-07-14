from __future__ import annotations

import dataclasses
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from holosoma.motion_gen.configs import DataCfg, DiffusionCfg, ModelCfg, TrainConfig
from holosoma.motion_gen.features import FeatureLayout
from holosoma.motion_gen.terrain import ScanGrid
from holosoma.motion_gen.tests.synthetic import make_synthetic_wbt_npz
from holosoma.motion_gen.training import Trainer, ValidationLossAccumulator


def _write_dataset(root: Path, *, num_frames: int = 40) -> dict[str, Path]:
    processed = root / "processed"
    metadata = root / "metadata"
    splits_dir = root / "splits"
    for directory in (processed, metadata, splits_dir):
        directory.mkdir(parents=True, exist_ok=True)

    stems = ("flat_train", "terrain_train", "flat_val", "terrain_val")
    grid = ScanGrid()
    for seed, stem in enumerate(stems):
        path = make_synthetic_wbt_npz(processed / f"{stem}.npz", num_frames=num_frames, seed=seed)
        scanned = stem.startswith("terrain")
        if scanned:
            with np.load(path, allow_pickle=True) as source:
                payload = {key: source[key] for key in source.files}
            payload["terrain_height"] = np.full((num_frames, grid.dim), 0.2, dtype=np.float32)
            payload["terrain_grid"] = grid.to_array()
            np.savez(path, **payload)
        (metadata / f"{stem}.json").write_text(json.dumps({"source": "synthetic", "flat_terrain": not scanned}))

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
    return {"processed": processed, "metadata": metadata, "splits": splits_file}


def _config(tmp_path: Path, run_name: str = "resume_hardening") -> TrainConfig:
    paths = _write_dataset(tmp_path / "data")
    return TrainConfig(
        run_name=run_name,
        out_root=str(tmp_path / "logs"),
        device="cpu",
        seed=17,
        data=DataCfg(
            processed_dir=str(paths["processed"]),
            metadata_dir=str(paths["metadata"]),
            splits_file=str(paths["splits"]),
            future_frames=5,
            train_stride=5,
            val_stride=10,
            terrain_dim=ScanGrid().dim,
            use_terrain_scan=True,
            terrain_train_fraction=0.5,
            stratified_validation=True,
        ),
        model=ModelCfg(d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.0),
        diffusion=DiffusionCfg(timesteps=10),
        batch_size=4,
        max_steps=10,
        warmup_steps=2,
        amp=False,
        use_ema=True,
        num_workers=0,
        val_num_workers=0,
        val_batches=8,
        val_sample_steps=2,
        norm_max_windows=20,
    )


def _close(trainer: Trainer) -> None:
    trainer.writer.close()
    trainer.wandb.finish()


def _advance_optimizer_and_scheduler(trainer: Trainer, steps: int) -> None:
    for _ in range(steps):
        trainer.optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().sum() for parameter in trainer.model.parameters()).backward()
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.step += 1


def _assert_numpy_rng_state_equal(actual, expected) -> None:
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def test_full_resume_restores_scheduler_rng_sampler_and_metrics(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    trainer = Trainer(cfg)
    try:
        _advance_optimizer_and_scheduler(trainer, 3)
        trainer.metrics_history = [{"step": 2, "val_loss": {"total": 0.25}}]
        assert trainer.train_sampler_generator is not None
        list(trainer.train_loader.sampler)
        random.random()
        np.random.random()
        torch.rand(3)
        checkpoint_path = trainer.save_checkpoint("stateful.pt")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    finally:
        _close(trainer)

    random.random()
    np.random.random()
    torch.rand(3)
    list(trainer.train_loader.sampler)

    resumed = Trainer(dataclasses.replace(cfg, resume=str(checkpoint_path)))
    try:
        assert resumed.step == 3
        assert resumed.metrics_history == saved["metrics_history"]
        assert resumed.scheduler.state_dict() == saved["scheduler"]
        assert random.getstate() == saved["rng_state"]["python"]
        _assert_numpy_rng_state_equal(np.random.get_state(), saved["rng_state"]["numpy"])
        torch.testing.assert_close(torch.get_rng_state(), saved["rng_state"]["torch_cpu"])
        assert resumed.train_sampler_generator is not None
        torch.testing.assert_close(
            resumed.train_sampler_generator.get_state(),
            saved["train_sampler_generator_state"],
        )
    finally:
        _close(resumed)


def test_weights_only_does_not_restore_trainer_state(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    trainer = Trainer(cfg)
    try:
        _advance_optimizer_and_scheduler(trainer, 2)
        trainer.metrics_history = [{"step": 1}]
        assert trainer.train_sampler_generator is not None
        list(trainer.train_loader.sampler)
        checkpoint_path = trainer.save_checkpoint("source.pt")
    finally:
        _close(trainer)

    restarted_cfg = dataclasses.replace(
        cfg,
        run_name="weights_only",
        resume=str(checkpoint_path),
        resume_weights_only=True,
    )
    restarted = Trainer(restarted_cfg)
    try:
        assert restarted.step == 0
        assert restarted.metrics_history == []
        assert restarted.scheduler.last_epoch == 0
        assert restarted.train_sampler_generator is not None
        expected_sampler = torch.Generator().manual_seed(cfg.seed)
        torch.testing.assert_close(restarted.train_sampler_generator.get_state(), expected_sampler.get_state())
    finally:
        _close(restarted)


def test_missing_ema_uses_loaded_model_and_legacy_scheduler_has_no_replay_warning(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    trainer = Trainer(cfg)
    try:
        _advance_optimizer_and_scheduler(trainer, 3)
        checkpoint_path = trainer.save_checkpoint("new.pt")
    finally:
        _close(trainer)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["ema"] = None
    for key in ("scheduler", "rng_state", "train_sampler_generator_state", "metrics_history"):
        checkpoint.pop(key)
    legacy_path = checkpoint_path.with_name("legacy_no_ema.pt")
    torch.save(checkpoint, legacy_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resumed = Trainer(dataclasses.replace(cfg, resume=str(legacy_path)))
    try:
        assert not any("lr_scheduler.step" in str(item.message) for item in caught)
        assert resumed.scheduler.last_epoch == resumed.step == 3
        assert resumed.scheduler.get_last_lr() == [group["lr"] for group in resumed.optimizer.param_groups]
        assert resumed.metrics_history == []
        assert resumed.ema_model is not None
        for model_value, ema_value in zip(resumed.model.state_dict().values(), resumed.ema_model.state_dict().values()):
            torch.testing.assert_close(model_value, ema_value)
    finally:
        _close(resumed)


def test_scratch_output_guard_and_explicit_normalizer_reuse(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = Trainer(cfg)
    expected_mean = first.normalizer.mean.clone()
    config_before = (first.out_dir / "config.yaml").read_text()
    _close(first)

    with pytest.raises(FileExistsError, match="Refusing to start a scratch run"):
        Trainer(cfg)
    assert (first.out_dir / "config.yaml").read_text() == config_before

    opted_in = Trainer(dataclasses.replace(cfg, allow_existing_output=True))
    try:
        torch.testing.assert_close(opted_in.normalizer.mean, expected_mean)
    finally:
        _close(opted_in)


def test_validation_reduction_scope_and_nonpersistent_loaders(tmp_path: Path) -> None:
    accumulator = ValidationLossAccumulator()
    first_batch = {
        "x": torch.zeros(2, 2, 1),
        "contact": torch.ones(2, 2, 1, dtype=torch.bool),
    }
    second_batch = {
        "x": torch.zeros(1, 2, 1),
        "contact": torch.zeros(1, 2, 1, dtype=torch.bool),
    }
    accumulator.update(
        {
            "root_pos": torch.tensor(1.0),
            "foot_slide": torch.tensor(3.0),
            "lower_body_terrain_tail": torch.tensor(2.0),
            "lower_body_terrain_tail_count": torch.tensor(2.0),
            "lower_body_max_penetration_m": torch.tensor(0.2),
            "terrain_penetration": torch.tensor(0.1),
        },
        first_batch,
    )
    accumulator.update(
        {
            "root_pos": torch.tensor(4.0),
            "foot_slide": torch.tensor(99.0),
            "lower_body_terrain_tail": torch.tensor(8.0),
            "lower_body_terrain_tail_count": torch.tensor(1.0),
            "lower_body_max_penetration_m": torch.tensor(0.5),
            "terrain_penetration": torch.tensor(0.4),
        },
        second_batch,
    )
    values, aggregation = accumulator.finalize()
    assert values["root_pos"] == pytest.approx(2.0)
    assert values["foot_slide"] == pytest.approx(3.0)
    assert values["lower_body_terrain_tail"] == pytest.approx(4.0)
    assert values["lower_body_terrain_tail_count"] == pytest.approx(3.0)
    assert values["lower_body_max_penetration_m"] == pytest.approx(0.5)
    assert values["terrain_penetration"] == pytest.approx(0.2)
    assert aggregation["num_samples"] == 3
    assert aggregation["sample_weight_approximations"] == ["terrain_penetration"]

    trainer = Trainer(_config(tmp_path / "integration", run_name="validation_scope"))
    try:
        assert trainer.val_loader.num_workers == 0
        assert not trainer.val_loader.persistent_workers
        assert all(not loader.persistent_workers for loader in trainer.val_stratum_loaders.values())
        entry = trainer.validate()
        assert entry["val_sample_scope"] == "first_batch"
        assert entry["val_loss_aggregation_scope"] == "full_loader"
        assert entry["val_loss_aggregation"]["num_samples"] == len(trainer.val_dataset)
        assert all(result["sample_scope"] == "first_batch" for result in entry["val_strata"].values())
    finally:
        _close(trainer)
