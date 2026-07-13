from __future__ import annotations

import dataclasses
from types import MethodType

import pytest
import torch

from holosoma.motion_gen.condition_noise import (
    apply_condition_noise,
    axis_angle_to_quat_wxyz,
    validate_condition_noise_config,
    warp_terrain_scan,
)
from holosoma.motion_gen.configs import (
    ConditionNoiseCfg,
    DataCfg,
    DiffusionCfg,
    ModelCfg,
    TrainConfig,
    terrain_robust_4090,
    terrain_robust_fk_4090,
)
from holosoma.motion_gen.diffusion import GaussianDiffusion
from holosoma.motion_gen.features import FeatureLayout, pack_features, quat_from_yaw, unpack_features
from holosoma.motion_gen.losses import LossWeights
from holosoma.motion_gen.normalization import FeatureNormalizer
from holosoma.motion_gen.sampling import _config_from_dict
from holosoma.motion_gen.terrain import ScanGrid
from holosoma.motion_gen.tests.synthetic import make_synthetic_dataset_dir
from holosoma.motion_gen.training import Trainer


def _physical_conditions(
    *,
    batch_size: int = 3,
    past_frames: int = 5,
    grid: ScanGrid | None = None,
) -> tuple[FeatureLayout, torch.Tensor, torch.Tensor]:
    layout = FeatureLayout()
    generator = torch.Generator().manual_seed(4)
    root_pos = torch.randn(batch_size, past_frames, 3, generator=generator)
    yaw = torch.linspace(-0.2, 0.2, past_frames).expand(batch_size, -1)
    root_quat = quat_from_yaw(yaw)
    joint_pos = torch.randn(
        batch_size,
        past_frames,
        layout.num_joints,
        generator=generator,
    )
    body_pos = torch.randn(
        batch_size,
        past_frames,
        layout.num_bodies,
        3,
        generator=generator,
    )
    scan_grid = grid or ScanGrid(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1, spacing=0.1)
    terrain = torch.randn(batch_size, scan_grid.dim, generator=generator)
    return layout, pack_features(root_pos, root_quat, joint_pos, body_pos), terrain


def test_zero_defaults_are_exact_noop_and_backward_compatible():
    grid = ScanGrid(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1, spacing=0.1)
    layout, past, terrain = _physical_conditions(grid=grid)
    past_before = past.clone()
    terrain_before = terrain.clone()

    noisy_past, noisy_terrain = apply_condition_noise(
        past,
        terrain,
        layout,
        ConditionNoiseCfg(),
        grid,
        generator=torch.Generator().manual_seed(1),
    )

    torch.testing.assert_close(noisy_past, past_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(noisy_terrain, terrain_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(past, past_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(terrain, terrain_before, rtol=0.0, atol=0.0)
    assert not ConditionNoiseCfg().is_enabled()

    old_checkpoint_cfg = dataclasses.asdict(TrainConfig())
    old_checkpoint_cfg.pop("condition_noise")
    restored = _config_from_dict(old_checkpoint_cfg)
    assert restored.condition_noise == ConditionNoiseCfg()


def test_wxyz_axis_angle_and_noisy_quaternion_manifold_continuity():
    quaternion = axis_angle_to_quat_wxyz(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([torch.pi]),
    )
    torch.testing.assert_close(
        quaternion.abs(),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )

    grid = ScanGrid(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1, spacing=0.1)
    layout, past, terrain = _physical_conditions(grid=grid)
    # Exercise reference sign preservation rather than assuming w >= 0.
    past[0, :, layout.root_quat_slice] *= -1.0
    original = past.clone()
    cfg = ConditionNoiseCfg(
        root_position_std_m=0.02,
        root_orientation_std_rad=0.3,
        joint_position_std_rad=0.04,
        body_position_std_m=0.03,
    )
    noisy, unchanged_terrain = apply_condition_noise(
        past,
        terrain,
        layout,
        cfg,
        grid,
        generator=torch.Generator().manual_seed(12),
    )
    clean_parts = unpack_features(original, layout)
    noisy_parts = unpack_features(noisy, layout)

    assert not torch.equal(noisy_parts["root_pos"], clean_parts["root_pos"])
    assert not torch.equal(noisy_parts["root_quat"], clean_parts["root_quat"])
    assert not torch.equal(noisy_parts["joint_pos"], clean_parts["joint_pos"])
    assert not torch.equal(noisy_parts["body_pos"], clean_parts["body_pos"])
    torch.testing.assert_close(
        noisy_parts["root_quat"].norm(dim=-1),
        torch.ones_like(noisy_parts["root_quat"][..., 0]),
        atol=1.0e-6,
        rtol=0.0,
    )
    adjacent_dots = (
        noisy_parts["root_quat"][:, 1:] * noisy_parts["root_quat"][:, :-1]
    ).sum(dim=-1)
    assert torch.all(adjacent_dots >= 0.0)
    first_reference_dots = (
        noisy_parts["root_quat"][:, 0] * clean_parts["root_quat"][:, 0]
    ).sum(dim=-1)
    assert torch.all(first_reference_dots >= 0.0)
    torch.testing.assert_close(unchanged_terrain, terrain, rtol=0.0, atol=0.0)
    torch.testing.assert_close(past, original, rtol=0.0, atol=0.0)


def test_condition_noise_is_seed_deterministic():
    grid = ScanGrid(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1, spacing=0.1)
    layout, past, terrain = _physical_conditions(grid=grid)
    cfg = ConditionNoiseCfg(
        root_position_std_m=0.01,
        root_orientation_std_rad=0.02,
        joint_position_std_rad=0.01,
        body_position_std_m=0.01,
        terrain_height_std_m=0.01,
        terrain_point_dropout_prob=0.2,
        terrain_height_bias_std_m=0.01,
        terrain_xy_std_m=0.01,
        terrain_yaw_std_rad=0.02,
    )

    first = apply_condition_noise(
        past,
        terrain,
        layout,
        cfg,
        grid,
        generator=torch.Generator().manual_seed(99),
    )
    second = apply_condition_noise(
        past,
        terrain,
        layout,
        cfg,
        grid,
        generator=torch.Generator().manual_seed(99),
    )

    torch.testing.assert_close(first[0], second[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first[1], second[1], rtol=0.0, atol=0.0)


def test_terrain_warp_preserves_flatten_order_and_applies_xy_yaw_queries():
    grid = ScanGrid(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, spacing=1.0)
    offsets = grid.offsets_tensor(device="cpu")
    scan = (offsets[:, 0] + 10.0 * offsets[:, 1]).unsqueeze(0)

    shifted = warp_terrain_scan(
        scan,
        grid,
        translation_xy=torch.tensor([[1.0, 0.0]]),
        yaw=torch.zeros(1),
    )
    # Centre output queries clean (x=1, y=0).
    assert shifted[0, 1 * grid.ny + 1].item() == pytest.approx(1.0)
    # Output (x=1, y=0) queries outside x=2 and receives ground height zero.
    assert shifted[0, 2 * grid.ny + 1].item() == pytest.approx(0.0)

    rotated = warp_terrain_scan(
        scan,
        grid,
        translation_xy=torch.zeros(1, 2),
        yaw=torch.tensor([torch.pi / 2.0]),
    )
    # Output p=(1,0) queries R(90deg)p=(0,1), whose encoded height is 10.
    assert rotated[0, 2 * grid.ny + 1].item() == pytest.approx(10.0, abs=1.0e-5)


def test_terrain_bias_is_per_scan_and_dropout_uses_ground_zero():
    grid = ScanGrid(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1, spacing=0.1)
    layout, past, _ = _physical_conditions(batch_size=4, grid=grid)
    zeros = torch.zeros(4, grid.dim)
    biased_cfg = ConditionNoiseCfg(terrain_height_bias_std_m=0.2)
    _, biased = apply_condition_noise(
        past,
        zeros,
        layout,
        biased_cfg,
        grid,
        generator=torch.Generator().manual_seed(8),
    )
    torch.testing.assert_close(biased, biased[:, :1].expand_as(biased))
    assert torch.any(biased != 0.0)

    dropout_cfg = ConditionNoiseCfg(
        terrain_height_std_m=1.0,
        terrain_height_bias_std_m=1.0,
        terrain_point_dropout_prob=1.0,
    )
    _, dropped = apply_condition_noise(
        past,
        zeros,
        layout,
        dropout_cfg,
        grid,
        generator=torch.Generator().manual_seed(8),
    )
    torch.testing.assert_close(dropped, torch.zeros_like(dropped), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "cfg",
    [
        ConditionNoiseCfg(root_position_std_m=-0.1),
        ConditionNoiseCfg(root_orientation_std_rad=float("inf")),
        ConditionNoiseCfg(terrain_point_dropout_prob=1.1),
    ],
)
def test_noise_config_rejects_invalid_magnitudes(cfg: ConditionNoiseCfg):
    grid = ScanGrid()
    with pytest.raises(ValueError, match="condition_noise"):
        validate_condition_noise_config(
            cfg,
            use_terrain_scan=True,
            terrain_dim=grid.dim,
            scan_grid=grid,
        )


def test_terrain_noise_requires_real_scan_contract():
    grid = ScanGrid()
    with pytest.raises(ValueError, match="use_terrain_scan"):
        validate_condition_noise_config(
            ConditionNoiseCfg(terrain_height_std_m=0.01),
            use_terrain_scan=False,
            terrain_dim=grid.dim,
            scan_grid=grid,
        )
    with pytest.raises(ValueError, match="terrain_dim"):
        validate_condition_noise_config(
            ConditionNoiseCfg(terrain_xy_std_m=0.01),
            use_terrain_scan=True,
            terrain_dim=3,
            scan_grid=grid,
        )


class _RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(self, x_t, t, past, heading, terrain, **kwargs):
        self.calls.append(
            {
                "x_t": x_t.detach().clone(),
                "t": t.detach().clone(),
                "past": past.detach().clone(),
                "terrain": terrain.detach().clone(),
            }
        )
        return x_t


def _minimal_trainer_for_forward(grid: ScanGrid) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.layout = FeatureLayout()
    trainer.cfg = TrainConfig(
        device="cpu",
        data=DataCfg(
            terrain_dim=grid.dim,
            use_terrain_scan=True,
            scan_grid=grid,
            future_frames=3,
        ),
        diffusion=DiffusionCfg(timesteps=10),
        condition_noise=ConditionNoiseCfg(
            root_position_std_m=0.1,
            terrain_height_std_m=0.1,
        ),
        cond_dropout=0.0,
    )
    trainer.normalizer = FeatureNormalizer(
        torch.full((trainer.layout.dim,), 2.0),
        torch.full((trainer.layout.dim,), 4.0),
    )
    trainer.diffusion = GaussianDiffusion(timesteps=10)
    trainer.model = _RecordingModel().train()
    return trainer


def test_forward_noise_precedes_normalization_and_keeps_targets_and_loss_terrain_clean(monkeypatch):
    grid = ScanGrid(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1, spacing=0.1)
    trainer = _minimal_trainer_for_forward(grid)
    layout, past, terrain = _physical_conditions(batch_size=2, past_frames=2, grid=grid)
    batch = {
        "past": past,
        "terrain": terrain,
        "x": torch.randn(2, 3, layout.dim),
        "heading": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "contact": torch.zeros(2, 3, 2, dtype=torch.bool),
        "flat": torch.zeros(2, dtype=torch.bool),
        "has_scan": torch.ones(2, dtype=torch.bool),
    }
    clean_batch = {key: value.clone() for key, value in batch.items()}
    captured_losses: list[dict[str, torch.Tensor]] = []

    def fake_compute_losses(pred, gt, feature_layout, weights, **kwargs):
        del feature_layout, weights
        captured_losses.append(
            {
                "gt": gt.detach().clone(),
                "terrain_scan": kwargs["terrain_scan"].detach().clone(),
            }
        )
        return {"total": pred.sum() * 0.0}

    monkeypatch.setattr("holosoma.motion_gen.training.compute_losses", fake_compute_losses)
    clean_diffusion_gen = torch.Generator().manual_seed(44)
    noisy_diffusion_gen = torch.Generator().manual_seed(44)
    condition_seed = 55
    trainer._forward_losses(
        batch,
        generator=clean_diffusion_gen,
        apply_condition_noise_input=False,
    )
    trainer._forward_losses(
        batch,
        generator=noisy_diffusion_gen,
        apply_condition_noise_input=True,
        condition_generator=torch.Generator().manual_seed(condition_seed),
    )

    clean_call, noisy_call = trainer.model.calls
    torch.testing.assert_close(clean_call["t"], noisy_call["t"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(clean_call["x_t"], noisy_call["x_t"], rtol=0.0, atol=0.0)
    assert not torch.equal(clean_call["past"], noisy_call["past"])
    assert not torch.equal(clean_call["terrain"], noisy_call["terrain"])

    expected_past, expected_terrain = apply_condition_noise(
        batch["past"],
        batch["terrain"],
        trainer.layout,
        trainer.cfg.condition_noise,
        grid,
        generator=torch.Generator().manual_seed(condition_seed),
    )
    torch.testing.assert_close(noisy_call["past"], trainer.normalizer.normalize(expected_past))
    torch.testing.assert_close(noisy_call["terrain"], expected_terrain)
    for captured in captured_losses:
        torch.testing.assert_close(captured["gt"], clean_batch["x"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            captured["terrain_scan"],
            clean_batch["terrain"],
            rtol=0.0,
            atol=0.0,
        )
    for key, value in batch.items():
        torch.testing.assert_close(value, clean_batch[key], rtol=0.0, atol=0.0)


def test_validation_pair_reuses_exact_ddim_initial_noise():
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    calls: list[tuple[bool, torch.Tensor]] = []

    def fake_generate(
        self,
        batch,
        generator,
        *,
        apply_condition_noise_input=False,
        condition_generator=None,
        init_noise=None,
    ):
        del self, batch, generator, condition_generator
        calls.append((apply_condition_noise_input, init_noise.clone()))
        return init_noise

    trainer._generate_for_batch = MethodType(fake_generate, trainer)
    batch = {"x": torch.zeros(2, 3, 4)}
    clean, noisy = trainer._generate_validation_pair(
        batch,
        torch.Generator().manual_seed(6),
        torch.Generator().manual_seed(7),
    )

    assert [call[0] for call in calls] == [False, True]
    torch.testing.assert_close(calls[0][1], calls[1][1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(clean, noisy, rtol=0.0, atol=0.0)


def test_noisy_training_backward_and_clean_noisy_validation_are_finite(tmp_path):
    dirs = make_synthetic_dataset_dir(tmp_path / "data", num_clips=2, num_frames=60)
    grid = ScanGrid(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1, spacing=0.1)
    cfg = TrainConfig(
        run_name="condition_noise_backward_smoke",
        out_root=str(tmp_path / "logs"),
        device="cpu",
        seed=3,
        data=DataCfg(
            processed_dir=str(dirs["processed"]),
            metadata_dir=str(dirs["metadata"]),
            splits_file=str(dirs["splits_file"]),
            train_stride=10,
            val_stride=20,
            terrain_dim=grid.dim,
            use_terrain_scan=True,
            scan_grid=grid,
        ),
        model=ModelCfg(d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0),
        diffusion=DiffusionCfg(timesteps=20),
        condition_noise=ConditionNoiseCfg(
            root_position_std_m=0.01,
            root_orientation_std_rad=0.02,
            joint_position_std_rad=0.01,
            body_position_std_m=0.01,
            terrain_height_std_m=0.01,
            terrain_point_dropout_prob=0.1,
            terrain_height_bias_std_m=0.01,
            terrain_xy_std_m=0.01,
            terrain_yaw_std_rad=0.02,
        ),
        loss=LossWeights(fk_consistency=0.1),
        fk_calibration_tolerance_m=None,
        batch_size=4,
        max_steps=1,
        warmup_steps=1,
        amp=False,
        use_ema=False,
        cond_dropout=0.0,
        num_workers=0,
        val_batches=1,
        val_sample_steps=2,
        norm_max_windows=20,
    )
    trainer = Trainer(cfg)
    batch = trainer._batch_to_device(next(iter(trainer.train_loader)))
    trainer.model.train()
    losses = trainer._forward_losses(batch)
    losses["total"].backward()

    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["fk_consistency"])
    assert torch.isfinite(losses["fk_body_error_m"])
    gradients = [parameter.grad for parameter in trainer.model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    entry = trainer.validate()
    assert entry["val_condition_noise"]["enabled"]
    assert entry["val_condition_noise"]["shared_ddim_initial_noise"]
    assert all(torch.isfinite(torch.tensor(value)) for value in entry["val_loss_noisy"].values())
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in entry["val_sample_metrics_noisy"].values()
    )
    assert torch.isfinite(
        torch.tensor(
            entry["val_condition_noise_delta"][
                "sample_condition_response_mean_abs_mixed_units"
            ]
        )
    )
    trainer.writer.close()


def test_terrain_robust_preset_is_explicit_and_scan_enabled():
    cfg = terrain_robust_4090()
    assert cfg.run_name == "terrain_robust_4090"
    assert cfg.data.use_terrain_scan
    assert cfg.data.terrain_dim == cfg.data.scan_grid.dim
    assert cfg.condition_noise.is_enabled()
    assert cfg.resume_weights_only
    validate_condition_noise_config(
        cfg.condition_noise,
        use_terrain_scan=cfg.data.use_terrain_scan,
        terrain_dim=cfg.data.terrain_dim,
        scan_grid=cfg.data.scan_grid,
    )

    combined = terrain_robust_fk_4090()
    assert combined.run_name == "terrain_robust_fk_4090"
    assert combined.condition_noise == cfg.condition_noise
    assert combined.loss.fk_consistency == pytest.approx(0.1)
    assert combined.fk_calibration_tolerance_m == pytest.approx(1.0e-3)
