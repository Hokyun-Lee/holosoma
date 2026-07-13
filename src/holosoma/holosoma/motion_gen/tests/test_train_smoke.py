"""End-to-end CPU smoke test: training, checkpointing, inference, export."""

import dataclasses

import numpy as np
import pytest
import torch

from holosoma.motion_gen.configs import DataCfg, DiffusionCfg, ModelCfg, TrainConfig
from holosoma.motion_gen.export import export_generated_qpos_npz, export_generated_raw_npz
from holosoma.motion_gen.features import FeatureLayout
from holosoma.motion_gen.sampling import MotionGenerator, MotionGeneratorInput
from holosoma.motion_gen.tests.synthetic import make_synthetic_dataset_dir
from holosoma.motion_gen.training import Trainer


@pytest.fixture(scope="module")
def smoke_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("motion_gen_data")
    dirs = make_synthetic_dataset_dir(root, num_clips=2, num_frames=80)
    out_root = tmp_path_factory.mktemp("logs")
    return TrainConfig(
        run_name="pytest_smoke",
        out_root=str(out_root),
        device="cpu",
        seed=0,
        data=DataCfg(
            processed_dir=str(dirs["processed"]),
            metadata_dir=str(dirs["metadata"]),
            splits_file=str(dirs["splits_file"]),
            train_stride=5,
            val_stride=20,
            terrain_dim=16,
        ),
        model=ModelCfg(d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0),
        diffusion=DiffusionCfg(timesteps=20),
        batch_size=4,
        max_steps=10,
        warmup_steps=2,
        amp=False,
        num_workers=0,
        log_interval=5,
        ckpt_interval=10,
        val_interval=5,
        sample_interval=10,
        val_batches=1,
        val_sample_steps=3,
        norm_max_windows=50,
    )


@pytest.fixture(scope="module")
def trained(smoke_cfg):
    trainer = Trainer(smoke_cfg)
    trainer.train()
    return trainer


def test_training_produces_outputs(trained, smoke_cfg):
    out = trained.out_dir
    assert (out / "config.yaml").exists()
    assert (out / "normalization_stats.npz").exists()
    assert (out / "checkpoints" / "latest.pt").exists()
    assert (out / "checkpoints" / "final.pt").exists()
    assert (out / "metrics.json").exists()
    plots = list((out / "plots").rglob("*.png"))
    samples = list((out / "samples").rglob("*_gen_raw.npz"))
    assert plots, "no plots produced"
    assert samples, "no sample npz produced"


def test_metrics_history_has_val_entries(trained):
    assert trained.metrics_history
    entry = trained.metrics_history[-1]
    assert "val_sample_metrics" in entry
    for key in ("root_pos_err_m", "root_quat_err_rad", "joint_pos_err_rad", "body_mpjpe_m"):
        assert np.isfinite(entry["val_sample_metrics"][key])
    assert entry["val_condition_noise"]["shared_ddim_initial_noise"]
    assert not entry["val_condition_noise"]["enabled"]
    assert entry["val_loss_clean"] == entry["val_loss_noisy"]
    assert entry["val_sample_metrics_clean"] == entry["val_sample_metrics_noisy"]
    assert all(value == 0.0 for value in entry["val_condition_noise_delta"].values())


def test_checkpoint_reload_matches(trained, smoke_cfg):
    ckpt_path = trained.out_dir / "checkpoints" / "final.pt"
    cfg2 = dataclasses.replace(smoke_cfg, resume=str(ckpt_path))
    trainer2 = Trainer(cfg2)
    assert trainer2.step == trained.step
    for p1, p2 in zip(trained.model.parameters(), trainer2.model.parameters()):
        assert torch.allclose(p1, p2)


def test_checkpoint_weights_only_restarts_optimizer_and_step(trained, smoke_cfg):
    ckpt_path = trained.out_dir / "checkpoints" / "final.pt"
    cfg2 = dataclasses.replace(
        smoke_cfg,
        run_name="pytest_weights_only",
        resume=str(ckpt_path),
        resume_weights_only=True,
    )
    trainer2 = Trainer(cfg2)
    assert trainer2.step == 0
    assert not trainer2.optimizer.state
    for p1, p2 in zip(trained.model.parameters(), trainer2.model.parameters()):
        assert torch.allclose(p1, p2)
    torch.testing.assert_close(trained.normalizer.mean, trainer2.normalizer.mean)
    torch.testing.assert_close(trained.normalizer.std, trainer2.normalizer.std)


def test_generator_inference_and_determinism(trained):
    ckpt_path = str(trained.out_dir / "checkpoints" / "final.pt")
    gen = MotionGenerator.from_checkpoint(ckpt_path, device="cpu")
    assert not gen.model.training
    assert not any(p.requires_grad for p in gen.model.parameters())
    layout = gen.layout
    past = trained.val_dataset[0]["past"].unsqueeze(0)

    outs = [
        gen.generate(MotionGeneratorInput(past_motion=past), num_steps=3, deterministic=True, seed=7)
        for _ in range(2)
    ]
    H = gen.cfg.data.future_frames
    assert outs[0].root_pos.shape == (1, H, 3)
    assert outs[0].joint_pos.shape == (1, H, layout.num_joints)
    assert outs[0].body_pos.shape == (1, H, layout.num_bodies, 3)
    assert torch.allclose(outs[0].features, outs[1].features), "deterministic sampling not reproducible"
    # world-frame continuity: first generated root should be near the anchor
    anchor = past[0, -1, :3]
    assert (outs[0].root_pos[0, 0] - anchor).norm() < 10.0


def test_receding_horizon_shapes(trained):
    gen = MotionGenerator.from_checkpoint(str(trained.out_dir / "checkpoints" / "final.pt"), device="cpu")
    past = trained.val_dataset[0]["past"]
    traj = gen.receding_horizon(past, num_cycles=3, replan_stride=5, num_steps=2, deterministic=True)
    assert traj.shape == (2 + 3 * 5, gen.layout.dim)
    assert torch.isfinite(traj).all()
    # fed-back quaternions must stay on the unit manifold
    quats = traj[2:, gen.layout.root_quat_slice]
    assert torch.allclose(quats.norm(dim=-1), torch.ones(15), atol=1e-4)


def test_generate_with_target_heading(trained):
    """Regression: world-frame heading rotation used to crash on shape mismatch."""
    gen = MotionGenerator.from_checkpoint(str(trained.out_dir / "checkpoints" / "final.pt"), device="cpu")
    past = trained.val_dataset[0]["past"].unsqueeze(0).repeat(3, 1, 1)
    heading = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    out = gen.generate(
        MotionGeneratorInput(past_motion=past, target_heading=heading),
        num_steps=2, deterministic=True,
    )
    assert out.features.shape == (3, gen.cfg.data.future_frames, gen.layout.dim)
    assert torch.isfinite(out.features).all()


def test_receding_horizon_with_heading_and_guidance(trained):
    gen = MotionGenerator.from_checkpoint(str(trained.out_dir / "checkpoints" / "final.pt"), device="cpu")
    past = trained.val_dataset[0]["past"]
    traj = gen.receding_horizon(
        past, num_cycles=2, replan_stride=4, num_steps=2,
        target_heading=torch.tensor([0.0, 1.0]), deterministic=True,
    )
    assert traj.shape == (2 + 2 * 4, gen.layout.dim)
    out = gen.generate(
        MotionGeneratorInput(past_motion=past.unsqueeze(0)),
        num_steps=2, deterministic=True, guidance_scale=2.0,
    )
    assert torch.isfinite(out.features).all()


def test_export_npz_files(trained, tmp_path):
    layout = FeatureLayout()
    feats = trained.val_dataset[0]["x"]
    raw = export_generated_raw_npz(feats, layout, 50.0, tmp_path / "gen_raw.npz", gt_features=feats)
    qpos = export_generated_qpos_npz(feats, layout, 50.0, tmp_path / "gen_qpos.npz")

    raw_data = np.load(raw)
    for key in ("fps", "root_pos", "root_quat_wxyz", "joint_pos", "body_pos", "joint_names", "body_names"):
        assert key in raw_data.files, key
    q = np.load(qpos)
    assert q["qpos"].shape == (feats.shape[0], 36)
    assert int(np.asarray(q["fps"]).reshape(-1)[0]) == 50
    norms = np.linalg.norm(q["qpos"][:, 3:7], axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)
