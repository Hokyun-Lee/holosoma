import pytest
import torch

from holosoma.motion_gen.features import FeatureLayout, pack_features, quat_normalize
from holosoma.motion_gen.kinematics import G1ForwardKinematics
from holosoma.motion_gen.losses import LossWeights, compute_losses


def _window(bsz=2, horizon=25, seed=0):
    layout = FeatureLayout()
    g = torch.Generator().manual_seed(seed)
    x = pack_features(
        torch.randn(bsz, horizon, 3, generator=g),
        quat_normalize(torch.randn(bsz, horizon, 4, generator=g)),
        torch.randn(bsz, horizon, layout.num_joints, generator=g),
        torch.randn(bsz, horizon, layout.num_bodies, 3, generator=g),
    )
    return layout, x


def test_losses_finite_and_nonnegative():
    layout, gt = _window()
    pred = gt + 0.1 * torch.randn_like(gt)
    losses = compute_losses(
        pred, gt, layout, LossWeights(),
        contact=torch.ones(2, 25, 2, dtype=torch.bool),
        flat=torch.tensor([True, False]),
    )
    for name, value in losses.items():
        assert torch.isfinite(value), name
        assert value >= 0, name


def test_perfect_prediction_gives_zero():
    layout, gt = _window()
    losses = compute_losses(
        gt.clone(), gt, layout, LossWeights(),
        contact=torch.zeros(2, 25, 2, dtype=torch.bool),
        flat=torch.tensor([False, False]),
    )
    for name in ("root_pos", "root_quat", "joint_pos", "body_pos", "quat_norm", "velocity", "bone_length"):
        assert losses[name] < 1e-6, name


def test_quat_loss_sign_invariant():
    layout, gt = _window()
    pred = gt.clone()
    pred[..., layout.root_quat_slice] = -pred[..., layout.root_quat_slice]
    losses = compute_losses(pred, gt, layout, LossWeights())
    assert losses["root_quat"] < 1e-6


def test_seq_mask_excludes_frames():
    layout, gt = _window()
    pred = gt.clone()
    pred[:, -5:] += 100.0  # corrupt only masked frames
    mask = torch.ones(2, 25, dtype=torch.bool)
    mask[:, -5:] = False
    losses = compute_losses(pred, gt, layout, LossWeights(), seq_mask=mask)
    assert losses["root_pos"] < 1e-6
    assert losses["joint_pos"] < 1e-6


def test_terrain_penetration_only_on_flat():
    layout, gt = _window()
    pred = gt.clone()
    body = pred[..., layout.body_pos_slice].reshape(2, 25, layout.num_bodies, 3)
    body[..., 2] = -1.0  # everything below ground
    flat_off = compute_losses(pred, gt, layout, LossWeights(), flat=torch.tensor([False, False]))
    flat_on = compute_losses(pred, gt, layout, LossWeights(), flat=torch.tensor([True, True]))
    assert flat_off["terrain_penetration"] < 1e-9
    assert flat_on["terrain_penetration"] > 0.1


def test_fk_consistency_is_zero_on_fk_body_positions_and_has_gradients():
    layout = FeatureLayout()
    fk = G1ForwardKinematics(dtype=torch.float64)
    root_pos = torch.tensor([[[0.1, -0.2, 0.8]]], dtype=torch.float64, requires_grad=True)
    root_quat = torch.tensor([[[0.9, 0.1, -0.2, 0.3]]], dtype=torch.float64, requires_grad=True)
    joint_pos = torch.linspace(-0.2, 0.3, layout.num_joints, dtype=torch.float64)
    joint_pos = joint_pos.view(1, 1, -1).requires_grad_()
    body_pos = fk(root_pos, root_quat, joint_pos).detach().requires_grad_()
    pred = pack_features(root_pos, root_quat, joint_pos, body_pos)
    weights = LossWeights(fk_consistency=1.0)

    exact = compute_losses(pred, pred.detach(), layout, weights, fk_model=fk)
    torch.testing.assert_close(exact["fk_consistency"], torch.zeros((), dtype=torch.float64))
    torch.testing.assert_close(exact["fk_body_error_m"], torch.zeros((), dtype=torch.float64))

    inconsistent = pred.clone()
    inconsistent[..., layout.body_pos_slice] += 0.05
    losses = compute_losses(inconsistent, pred.detach(), layout, weights, fk_model=fk)
    assert losses["fk_consistency"].item() == pytest.approx(0.05**2)
    assert losses["fk_body_error_m"].item() == pytest.approx(3.0**0.5 * 0.05)
    losses["total"].backward()
    for value in (root_pos, root_quat, joint_pos, body_pos):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_nonzero_fk_weight_requires_fk_model():
    layout, pred = _window(bsz=1, horizon=2)
    with pytest.raises(ValueError, match="fk_model is required"):
        compute_losses(pred, pred, layout, LossWeights(fk_consistency=0.1))
