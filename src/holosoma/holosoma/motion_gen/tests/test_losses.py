import torch

from holosoma.motion_gen.features import FeatureLayout, pack_features, quat_normalize
from holosoma.motion_gen.losses import LossWeights, compute_losses


def _window(bsz=2, H=25, seed=0):
    layout = FeatureLayout()
    g = torch.Generator().manual_seed(seed)
    x = pack_features(
        torch.randn(bsz, H, 3, generator=g),
        quat_normalize(torch.randn(bsz, H, 4, generator=g)),
        torch.randn(bsz, H, layout.num_joints, generator=g),
        torch.randn(bsz, H, layout.num_bodies, 3, generator=g),
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
