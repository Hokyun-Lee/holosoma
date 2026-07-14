from __future__ import annotations

import pytest
import torch

from holosoma.motion_gen.features import FeatureLayout, pack_features, quat_normalize
from holosoma.motion_gen.kinematics import G1ForwardKinematics
from holosoma.motion_gen.losses import LossWeights, compute_losses
from holosoma.motion_gen.terrain import ScanGrid


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


def _fk_pose_window(
    *,
    root_z: float = 0.8,
    root_x: float = 0.0,
    horizon: int = 1,
    joint0_values: tuple[float, ...] | None = None,
    dtype: torch.dtype = torch.float64,
):
    """Create leaf root/joint tensors and a matching detached FK body head."""
    layout = FeatureLayout()
    fk = G1ForwardKinematics(dtype=dtype)
    root_pos = torch.zeros(1, horizon, 3, dtype=dtype)
    root_pos[..., 0] = root_x
    root_pos[..., 2] = root_z
    root_pos.requires_grad_()
    root_quat = torch.zeros(1, horizon, 4, dtype=dtype)
    root_quat[..., 0] = 1.0
    root_quat.requires_grad_()
    joint_pos = torch.zeros(1, horizon, layout.num_joints, dtype=dtype)
    if joint0_values is not None:
        if len(joint0_values) != horizon:
            raise ValueError("joint0_values must provide one value per frame")
        joint_pos[0, :, 0] = torch.tensor(joint0_values, dtype=dtype)
    joint_pos.requires_grad_()
    body_pos = fk(root_pos, root_quat, joint_pos).detach()
    features = pack_features(root_pos, root_quat, joint_pos, body_pos)
    return layout, fk, features, root_pos, root_quat, joint_pos


def _symmetric_joint_limits(layout: FeatureLayout, dtype=torch.float64):
    return torch.stack(
        [
            -torch.ones(layout.num_joints, dtype=dtype),
            torch.ones(layout.num_joints, dtype=dtype),
        ]
    )


def _weights_only(**overrides: float) -> LossWeights:
    values = dict.fromkeys(LossWeights.__dataclass_fields__, 0.0)
    values.update(overrides)
    return LossWeights(**values)


class _ControlledCollisionProxyFK(torch.nn.Module):
    """Minimal FK API exposing twelve distinct, root-linked proxy depths."""

    def __init__(self, depths: torch.Tensor):
        super().__init__()
        offsets = torch.zeros(depths.numel(), 3, dtype=depths.dtype)
        offsets[:, 2] = -depths
        self.register_buffer("offsets", offsets)
        self.register_buffer("radii", torch.zeros_like(depths))

    def tracked_body_transforms(self, root_pos, root_quat, joint_pos):
        del joint_pos
        num_bodies = FeatureLayout().num_bodies
        body_pos = root_pos.unsqueeze(-2).expand(*root_pos.shape[:-1], num_bodies, 3)
        body_quat = root_quat.unsqueeze(-2).expand(*root_quat.shape[:-1], num_bodies, 4)
        return body_pos, body_quat

    def lower_body_collision_spheres_from_tracked_transforms(self, body_pos, body_quat):
        del body_quat
        centers = body_pos[..., :1, :] + self.offsets
        return centers, self.radii


def _controlled_proxy_window(depths: torch.Tensor):
    layout = FeatureLayout()
    fk = _ControlledCollisionProxyFK(depths)
    root_pos = torch.zeros(1, 1, 3, dtype=depths.dtype, requires_grad=True)
    root_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=depths.dtype, requires_grad=True)
    joint_pos = torch.zeros(1, 1, layout.num_joints, dtype=depths.dtype, requires_grad=True)
    body_pos = fk.tracked_body_transforms(root_pos, root_quat, joint_pos)[0].detach()
    pred = pack_features(root_pos, root_quat, joint_pos, body_pos)
    return layout, fk, pred, root_pos


def test_losses_finite_and_nonnegative():
    layout, gt = _window()
    pred = gt + 0.1 * torch.randn_like(gt)
    losses = compute_losses(
        pred,
        gt,
        layout,
        LossWeights(),
        contact=torch.ones(2, 25, 2, dtype=torch.bool),
        flat=torch.tensor([True, False]),
    )
    for name, value in losses.items():
        assert torch.isfinite(value), name
        assert value >= 0, name


def test_perfect_prediction_gives_zero():
    layout, gt = _window()
    losses = compute_losses(
        gt.clone(),
        gt,
        layout,
        LossWeights(),
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


def test_joint_limit_loss_is_zero_inside_ranges():
    layout, _, pred, *_ = _fk_pose_window()
    limits = _symmetric_joint_limits(layout)

    losses = compute_losses(
        pred,
        pred.detach(),
        layout,
        LossWeights(joint_limit=1.0),
        joint_limits=limits,
    )

    torch.testing.assert_close(losses["joint_limit"], torch.zeros((), dtype=pred.dtype))
    torch.testing.assert_close(losses["joint_limit_max_violation_rad"], torch.zeros((), dtype=pred.dtype))
    assert losses["joint_limit_frame_rate"].item() == 0.0


def test_joint_limit_margin_penalizes_inside_strict_range():
    layout, _, pred, *_ = _fk_pose_window(joint0_values=(0.998,))
    limits = _symmetric_joint_limits(layout)

    losses = compute_losses(
        pred,
        pred.detach(),
        layout,
        LossWeights(joint_limit=1.0),
        joint_limits=limits,
        joint_limit_margin_rad=0.005,
    )

    assert losses["joint_limit"].item() == pytest.approx(0.003**2)
    # Margin pressure is distinct from strict MJCF limit violation diagnostics.
    assert losses["joint_limit_max_violation_rad"].item() == 0.0
    assert losses["joint_limit_frame_rate"].item() == 0.0


def test_joint_limit_seq_mask_excludes_invalid_frames():
    layout, _, pred, *_ = _fk_pose_window(horizon=2, joint0_values=(0.0, 1.2))
    limits = _symmetric_joint_limits(layout)
    seq_mask = torch.tensor([[True, False]])

    losses = compute_losses(
        pred,
        pred.detach(),
        layout,
        LossWeights(joint_limit=1.0),
        joint_limits=limits,
        seq_mask=seq_mask,
    )

    assert losses["joint_limit"].item() == 0.0
    assert losses["joint_limit_frame_rate"].item() == 0.0
    assert losses["joint_limit_max_violation_rad"].item() == 0.0


def test_joint_limit_loss_has_exact_gradient_on_offending_joint():
    layout, _, pred, _, _, joint_pos = _fk_pose_window(joint0_values=(1.2,))
    limits = _symmetric_joint_limits(layout)

    losses = compute_losses(
        pred,
        pred.detach(),
        layout,
        LossWeights(joint_limit=1.0),
        joint_limits=limits,
    )
    losses["joint_limit"].backward()

    assert joint_pos.grad is not None
    assert joint_pos.grad[..., 0].item() == pytest.approx(0.4)
    assert torch.count_nonzero(joint_pos.grad[..., 1:]) == 0


def test_nonzero_joint_limit_weight_requires_limits():
    layout, _, pred, *_ = _fk_pose_window()
    with pytest.raises(ValueError, match="joint_limits are required"):
        compute_losses(pred, pred.detach(), layout, LossWeights(joint_limit=1.0))


def test_lower_body_proxy_flat_loss_tolerance_and_gradients():
    layout, fk, pred, root_pos, root_quat, joint_pos = _fk_pose_window(root_z=0.78)
    weights = LossWeights(lower_body_terrain_penetration=1.0)
    tolerance = 0.005

    centers, radii = fk.lower_body_collision_spheres(root_pos, root_quat, joint_pos)
    raw_depth = torch.relu(-(centers[..., 2] - radii.view(1, 1, -1)))
    expected = torch.relu(raw_depth - tolerance).square().mean()
    losses = compute_losses(
        pred,
        pred.detach(),
        layout,
        weights,
        flat=torch.tensor([True]),
        fk_model=fk,
        terrain_penetration_tolerance_m=tolerance,
    )

    torch.testing.assert_close(losses["lower_body_terrain_penetration"], expected)
    torch.testing.assert_close(losses["lower_body_max_penetration_m"], raw_depth.max())
    torch.testing.assert_close(
        losses["lower_body_penetration_value_rate_5mm"],
        (raw_depth > 0.005).to(pred.dtype).mean(),
    )
    losses["lower_body_terrain_penetration"].backward()
    for value in (root_pos, root_quat, joint_pos):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
    assert root_pos.grad[..., 2].item() < 0.0
    assert torch.count_nonzero(joint_pos.grad[..., :12]) > 0
    assert torch.count_nonzero(joint_pos.grad[..., 12:]) == 0

    above_tolerance = compute_losses(
        pred.detach(),
        pred.detach(),
        layout,
        weights,
        flat=torch.tensor([True]),
        fk_model=fk,
        terrain_penetration_tolerance_m=float(raw_depth.max()) + 1.0e-4,
    )
    assert above_tolerance["lower_body_terrain_penetration"].item() == 0.0
    # Diagnostic depth remains raw and is intentionally not tolerance-adjusted.
    torch.testing.assert_close(above_tolerance["lower_body_max_penetration_m"], raw_depth.max())


def test_lower_body_proxy_scan_loss_uses_all_12_spheres():
    layout, fk, pred, root_pos, root_quat, joint_pos = _fk_pose_window(root_z=0.98)
    grid = ScanGrid()
    scan_height = 0.2
    terrain_scan = torch.full((1, grid.dim), scan_height, dtype=pred.dtype)

    centers, radii = fk.lower_body_collision_spheres(root_pos, root_quat, joint_pos)
    raw_depth = torch.relu(scan_height - (centers[..., 2] - radii.view(1, 1, -1)))
    losses = compute_losses(
        pred,
        pred.detach(),
        layout,
        LossWeights(lower_body_terrain_penetration=1.0),
        flat=torch.tensor([False]),
        terrain_scan=terrain_scan,
        has_scan=torch.tensor([True]),
        scan_grid=grid,
        fk_model=fk,
    )

    assert raw_depth.shape[-1] == fk.num_lower_body_collision_proxies == 12
    torch.testing.assert_close(losses["lower_body_terrain_penetration"], raw_depth.square().mean())
    torch.testing.assert_close(losses["lower_body_max_penetration_m"], raw_depth.max())


def test_lower_body_proxy_scan_masks_points_outside_grid():
    layout, fk, pred, *_ = _fk_pose_window(root_z=0.78, root_x=10.0)
    grid = ScanGrid()
    terrain_scan = torch.full((1, grid.dim), 1.0, dtype=pred.dtype)

    losses = compute_losses(
        pred,
        pred.detach(),
        layout,
        LossWeights(lower_body_terrain_penetration=1.0),
        flat=torch.tensor([False]),
        terrain_scan=terrain_scan,
        has_scan=torch.tensor([True]),
        scan_grid=grid,
        fk_model=fk,
    )

    assert losses["lower_body_terrain_penetration"].item() == 0.0
    assert losses["lower_body_max_penetration_m"].item() == 0.0
    assert losses["lower_body_penetration_value_rate_5mm"].item() == 0.0


def test_lower_body_terrain_tail_uses_exact_top_fraction_and_gradient():
    depths = torch.arange(0.01, 0.121, 0.01, dtype=torch.float64)
    layout, fk, pred, root_pos = _controlled_proxy_window(depths)
    tolerance = 0.005

    losses = compute_losses(
        pred,
        pred.detach(),
        layout,
        _weights_only(lower_body_terrain_tail=1.0),
        flat=torch.tensor([True]),
        fk_model=fk,
        terrain_penetration_tolerance_m=tolerance,
        terrain_penetration_tail_fraction=0.25,
    )

    # ceil(12 * 0.25) selects the unique 10, 11, and 12 cm proxy depths.
    expected_values = torch.tensor([0.095, 0.105, 0.115], dtype=pred.dtype)
    expected_tail = expected_values.square().mean()
    torch.testing.assert_close(losses["lower_body_terrain_tail_count"], pred.new_tensor(3.0))
    torch.testing.assert_close(losses["lower_body_terrain_tail"], expected_tail)

    losses["lower_body_terrain_tail"].backward()
    assert root_pos.grad is not None
    torch.testing.assert_close(root_pos.grad[..., :2], torch.zeros_like(root_pos.grad[..., :2]))
    # Each selected depth is (constant - root_z - tolerance).
    torch.testing.assert_close(root_pos.grad[..., 2], -2.0 * expected_values.mean().reshape(1, 1))


@pytest.mark.parametrize(
    "fraction",
    [0.0, -0.1, 1.0001, float("nan"), float("inf"), -float("inf")],
)
def test_lower_body_terrain_tail_fraction_validation(fraction):
    layout, pred = _window(bsz=1, horizon=1)

    with pytest.raises(ValueError, match="terrain_penetration_tail_fraction"):
        compute_losses(
            pred,
            pred,
            layout,
            LossWeights(),
            terrain_penetration_tail_fraction=fraction,
        )


def test_lower_body_terrain_tail_is_zero_without_valid_proxy_values():
    depths = torch.arange(0.01, 0.121, 0.01, dtype=torch.float64)
    layout, fk, pred, _ = _controlled_proxy_window(depths)

    losses = compute_losses(
        pred,
        pred.detach(),
        layout,
        _weights_only(lower_body_terrain_tail=2.0),
        flat=torch.tensor([True]),
        seq_mask=torch.tensor([[False]]),
        fk_model=fk,
        terrain_penetration_tail_fraction=0.25,
    )

    assert losses["lower_body_terrain_penetration"].item() == 0.0
    assert losses["lower_body_terrain_tail"].item() == 0.0
    assert losses["lower_body_terrain_tail_count"].item() == 0.0
    assert losses["total"].item() == 0.0


def test_lower_body_mean_and_tail_weights_are_independent():
    depths = torch.arange(0.01, 0.121, 0.01, dtype=torch.float64)
    layout, fk, pred, _ = _controlled_proxy_window(depths)
    common = {
        "flat": torch.tensor([True]),
        "fk_model": fk,
        "terrain_penetration_tail_fraction": 0.25,
    }

    mean_only = compute_losses(
        pred,
        pred.detach(),
        layout,
        _weights_only(lower_body_terrain_penetration=2.0),
        **common,
    )
    tail_only = compute_losses(
        pred,
        pred.detach(),
        layout,
        _weights_only(lower_body_terrain_tail=3.0),
        **common,
    )
    both = compute_losses(
        pred,
        pred.detach(),
        layout,
        _weights_only(lower_body_terrain_penetration=2.0, lower_body_terrain_tail=3.0),
        **common,
    )

    mean_term = mean_only["lower_body_terrain_penetration"]
    tail_term = tail_only["lower_body_terrain_tail"]
    torch.testing.assert_close(mean_only["total"], 2.0 * mean_term)
    torch.testing.assert_close(tail_only["total"], 3.0 * tail_term)
    torch.testing.assert_close(both["lower_body_terrain_penetration"], mean_term)
    torch.testing.assert_close(both["lower_body_terrain_tail"], tail_term)
    torch.testing.assert_close(both["total"], 2.0 * mean_term + 3.0 * tail_term)
