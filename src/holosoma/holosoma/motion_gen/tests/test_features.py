import torch

from holosoma.motion_gen.features import (
    FeatureLayout,
    canonicalize_window,
    compute_target_heading,
    decanonicalize_window,
    pack_features,
    quat_enforce_continuity,
    quat_from_yaw,
    quat_mul,
    quat_normalize,
    quat_rotate,
    quat_yaw,
    unpack_features,
)


def _random_window(bsz=3, T=27, seed=0):
    layout = FeatureLayout()
    g = torch.Generator().manual_seed(seed)
    root_pos = torch.randn(bsz, T, 3, generator=g)
    root_quat = quat_enforce_continuity(quat_normalize(torch.randn(bsz, T, 4, generator=g)))
    joint_pos = torch.randn(bsz, T, layout.num_joints, generator=g)
    body_pos = torch.randn(bsz, T, layout.num_bodies, 3, generator=g)
    return layout, pack_features(root_pos, root_quat, joint_pos, body_pos)


def test_layout_dims():
    layout = FeatureLayout()
    assert layout.num_joints == 29
    assert layout.num_bodies == 14
    assert layout.dim == 3 + 4 + 29 + 42
    meta = layout.to_metadata()
    assert FeatureLayout.from_metadata(meta).dim == layout.dim


def test_pack_unpack_roundtrip():
    layout, x = _random_window()
    parts = unpack_features(x, layout)
    x2 = pack_features(parts["root_pos"], parts["root_quat"], parts["joint_pos"], parts["body_pos"])
    assert torch.allclose(x, x2)


def test_quat_yaw_of_yaw_quat():
    yaw = torch.tensor([0.3, -1.2, 2.5])
    assert torch.allclose(quat_yaw(quat_from_yaw(yaw)), yaw, atol=1e-6)


def test_quat_rotate_matches_mul():
    g = torch.Generator().manual_seed(1)
    q = quat_normalize(torch.randn(5, 4, generator=g))
    v = torch.randn(5, 3, generator=g)
    qv = torch.cat([torch.zeros(5, 1), v], dim=-1)
    expected = quat_mul(quat_mul(q, qv), torch.cat([q[:, :1], -q[:, 1:]], dim=-1))[:, 1:]
    assert torch.allclose(quat_rotate(q, v), expected, atol=1e-5)


def test_canonicalize_roundtrip():
    layout, x = _random_window()
    canon, transform = canonicalize_window(x, layout, anchor_index=1)
    x2 = decanonicalize_window(canon, layout, transform)
    # canonicalization may flip the whole window's quaternion sign (q == -q)
    q, q2 = x[..., layout.root_quat_slice], x2[..., layout.root_quat_slice]
    sign = torch.sign((q * q2).sum(dim=-1, keepdim=True))
    assert torch.allclose(q, sign * q2, atol=1e-4)
    for sl in (layout.root_pos_slice, layout.joint_pos_slice, layout.body_pos_slice):
        assert torch.allclose(x[..., sl], x2[..., sl], atol=1e-4)


def test_canonical_anchor_at_origin_and_zero_yaw():
    layout, x = _random_window()
    canon, _ = canonicalize_window(x, layout, anchor_index=1)
    parts = unpack_features(canon, layout)
    assert torch.allclose(parts["root_pos"][:, 1, :2], torch.zeros(3, 2), atol=1e-5)
    assert torch.allclose(quat_yaw(parts["root_quat"][:, 1]), torch.zeros(3), atol=1e-5)
    # z stays absolute
    assert torch.allclose(parts["root_pos"][:, 1, 2], unpack_features(x, layout)["root_pos"][:, 1, 2])


def test_heading_unit_norm_and_fallback():
    layout, x = _random_window()
    canon, _ = canonicalize_window(x, layout, anchor_index=1)
    h = compute_target_heading(canon, layout, anchor_index=1)
    assert torch.allclose(h.norm(dim=-1), torch.ones(3), atol=1e-5)
    # static window falls back to (1, 0)
    static = canon.clone()
    static[..., :3] = 0.0
    h2 = compute_target_heading(static, layout, anchor_index=1)
    assert torch.allclose(h2, torch.tensor([[1.0, 0.0]]).expand(3, 2))


def test_canonical_quat_sign_stable_across_yaw_branch_cut():
    """Windows with anchor yaw just above/below +-pi must give same-sign targets."""
    layout = FeatureLayout()
    T = 4
    for yaw0 in (torch.pi - 1e-3, -torch.pi + 1e-3):
        yaw = torch.full((1, T), yaw0)
        x = pack_features(
            torch.zeros(1, T, 3),
            quat_from_yaw(yaw),
            torch.zeros(1, T, layout.num_joints),
            torch.zeros(1, T, layout.num_bodies, 3),
        )
        canon, _ = canonicalize_window(x, layout, anchor_index=1)
        w = unpack_features(canon, layout)["root_quat"][..., 0]
        assert (w > 0.99).all(), f"anchor yaw {yaw0}: canonical quat w={w}"


def test_quat_continuity():
    q = quat_normalize(torch.randn(1, 10, 4))
    q[:, 5:] = -q[:, 5:]
    qc = quat_enforce_continuity(q)
    dots = (qc[:, 1:] * qc[:, :-1]).sum(-1)
    assert (dots >= -1e-6).all()
