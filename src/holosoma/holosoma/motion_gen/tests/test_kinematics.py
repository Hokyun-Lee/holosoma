"""Ground-truth and autograd tests for G1 differentiable FK."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from holosoma.motion_gen.features import DEFAULT_BODY_NAMES, DEFAULT_JOINT_NAMES
from holosoma.motion_gen.kinematics import (
    G1_29DOF_BODY_NAMES,
    G1_29DOF_JOINT_BODY_NAMES,
    G1_29DOF_JOINT_NAMES,
    G1ForwardKinematics,
    validate_source_mjcf,
)


def _source_mjcf() -> Path:
    repo_root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "src/holosoma_retargeting").is_dir()
    )
    return repo_root / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"


def test_fixed_model_mapping_and_non_trainable_buffers(tmp_path: Path) -> None:
    source = _source_mjcf()
    validate_source_mjcf(source)
    fk = G1ForwardKinematics(source_mjcf_path=source)

    assert fk.joint_names == tuple(DEFAULT_JOINT_NAMES) == G1_29DOF_JOINT_NAMES
    assert fk.body_names == tuple(DEFAULT_BODY_NAMES) == G1_29DOF_BODY_NAMES
    assert len(G1_29DOF_JOINT_BODY_NAMES) == 29
    assert not list(fk.parameters())
    assert all(not value.requires_grad for value in fk.buffers())
    assert set(fk.state_dict()) == {
        "parent_node_indices",
        "local_positions",
        "local_quaternions_wxyz",
        "joint_axes",
        "joint_pivots",
        "tracked_body_node_indices",
    }

    changed = tmp_path / "changed.xml"
    changed.write_bytes(source.read_bytes() + b"\n<!-- changed -->\n")
    with pytest.raises(ValueError, match="Unsupported G1 MJCF SHA-256"):
        validate_source_mjcf(changed)


def test_mapping_and_model_shape_constraints_fail_fast() -> None:
    wrong_joints = list(G1_29DOF_JOINT_NAMES)
    wrong_joints[0], wrong_joints[1] = wrong_joints[1], wrong_joints[0]
    with pytest.raises(ValueError, match="exact 29-joint"):
        G1ForwardKinematics(joint_names=wrong_joints)

    with pytest.raises(ValueError, match="exact 14-body"):
        G1ForwardKinematics(body_names=G1_29DOF_BODY_NAMES[:-1])

    fk = G1ForwardKinematics()
    with pytest.raises(ValueError, match=r"joint_positions must have shape \(\.\.\., 29\)"):
        fk(torch.zeros(2, 3), torch.zeros(2, 4), torch.zeros(2, 28))
    with pytest.raises(ValueError, match="share batch dimensions"):
        fk(torch.zeros(2, 3), torch.zeros(1, 4), torch.zeros(2, 29))
    with pytest.raises(TypeError, match="must be floating point"):
        fk(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 4), torch.zeros(2, 29))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_shape_dtype_and_quaternion_normalization(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(7)
    fk = G1ForwardKinematics(dtype=dtype)
    root_position = torch.randn(2, 3, 3, generator=generator, dtype=dtype)
    root_quaternion = torch.randn(2, 3, 4, generator=generator, dtype=dtype)
    joint_positions = torch.randn(2, 3, 29, generator=generator, dtype=dtype)

    actual = fk(root_position, root_quaternion, joint_positions)
    scaled_quaternion = fk(root_position, 3.7 * root_quaternion, joint_positions)

    assert actual.shape == (2, 3, 14, 3)
    assert actual.dtype == dtype
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, scaled_quaternion)


def test_gradcheck_and_gradients_are_finite() -> None:
    fk = G1ForwardKinematics(dtype=torch.float64)
    root_position = torch.tensor([[0.1, -0.2, 0.8]], dtype=torch.float64, requires_grad=True)
    root_quaternion = torch.tensor([[0.9, 0.2, -0.1, 0.3]], dtype=torch.float64, requires_grad=True)
    joint_positions = torch.linspace(-0.3, 0.4, 29, dtype=torch.float64).unsqueeze(0).requires_grad_()

    assert torch.autograd.gradcheck(
        fk,
        (root_position, root_quaternion, joint_positions),
        eps=1.0e-6,
        atol=2.0e-5,
        rtol=1.0e-3,
    )
    fk(root_position, root_quaternion, joint_positions).square().mean().backward()
    for value in (root_position, root_quaternion, joint_positions):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
    assert torch.count_nonzero(joint_positions.grad) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_device_and_backward() -> None:
    device = torch.device("cuda")
    fk = G1ForwardKinematics(device=device)
    root_position = torch.randn(4, 3, device=device, requires_grad=True)
    root_quaternion = torch.randn(4, 4, device=device, requires_grad=True)
    joint_positions = torch.randn(4, 29, device=device, requires_grad=True)

    result = fk(root_position, root_quaternion, joint_positions)
    assert result.device == root_position.device
    result.sum().backward()
    assert joint_positions.grad is not None
    assert torch.isfinite(joint_positions.grad).all()


def test_against_mujoco_ground_truth() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(_source_mjcf()))
    data = mujoco.MjData(model)

    joint_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, model.njnt)
    )
    joint_body_names = tuple(
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.jnt_bodyid[index]),
        )
        for index in range(1, model.njnt)
    )
    assert joint_names == G1_29DOF_JOINT_NAMES
    assert joint_body_names == G1_29DOF_JOINT_BODY_NAMES
    assert np.all(model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE)
    np.testing.assert_array_equal(model.jnt_pos[1:], np.zeros((29, 3)))

    rng = np.random.default_rng(123)
    batch_size = 32
    root_position = rng.normal(size=(batch_size, 3))
    root_quaternion = rng.normal(size=(batch_size, 4))
    root_quaternion /= np.linalg.norm(root_quaternion, axis=-1, keepdims=True)
    joint_positions = rng.uniform(-0.7, 0.7, size=(batch_size, 29))
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in G1_29DOF_BODY_NAMES
    ]
    expected = []
    for batch_index in range(batch_size):
        data.qpos[:3] = root_position[batch_index]
        data.qpos[3:7] = root_quaternion[batch_index]
        data.qpos[7:] = joint_positions[batch_index]
        mujoco.mj_forward(model, data)
        expected.append(data.xpos[body_ids].copy())

    fk = G1ForwardKinematics(dtype=torch.float64, source_mjcf_path=_source_mjcf())
    actual = fk(
        torch.from_numpy(root_position),
        torch.from_numpy(root_quaternion),
        torch.from_numpy(joint_positions),
    ).detach().numpy()
    np.testing.assert_allclose(actual, np.stack(expected), atol=5.0e-12, rtol=5.0e-12)
