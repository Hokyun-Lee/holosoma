"""Differentiable forward kinematics for the retargeting G1 29-DoF MJCF.

The constants in this module are pinned to
``holosoma_retargeting/models/g1/g1_29dof.xml`` (SHA-256 stored below).  The
model currently has one free root followed by 29 hinge joints, every hinge
joint is attached at the origin of its body, and joint order matches the WBT
dataset.  These are model constraints, not values reported by the paper.

Inputs and outputs use the motion-generator conventions: arbitrary batch
dimensions, root quaternion in ``wxyz`` order, and the 14 tracked body
positions in :data:`holosoma.motion_gen.features.DEFAULT_BODY_NAMES` order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from holosoma.motion_gen.features import (
    DEFAULT_BODY_NAMES,
    DEFAULT_JOINT_NAMES,
    quat_mul,
    quat_normalize,
    quat_rotate,
)

SOURCE_MJCF_SHA256 = "8c586e4747da85804180fe44d8692e0fd8231356728b6327e256dca498087a78"

G1_29DOF_JOINT_NAMES = tuple(DEFAULT_JOINT_NAMES)
G1_29DOF_BODY_NAMES = tuple(DEFAULT_BODY_NAMES)
G1_29DOF_JOINT_BODY_NAMES = (
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

# Node zero is the pelvis; node j + 1 is joint j's body.  Parent indices are
# topologically ordered so the forward pass needs no dynamic graph traversal.
_PARENT_NODE_INDICES = (
    0, 1, 2, 3, 4, 5,
    0, 7, 8, 9, 10, 11,
    0, 13, 14,
    15, 16, 17, 18, 19, 20, 21,
    15, 23, 24, 25, 26, 27, 28,
)

_LOCAL_POSITIONS = (
    (0.0, 0.064452, -0.1027),
    (0.0, 0.052, -0.030465),
    (0.025001, 0.0, -0.12412),
    (-0.078273, 0.0021489, -0.17734),
    (0.0, -9.4445e-05, -0.30001),
    (0.0, 0.0, -0.017558),
    (0.0, -0.064452, -0.1027),
    (0.0, -0.052, -0.030465),
    (0.025001, 0.0, -0.12412),
    (-0.078273, -0.0021489, -0.17734),
    (0.0, 9.4445e-05, -0.30001),
    (0.0, 0.0, -0.017558),
    (0.0, 0.0, 0.0),
    (-0.0039635, 0.0, 0.035),
    (0.0, 0.0, 0.019),
    (0.0039563, 0.10022, 0.23778),
    (0.0, 0.038, -0.013831),
    (0.0, 0.00624, -0.1032),
    (0.015783, 0.0, -0.080518),
    (0.1, 0.00188791, -0.01),
    (0.038, 0.0, 0.0),
    (0.046, 0.0, 0.0),
    (0.0039563, -0.10021, 0.23778),
    (0.0, -0.038, -0.013831),
    (0.0, -0.00624, -0.1032),
    (0.015783, 0.0, -0.080518),
    (0.1, -0.00188791, -0.01),
    (0.038, 0.0, 0.0),
    (0.046, 0.0, 0.0),
)

# MuJoCo normalizes body quaternions while compiling the XML.  Store those
# normalized values to match ``data.xpos`` down to floating-point precision.
_LOCAL_QUATERNIONS_WXYZ = (
    (1.0, 0.0, 0.0, 0.0),
    (0.996178685660368, 0.0, -0.08733857244071258, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.996178685660368, 0.0, 0.08733857244071258, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.996178685660368, 0.0, -0.08733857244071258, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.996178685660368, 0.0, 0.08733857244071258, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.9902641396131312, 0.13920101962536, 1.387220195578278e-05, -9.868681391343435e-05),
    (0.9902682191424201, -0.1391720307982171, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.9902641396131312, -0.13920101962536, 1.387220195578278e-05, 9.868681391343435e-05),
    (0.9902682191424201, 0.1391720307982171, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
)

_JOINT_AXES = (
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

# Indices into [pelvis, joint-body 0, ..., joint-body 28].
_TRACKED_BODY_NODE_INDICES = (0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29)


def validate_source_mjcf(path: str | Path) -> None:
    """Fail if ``path`` is not the exact MJCF used to define these buffers."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_MJCF_SHA256:
        raise ValueError(
            f"Unsupported G1 MJCF SHA-256 {digest}; expected {SOURCE_MJCF_SHA256} for {path}."
        )


class G1ForwardKinematics(nn.Module):
    """Batched differentiable FK for the fixed G1 29-DoF dataset model."""

    num_joints = 29
    num_bodies = 14

    def __init__(
        self,
        joint_names: Sequence[str] = G1_29DOF_JOINT_NAMES,
        body_names: Sequence[str] = G1_29DOF_BODY_NAMES,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        source_mjcf_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.joint_names = tuple(joint_names)
        self.body_names = tuple(body_names)
        if self.joint_names != G1_29DOF_JOINT_NAMES:
            raise ValueError(
                "G1 FK requires the exact 29-joint MJCF/dataset order; "
                f"got {self.joint_names}."
            )
        if self.body_names != G1_29DOF_BODY_NAMES:
            raise ValueError(
                "G1 FK requires the exact 14-body motion-generator order; "
                f"got {self.body_names}."
            )
        if source_mjcf_path is not None:
            validate_source_mjcf(source_mjcf_path)

        if dtype is None:
            dtype = torch.get_default_dtype()
        if not dtype.is_floating_point:
            raise TypeError(f"FK buffer dtype must be floating point, got {dtype}.")
        tensor_kwargs = {"device": device, "dtype": dtype}
        self.register_buffer("parent_node_indices", torch.tensor(_PARENT_NODE_INDICES, device=device))
        self.register_buffer("local_positions", torch.tensor(_LOCAL_POSITIONS, **tensor_kwargs))
        self.register_buffer("local_quaternions_wxyz", torch.tensor(_LOCAL_QUATERNIONS_WXYZ, **tensor_kwargs))
        self.register_buffer("joint_axes", torch.tensor(_JOINT_AXES, **tensor_kwargs))
        self.register_buffer("joint_pivots", torch.zeros(self.num_joints, 3, **tensor_kwargs))
        self.register_buffer("tracked_body_node_indices", torch.tensor(_TRACKED_BODY_NODE_INDICES, device=device))
        self._validate_fixed_model()

    def _validate_fixed_model(self) -> None:
        """Validate constants whose violation would require a different FK implementation."""
        expected_nodes = torch.arange(1, self.num_joints + 1, device=self.parent_node_indices.device)
        if self.parent_node_indices.shape != (self.num_joints,) or not bool(
            torch.all((self.parent_node_indices >= 0) & (self.parent_node_indices < expected_nodes))
        ):
            raise RuntimeError("G1 FK parent indices must be topologically ordered.")
        expected_shapes = {
            "local_positions": (self.num_joints, 3),
            "local_quaternions_wxyz": (self.num_joints, 4),
            "joint_axes": (self.num_joints, 3),
            "joint_pivots": (self.num_joints, 3),
            "tracked_body_node_indices": (self.num_bodies,),
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise RuntimeError(f"G1 FK {name} must have shape {shape}.")
        one = torch.ones(self.num_joints, device=self.local_positions.device, dtype=self.local_positions.dtype)
        if not torch.allclose(torch.linalg.vector_norm(self.local_quaternions_wxyz, dim=-1), one):
            raise RuntimeError("G1 FK local body quaternions must be normalized.")
        if not torch.allclose(torch.linalg.vector_norm(self.joint_axes, dim=-1), one):
            raise RuntimeError("G1 FK supports only normalized hinge axes.")
        if bool(torch.count_nonzero(self.joint_pivots)):
            raise RuntimeError("G1 FK currently supports only zero-position MJCF hinge joints.")

    def forward(
        self,
        root_position: torch.Tensor,
        root_quaternion_wxyz: torch.Tensor,
        joint_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return tracked world body positions with shape ``(..., 14, 3)``."""
        self._validate_inputs(root_position, root_quaternion_wxyz, joint_positions)
        root_quaternion_wxyz = quat_normalize(root_quaternion_wxyz)

        node_positions = [root_position]
        node_quaternions = [root_quaternion_wxyz]
        half_angles = 0.5 * joint_positions
        for joint_index, parent_index in enumerate(_PARENT_NODE_INDICES):
            parent_position = node_positions[parent_index]
            parent_quaternion = node_quaternions[parent_index]
            local_position = self.local_positions[joint_index].expand_as(root_position)
            world_position = parent_position + quat_rotate(parent_quaternion, local_position)

            sin_half = torch.sin(half_angles[..., joint_index]).unsqueeze(-1)
            joint_quaternion = torch.cat(
                [
                    torch.cos(half_angles[..., joint_index]).unsqueeze(-1),
                    self.joint_axes[joint_index] * sin_half,
                ],
                dim=-1,
            )
            local_quaternion = self.local_quaternions_wxyz[joint_index].expand_as(root_quaternion_wxyz)
            world_quaternion = quat_mul(parent_quaternion, quat_mul(local_quaternion, joint_quaternion))
            node_positions.append(world_position)
            node_quaternions.append(world_quaternion)

        selected = [node_positions[index] for index in _TRACKED_BODY_NODE_INDICES]
        return torch.stack(selected, dim=-2)

    def _validate_inputs(
        self,
        root_position: torch.Tensor,
        root_quaternion_wxyz: torch.Tensor,
        joint_positions: torch.Tensor,
    ) -> None:
        tensors = {
            "root_position": root_position,
            "root_quaternion_wxyz": root_quaternion_wxyz,
            "joint_positions": joint_positions,
        }
        if root_position.shape[-1:] != (3,):
            raise ValueError(f"root_position must have shape (..., 3), got {tuple(root_position.shape)}.")
        if root_quaternion_wxyz.shape[-1:] != (4,):
            raise ValueError(
                "root_quaternion_wxyz must have shape (..., 4), "
                f"got {tuple(root_quaternion_wxyz.shape)}."
            )
        if joint_positions.shape[-1:] != (self.num_joints,):
            raise ValueError(
                f"joint_positions must have shape (..., {self.num_joints}), "
                f"got {tuple(joint_positions.shape)}."
            )
        leading_shape = root_position.shape[:-1]
        if root_quaternion_wxyz.shape[:-1] != leading_shape or joint_positions.shape[:-1] != leading_shape:
            raise ValueError("root position, root quaternion, and joint positions must share batch dimensions.")
        for name, value in tensors.items():
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point, got {value.dtype}.")
            if value.device != self.local_positions.device or value.dtype != self.local_positions.dtype:
                raise TypeError(
                    f"{name} must match FK buffers on {self.local_positions.device}/{self.local_positions.dtype}, "
                    f"got {value.device}/{value.dtype}."
                )


__all__ = [
    "G1_29DOF_BODY_NAMES",
    "G1_29DOF_JOINT_BODY_NAMES",
    "G1_29DOF_JOINT_NAMES",
    "SOURCE_MJCF_SHA256",
    "G1ForwardKinematics",
    "validate_source_mjcf",
]
