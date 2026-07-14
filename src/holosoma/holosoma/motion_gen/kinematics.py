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

# Exact lower-body collision-marker bodies from the pinned MJCF, in XML order.
# The two ``ankle_intermediate`` spheres are rigid children of the knee links;
# the remaining ten spheres sample the soles as rigid children of the ankle-roll
# links.  These are implementation/model constants, not values from the paper.
G1_LOWER_BODY_COLLISION_PROXY_NAMES = (
    "left_ankle_intermediate_1_link",
    "left_ankle_roll_sphere_3_link",
    "left_ankle_roll_sphere_4_link",
    "left_ankle_roll_sphere_5_link",
    "left_ankle_roll_sphere_1_link",
    "left_ankle_roll_sphere_2_link",
    "right_ankle_intermediate_1_link",
    "right_ankle_roll_sphere_3_link",
    "right_ankle_roll_sphere_4_link",
    "right_ankle_roll_sphere_5_link",
    "right_ankle_roll_sphere_1_link",
    "right_ankle_roll_sphere_2_link",
)

# Node zero is the pelvis; node j + 1 is joint j's body.  Parent indices are
# topologically ordered so the forward pass needs no dynamic graph traversal.
_PARENT_NODE_INDICES = (
    0,
    1,
    2,
    3,
    4,
    5,
    0,
    7,
    8,
    9,
    10,
    11,
    0,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    15,
    23,
    24,
    25,
    26,
    27,
    28,
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
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

# Indices into [pelvis, joint-body 0, ..., joint-body 28].
_TRACKED_BODY_NODE_INDICES = (0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29)

# Indices into the 14 tracked bodies.  Their order matches
# G1_LOWER_BODY_COLLISION_PROXY_NAMES above.
_LOWER_BODY_PROXY_PARENT_TRACKED_INDICES = (2, 3, 3, 3, 3, 3, 5, 6, 6, 6, 6, 6)
_LOWER_BODY_PROXY_LOCAL_POSITIONS = (
    (0.0, 0.0, -0.28),
    (0.12, 0.03, -0.03),
    (0.12, -0.03, -0.03),
    (0.14, 0.0, -0.03),
    (-0.05, 0.025, -0.03),
    (-0.05, -0.025, -0.03),
    (0.0, 0.0, -0.28),
    (0.12, 0.03, -0.03),
    (0.12, -0.03, -0.03),
    (0.14, 0.0, -0.03),
    (-0.05, 0.025, -0.03),
    (-0.05, -0.025, -0.03),
)
_LOWER_BODY_PROXY_RADII = (0.01, 0.005, 0.005, 0.005, 0.005, 0.005) * 2


def validate_source_mjcf(path: str | Path) -> None:
    """Fail if ``path`` is not the exact MJCF used to define these buffers."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_MJCF_SHA256:
        raise ValueError(f"Unsupported G1 MJCF SHA-256 {digest}; expected {SOURCE_MJCF_SHA256} for {path}.")


class G1ForwardKinematics(nn.Module):
    """Batched differentiable FK for the fixed G1 29-DoF dataset model."""

    num_joints = 29
    num_bodies = 14
    num_lower_body_collision_proxies = len(G1_LOWER_BODY_COLLISION_PROXY_NAMES)

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
            raise ValueError(f"G1 FK requires the exact 29-joint MJCF/dataset order; got {self.joint_names}.")
        if self.body_names != G1_29DOF_BODY_NAMES:
            raise ValueError(f"G1 FK requires the exact 14-body motion-generator order; got {self.body_names}.")
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
        self.register_buffer(
            "lower_body_proxy_parent_tracked_indices",
            torch.tensor(_LOWER_BODY_PROXY_PARENT_TRACKED_INDICES, device=device),
        )
        self.register_buffer(
            "lower_body_proxy_local_positions",
            torch.tensor(_LOWER_BODY_PROXY_LOCAL_POSITIONS, **tensor_kwargs),
        )
        self.register_buffer(
            "lower_body_proxy_radii",
            torch.tensor(_LOWER_BODY_PROXY_RADII, **tensor_kwargs),
        )
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
            "lower_body_proxy_parent_tracked_indices": (self.num_lower_body_collision_proxies,),
            "lower_body_proxy_local_positions": (
                self.num_lower_body_collision_proxies,
                3,
            ),
            "lower_body_proxy_radii": (self.num_lower_body_collision_proxies,),
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
        if not bool(
            torch.all(
                (self.lower_body_proxy_parent_tracked_indices >= 0)
                & (self.lower_body_proxy_parent_tracked_indices < self.num_bodies)
            )
        ):
            raise RuntimeError("G1 collision-proxy parent indices must select tracked bodies.")
        if not bool(torch.all(self.lower_body_proxy_radii > 0.0)):
            raise RuntimeError("G1 collision-proxy sphere radii must be positive.")

    def forward(
        self,
        root_position: torch.Tensor,
        root_quaternion_wxyz: torch.Tensor,
        joint_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return tracked world body positions with shape ``(..., 14, 3)``."""
        positions, _ = self.tracked_body_transforms(
            root_position,
            root_quaternion_wxyz,
            joint_positions,
        )
        return positions

    def tracked_body_transforms(
        self,
        root_position: torch.Tensor,
        root_quaternion_wxyz: torch.Tensor,
        joint_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tracked body positions and quaternions.

        Returns tensors with shapes ``(..., 14, 3)`` and ``(..., 14, 4)``.
        Quaternions use normalized ``wxyz`` order.  Exposing both transforms
        lets terrain losses reuse one FK traversal for body consistency and
        rigid collision-marker placement.
        """
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

        selected_positions = [node_positions[index] for index in _TRACKED_BODY_NODE_INDICES]
        selected_quaternions = [node_quaternions[index] for index in _TRACKED_BODY_NODE_INDICES]
        return torch.stack(selected_positions, dim=-2), torch.stack(selected_quaternions, dim=-2)

    def lower_body_collision_spheres_from_tracked_transforms(
        self,
        tracked_positions: torch.Tensor,
        tracked_quaternions_wxyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Place the exact 12 lower-body MJCF marker spheres.

        Args:
            tracked_positions: ``(..., 14, 3)`` from
                :meth:`tracked_body_transforms`.
            tracked_quaternions_wxyz: ``(..., 14, 4)`` from the same call.
        Returns:
            ``(centers, radii)`` where centers have shape ``(..., 12, 3)``
            and radii is the fixed ``(12,)`` non-trainable buffer.  Proxy order
            is :data:`G1_LOWER_BODY_COLLISION_PROXY_NAMES`.
        """
        expected_position_shape = (*tracked_positions.shape[:-2], self.num_bodies, 3)
        expected_quaternion_shape = (*tracked_positions.shape[:-2], self.num_bodies, 4)
        if tuple(tracked_positions.shape) != expected_position_shape:
            raise ValueError(f"tracked_positions must have shape (..., 14, 3), got {tuple(tracked_positions.shape)}.")
        if tuple(tracked_quaternions_wxyz.shape) != expected_quaternion_shape:
            raise ValueError(
                "tracked_quaternions_wxyz must have shape (..., 14, 4) matching "
                f"tracked_positions, got {tuple(tracked_quaternions_wxyz.shape)}."
            )
        for name, value in {
            "tracked_positions": tracked_positions,
            "tracked_quaternions_wxyz": tracked_quaternions_wxyz,
        }.items():
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point, got {value.dtype}.")
            if value.device != self.local_positions.device or value.dtype != self.local_positions.dtype:
                raise TypeError(
                    f"{name} must match FK buffers on "
                    f"{self.local_positions.device}/{self.local_positions.dtype}, "
                    f"got {value.device}/{value.dtype}."
                )

        parent_indices = self.lower_body_proxy_parent_tracked_indices
        parent_positions = tracked_positions[..., parent_indices, :]
        parent_quaternions = tracked_quaternions_wxyz[..., parent_indices, :]
        local_positions = self.lower_body_proxy_local_positions.expand_as(parent_positions)
        centers = parent_positions + quat_rotate(parent_quaternions, local_positions)
        return centers, self.lower_body_proxy_radii

    def lower_body_collision_spheres(
        self,
        root_position: torch.Tensor,
        root_quaternion_wxyz: torch.Tensor,
        joint_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run FK and return the exact lower-body marker centers/radii."""
        tracked_positions, tracked_quaternions = self.tracked_body_transforms(
            root_position,
            root_quaternion_wxyz,
            joint_positions,
        )
        return self.lower_body_collision_spheres_from_tracked_transforms(
            tracked_positions,
            tracked_quaternions,
        )

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
            raise ValueError(f"root_quaternion_wxyz must have shape (..., 4), got {tuple(root_quaternion_wxyz.shape)}.")
        if joint_positions.shape[-1:] != (self.num_joints,):
            raise ValueError(
                f"joint_positions must have shape (..., {self.num_joints}), got {tuple(joint_positions.shape)}."
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
    "G1_LOWER_BODY_COLLISION_PROXY_NAMES",
    "SOURCE_MJCF_SHA256",
    "G1ForwardKinematics",
    "validate_source_mjcf",
]
