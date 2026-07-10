"""Motion representation for the diffusion motion generator.

Per-frame feature vector (all float32, in a heading-normalized canonical frame):

    [ root_pos (3) | root_quat wxyz (4) | joint_pos (J) | body_pos (B * 3) ]

The paper (arXiv:2604.17335, Sec. III-B2) uses root position (R^3), root
orientation (R^4), joint positions (R^23) and body link positions (R^{23x3})
for a 23-DoF G1. The HoloSoma G1 model is the 29-DoF variant, so J=29 here,
and body positions use the 14 bodies tracked by the HoloSoma WBT task
(implementation choice; documented in docs/motion_generator_implementation_notes.md).

Conventions:
    - World up axis: +Z, quaternions are wxyz (MuJoCo convention, matches the
      HoloSoma ``*_mj.npz`` motion files).
    - Canonical frame: the window is translated so the anchor frame (the last
      past/conditioning frame) root is at xy=(0,0) (z stays absolute), and
      rotated about z so the anchor root heading (yaw) is zero.
    - Target heading: unit xy vector from the anchor root position to the last
      future-frame root position, expressed in the canonical frame. Falls back
      to (1, 0) when the displacement is below ``min_disp`` (the paper states
      the heading is computed "from base pose difference" without details).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Bodies tracked by the HoloSoma G1 WBT task (config_values/wbt/g1/command.py).
# Reused here so generated body positions line up with future tracker rewards.
DEFAULT_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

# G1 29-DoF joint order used by holosoma_retargeting (MuJoCo model order).
DEFAULT_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Kinematic-chain pairs among the tracked bodies, used for the bone-length
# consistency loss (surrogate for an FK-consistency loss; no differentiable FK).
DEFAULT_BONE_PAIRS = [
    ("pelvis", "torso_link"),
    ("pelvis", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("torso_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]

FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


@dataclass(frozen=True)
class FeatureLayout:
    """Slice layout of the per-frame feature vector."""

    joint_names: tuple[str, ...] = tuple(DEFAULT_JOINT_NAMES)
    body_names: tuple[str, ...] = tuple(DEFAULT_BODY_NAMES)
    bone_pairs: tuple[tuple[str, str], ...] = field(
        default=tuple(DEFAULT_BONE_PAIRS),
    )

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)

    @property
    def dim(self) -> int:
        return 3 + 4 + self.num_joints + 3 * self.num_bodies

    @property
    def root_pos_slice(self) -> slice:
        return slice(0, 3)

    @property
    def root_quat_slice(self) -> slice:
        return slice(3, 7)

    @property
    def joint_pos_slice(self) -> slice:
        return slice(7, 7 + self.num_joints)

    @property
    def body_pos_slice(self) -> slice:
        return slice(7 + self.num_joints, self.dim)

    def bone_pair_indices(self) -> list[tuple[int, int]]:
        name_to_idx = {n: i for i, n in enumerate(self.body_names)}
        return [(name_to_idx[a], name_to_idx[b]) for a, b in self.bone_pairs]

    def foot_body_indices(self) -> list[int]:
        return [self.body_names.index(n) for n in FOOT_BODY_NAMES if n in self.body_names]

    def to_metadata(self) -> dict:
        return {
            "joint_names": list(self.joint_names),
            "body_names": list(self.body_names),
            "bone_pairs": [list(p) for p in self.bone_pairs],
            "dim": self.dim,
            "quat_order": "wxyz",
            "up_axis": "z",
            "frame": "heading_normalized_anchor",
        }

    @staticmethod
    def from_metadata(meta: dict) -> "FeatureLayout":
        return FeatureLayout(
            joint_names=tuple(meta["joint_names"]),
            body_names=tuple(meta["body_names"]),
            bone_pairs=tuple(tuple(p) for p in meta["bone_pairs"]),
        )


# ---------------------------------------------------------------------------
# Quaternion utilities (wxyz, Hamilton convention, torch, batched)
# ---------------------------------------------------------------------------


def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by quaternions q (..., 4)."""
    qw = q[..., :1]
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def quat_yaw(q: torch.Tensor) -> torch.Tensor:
    """Yaw (rotation about world z) of quaternions q (..., 4) wxyz."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    zeros = torch.zeros_like(yaw)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def quat_angle(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Geodesic angle (rad) between two quaternions, sign-invariant."""
    dot = (quat_normalize(a) * quat_normalize(b)).sum(dim=-1).abs().clamp(max=1.0 - eps)
    return 2.0 * torch.acos(dot)


def quat_enforce_continuity(q: torch.Tensor) -> torch.Tensor:
    """Flip quaternion signs along the time axis (dim=-2) for continuity.

    The first frame is flipped to w >= 0; every following frame is flipped if
    its dot product with the previous (already fixed) frame is negative.
    Sign flips represent the same rotation, but element-wise operations
    (interpolation, diffusion noise) require a continuous sequence.
    """
    q = q.clone()
    first = q[..., 0, :]
    first = torch.where(first[..., :1] < 0, -first, first)
    q[..., 0, :] = first
    for t in range(1, q.shape[-2]):
        dot = (q[..., t, :] * q[..., t - 1, :]).sum(dim=-1, keepdim=True)
        q[..., t, :] = torch.where(dot < 0, -q[..., t, :], q[..., t, :])
    return q


def yaw_rotate_xy(yaw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) about z by yaw (broadcast over v's leading dims)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    x = c * v[..., 0] - s * v[..., 1]
    y = s * v[..., 0] + c * v[..., 1]
    return torch.stack([x, y, v[..., 2]], dim=-1)


# ---------------------------------------------------------------------------
# Feature packing / canonicalization
# ---------------------------------------------------------------------------


def pack_features(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    body_pos: torch.Tensor,
) -> torch.Tensor:
    """Pack (…, T, ·) motion arrays into (…, T, D) feature vectors."""
    flat_body = body_pos.reshape(*body_pos.shape[:-2], body_pos.shape[-2] * 3)
    return torch.cat([root_pos, root_quat, joint_pos, flat_body], dim=-1)


def unpack_features(x: torch.Tensor, layout: FeatureLayout) -> dict[str, torch.Tensor]:
    """Inverse of :func:`pack_features`. Returns a dict of views."""
    body = x[..., layout.body_pos_slice]
    return {
        "root_pos": x[..., layout.root_pos_slice],
        "root_quat": x[..., layout.root_quat_slice],
        "joint_pos": x[..., layout.joint_pos_slice],
        "body_pos": body.reshape(*body.shape[:-1], layout.num_bodies, 3),
    }


@dataclass
class CanonicalTransform:
    """World-from-canonical transform of a window (per batch element)."""

    anchor_xy: torch.Tensor  # (..., 2) world xy of anchor root
    anchor_yaw: torch.Tensor  # (...,) world yaw of anchor root

    def to_dict(self) -> dict:
        return {"anchor_xy": self.anchor_xy, "anchor_yaw": self.anchor_yaw}


def canonicalize_window(
    x: torch.Tensor,
    layout: FeatureLayout,
    anchor_index: int,
) -> tuple[torch.Tensor, CanonicalTransform]:
    """Express a world-frame feature window in the anchor-frame canonical frame.

    Args:
        x: (..., T, D) world-frame features (quats must be sign-continuous).
        anchor_index: index along T of the anchor frame (last past frame).
    Returns:
        (canonical features (..., T, D), transform to undo the mapping).
    """
    parts = unpack_features(x, layout)
    anchor_pos = parts["root_pos"][..., anchor_index, :]
    anchor_yaw = quat_yaw(parts["root_quat"][..., anchor_index, :])

    offset = torch.cat([anchor_pos[..., :2], torch.zeros_like(anchor_pos[..., :1])], dim=-1)
    inv_yaw = (-anchor_yaw).unsqueeze(-1)  # broadcast over T

    root_pos = yaw_rotate_xy(inv_yaw, parts["root_pos"] - offset.unsqueeze(-2))
    q_inv = quat_from_yaw(inv_yaw).expand(*parts["root_quat"].shape[:-1], 4)
    root_quat = quat_mul(q_inv, parts["root_quat"])
    # quat_from_yaw flips sign across the yaw = +-pi branch cut; re-anchor the
    # (sign-continuous) window to w >= 0 at the anchor so near-identical inputs
    # cannot produce opposite-sign diffusion targets.
    anchor_w = root_quat[..., anchor_index, :1]
    root_quat = torch.where(anchor_w.unsqueeze(-2) < 0, -root_quat, root_quat)
    body_pos = yaw_rotate_xy(
        inv_yaw.unsqueeze(-1), parts["body_pos"] - offset.unsqueeze(-2).unsqueeze(-2)
    )
    canon = pack_features(root_pos, root_quat, parts["joint_pos"], body_pos)
    return canon, CanonicalTransform(anchor_xy=anchor_pos[..., :2], anchor_yaw=anchor_yaw)


def decanonicalize_window(
    x: torch.Tensor,
    layout: FeatureLayout,
    transform: CanonicalTransform,
) -> torch.Tensor:
    """Inverse of :func:`canonicalize_window` (canonical -> world frame)."""
    parts = unpack_features(x, layout)
    yaw = transform.anchor_yaw.unsqueeze(-1)
    offset = torch.cat(
        [transform.anchor_xy, torch.zeros_like(transform.anchor_xy[..., :1])], dim=-1
    )
    root_pos = yaw_rotate_xy(yaw, parts["root_pos"]) + offset.unsqueeze(-2)
    q_fwd = quat_from_yaw(yaw).expand(*parts["root_quat"].shape[:-1], 4)
    root_quat = quat_mul(q_fwd, parts["root_quat"])
    body_pos = yaw_rotate_xy(yaw.unsqueeze(-1), parts["body_pos"]) + offset.unsqueeze(-2).unsqueeze(-2)
    return pack_features(root_pos, root_quat, parts["joint_pos"], body_pos)


def compute_target_heading(
    canon_features: torch.Tensor,
    layout: FeatureLayout,
    anchor_index: int,
    min_disp: float = 0.05,
) -> torch.Tensor:
    """Unit xy heading vector from anchor to last frame, canonical frame.

    Falls back to (1, 0) — the anchor facing direction after heading
    normalization — when the root displacement is below ``min_disp`` meters.
    """
    root_pos = canon_features[..., layout.root_pos_slice]
    disp = root_pos[..., -1, :2] - root_pos[..., anchor_index, :2]
    norm = disp.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(disp)
    fallback[..., 0] = 1.0
    heading = torch.where(norm > min_disp, disp / norm.clamp_min(1e-8), fallback)
    return heading
