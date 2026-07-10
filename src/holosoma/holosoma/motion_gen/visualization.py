"""Matplotlib comparison plots for generated vs. ground-truth windows."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from holosoma.motion_gen.features import FeatureLayout, unpack_features  # noqa: E402

# A readable subset for joint-angle plots (legs + one arm).
_PLOT_JOINTS = [
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_elbow_joint",
]


def plot_window_comparison(
    pred: torch.Tensor,
    gt: torch.Tensor,
    layout: FeatureLayout,
    path: str | Path,
) -> Path:
    """One figure per window: root top-down path, root/feet heights, joints."""
    path = Path(path)
    p = unpack_features(pred, layout)
    g = unpack_features(gt, layout)
    joints = [j for j in _PLOT_JOINTS if j in layout.joint_names]

    n_rows = 2 + (len(joints) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3 * n_rows))

    ax = axes[0, 0]
    ax.plot(g["root_pos"][:, 0], g["root_pos"][:, 1], "k-", label="GT")
    ax.plot(p["root_pos"][:, 0], p["root_pos"][:, 1], "C0--", label="generated")
    ax.set_title("root xy (top-down)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(g["root_pos"][:, 2], "k-", label="GT")
    ax.plot(p["root_pos"][:, 2], "C0--", label="generated")
    ax.set_title("root height z [m]")
    ax.legend()

    ax = axes[0, 2]
    foot_idx = layout.foot_body_indices()
    for f, name in zip(foot_idx, ["L foot", "R foot"]):
        ax.plot(g["body_pos"][:, f, 2], "-", label=f"GT {name}")
        ax.plot(p["body_pos"][:, f, 2], "--", label=f"gen {name}")
    ax.set_title("foot height z [m]")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(g["root_quat"], "-")
    ax.set_prop_cycle(None)
    ax.plot(p["root_quat"], "--")
    ax.set_title("root quat wxyz (solid GT / dashed gen)")

    ax = axes[1, 1]
    err = (p["body_pos"] - g["body_pos"]).norm(dim=-1).mean(dim=-1)
    ax.plot(err)
    ax.set_title("body MPJPE per frame [m]")

    ax = axes[1, 2]
    ax.plot(p["root_quat"].norm(dim=-1))
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_title("generated quat norm")

    for i, jname in enumerate(joints):
        ax = axes[2 + i // 3, i % 3]
        j = layout.joint_names.index(jname)
        ax.plot(g["joint_pos"][:, j], "k-", label="GT")
        ax.plot(p["joint_pos"][:, j], "C0--", label="gen")
        ax.set_title(jname, fontsize=8)
    for i in range(len(joints), (n_rows - 2) * 3):
        axes[2 + i // 3, i % 3].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_long_rollout(
    features: torch.Tensor,
    layout: FeatureLayout,
    path: str | Path,
    title: str = "receding-horizon rollout",
) -> Path:
    """Top-down root path + heights for a stitched long generation (T, D)."""
    path = Path(path)
    parts = unpack_features(features, layout)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(parts["root_pos"][:, 0], parts["root_pos"][:, 1], "C0-")
    axes[0].plot(parts["root_pos"][0, 0], parts["root_pos"][0, 1], "go", label="start")
    axes[0].set_title(f"{title}: root xy")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend()

    axes[1].plot(parts["root_pos"][:, 2], label="root z")
    for f, name in zip(layout.foot_body_indices(), ["L foot", "R foot"]):
        axes[1].plot(parts["body_pos"][:, f, 2], label=f"{name} z")
    axes[1].set_title("heights [m]")
    axes[1].legend(fontsize=8)

    axes[2].plot(parts["root_quat"].norm(dim=-1))
    axes[2].axhline(1.0, color="k", lw=0.5)
    axes[2].set_title("quat norm")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
