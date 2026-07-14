from __future__ import annotations

import pytest
import torch

from holosoma.motion_gen.evaluation import compute_metrics
from holosoma.motion_gen.features import FeatureLayout, pack_features
from holosoma.motion_gen.terrain import ScanGrid


def _features(body_z: torch.Tensor) -> tuple[FeatureLayout, torch.Tensor]:
    """Pack zero poses with caller-controlled ``(B, H, bodies)`` heights."""
    layout = FeatureLayout()
    batch, horizon, num_bodies = body_z.shape
    assert num_bodies == layout.num_bodies
    root_pos = torch.zeros(batch, horizon, 3)
    root_quat = torch.zeros(batch, horizon, 4)
    root_quat[..., 0] = 1.0
    joint_pos = torch.zeros(batch, horizon, layout.num_joints)
    body_pos = torch.zeros(batch, horizon, layout.num_bodies, 3)
    body_pos[..., 2] = body_z
    return layout, pack_features(root_pos, root_quat, joint_pos, body_pos)


def test_evaluation_flat_penetration_denominator_excludes_nonflat_values():
    body_z = torch.empty(2, 2, FeatureLayout().num_bodies)
    body_z[0] = -0.25
    body_z[1] = -100.0
    layout, pred = _features(body_z)

    metrics = compute_metrics(
        pred,
        pred,
        layout,
        flat=torch.tensor([True, False]),
    )

    assert metrics["terrain_penetration_m"] == pytest.approx(0.25)


def test_evaluation_rejects_flat_scan_overlap_before_reporting_metrics():
    body_z = torch.zeros(2, 2, FeatureLayout().num_bodies)
    layout, pred = _features(body_z)
    grid = ScanGrid()

    with pytest.raises(ValueError, match=r"mutually exclusive.*indices \[0\]"):
        compute_metrics(
            pred,
            pred,
            layout,
            flat=torch.tensor([True, False]),
            terrain_scan=torch.zeros(2, grid.dim),
            has_scan=torch.tensor([True, True]),
            scan_grid=grid,
        )
