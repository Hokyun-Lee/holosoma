import numpy as np
import pytest
import torch

from holosoma.motion_gen.dataset import MotionWindowDataset, load_wbt_motion
from holosoma.motion_gen.features import FeatureLayout
from holosoma.motion_gen.tests.synthetic import make_synthetic_wbt_npz


@pytest.fixture
def clip(tmp_path):
    path = make_synthetic_wbt_npz(tmp_path / "clip.npz", num_frames=100)
    return load_wbt_motion(path, FeatureLayout(), flat_terrain=True, source="synthetic")


def test_load_schema_and_shapes(clip):
    layout = FeatureLayout()
    assert clip.features.shape == (100, layout.dim)
    assert clip.foot_contact.shape == (100, 2)
    assert clip.flat_terrain


def test_load_rejects_missing_keys(tmp_path):
    np.savez(tmp_path / "bad.npz", fps=np.array([50]))
    with pytest.raises(ValueError, match="missing npz keys"):
        load_wbt_motion(tmp_path / "bad.npz", FeatureLayout())


def test_load_rejects_wrong_fps(tmp_path):
    path = make_synthetic_wbt_npz(tmp_path / "clip30.npz", num_frames=50, fps=30)
    with pytest.raises(ValueError, match="fps"):
        load_wbt_motion(path, FeatureLayout(), expected_fps=50.0)


def test_window_count_and_no_boundary_crossing(clip):
    ds = MotionWindowDataset([clip], FeatureLayout(), past_frames=2, future_frames=25, stride=1)
    assert len(ds) == 100 - 27 + 1
    last = ds[len(ds) - 1]
    assert int(last["start"]) + 27 <= clip.num_frames


def test_item_shapes_and_types(clip):
    layout = FeatureLayout()
    ds = MotionWindowDataset([clip], layout, past_frames=2, future_frames=25, terrain_dim=121)
    item = ds[0]
    assert item["x"].shape == (25, layout.dim)
    assert item["past"].shape == (2, layout.dim)
    assert item["heading"].shape == (2,)
    assert torch.allclose(item["heading"].norm(), torch.tensor(1.0), atol=1e-5)
    assert item["terrain"].shape == (121,)
    assert item["contact"].shape == (25, 2)
    assert item["contact"].dtype == torch.bool
    assert bool(item["flat"])


def test_window_too_long_raises(clip):
    with pytest.raises(ValueError, match="No valid windows"):
        MotionWindowDataset([clip], FeatureLayout(), past_frames=2, future_frames=200)
