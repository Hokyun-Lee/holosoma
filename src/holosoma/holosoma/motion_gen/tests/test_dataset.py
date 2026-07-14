import numpy as np
import pytest
import torch

from holosoma.motion_gen.dataset import MotionWindowDataset, load_wbt_motion
from holosoma.motion_gen.features import FeatureLayout
from holosoma.motion_gen.terrain import ScanGrid
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


def _add_scan(path, grid: ScanGrid, *, include_grid: bool = True):
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["terrain_height"] = np.zeros((payload["joint_pos"].shape[0], grid.dim), dtype=np.float32)
    if include_grid:
        payload["terrain_grid"] = grid.to_array()
    np.savez(path, **payload)


def test_scanned_clip_grid_contract_accepts_exact_custom_grid(tmp_path):
    grid = ScanGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, spacing=0.2)
    path = make_synthetic_wbt_npz(tmp_path / "exact_grid.npz", num_frames=30)
    _add_scan(path, grid)
    scanned_clip = load_wbt_motion(path, FeatureLayout())

    dataset = MotionWindowDataset(
        [scanned_clip],
        FeatureLayout(),
        past_frames=2,
        future_frames=5,
        terrain_dim=grid.dim,
        use_terrain_scan=True,
        scan_grid=grid,
    )

    item = dataset[0]
    assert bool(item["has_scan"])
    assert item["terrain"].shape == (grid.dim,)


@pytest.mark.parametrize(
    "configured_grid",
    [
        # Same 17x17/dim=289, shifted extents.
        ScanGrid(x_min=-0.2, x_max=1.4, y_min=-0.8, y_max=0.8, spacing=0.1),
        # Same 17x17/dim=289, different extents and spacing.
        ScanGrid(x_min=-0.6, x_max=2.6, y_min=-1.6, y_max=1.6, spacing=0.2),
    ],
)
def test_scanned_clip_rejects_same_dimension_grid_contract_mismatch(tmp_path, configured_grid):
    stored_grid = ScanGrid()
    path = make_synthetic_wbt_npz(tmp_path / "mismatched_grid.npz", num_frames=30)
    _add_scan(path, stored_grid)
    scanned_clip = load_wbt_motion(path, FeatureLayout())
    assert stored_grid.dim == configured_grid.dim

    with pytest.raises(ValueError, match="terrain_grid contract mismatch") as exc_info:
        MotionWindowDataset(
            [scanned_clip],
            FeatureLayout(),
            past_frames=2,
            future_frames=5,
            terrain_dim=configured_grid.dim,
            use_terrain_scan=True,
            scan_grid=configured_grid,
        )

    message = str(exc_info.value)
    assert str(path) in message
    assert "mismatched_grid" in message
    assert "stored=" in message and "configured=" in message


def test_scanned_clip_rejects_missing_grid_metadata_with_path(tmp_path):
    path = make_synthetic_wbt_npz(tmp_path / "missing_grid.npz", num_frames=30)
    _add_scan(path, ScanGrid(), include_grid=False)

    with pytest.raises(ValueError, match="terrain_grid is missing") as exc_info:
        load_wbt_motion(path, FeatureLayout())

    assert str(path) in str(exc_info.value)


def test_legacy_no_scan_clip_remains_compatible_when_scan_mode_enabled(clip):
    grid = ScanGrid()
    dataset = MotionWindowDataset(
        [clip],
        FeatureLayout(),
        past_frames=2,
        future_frames=5,
        terrain_dim=grid.dim,
        use_terrain_scan=True,
        scan_grid=grid,
    )

    item = dataset[0]
    assert not bool(item["has_scan"])
    assert torch.count_nonzero(item["terrain"]) == 0
