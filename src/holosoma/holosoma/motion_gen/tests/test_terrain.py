import numpy as np
import pytest
import torch

from holosoma.motion_gen.terrain import BoxTerrain, ScanGrid, interpolate_scan_heights


def _write_box_obj(path, corners_xy, z0, z1):
    lines = []
    for x, y in corners_xy:
        lines.append(f"v {x} {y} {z0}")
        lines.append(f"v {x} {y} {z1}")
    lines.append("f 1 3 5")  # faces unused by the parser
    path.write_text("\n".join(lines))


@pytest.fixture
def terrain(tmp_path):
    """One axis-aligned box [0,1]x[0,1] h=0.5 and one yaw-rotated box at 0.3."""
    box_dir = tmp_path / "box_models"
    box_dir.mkdir()
    _write_box_obj(box_dir / "box1.obj", [(0, 0), (1, 0), (1, 1), (0, 1)], 0.0, 0.5)
    c, s = np.cos(0.5), np.sin(0.5)
    rot = [(2 + c * x - s * y, 1 + s * x + c * y) for x, y in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]]
    _write_box_obj(box_dir / "box2.obj", rot, 0.0, 0.3)
    urdf = tmp_path / "multi_boxes_z_scale_1.0.urdf"
    urdf.write_text(
        '<robot name="multi_boxes">'
        '<mesh filename="box_models/box1.obj" scale="1.0 1.0 1.0"/>'
        '<mesh filename="box_models/box2.obj" scale="1.0 1.0 1.0"/>'
        "</robot>"
    )
    return BoxTerrain.from_urdf(urdf)


def test_height_queries(terrain):
    pts = np.array([
        [0.5, 0.5],   # on box1
        [2.0, 1.0],   # center of rotated box2
        [-1.0, -1.0], # ground
        [1.5, 0.5],   # between boxes -> ground
    ])
    h = terrain.height_at(pts)
    assert h[0] == pytest.approx(0.5)
    assert h[1] == pytest.approx(0.3)
    assert h[2] == 0.0
    assert h[3] == 0.0


def test_scale_applies(tmp_path, terrain):
    # z-scale variant: same footprint, scaled top height
    box_dir = tmp_path / "box_models"
    urdf = tmp_path / "multi_boxes_z_scale_0.8.urdf"
    urdf.write_text(
        '<robot name="multi_boxes">'
        '<mesh filename="box_models/box1.obj" scale="1.0 1.0 0.8"/>'
        "</robot>"
    )
    t = BoxTerrain.from_urdf(urdf)
    assert t.height_at(np.array([[0.5, 0.5]]))[0] == pytest.approx(0.4)


def test_scan_heading_alignment(terrain):
    """A scan taken facing the box must see it in front (+x rows)."""
    grid = ScanGrid()
    # root behind box1 at (-0.35, 0.5), facing +x -> box occupies x in [0.35, 1.35] locally
    scan = terrain.sample_scan(np.array([-0.35, 0.5]), 0.0, grid).reshape(grid.nx, grid.ny)
    xs = grid.x_min + grid.spacing * np.arange(grid.nx)
    on_box = (xs > 0.36) & (xs < 1.3)
    mid_y = grid.ny // 2
    assert np.allclose(scan[on_box, mid_y], 0.5)
    assert scan[0, mid_y] == 0.0  # behind the root: ground
    # rotate the root by pi/2: the same grid indices now point along world +y (ground)
    scan_rot = terrain.sample_scan(np.array([-0.35, 0.5]), np.pi / 2, grid).reshape(grid.nx, grid.ny)
    assert scan_rot[on_box, mid_y].max() == 0.0


def test_interpolate_scan_heights_and_mask():
    grid = ScanGrid()
    scan = torch.zeros(1, grid.dim)
    scan2d = scan.view(1, grid.nx, grid.ny)
    scan2d[:, :, :] = 0.2  # constant height
    q = torch.tensor([[[0.0, 0.0], [0.5, 0.1], [5.0, 5.0]]])  # last point outside
    h, valid = interpolate_scan_heights(scan, q, grid)
    assert h[0, 0] == pytest.approx(0.2)
    assert h[0, 1] == pytest.approx(0.2)
    assert bool(valid[0, 0]) and bool(valid[0, 1])
    assert not bool(valid[0, 2]) and h[0, 2] == 0.0


def test_grid_roundtrip():
    g = ScanGrid()
    g2 = ScanGrid.from_array(g.to_array())
    assert g2 == g
    assert g.dim == g.nx * g.ny == 17 * 17
