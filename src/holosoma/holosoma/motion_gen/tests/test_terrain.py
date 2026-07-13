import numpy as np
import pytest
import torch

from holosoma.motion_gen.terrain import (
    BoxTerrain,
    ScanGrid,
    interpolate_scan_heights,
    local_scan_world_xy,
    save_height_scan_debug,
)


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
    offsets = g.offsets_tensor(device="cpu", dtype=torch.float64)
    assert offsets.dtype == torch.float64
    expected = {
        0: (-0.3, -0.8),
        1: (-0.3, -0.7),
        16: (-0.3, 0.8),
        17: (-0.2, -0.8),
        59: (0.0, 0.0),
        288: (1.3, 0.8),
    }
    for index, xy in expected.items():
        torch.testing.assert_close(offsets[index], torch.tensor(xy, dtype=torch.float64))


@pytest.mark.parametrize("yaw", [0.0, np.pi / 2, -1.2])
def test_torch_local_scan_matches_dataset_box_terrain(terrain, yaw):
    grid = ScanGrid()
    root_xy = torch.tensor([[0.3, -0.4]], dtype=torch.float64)
    root_yaw = torch.tensor([yaw], dtype=torch.float64)
    world_xy = local_scan_world_xy(root_xy, root_yaw, grid)
    runtime_scan = terrain.height_at(world_xy[0].numpy())
    dataset_scan = terrain.sample_scan(root_xy[0].numpy(), yaw, grid)
    np.testing.assert_allclose(runtime_scan, dataset_scan, atol=1e-12, rtol=0.0)


def _validation_obstacle_heights(local_xy: torch.Tensor) -> torch.Tensor:
    """Three disjoint validation lanes: box, ascending stairs, and hurdle."""
    x, y = local_xy.unbind(-1)
    heights = torch.zeros_like(x)

    box = (x >= 0.15) & (x <= 0.55) & (y >= -0.75) & (y <= -0.25)
    heights = torch.where(box, 0.3, heights)

    stair_lane = (y >= -0.15) & (y <= 0.15)
    heights = torch.where(stair_lane & (x >= 0.15) & (x <= 0.45), 0.1, heights)
    heights = torch.where(stair_lane & (x >= 0.45) & (x <= 0.75), 0.2, heights)
    heights = torch.where(stair_lane & (x >= 0.75) & (x <= 1.05), 0.3, heights)

    hurdle = (x >= 0.65) & (x <= 0.95) & (y >= 0.25) & (y <= 0.75)
    return torch.where(hurdle, 0.6, heights)


def test_local_scan_flat_obstacles_order_and_yaw():
    grid = ScanGrid()
    root_xy = torch.tensor([[0.0, 0.0]])
    root_yaw = torch.tensor([0.0])
    world_xy = local_scan_world_xy(root_xy, root_yaw, grid)

    # Flat query: absolute world terrain Z is exactly zero.
    flat = torch.zeros(world_xy.shape[:-1])
    assert flat.shape == (1, 289)
    assert flat.abs().max() < 1e-7

    # The torch grid must match the dataset's numpy meshgrid(indexing="ij")
    # flatten exactly: x-major, with y changing fastest.
    np.testing.assert_allclose(world_xy[0].numpy(), grid.offsets(), atol=1e-7)
    scan = _validation_obstacle_heights(world_xy).view(grid.nx, grid.ny)
    xs = grid.x_min + grid.spacing * torch.arange(grid.nx)
    ys = grid.y_min + grid.spacing * torch.arange(grid.ny)
    box_x = (xs >= 0.15) & (xs <= 0.55)
    box_y = (ys >= -0.75) & (ys <= -0.25)
    torch.testing.assert_close(
        scan[box_x][:, box_y],
        torch.full_like(scan[box_x][:, box_y], 0.3),
    )
    assert scan[(xs - 0.5).abs().argmin(), (ys + 0.5).abs().argmin()] == pytest.approx(0.3)
    assert scan[(xs - 0.3).abs().argmin(), (ys - 0.0).abs().argmin()] == pytest.approx(0.1)
    assert scan[(xs - 0.6).abs().argmin(), (ys - 0.0).abs().argmin()] == pytest.approx(0.2)
    assert scan[(xs - 0.9).abs().argmin(), (ys - 0.0).abs().argmin()] == pytest.approx(0.3)
    assert scan[(xs - 0.8).abs().argmin(), (ys - 0.5).abs().argmin()] == pytest.approx(0.6)

    # Rotate both the robot frame and validation scene by 90 degrees in world.
    # The local scan indices and values must remain unchanged.
    rotated_root = torch.tensor([[2.0, -3.0]])
    rotated_yaw = torch.tensor([torch.pi / 2])
    rotated_world = local_scan_world_xy(rotated_root, rotated_yaw, grid)
    delta = rotated_world - rotated_root[:, None, :]
    rotated_back_to_local = torch.stack([delta[..., 1], -delta[..., 0]], dim=-1)
    rotated_scan = _validation_obstacle_heights(rotated_back_to_local)
    torch.testing.assert_close(rotated_scan, scan.reshape(1, -1), atol=1e-6, rtol=0.0)


def test_height_scan_debug_snapshot(tmp_path):
    grid = ScanGrid()
    root_xy = torch.tensor([[1.0, 2.0]])
    root_yaw = torch.tensor([0.25])
    world_xy = local_scan_world_xy(root_xy, root_yaw, grid)
    terrain_height = _validation_obstacle_heights(grid.offsets_tensor(device="cpu")).unsqueeze(0)
    output = save_height_scan_debug(
        tmp_path / "scan_debug.npz",
        root_xy=root_xy,
        root_yaw=root_yaw,
        world_xy=world_xy,
        terrain_height=terrain_height,
        grid=grid,
    )

    data = np.load(output)
    assert tuple(data["terrain_height"].shape) == (1, 289)
    assert tuple(data["world_xy"].shape) == (1, 289, 2)
    assert data["flatten_order"].item() == "x-major,y-fastest"
    assert data["height_units"].item() == "absolute-world-z-metres"
    np.testing.assert_allclose(data["local_xy"], grid.offsets())
