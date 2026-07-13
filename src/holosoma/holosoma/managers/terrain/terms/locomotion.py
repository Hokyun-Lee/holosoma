"""Terrain hooks for locomotion tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from holosoma.managers.terrain.base import TerrainTermBase
from holosoma.motion_gen.terrain import (
    ScanGrid,
    local_scan_world_xy,
    save_height_scan_debug,
)
from holosoma.simulator.shared.terrain import Terrain
from holosoma.utils import draw, warp_utils
from holosoma.utils.rotations import quat_apply_yaw
from holosoma.utils.safe_torch_import import torch

ATTACHMENT_POS = (0.0, 0.0, 0.25)


def interquartile_mean(x, dim=-1):
    """Compute interquartile mean (mean of values between 25th and 75th percentile)"""
    sorted_x, _ = torch.sort(x, dim=dim)
    n = sorted_x.shape[dim]

    # Calculate quartile indices
    q1_idx = max(1, n // 4)  # 25th percentile index
    q3_idx = min(n - 1, 3 * n // 4)  # 75th percentile index

    # Extract interquartile range and compute mean
    if dim == -1 or dim == sorted_x.dim() - 1:
        iqr_values = sorted_x[..., q1_idx : q3_idx + 1]
    else:
        # Handle arbitrary dimension
        indices = [slice(None)] * sorted_x.dim()
        indices[dim] = slice(q1_idx, q3_idx + 1)
        iqr_values = sorted_x[tuple(indices)]

    return torch.mean(iqr_values, dim=dim)


class TerrainLocomotion(TerrainTermBase):
    """Stateful terrain term that owns terrain buffers and updates them each step."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        self._terrain = Terrain(self._cfg, self.num_envs)
        assert hasattr(self._terrain, "mesh")
        self._warp_mesh = warp_utils.convert_to_wp_mesh(
            self._terrain.mesh.vertices, self._terrain.mesh.faces, self.device
        )
        self._env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self._get_env_origins()

    @property
    def terrain(self):
        return self._terrain

    @property
    def mesh(self):
        return self._terrain.mesh

    @property
    def warp_mesh(self):
        return self._warp_mesh

    def setup(self) -> None:
        self._base_heights = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self._base_height_points, self._ray_directions, self._num_base_height_points = self._init_base_height_points()
        self._ray_hits_world_base = torch.zeros(
            self.num_envs, self._num_base_height_points, 3, device=self.device, requires_grad=False
        )
        self._offset_pos = torch.tensor(list(ATTACHMENT_POS), device=self.device)

        if hasattr(self.env, "feet_height_indices"):
            # Environment requires feet heights
            self._compute_feet_heights = True
            self._feet_heights = torch.zeros(
                self.num_envs, len(self.env.feet_height_indices), device=self.device, requires_grad=False
            )
            self._ray_directions_feet = torch.zeros(
                self.num_envs, len(self.env.feet_height_indices), 3, device=self.device, requires_grad=False
            )
            self._ray_directions_feet[..., :] = torch.tensor([0.0, 0.0, -1.0], device=self.device)
            self._ray_hits_world_feet = torch.zeros(
                self.num_envs, len(self.env.feet_height_indices), 3, device=self.device, requires_grad=False
            )
        else:
            # Initialize not-used feet heights tensors with empty tensors
            self._compute_feet_heights = False
            self._feet_heights = torch.zeros(0, device=self.device, requires_grad=False)
            self._ray_directions_feet = torch.zeros(0, 0, 3, device=self.device, requires_grad=False)
            self._ray_hits_world_feet = torch.zeros(0, 0, 3, device=self.device, requires_grad=False)

    @property
    def env_origins(self) -> torch.Tensor:
        return self._env_origins

    @property
    def custom_origins(self) -> bool:
        return self._custom_origins

    @property
    def base_heights(self) -> torch.Tensor:
        return self._base_heights

    @property
    def feet_heights(self) -> torch.Tensor:
        return self._feet_heights

    @property
    def local_height_scan(self) -> torch.Tensor:
        if not hasattr(self, "_local_height_scan"):
            raise RuntimeError("Local height scan is not configured. Call configure_local_height_scan() first.")
        return self._local_height_scan

    @property
    def local_height_scan_configured(self) -> bool:
        return hasattr(self, "_local_scan_grid")

    @property
    def local_height_scan_valid(self) -> torch.Tensor:
        if not hasattr(self, "_local_height_scan_valid"):
            raise RuntimeError("Local height scan is not configured. Call configure_local_height_scan() first.")
        return self._local_height_scan_valid

    @property
    def local_height_scan_root_xy(self) -> torch.Tensor:
        if not hasattr(self, "_local_scan_root_xy"):
            raise RuntimeError("Local height scan is not configured. Call configure_local_height_scan() first.")
        return self._local_scan_root_xy

    @property
    def local_height_scan_root_yaw(self) -> torch.Tensor:
        if not hasattr(self, "_local_scan_root_yaw"):
            raise RuntimeError("Local height scan is not configured. Call configure_local_height_scan() first.")
        return self._local_scan_root_yaw

    def update_heights(self, env_ids=None):
        idx = env_ids if env_ids is not None else slice(None)
        self._base_heights[idx], self._ray_hits_world_base[idx] = self._get_base_heights(env_ids)
        if self._compute_feet_heights:
            self._feet_heights[idx], self._ray_hits_world_feet[idx] = self._get_feet_heights(env_ids)

    def configure_local_height_scan(self, grid: ScanGrid) -> None:
        """Allocate the checkpoint-defined generator scan buffers.

        This may run before :meth:`setup` because generated-motion command
        setup precedes terrain setup in ``BaseTask``.
        """
        existing = getattr(self, "_local_scan_grid", None)
        if existing is not None and existing != grid:
            raise ValueError(f"Local height scan already configured with {existing}, cannot replace it with {grid}")
        if existing is not None:
            return

        self._local_scan_grid = grid
        self._local_height_scan = torch.zeros(self.num_envs, grid.dim, device=self.device)
        self._local_scan_world_xy = torch.zeros(self.num_envs, grid.dim, 2, device=self.device)
        self._local_scan_root_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._local_scan_root_yaw = torch.zeros(self.num_envs, device=self.device)
        self._local_height_scan_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def query_local_height_scan(
        self,
        root_xy: torch.Tensor,
        root_yaw: torch.Tensor,
        grid: ScanGrid,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return absolute world-Z scan values and their world XY locations."""
        world_xy = local_scan_world_xy(root_xy, root_yaw, grid)
        heights = self.query_terrain_heights(world_xy.reshape(-1, 2)).reshape(root_xy.shape[0], grid.dim)
        return heights, world_xy

    def update_local_height_scan(
        self,
        root_xy: torch.Tensor,
        root_yaw: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Query and cache scans for all environments or an explicit subset."""
        if not hasattr(self, "_local_scan_grid"):
            raise RuntimeError("Local height scan is not configured. Call configure_local_height_scan() first.")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if root_xy.shape[0] != env_ids.numel():
            raise ValueError(
                f"root batch ({root_xy.shape[0]}) must match env_ids ({env_ids.numel()})"
            )

        heights, world_xy = self.query_local_height_scan(root_xy, root_yaw, self._local_scan_grid)
        self._local_height_scan[env_ids] = heights
        self._local_scan_world_xy[env_ids] = world_xy
        self._local_scan_root_xy[env_ids] = root_xy
        self._local_scan_root_yaw[env_ids] = root_yaw
        self._local_height_scan_valid[env_ids] = True
        return heights

    def invalidate_local_height_scan(self, env_ids: torch.Tensor | None = None) -> None:
        """Mark cached scans stale, typically immediately before an env reset."""
        if not hasattr(self, "_local_height_scan_valid"):
            return
        if env_ids is None:
            self._local_height_scan_valid[:] = False
        else:
            self._local_height_scan_valid[env_ids.to(device=self.device, dtype=torch.long)] = False

    def save_local_height_scan_debug(self, path: str | Path, env_id: int = 0) -> Path:
        """Persist one cached scan for coordinate/order inspection."""
        if not hasattr(self, "_local_scan_grid"):
            raise RuntimeError("Local height scan is not configured. Call configure_local_height_scan() first.")
        if not 0 <= env_id < self.num_envs:
            raise IndexError(f"env_id must be in [0, {self.num_envs}), got {env_id}")
        if not bool(self._local_height_scan_valid[env_id]):
            raise RuntimeError(f"Local height scan for env {env_id} has not been updated yet.")
        sl = slice(env_id, env_id + 1)
        return save_height_scan_debug(
            path,
            root_xy=self._local_scan_root_xy[sl],
            root_yaw=self._local_scan_root_yaw[sl],
            world_xy=self._local_scan_world_xy[sl],
            terrain_height=self._local_height_scan[sl],
            grid=self._local_scan_grid,
        )

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        self._custom_origins = True

        if self._cfg.spawn.randomize_tiles:
            # Training mode: random terrain tiles for curriculum learning
            self._env_origins[:] = torch.from_numpy(self.terrain.sample_env_origins()).to(self.device).to(torch.float)
        else:
            # Eval mode: all robots at tile (0,0) for deterministic evaluation
            origin_0_0 = torch.from_numpy(self.terrain._env_origins[0, 0]).to(self.device).to(torch.float)
            self._env_origins[:] = origin_0_0  # Broadcast to all robots

    def _init_base_height_points(self):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        # Sampling spacing should match terrain resolution for meaningful measurements
        terrain_res = self._cfg.horizontal_scale
        base_spacing = terrain_res  # Match terrain grid resolution

        # Create sampling range: use terrain resolution as spacing
        # For 0.05m terrain: sample at [..., -0.10, -0.05, 0.0, 0.05, 0.10, ...]
        # For 0.10m terrain: sample at [..., -0.10, 0.0, 0.10, ...]
        max_range = 0.15  # 15cm radius around robot
        num_points = int(2 * max_range / base_spacing) + 1

        # Ensure odd number for symmetric sampling around center
        if num_points % 2 == 0:
            num_points += 1

        # Create symmetric range aligned with terrain grid
        half_range = (num_points - 1) * base_spacing / 2
        points_1d = torch.linspace(-half_range, half_range, num_points, device=self.device, requires_grad=False)

        y = points_1d
        x = points_1d

        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        num_base_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            num_base_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        ray_directions = torch.zeros((self.num_envs, num_base_height_points, 3), device=self.device)
        ray_directions[..., :] = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        return points, ray_directions, num_base_height_points

    def _get_base_heights(self, env_ids=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get base heights from the terrain."""
        idx = env_ids if env_ids is not None else slice(None)
        base_positions = quat_apply_yaw(
            self.env.base_quat[idx].repeat(1, self._num_base_height_points),
            self._base_height_points[idx],
            True,
        ) + (self.env.simulator.robot_root_states[idx, :3]).unsqueeze(1)
        ray_starts_world = base_positions + self._offset_pos[None, None, ...]
        ray_directions_world = self._ray_directions[idx]
        ray_hits_world = warp_utils.ray_cast(ray_starts_world, ray_directions_world, self.warp_mesh)
        base_heights = (base_positions - ray_hits_world)[..., 2]
        return interquartile_mean(base_heights, dim=1), ray_hits_world

    def _get_feet_heights(self, env_ids=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get feet heights from the terrain."""
        idx = env_ids if env_ids is not None else slice(None)
        foot_positions = self.env.simulator._rigid_body_pos[idx, self.env.feet_height_indices, :].clone()
        ray_starts_world = foot_positions + self._offset_pos[None, None, ...]
        ray_directions_world = self._ray_directions_feet[idx]
        ray_hits_world = warp_utils.ray_cast(ray_starts_world, ray_directions_world, self.warp_mesh)
        return (foot_positions - ray_hits_world)[..., 2], ray_hits_world

    def query_terrain_heights(
        self,
        xy_positions: torch.Tensor,
        use_grid_sampling: bool = False,
        grid_size: int = 3,
        grid_spacing: float = 0.3,
    ) -> torch.Tensor:
        """Query terrain height at arbitrary XY positions using ray casting.

        This method casts rays straight down from high above the terrain to find
        the actual terrain height at each XY position. Useful for determining
        spawn heights, checking clearances, or any arbitrary height query.

        Args:
            xy_positions: XY coordinates to query. Shape: (N, 2)
            use_grid_sampling: If True, sample a grid around each position and return max height.
                              This ensures the robot clears all terrain within its footprint.
            grid_size: Number of points per dimension in the grid (e.g., 3 = 3x3 = 9 points)
            grid_spacing: Spacing between grid points in meters (e.g., 0.3m)

        Returns:
            Terrain heights (Z coordinates). Shape: (N,)

        Example:
            >>> xy = torch.tensor([[0.5, 0.5], [1.0, 1.0]], device="cuda")
            >>> heights = terrain.query_terrain_heights(xy)
            >>> heights.shape  # (2,)

            >>> # With grid sampling for safer spawn heights
            >>> heights = terrain.query_terrain_heights(xy, use_grid_sampling=True)
            >>> heights.shape  # (2,) - max height from 3x3 grid around each position
        """
        num_robots = xy_positions.shape[0]

        if use_grid_sampling:
            # Generate grid offsets centered at origin
            # For grid_size=3, spacing=0.3: offsets = [-0.3, 0.0, 0.3]
            half_extent = (grid_size - 1) * grid_spacing / 2.0
            offsets_1d = torch.linspace(-half_extent, half_extent, grid_size, device=self.device)
            grid_x, grid_y = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
            grid_offsets = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
            # Shape: (grid_size^2, 2) e.g., (9, 2) for 3x3

            # Expand positions to include all grid points
            # xy_positions: (N, 2) -> (N, 1, 2)
            # grid_offsets: (9, 2) -> (1, 9, 2)
            # Result: (N, 9, 2) via broadcasting
            xy_expanded = xy_positions.unsqueeze(1) + grid_offsets.unsqueeze(0)

            # Flatten for batched ray casting
            xy_flat = xy_expanded.reshape(-1, 2)  # (N*grid_size^2, 2)
            num_points = len(xy_flat)
        else:
            xy_flat = xy_positions
            num_points = num_robots

        # Create ray starts high above terrain (100m should be above any realistic terrain)
        ray_starts = torch.zeros(num_points, 3, device=self.device)
        ray_starts[:, :2] = xy_flat
        ray_starts[:, 2] = 100.0  # Start 100m above ground for safety

        # Ray directions pointing straight down
        ray_directions = torch.zeros(num_points, 3, device=self.device)
        ray_directions[:, 2] = -1.0

        # Cast rays to find terrain height
        ray_hits = warp_utils.ray_cast(ray_starts, ray_directions, self.warp_mesh)

        if use_grid_sampling:
            # Reshape and take max per robot.
            terrain_heights_raw = ray_hits[:, 2]
            grid_points_per_robot = grid_size * grid_size
            terrain_heights_grid = terrain_heights_raw.reshape(num_robots, grid_points_per_robot)
            # Takes max height across grid to improve likelihood robot clears all terrain
            return terrain_heights_grid.max(dim=1)[0]  # Shape: (N,)
        # Extract Z coordinates (assumes above terrain heights)
        return ray_hits[:, 2]

    def draw_debug_viz(self):
        env_id = 0
        for j in range(self._num_base_height_points):
            position = self._ray_hits_world_base[env_id, j].detach().cpu().numpy()
            draw.draw_sphere(self.env.simulator, position, 0.02, (1, 0, 0), env_id)
        if self._compute_feet_heights:
            for j in range(len(self.env.feet_height_indices)):
                position = self._ray_hits_world_feet[env_id, j].detach().cpu().numpy()
                draw.draw_sphere(self.env.simulator, position, 0.02, (0, 1, 0), env_id)
        if hasattr(self, "_local_height_scan_valid") and bool(self._local_height_scan_valid[env_id]):
            scan_points = torch.cat(
                [
                    self._local_scan_world_xy[env_id],
                    self._local_height_scan[env_id, :, None],
                ],
                dim=-1,
            )
            for position in scan_points.detach().cpu().numpy():
                draw.draw_sphere(self.env.simulator, position, 0.012, (0, 0.6, 1), env_id)
