from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from holosoma.config_values.wbt.g1.command_ablation import fixed_reference_heading_motion_config
from holosoma.managers.command.terms.wbt import (
    MotionCommand,
    configure_motion_terrain_scan,
    select_reference_heading,
)
from holosoma.managers.reward.terms.wbt import motion_heading_alignment


def test_reference_heading_uses_velocity_lookahead_then_stationary_fallback() -> None:
    velocity = torch.tensor([[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    lookahead = torch.tensor([[1.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    fallback = torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])

    heading = select_reference_heading(
        velocity,
        lookahead,
        fallback,
        source="velocity_then_lookahead",
        speed_threshold=0.05,
        displacement_threshold_m=0.02,
    )

    torch.testing.assert_close(
        heading,
        torch.tensor([[0.0, 1.0], [0.6, 0.8], [0.0, 1.0]]),
    )


def test_reference_heading_priority_is_configurable() -> None:
    heading = select_reference_heading(
        torch.tensor([[0.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([[-1.0, 0.0]]),
        source="lookahead_then_velocity",
        speed_threshold=0.05,
        displacement_threshold_m=0.02,
    )

    torch.testing.assert_close(heading, torch.tensor([[0.6, 0.8]]))


def test_zero_thresholds_do_not_turn_zero_vectors_into_headings() -> None:
    heading = select_reference_heading(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.tensor([[0.0, 1.0]]),
        source="velocity_then_lookahead",
        speed_threshold=0.0,
        displacement_threshold_m=0.0,
    )

    torch.testing.assert_close(heading, torch.tensor([[0.0, 1.0]]))


@pytest.mark.parametrize("source", ["random", "", "velocity"])
def test_reference_heading_rejects_unknown_source(source: str) -> None:
    values = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="reference_heading_source"):
        select_reference_heading(
            values,
            values,
            values,
            source=source,
            speed_threshold=0.05,
            displacement_threshold_m=0.02,
        )


def test_heading_reward_accepts_fixed_motion_command_interface() -> None:
    command = MotionCommand.__new__(MotionCommand)
    command.motion_cfg = fixed_reference_heading_motion_config
    command.time_steps = torch.tensor([0, 0])
    command.motion_ids = torch.tensor([0, 0])
    root_states = torch.zeros(2, 13)
    root_states[:, 7:9] = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    command._env = SimpleNamespace(dt=0.02, simulator=SimpleNamespace(robot_root_states=root_states))

    root_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]]])
    root_pos = torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    root_vel = torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    command.motion = SimpleNamespace(
        motion_end_idx=torch.tensor([2]),
        body_pos_w=root_pos,
        body_quat_w=root_quat,
        body_lin_vel_w=root_vel,
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_state=lambda _: command))

    reward = motion_heading_alignment(env)

    epsilon = fixed_reference_heading_motion_config.heading_reward_epsilon
    torch.testing.assert_close(reward, torch.tensor([1.0 / (1.0 + epsilon), -1.0 / (1.0 + epsilon)]))


def test_fixed_command_configures_dataset_ordered_tracker_scan() -> None:
    configured_grids = []
    terrain_state = SimpleNamespace(configure_local_height_scan=configured_grids.append)
    env = SimpleNamespace(terrain_manager=SimpleNamespace(get_state=lambda _: terrain_state))

    grid = configure_motion_terrain_scan(fixed_reference_heading_motion_config, env)

    assert grid is not None
    assert configured_grids == [grid]
    assert (grid.nx, grid.ny, grid.dim) == (17, 17, 289)
    offsets = grid.offsets_tensor(device="cpu")
    torch.testing.assert_close(offsets[0], torch.tensor([-0.3, -0.8]))
    torch.testing.assert_close(offsets[1], torch.tensor([-0.3, -0.7]))
    torch.testing.assert_close(offsets[17], torch.tensor([-0.2, -0.8]))
