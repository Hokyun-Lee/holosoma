"""Evaluation phase sampling semantics for fixed and generated WBT parents."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from holosoma.config_types.command import MotionConfig
from holosoma.managers.command.terms.wbt import MotionCommand


class _ForbiddenAdaptiveSampler:
    def sample_global_time_steps(self, _count: int) -> torch.Tensor:
        raise AssertionError("evaluation must not sample the adaptive distribution")


def _evaluation_command(
    mode: str,
    *,
    num_envs: int = 8,
    frame_count: int = 100,
    phase_horizon_steps: int = 0,
    reanchor: bool = False,
    is_evaluating: bool = True,
) -> MotionCommand:
    command = MotionCommand.__new__(MotionCommand)
    command.num_envs = num_envs
    command.device = torch.device("cpu")
    command.motion_cfg = MotionConfig(
        motion_file="unused.npz",
        body_name_ref=["root"],
        body_names_to_track=["root"],
        evaluation_phase_mode=mode,
        reanchor_motion_xy_on_reset=reanchor,
        phase_horizon_steps=phase_horizon_steps,
        use_adaptive_timesteps_sampler=is_evaluating,
    )
    command.init_pose_cfg = command.motion_cfg.noise_to_initial_pose
    identity_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(frame_count, 1, 1)
    body_pos = torch.zeros(frame_count, 1, 3)
    body_pos[:, 0, 0] = torch.arange(frame_count, dtype=torch.float32) * 0.1 + 3.0
    body_pos[:, 0, 1] = torch.arange(frame_count, dtype=torch.float32) * -0.2 + 4.0
    body_pos[:, 0, 2] = 0.8
    command.motion = SimpleNamespace(
        num_motions=1,
        motion_start_idx=torch.tensor([0]),
        motion_end_idx=torch.tensor([frame_count]),
        joint_pos=torch.zeros(frame_count, 1),
        joint_vel=torch.zeros(frame_count, 1),
        body_pos_w=body_pos,
        body_quat_w=identity_xyzw,
        body_lin_vel_w=torch.zeros(frame_count, 1, 3),
        body_ang_vel_w=torch.zeros(frame_count, 1, 3),
        object_pos_w=body_pos[:, 0] + torch.tensor([0.5, -0.25, 0.1]),
        has_object=False,
    )
    command.motion_ids = torch.zeros(num_envs, dtype=torch.long)
    command.time_steps = torch.zeros(num_envs, dtype=torch.long)
    command._motion_reanchor_xy = torch.zeros(num_envs, 2)
    command.tracked_body_indexes = torch.tensor([0])
    command.ref_body_index = 0
    command.adaptive_timesteps_sampler = _ForbiddenAdaptiveSampler()
    simulator = SimpleNamespace(
        scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)),
        dof_pos=torch.zeros(num_envs, 1),
        dof_vel=torch.zeros(num_envs, 1),
        dof_pos_limits=torch.tensor([[-1.0, 1.0]]),
        robot_root_states=torch.zeros(num_envs, 13),
    )
    command._env = SimpleNamespace(is_evaluating=is_evaluating, simulator=simulator)
    return command


def test_evaluation_zero_phase_preserves_legacy_first_frame() -> None:
    command = _evaluation_command("zero")
    command.reset(None)
    assert command.time_steps.tolist() == [0] * command.num_envs


def test_evaluation_uniform_phase_is_seeded_and_not_forced_to_frame_zero() -> None:
    first = _evaluation_command("uniform")
    second = _evaluation_command("uniform")
    torch.manual_seed(123)
    first.reset(None)
    torch.manual_seed(123)
    second.reset(None)
    assert first.time_steps.tolist() == second.time_steps.tolist()
    assert any(time_step > 0 for time_step in first.time_steps.tolist())
    assert len(set(first.time_steps.tolist())) > 1
    assert all(0 <= time_step < 99 for time_step in first.time_steps.tolist())


def test_phase_horizon_includes_last_safe_start_without_clip_wrap() -> None:
    command = _evaluation_command(
        "uniform",
        num_envs=4096,
        frame_count=1001,
        phase_horizon_steps=500,
    )
    torch.manual_seed(7)
    command.reset(None)

    assert int(command.time_steps.max()) == 500
    assert bool((command.time_steps + 500 < command.motion.motion_end_idx[0]).all())


def test_phase_horizon_also_constrains_training_uniform_sampling() -> None:
    command = _evaluation_command(
        "zero",
        num_envs=4096,
        frame_count=1001,
        phase_horizon_steps=500,
        is_evaluating=False,
    )
    torch.manual_seed(7)
    command.reset(None)

    assert int(command.time_steps.max()) == 500
    assert bool((command.time_steps + 500 < command.motion.motion_end_idx[0]).all())


def test_phase_horizon_rejects_clip_without_one_safe_start() -> None:
    command = _evaluation_command("uniform", frame_count=500, phase_horizon_steps=500)
    with pytest.raises(ValueError, match="more than 500 frames"):
        command.reset(None)


def test_reanchor_places_selected_root_at_env_origin_and_preserves_trajectory() -> None:
    command = _evaluation_command("zero", num_envs=2, frame_count=10, reanchor=True)
    command._env.simulator.scene.env_origins[:] = torch.tensor(
        [[10.0, -5.0, 0.0], [-2.0, 8.0, 0.0]]
    )
    command.reset(None)

    torch.testing.assert_close(
        command._env.simulator.robot_root_states[:, :2],
        command._env.simulator.scene.env_origins[:, :2],
    )
    command.time_steps[:] = 2
    expected_delta = torch.tensor([0.2, -0.4])
    torch.testing.assert_close(
        command.root_pos_w[:, :2],
        command._env.simulator.scene.env_origins[:, :2] + expected_delta,
    )
    torch.testing.assert_close(command.body_pos_w[:, 0], command.root_pos_w)
    torch.testing.assert_close(command.ref_pos_w, command.root_pos_w)
    torch.testing.assert_close(
        command.object_pos_w[:, :2],
        command.root_pos_w[:, :2] + torch.tensor([0.5, -0.25]),
    )
