"""Evaluation phase sampling semantics for fixed and generated WBT parents."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from holosoma.config_types.command import MotionConfig
from holosoma.managers.command.terms.wbt import MotionCommand


class _ForbiddenAdaptiveSampler:
    def sample_global_time_steps(self, _count: int) -> torch.Tensor:
        raise AssertionError("evaluation must not sample the adaptive distribution")


def _evaluation_command(mode: str, *, num_envs: int = 8) -> MotionCommand:
    command = MotionCommand.__new__(MotionCommand)
    command.num_envs = num_envs
    command.device = torch.device("cpu")
    command.motion_cfg = MotionConfig(
        motion_file="unused.npz",
        body_name_ref=["root"],
        body_names_to_track=["root"],
        evaluation_phase_mode=mode,
        use_adaptive_timesteps_sampler=True,
    )
    command.init_pose_cfg = command.motion_cfg.noise_to_initial_pose
    frame_count = 100
    identity_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(frame_count, 1, 1)
    command.motion = SimpleNamespace(
        num_motions=1,
        motion_start_idx=torch.tensor([0]),
        motion_end_idx=torch.tensor([frame_count]),
        joint_pos=torch.zeros(frame_count, 1),
        joint_vel=torch.zeros(frame_count, 1),
        body_pos_w=torch.zeros(frame_count, 1, 3),
        body_quat_w=identity_xyzw,
        body_lin_vel_w=torch.zeros(frame_count, 1, 3),
        body_ang_vel_w=torch.zeros(frame_count, 1, 3),
        has_object=False,
    )
    command.motion_ids = torch.zeros(num_envs, dtype=torch.long)
    command.time_steps = torch.zeros(num_envs, dtype=torch.long)
    command.adaptive_timesteps_sampler = _ForbiddenAdaptiveSampler()
    simulator = SimpleNamespace(
        scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)),
        dof_pos=torch.zeros(num_envs, 1),
        dof_vel=torch.zeros(num_envs, 1),
        dof_pos_limits=torch.tensor([[-1.0, 1.0]]),
        robot_root_states=torch.zeros(num_envs, 13),
    )
    command._env = SimpleNamespace(is_evaluating=True, simulator=simulator)
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
