"""Measured-history and replan scheduling tests for generated WBT commands."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from holosoma.managers.command.terms.wbt_gen import GeneratedMotionCommand
from holosoma.motion_gen.features import FeatureLayout
from holosoma.motion_gen.sampling import MotionGeneratorOutput


def _scheduler_command(num_envs: int = 3) -> tuple[GeneratedMotionCommand, list[torch.Tensor]]:
    cmd = GeneratedMotionCommand.__new__(GeneratedMotionCommand)
    cmd.num_envs = num_envs
    cmd.device = torch.device("cpu")
    cmd._past = 2
    cmd._feat_dim = 1
    cmd._replan_steps = 25
    cmd._history = torch.full((num_envs, cmd._past, cmd._feat_dim), -1.0)
    cmd.window_idx = torch.zeros(num_envs, dtype=torch.long)
    cmd._measured_history_valid_count = torch.zeros(num_envs, dtype=torch.long)
    cmd._has_closed_loop_replan = torch.zeros(num_envs, dtype=torch.bool)
    cmd._bootstrap_replan_count = torch.zeros(num_envs, dtype=torch.long)
    cmd._closed_loop_replan_count = torch.zeros(num_envs, dtype=torch.long)
    cmd._last_replan_interval_steps = torch.zeros(num_envs, dtype=torch.long)

    generated_histories: list[torch.Tensor] = []

    def _capture_window(self, env_ids: torch.Tensor) -> None:
        generated_histories.append(self._history[env_ids].clone())

    cmd._generate_window = MethodType(_capture_window, cmd)
    return cmd, generated_histories


def _set_measured_frame(cmd: GeneratedMotionCommand, values: list[float]) -> None:
    frame = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

    def _measured_frame(_self) -> torch.Tensor:
        return frame.clone()

    cmd._sim_features = MethodType(_measured_frame, cmd)


def test_first_closed_loop_replan_uses_two_distinct_measured_frames() -> None:
    cmd, generated_histories = _scheduler_command()
    env_ids = torch.arange(cmd.num_envs)
    cmd._reset_closed_loop_tracking(env_ids)
    cmd._replan(env_ids, bootstrap=True)
    assert cmd._bootstrap_replan_count.tolist() == [1, 1, 1]
    assert generated_histories[0].squeeze(-1).tolist() == [[-1.0, -1.0]] * 3

    _set_measured_frame(cmd, [10.0, 20.0, 30.0])
    cmd._advance_measured_history_and_replan()
    assert cmd._measured_history_valid_count.tolist() == [1, 1, 1]
    assert cmd._history.squeeze(-1).tolist() == [[-1.0, 10.0], [-1.0, 20.0], [-1.0, 30.0]]
    assert len(generated_histories) == 1

    _set_measured_frame(cmd, [11.0, 21.0, 31.0])
    cmd._advance_measured_history_and_replan()

    assert len(generated_histories) == 2
    assert generated_histories[1].squeeze(-1).tolist() == [
        [10.0, 11.0],
        [20.0, 21.0],
        [30.0, 31.0],
    ]
    assert cmd._closed_loop_replan_count.tolist() == [1, 1, 1]
    assert cmd._last_replan_interval_steps.tolist() == [2, 2, 2]
    assert cmd.window_idx.tolist() == [0, 0, 0]


def test_periodic_replan_occurs_25_steps_after_first_closed_loop_call() -> None:
    cmd, generated_histories = _scheduler_command()
    cmd._replan(torch.arange(cmd.num_envs), bootstrap=True)

    for step in range(2):
        _set_measured_frame(cmd, [100.0 + step, 200.0 + step, 300.0 + step])
        cmd._advance_measured_history_and_replan()
    assert len(generated_histories) == 2

    for step in range(24):
        value = float(step + 2)
        _set_measured_frame(cmd, [100.0 + value, 200.0 + value, 300.0 + value])
        cmd._advance_measured_history_and_replan()
    assert len(generated_histories) == 2
    assert cmd.window_idx.tolist() == [24, 24, 24]

    _set_measured_frame(cmd, [126.0, 226.0, 326.0])
    cmd._advance_measured_history_and_replan()
    assert len(generated_histories) == 3
    assert cmd._closed_loop_replan_count.tolist() == [2, 2, 2]
    assert cmd._last_replan_interval_steps.tolist() == [25, 25, 25]
    assert generated_histories[-1].squeeze(-1).tolist() == [
        [125.0, 126.0],
        [225.0, 226.0],
        [325.0, 326.0],
    ]


def test_nonbootstrap_replan_rejects_seed_measured_mixture() -> None:
    cmd, generated_histories = _scheduler_command()
    cmd._measured_history_valid_count[:] = torch.tensor([2, 1, 2])

    with pytest.raises(RuntimeError, match="fully measured history"):
        cmd._replan(torch.tensor([0, 1, 2]), bootstrap=False)

    assert generated_histories == []
    assert cmd._closed_loop_replan_count.tolist() == [0, 0, 0]


def test_partial_reset_keeps_other_environment_history_and_counters() -> None:
    cmd, _ = _scheduler_command()
    cmd._history[:, :, 0] = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    cmd._measured_history_valid_count[:] = 2
    cmd._has_closed_loop_replan[:] = True
    cmd._bootstrap_replan_count[:] = 1
    cmd._closed_loop_replan_count[:] = torch.tensor([4, 5, 6])
    cmd._last_replan_interval_steps[:] = 25
    cmd.window_idx[:] = torch.tensor([7, 8, 9])

    cmd._reset_closed_loop_tracking(torch.tensor([1]))

    assert cmd._history.squeeze(-1).tolist() == [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
    assert cmd._measured_history_valid_count.tolist() == [2, 0, 2]
    assert cmd._has_closed_loop_replan.tolist() == [True, False, True]
    assert cmd._bootstrap_replan_count.tolist() == [1, 0, 1]
    assert cmd._closed_loop_replan_count.tolist() == [4, 0, 6]
    assert cmd._last_replan_interval_steps.tolist() == [25, 0, 25]
    assert cmd.window_idx.tolist() == [7, 0, 9]


def test_replan_metrics_include_configured_two_denoise_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    cmd, _ = _scheduler_command(num_envs=2)
    cmd.metrics = {}
    cmd.gen_cfg = SimpleNamespace(
        denoise_steps=2,
        heading_error_speed_threshold=0.0,
        heading_reward_epsilon=1e-6,
    )
    cmd._use_sim_terrain_scan = False
    cmd._bootstrap_replan_count[:] = torch.tensor([1, 1])
    cmd._closed_loop_replan_count[:] = torch.tensor([2, 3])
    cmd._measured_history_valid_count[:] = 2
    cmd.window_idx[:] = torch.tensor([4, 5])
    cmd._last_replan_interval_steps[:] = torch.tensor([25, 2])
    cmd._headings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cmd._env = SimpleNamespace(
        simulator=SimpleNamespace(robot_root_states=torch.zeros(2, 13)),
    )

    def _skip_parent_metrics(_self) -> None:
        return None

    monkeypatch.setattr(
        "holosoma.managers.command.terms.wbt.MotionCommand.update_metrics", _skip_parent_metrics
    )
    cmd.update_metrics()

    assert cmd.metrics["generator/bootstrap_replan_count"].tolist() == [1.0, 1.0]
    assert cmd.metrics["generator/closed_loop_replan_count"].tolist() == [2.0, 3.0]
    assert cmd.metrics["generator/measured_history_valid_count"].tolist() == [2.0, 2.0]
    assert cmd.metrics["generator/window_index"].tolist() == [4.0, 5.0]
    assert cmd.metrics["generator/observed_replan_interval_steps"].tolist() == [25.0, 2.0]
    assert cmd.metrics["generator/denoise_steps"].tolist() == [2.0, 2.0]


def test_generated_window_calls_two_step_denoising() -> None:
    cmd = GeneratedMotionCommand.__new__(GeneratedMotionCommand)
    layout = FeatureLayout(
        joint_names=("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"),
        body_names=("torso_link",),
        bone_pairs=(),
    )
    horizon = 2
    cmd.layout = layout
    cmd._history = torch.zeros(1, 2, layout.dim)
    cmd._history[..., layout.root_quat_slice.start] = 1.0
    cmd._headings = torch.tensor([[1.0, 0.0]])
    cmd._use_sim_terrain_scan = False
    cmd._terrain_zeros = torch.zeros(1, 0)
    cmd.gen_cfg = SimpleNamespace(past_noise_std=0.0, denoise_steps=2)
    cmd._env = SimpleNamespace(dt=0.02)
    cmd._sim_j_from_lay = torch.arange(layout.num_joints)
    cmd._tracked_b_from_lay = torch.arange(layout.num_bodies)
    cmd._waist_lay_idx = [0, 1, 2]

    cmd._win_root_pos = torch.zeros(1, horizon, 3)
    cmd._win_root_quat = torch.zeros(1, horizon, 4)
    cmd._win_root_lin_vel = torch.zeros(1, horizon, 3)
    cmd._win_root_ang_vel = torch.zeros(1, horizon, 3)
    cmd._win_joint_pos = torch.zeros(1, horizon, layout.num_joints)
    cmd._win_joint_vel = torch.zeros_like(cmd._win_joint_pos)
    cmd._win_body_pos = torch.zeros(1, horizon, layout.num_bodies, 3)
    cmd._win_body_lin_vel = torch.zeros_like(cmd._win_body_pos)
    cmd._win_ref_quat = torch.zeros(1, horizon, 4)

    calls: list[tuple[int, bool, torch.Tensor]] = []

    class _RecordingGenerator:
        def generate(self, inp, num_steps: int, deterministic: bool) -> MotionGeneratorOutput:
            calls.append((num_steps, deterministic, inp.past_motion.clone()))
            root_quat = torch.zeros(1, horizon, 4)
            root_quat[..., 0] = 1.0
            return MotionGeneratorOutput(
                root_pos=torch.zeros(1, horizon, 3),
                root_quat=root_quat,
                joint_pos=torch.zeros(1, horizon, layout.num_joints),
                body_pos=torch.zeros(1, horizon, layout.num_bodies, 3),
                features=torch.zeros(1, horizon, layout.dim),
            )

    cmd.generator = _RecordingGenerator()
    cmd._generate_window(torch.tensor([0]))

    assert len(calls) == 1
    assert calls[0][0] == 2
    assert calls[0][1] is False
    torch.testing.assert_close(calls[0][2], cmd._history)
