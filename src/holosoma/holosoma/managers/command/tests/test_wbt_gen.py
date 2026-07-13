"""Measured-history and replan scheduling tests for generated WBT commands."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from holosoma.config_values.wbt.g1.command_gen import gen_motion_config
from holosoma.config_values.wbt.g1.command_gen_terrain import terrain_gen_motion_config
from holosoma.managers.command.terms.wbt_gen import (
    GeneratedMotionCommand,
    _body_origin_penetration_proxy,
    _validate_nonnegative_finite,
    derive_per_env_sampling_seed,
)
from holosoma.motion_gen.features import FeatureLayout
from holosoma.motion_gen.sampling import MotionGeneratorOutput
from holosoma.motion_gen.terrain import ScanGrid


def _scheduler_command(
    num_envs: int = 3,
    *,
    require_fully_measured_history: bool = True,
) -> tuple[GeneratedMotionCommand, list[torch.Tensor]]:
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
    cmd._replan_ordinal = torch.zeros(num_envs, dtype=torch.long)
    cmd.gen_cfg = SimpleNamespace(
        require_fully_measured_history=require_fully_measured_history
    )

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


def test_first_closed_loop_replan_stays_at_2hz_and_uses_measured_frames() -> None:
    cmd, generated_histories = _scheduler_command()
    env_ids = torch.arange(cmd.num_envs)
    cmd._reset_closed_loop_tracking(env_ids)
    cmd._replan(env_ids, bootstrap=True)
    assert cmd._bootstrap_replan_count.tolist() == [1, 1, 1]
    assert generated_histories[0].squeeze(-1).tolist() == [[-1.0, -1.0]] * 3

    for step in range(25):
        _set_measured_frame(cmd, [10.0 + step, 20.0 + step, 30.0 + step])
        cmd._advance_measured_history_and_replan()
        if step < 24:
            assert len(generated_histories) == 1

    assert len(generated_histories) == 2
    assert generated_histories[1].squeeze(-1).tolist() == [
        [33.0, 34.0],
        [43.0, 44.0],
        [53.0, 54.0],
    ]
    assert cmd._measured_history_valid_count.tolist() == [2, 2, 2]
    assert cmd._closed_loop_replan_count.tolist() == [1, 1, 1]
    assert cmd._last_replan_interval_steps.tolist() == [25, 25, 25]
    assert cmd.window_idx.tolist() == [0, 0, 0]


def test_periodic_replan_occurs_25_steps_after_first_closed_loop_call() -> None:
    cmd, generated_histories = _scheduler_command()
    cmd._replan(torch.arange(cmd.num_envs), bootstrap=True)

    for step in range(25):
        _set_measured_frame(cmd, [100.0 + step, 200.0 + step, 300.0 + step])
        cmd._advance_measured_history_and_replan()
    assert len(generated_histories) == 2

    for step in range(24):
        value = float(step + 25)
        _set_measured_frame(cmd, [100.0 + value, 200.0 + value, 300.0 + value])
        cmd._advance_measured_history_and_replan()
    assert len(generated_histories) == 2
    assert cmd.window_idx.tolist() == [24, 24, 24]

    _set_measured_frame(cmd, [149.0, 249.0, 349.0])
    cmd._advance_measured_history_and_replan()
    assert len(generated_histories) == 3
    assert cmd._closed_loop_replan_count.tolist() == [2, 2, 2]
    assert cmd._last_replan_interval_steps.tolist() == [25, 25, 25]
    assert generated_histories[-1].squeeze(-1).tolist() == [
        [148.0, 149.0],
        [248.0, 249.0],
        [348.0, 349.0],
    ]


def test_nonbootstrap_replan_rejects_seed_measured_mixture() -> None:
    cmd, generated_histories = _scheduler_command()
    cmd._measured_history_valid_count[:] = torch.tensor([2, 1, 2])

    with pytest.raises(RuntimeError, match="fully measured history"):
        cmd._replan(torch.tensor([0, 1, 2]), bootstrap=False)

    assert generated_histories == []
    assert cmd._closed_loop_replan_count.tolist() == [0, 0, 0]


def test_flat_legacy_schedule_keeps_25_step_clock_without_strict_guard() -> None:
    cmd, generated_histories = _scheduler_command(require_fully_measured_history=False)
    env_ids = torch.arange(cmd.num_envs)
    cmd._replan(env_ids, bootstrap=True)

    for step in range(24):
        _set_measured_frame(cmd, [float(step)] * cmd.num_envs)
        cmd._advance_measured_history_and_replan()
    assert len(generated_histories) == 1
    assert cmd.window_idx.tolist() == [24, 24, 24]

    _set_measured_frame(cmd, [24.0] * cmd.num_envs)
    cmd._advance_measured_history_and_replan()
    assert len(generated_histories) == 2
    assert cmd._last_replan_interval_steps.tolist() == [25, 25, 25]

    # The flat-compatible option controls only the assertion.  It does not
    # change the periodic clock and permits legacy direct non-bootstrap calls.
    cmd._measured_history_valid_count.zero_()
    cmd._replan(env_ids, bootstrap=False)
    assert len(generated_histories) == 3


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


def test_generated_window_calls_two_step_denoising_and_preserves_legacy_seed() -> None:
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
    cmd.gen_cfg = SimpleNamespace(
        past_noise_std=0.1,
        denoise_steps=2,
        deterministic_sampling=True,
        sampling_seed=17,
    )
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

    calls: list[tuple[int, bool, int, torch.Tensor, torch.Tensor | None]] = []

    class _RecordingGenerator:
        def generate(
            self,
            inp,
            num_steps: int,
            deterministic: bool,
            seed: int,
            per_sample_seeds: torch.Tensor | None,
        ) -> MotionGeneratorOutput:
            calls.append((num_steps, deterministic, seed, inp.past_motion.clone(), per_sample_seeds))
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
    cmd._generate_window(torch.tensor([0]))

    assert len(calls) == 2
    assert calls[0][0] == 2
    assert calls[0][1] is True
    assert calls[0][2] == 17
    torch.testing.assert_close(calls[0][3], calls[1][3])
    assert calls[0][4] is None
    assert not torch.equal(calls[0][3], cmd._history)

    cmd.gen_cfg.deterministic_per_env_sampling = True
    cmd.device = torch.device("cpu")
    cmd._episode_index = torch.tensor([3])
    cmd._replan_ordinal = torch.tensor([4])
    cmd._generate_window(torch.tensor([0]))
    torch.testing.assert_close(
        calls[2][4],
        torch.tensor([derive_per_env_sampling_seed(17, env_id=0, episode=3, replan=4)]),
    )


def test_per_env_seed_changes_for_each_identity_axis() -> None:
    baseline = derive_per_env_sampling_seed(17, env_id=2, episode=3, replan=4)
    seeds = {
        baseline,
        derive_per_env_sampling_seed(18, env_id=2, episode=3, replan=4),
        derive_per_env_sampling_seed(17, env_id=3, episode=3, replan=4),
        derive_per_env_sampling_seed(17, env_id=2, episode=4, replan=4),
        derive_per_env_sampling_seed(17, env_id=2, episode=3, replan=5),
    }
    assert len(seeds) == 5
    assert all(0 <= seed <= (1 << 63) - 1 for seed in seeds)


def test_deterministic_sampling_counter_reset_is_per_environment() -> None:
    cmd = GeneratedMotionCommand.__new__(GeneratedMotionCommand)
    cmd.num_envs = 3
    cmd.device = torch.device("cpu")
    cmd._episode_index = torch.tensor([4, 5, 6])
    cmd._replan_ordinal = torch.tensor([7, 8, 9])

    cmd.reset_deterministic_sampling_counters(torch.tensor([1]))

    assert cmd._episode_index.tolist() == [4, -1, 6]
    assert cmd._replan_ordinal.tolist() == [7, 0, 9]


def test_body_origin_penetration_proxy_uses_scan_anchor_yaw_and_validity() -> None:
    grid = ScanGrid(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1, spacing=0.1)
    scan = torch.full((1, grid.dim), 0.3)
    anchor_xy = torch.tensor([[10.0, 20.0]])
    anchor_yaw = torch.tensor([torch.pi / 2])
    # At yaw=pi/2, local +x maps to world +y.  The final point is outside
    # the scan and therefore must not be counted as zero penetration.
    body_pos_w = torch.tensor(
        [[[10.0, 20.0, 0.2], [10.0, 20.05, 0.4], [10.0, 20.2, 0.1]]]
    )

    penetration, valid = _body_origin_penetration_proxy(
        scan,
        body_pos_w,
        anchor_xy,
        anchor_yaw,
        grid,
    )

    torch.testing.assert_close(penetration, torch.tensor([[0.1, 0.0, 0.0]]))
    assert valid.tolist() == [[True, True, False]]


@pytest.mark.parametrize("value", [-0.01, float("inf"), float("nan")])
def test_body_origin_metric_threshold_validation_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        _validate_nonnegative_finite(value, "threshold")


def test_terrain_preset_exposes_body_origin_proxy_thresholds() -> None:
    assert not gen_motion_config.require_fully_measured_history
    assert terrain_gen_motion_config.require_fully_measured_history
    assert terrain_gen_motion_config.body_origin_penetration_threshold_m == 0.02
    assert terrain_gen_motion_config.body_origin_correction_min_improvement_m == 0.01


def test_terrain_metrics_report_tracker_correction_proxy_and_existing_contacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cmd, _ = _scheduler_command(num_envs=2)
    grid = ScanGrid(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1, spacing=0.1)
    cmd.metrics = {}
    cmd.gen_cfg = SimpleNamespace(
        denoise_steps=2,
        heading_error_speed_threshold=0.0,
        heading_reward_epsilon=1e-6,
        body_origin_penetration_threshold_m=0.05,
        body_origin_correction_min_improvement_m=0.02,
    )
    cmd._use_sim_terrain_scan = True
    cmd._scan_grid = grid
    cmd._horizon = 1
    cmd._arange = torch.arange(2)
    cmd._seed_mode = False
    cmd.window_idx.zero_()
    cmd._headings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    reference_body_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.20], [0.05, 0.0, 0.40]],
            [[10.0, 0.0, 0.29], [10.0, 0.05, 0.40]],
        ]
    )
    robot_body_pos = reference_body_pos.clone()
    robot_body_pos[0, 0, 2] = 0.26
    robot_body_pos[1, 0, 2] = 0.28
    cmd._win_body_pos = reference_body_pos.unsqueeze(1)
    cmd.tracked_body_indexes = torch.tensor([0, 1])

    robot_root_states = torch.zeros(2, 13)
    robot_root_states[:, :2] = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    robot_root_states[0, 6] = 1.0  # xyzw identity
    robot_root_states[1, 5:7] = torch.tensor([2**-0.5, 2**-0.5])  # yaw=pi/2
    cmd._env = SimpleNamespace(
        simulator=SimpleNamespace(
            robot_root_states=robot_root_states,
            _rigid_body_pos=robot_body_pos,
        )
    )
    cmd._terrain_state = SimpleNamespace(
        local_height_scan=torch.full((2, grid.dim), 0.3),
        local_height_scan_valid=torch.tensor([True, True]),
        local_height_scan_root_xy=torch.tensor([[0.0, 0.0], [10.0, 0.0]]),
        local_height_scan_root_yaw=torch.tensor([0.0, torch.pi / 2]),
    )

    class _ExistingContactTerm:
        def __call__(self, _env) -> torch.Tensor:
            return torch.tensor([0, 2])

    cmd._undesired_contacts_term = _ExistingContactTerm()

    def _skip_parent_metrics(_self) -> None:
        return None

    monkeypatch.setattr(
        "holosoma.managers.command.terms.wbt.MotionCommand.update_metrics",
        _skip_parent_metrics,
    )
    cmd.update_metrics()

    torch.testing.assert_close(
        cmd.metrics["terrain/reference_body_origin_penetration_max_m"],
        torch.tensor([0.10, 0.01]),
    )
    torch.testing.assert_close(
        cmd.metrics["terrain/robot_body_origin_penetration_max_m"],
        torch.tensor([0.04, 0.02]),
    )
    torch.testing.assert_close(
        cmd.metrics["terrain/tracker_body_origin_penetration_improvement_m"],
        torch.tensor([0.06, -0.01]),
    )
    torch.testing.assert_close(
        cmd.metrics["terrain/tracker_body_origin_correction_proxy_m"],
        torch.tensor([0.06, 0.0]),
    )
    assert cmd.metrics["terrain/tracker_body_origin_correction_case"].tolist() == [1.0, 0.0]
    assert cmd.metrics["terrain/body_origin_paired_scan_coverage"].tolist() == [1.0, 1.0]
    assert cmd.metrics["terrain/undesired_contact_body_count"].tolist() == [0.0, 2.0]
    assert cmd.metrics["terrain/undesired_contact_any"].tolist() == [0.0, 1.0]
