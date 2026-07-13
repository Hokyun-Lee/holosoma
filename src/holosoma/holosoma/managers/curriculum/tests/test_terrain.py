from __future__ import annotations

from copy import deepcopy
from types import MethodType, SimpleNamespace

import pytest
import torch

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.curriculum.manager import CurriculumManager
from holosoma.managers.curriculum.terms.terrain import TerrainCurriculum


class _FakeTerrainState:
    def __init__(
        self,
        levels: torch.Tensor | None = None,
        type_ids: torch.Tensor | None = None,
    ):
        self.terrain_levels = levels.clone() if levels is not None else torch.tensor([1, 1, 1, 1], dtype=torch.long)
        self.terrain_type_ids = (
            type_ids.clone() if type_ids is not None else torch.tensor([0, 1, 2, 3], dtype=torch.long)
        )
        self.terrain_type_names = ("flat", "box", "stair", "hurdle")
        self.num_curriculum_levels = 4
        self.origin_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def set_curriculum_origins(self, env_ids: torch.Tensor, levels: torch.Tensor) -> None:
        self.origin_calls.append((env_ids.clone(), levels.clone()))
        self.terrain_levels[env_ids] = levels


class _FakeEnv:
    def __init__(self, terrain_state: _FakeTerrainState | None = None):
        self.num_envs = 4
        self.device = "cpu"
        self.max_episode_length = 10
        self.log_dict: dict[str, torch.Tensor] = {}
        self._update_tasks_before_termination = True
        self.terrain_state = terrain_state or _FakeTerrainState()
        self.terrain_manager = SimpleNamespace(get_state=lambda _: self.terrain_state)
        self.termination_manager = SimpleNamespace(
            terminated=torch.zeros(self.num_envs, dtype=torch.bool),
            time_outs=torch.zeros(self.num_envs, dtype=torch.bool),
        )
        self.simulator = SimpleNamespace(robot_root_states=torch.zeros(self.num_envs, 13, dtype=torch.float32))
        self.target_heading_w = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=torch.float32,
        )
        motion_command = SimpleNamespace(target_heading_w=self.target_heading_w)
        self.command_manager = SimpleNamespace(
            get_state=lambda name: motion_command if name == "motion_command" else None
        )


def _make_term(env: _FakeEnv, **params) -> TerrainCurriculum:
    defaults = {
        "enabled": True,
        "initial_level": 1,
        "success_min_episode_fraction": 0.5,
        "promote_success_streak": 2,
        "demote_failure_streak": 2,
        "skip_first_episode": False,
    }
    defaults.update(params)
    term = TerrainCurriculum(SimpleNamespace(params=defaults), env)
    term.setup()
    return term


def _set_outcome(env: _FakeEnv, env_id: int, *, failure: bool, timeout: bool) -> None:
    env.termination_manager.terminated.zero_()
    env.termination_manager.time_outs.zero_()
    env.termination_manager.terminated[env_id] = failure
    env.termination_manager.time_outs[env_id] = timeout


def test_base_task_before_reset_order_precedes_command_and_regular_curriculum_reset():
    events: list[str] = []
    env = BaseTask.__new__(BaseTask)
    env.simulator = SimpleNamespace(
        on_episode_end=lambda _: events.append("episode_end"),
        on_episode_start=lambda _: events.append("episode_start"),
    )
    env.observation_manager = SimpleNamespace(reset=lambda _: events.append("observation"))
    env._pending_episode_lengths = torch.zeros(1, dtype=torch.long)
    env._pending_episode_update_mask = torch.zeros(1, dtype=torch.bool)
    env.episode_length_buf = torch.ones(1, dtype=torch.long)
    env._reset_envs_idx_impl = MethodType(lambda _self, _ids, _states, _buf: events.append("reset_impl"), env)
    env.randomization_manager = SimpleNamespace(reset=lambda _: events.append("randomization"))
    env.action_manager = SimpleNamespace(reset=lambda _: events.append("action"))
    env.command_manager = SimpleNamespace(reset=lambda _: events.append("command"))
    env.curriculum_manager = SimpleNamespace(
        before_reset=lambda _: events.append("curriculum_before"),
        reset=lambda _: events.append("curriculum_reset"),
    )
    env.termination_manager = SimpleNamespace(reset=lambda _: events.append("termination"))
    env.reset_manager = SimpleNamespace(reset_scene=lambda _: events.append("reset_scene"))

    env.reset_envs_idx(torch.tensor([0]))

    assert events.index("reset_impl") < events.index("curriculum_before")
    assert events.index("curriculum_before") < events.index("command")
    assert events.index("command") < events.index("curriculum_reset")


def test_curriculum_manager_dispatches_before_reset_to_stateful_terms():
    calls: list[torch.Tensor] = []
    manager = CurriculumManager.__new__(CurriculumManager)
    manager._class_instances = [SimpleNamespace(before_reset=lambda ids: calls.append(ids.clone()))]
    ids = torch.tensor([1, 3])

    manager.before_reset(ids)

    assert len(calls) == 1
    torch.testing.assert_close(calls[0], ids)


def test_success_promotion_failure_demotion_clamp_and_subset_isolation():
    env = _FakeEnv()
    term = _make_term(env)

    # Two eligible successes promote only env 0 from level 1 to 2.
    for _ in range(2):
        term.actual_episode_steps[0] = term._success_min_steps
        term.success_eligible[0] = True
        _set_outcome(env, 0, failure=False, timeout=True)
        term.before_reset(torch.tensor([0]))
    assert env.terrain_state.terrain_levels.tolist() == [2, 1, 1, 1]
    assert term.success_streaks.tolist() == [0, 0, 0, 0]

    # Two failures demote env 1; other environments retain level and streak state.
    term.success_streaks[2] = 1
    for _ in range(2):
        _set_outcome(env, 1, failure=True, timeout=False)
        term.before_reset(torch.tensor([1]))
    assert env.terrain_state.terrain_levels.tolist() == [2, 0, 1, 1]
    assert term.success_streaks[2].item() == 1

    # Repeated failures at the minimum level stay clamped without affecting peers.
    for _ in range(2):
        _set_outcome(env, 1, failure=True, timeout=False)
        term.before_reset(torch.tensor([1]))
    assert env.terrain_state.terrain_levels.tolist() == [2, 0, 1, 1]

    # Promotion also clamps at the terrain state's maximum level.
    term._set_origins(torch.tensor([3]), torch.tensor([3]))
    for _ in range(2):
        term.actual_episode_steps[3] = term._success_min_steps
        term.success_eligible[3] = True
        _set_outcome(env, 3, failure=False, timeout=True)
        term.before_reset(torch.tensor([3]))
    assert env.terrain_state.terrain_levels.tolist() == [2, 0, 1, 3]
    assert term.type_episode_counts.tolist() == [2, 4, 0, 2]
    assert term.type_success_counts.tolist() == [2, 0, 0, 2]


def test_initial_and_manual_reset_without_outcome_are_not_counted():
    env = _FakeEnv()
    term = _make_term(env, skip_first_episode=True)
    term.actual_episode_steps[2] = 4
    term.success_streaks[2] = 1

    term.before_reset(torch.tensor([2]))

    assert term.type_episode_counts.sum().item() == 0
    assert term.type_success_counts.sum().item() == 0
    assert term.success_streaks[2].item() == 1
    assert term.actual_episode_steps[2].item() == 0
    assert not term.outcome_eligible[2]


def test_randomized_first_timeout_and_short_timeout_are_not_successes():
    env = _FakeEnv()
    env._update_tasks_before_termination = False
    term = _make_term(
        env,
        success_min_episode_fraction=0.8,
        promote_success_streak=1,
        skip_first_episode=True,
    )

    # A randomized episode_length_buf creates an initial fragment, not a full
    # episode.  Even a long fragment is explicitly ineligible as an outcome.
    term.actual_episode_steps[0] = term._success_min_steps
    term.success_eligible[0] = True
    _set_outcome(env, 0, failure=False, timeout=True)
    term.before_reset(torch.tensor([0]))
    assert env.terrain_state.terrain_levels[0].item() == 1
    assert term.type_episode_counts[0].item() == 0
    assert not term.success_eligible[0]
    assert term.outcome_eligible[0]

    # The immediate post-reset curriculum step belongs to the terminal frame
    # and must not leak one step into the next episode.
    term.step()
    assert term.actual_episode_steps[0].item() == 0

    # A later timeout is still ignored until the actual-step threshold is met.
    term.actual_episode_steps[0] = term._success_min_steps - 2
    _set_outcome(env, 0, failure=False, timeout=True)
    term.before_reset(torch.tensor([0]))
    assert env.terrain_state.terrain_levels[0].item() == 1
    assert term.type_episode_counts[0].item() == 0

    # The terminal step makes this episode exactly eligible.
    term.actual_episode_steps[0] = term._success_min_steps - 1
    _set_outcome(env, 0, failure=False, timeout=True)
    term.before_reset(torch.tensor([0]))
    assert env.terrain_state.terrain_levels[0].item() == 2
    assert term.type_episode_counts[0].item() == 1


def test_type_rates_level_and_sampling_balance_metrics_are_scalar_tensors():
    terrain = _FakeTerrainState(type_ids=torch.tensor([0, 1, 1, 2], dtype=torch.long))
    env = _FakeEnv(terrain)
    term = _make_term(env, promote_success_streak=4)

    term.actual_episode_steps[:2] = term._success_min_steps
    term.success_eligible[:2] = True
    env.termination_manager.time_outs[0] = True
    env.termination_manager.terminated[1] = True
    term.before_reset(torch.tensor([0, 1]))
    term.step()

    flat = "terrain_curriculum/type/0_flat"
    box = "terrain_curriculum/type/1_box"
    stair = "terrain_curriculum/type/2_stair"
    assert env.log_dict[f"{flat}/success_rate"].item() == 1.0
    assert env.log_dict[f"{box}/success_rate"].item() == 0.0
    assert env.log_dict[f"{stair}/success_rate"].item() == 0.0
    assert env.log_dict[f"{flat}/env_fraction"].item() == 0.25
    assert env.log_dict[f"{box}/env_fraction"].item() == 0.5
    assert env.log_dict["terrain_curriculum/type_env_fraction_min"].item() == 0.0
    assert env.log_dict["terrain_curriculum/type_env_fraction_max"].item() == 0.5

    level_fraction_keys = [f"terrain_curriculum/level/{level}/env_fraction" for level in range(4)]
    assert sum(env.log_dict[key].item() for key in level_fraction_keys) == pytest.approx(1.0)
    assert all(torch.is_tensor(value) and value.ndim == 0 for value in env.log_dict.values())


def test_progress_gated_success_requires_crossing_and_logs_type_level_rates():
    env = _FakeEnv()
    term = _make_term(
        env,
        crossing_distance_m=1.5,
        promote_success_streak=1,
        demote_failure_streak=2,
    )
    term.reset(torch.arange(env.num_envs))

    # A full timeout after only one metre is survival, but not a crossing
    # success and therefore must not promote the terrain level.
    env.simulator.robot_root_states[0, :2] = torch.tensor([1.0, 0.0])
    term.step()
    term.actual_episode_steps[0] = term._success_min_steps
    term.success_eligible[0] = True
    _set_outcome(env, 0, failure=False, timeout=True)
    term.before_reset(torch.tensor([0]))
    assert env.terrain_state.terrain_levels[0].item() == 1
    assert term.type_episode_counts[0].item() == 1
    assert term.type_survival_counts[0].item() == 1
    assert term.type_crossing_counts[0].item() == 0
    assert term.type_success_counts[0].item() == 0
    assert term.type_failure_counts[0].item() == 1

    # Simulate a new command heading and reset at a new pose, then travel
    # farther than 1.5 m along that captured heading.
    env.simulator.robot_root_states[0, :2] = torch.tensor([4.0, -2.0])
    env.target_heading_w[0] = torch.tensor([0.0, -1.0])
    term.reset(torch.tensor([0]))
    env.simulator.robot_root_states[0, :2] = torch.tensor([4.0, -3.6])
    term.step()
    term.actual_episode_steps[0] = term._success_min_steps
    term.success_eligible[0] = True
    _set_outcome(env, 0, failure=False, timeout=True)
    term.before_reset(torch.tensor([0]))

    assert env.terrain_state.terrain_levels[0].item() == 2
    assert term.type_episode_counts[0].item() == 2
    assert term.type_survival_counts[0].item() == 2
    assert term.type_crossing_counts[0].item() == 1
    assert term.type_success_counts[0].item() == 1
    assert term.type_failure_counts[0].item() == 1
    assert term.type_level_episode_counts[0, 1].item() == 2
    assert term.type_level_success_counts[0, 1].item() == 1
    assert env.log_dict["terrain_curriculum/type/0_flat/success_rate"].item() == 0.5
    assert env.log_dict["terrain_curriculum/type/0_flat/survival_rate"].item() == 1.0
    assert env.log_dict["terrain_curriculum/type/0_flat/crossing_rate"].item() == 0.5
    assert env.log_dict["terrain_curriculum/type/0_flat/failure_rate"].item() == 0.5


def test_progress_buffers_are_isolated_and_reset_only_for_selected_envs():
    env = _FakeEnv()
    term = _make_term(env, crossing_distance_m=1.5)
    term.reset(torch.arange(env.num_envs))
    env.simulator.robot_root_states[:, :2] = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [0.0, -4.0]])
    term.step()
    torch.testing.assert_close(term.max_episode_forward_progress_m, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    env.simulator.robot_root_states[1, :2] = torch.tensor([10.0, 10.0])
    env.target_heading_w[1] = torch.tensor([1.0, 0.0])
    term.reset(torch.tensor([1]))
    torch.testing.assert_close(term.max_episode_forward_progress_m, torch.tensor([1.0, 0.0, 3.0, 4.0]))
    torch.testing.assert_close(term.episode_start_root_xy[1], torch.tensor([10.0, 10.0]))
    torch.testing.assert_close(term.episode_target_heading_w[1], torch.tensor([1.0, 0.0]))


def test_lateral_and_backward_motion_do_not_satisfy_forward_progress_gate():
    env = _FakeEnv()
    term = _make_term(env, crossing_distance_m=1.5)
    env.target_heading_w[0] = torch.tensor([1.0, 0.0])
    term.reset(torch.tensor([0]))

    env.simulator.robot_root_states[0, :2] = torch.tensor([0.0, 3.0])
    term.step()
    env.simulator.robot_root_states[0, :2] = torch.tensor([-2.0, 0.0])
    term.step()

    assert term.max_episode_forward_progress_m[0].item() == 0.0


def test_root_state_tensor_proxy_is_supported() -> None:
    class _Proxy:
        def __init__(self, tensor: torch.Tensor):
            self.tensor = tensor

        def __getitem__(self, index):
            return self.tensor[index]

    env = _FakeEnv()
    env.simulator.robot_root_states = _Proxy(torch.zeros(env.num_envs, 13))
    term = _make_term(env, crossing_distance_m=1.5)

    assert term.episode_start_root_xy.shape == (env.num_envs, 2)


def test_state_roundtrip_is_strict_and_reapplies_restored_origins():
    env = _FakeEnv()
    term = _make_term(env)
    term._set_origins(torch.arange(4), torch.tensor([0, 1, 2, 3]))
    term.success_streaks.copy_(torch.tensor([1, 0, 1, 0]))
    term.failure_streaks.copy_(torch.tensor([0, 1, 0, 1]))
    term.actual_episode_steps.copy_(torch.tensor([5, 2, 7, 0]))
    term.success_eligible.copy_(torch.tensor([True, False, True, False]))
    term.outcome_eligible.copy_(torch.tensor([True, True, False, True]))
    term._skip_next_step_increment.copy_(torch.tensor([False, True, False, False]))
    term.type_success_counts.copy_(torch.tensor([2, 1, 0, 0]))
    term.type_episode_counts.copy_(torch.tensor([3, 4, 0, 1]))
    term.type_survival_counts.copy_(torch.tensor([2, 3, 0, 1]))
    term.type_crossing_counts.copy_(torch.tensor([1, 2, 0, 0]))
    term.type_failure_counts.copy_(torch.tensor([1, 1, 0, 0]))
    term.type_level_episode_counts[0, 0] = 3
    term.type_level_success_counts[0, 0] = 2
    term.type_level_survival_counts[0, 0] = 2
    term.type_level_crossing_counts[0, 0] = 1
    term.type_level_failure_counts[0, 0] = 1
    term.episode_start_root_xy.copy_(torch.arange(8, dtype=torch.float32).view(4, 2))
    term.episode_target_heading_w.copy_(env.target_heading_w)
    term.max_episode_forward_progress_m.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    saved = term.state_dict()

    restored_env = _FakeEnv()
    restored = _make_term(restored_env)
    restored.load_state_dict(saved)

    assert restored_env.terrain_state.terrain_levels.tolist() == [0, 1, 2, 3]
    assert restored_env.terrain_state.origin_calls[-1][0].tolist() == [0, 1, 2, 3]
    for name in (
        "success_streaks",
        "failure_streaks",
        "actual_episode_steps",
        "success_eligible",
        "outcome_eligible",
        "_skip_next_step_increment",
        "type_success_counts",
        "type_episode_counts",
        "type_survival_counts",
        "type_crossing_counts",
        "type_failure_counts",
        "type_level_episode_counts",
        "type_level_success_counts",
        "type_level_survival_counts",
        "type_level_crossing_counts",
        "type_level_failure_counts",
        "episode_start_root_xy",
        "episode_target_heading_w",
        "max_episode_forward_progress_m",
    ):
        torch.testing.assert_close(getattr(restored, name), getattr(term, name))

    before = restored.state_dict()
    bad_version = deepcopy(saved)
    bad_version["version"] = 99
    with pytest.raises(ValueError, match="version"):
        restored.load_state_dict(bad_version)
    bad_shape = deepcopy(saved)
    bad_shape["terrain_levels"] = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="shape"):
        restored.load_state_dict(bad_shape)
    torch.testing.assert_close(restored.state_dict()["terrain_levels"], before["terrain_levels"])


def test_version_one_state_migrates_without_inventing_crossing_history():
    env = _FakeEnv()
    term = _make_term(env)
    term.type_success_counts.copy_(torch.tensor([2, 1, 0, 0]))
    term.type_episode_counts.copy_(torch.tensor([3, 4, 0, 1]))
    legacy = term.state_dict()
    legacy["version"] = 1
    for key in tuple(legacy):
        if key.startswith("type_level_") or key in {
            "type_survival_counts",
            "type_crossing_counts",
            "type_failure_counts",
            "episode_start_root_xy",
            "episode_target_heading_w",
            "max_episode_forward_progress_m",
            "progress_semantics",
        }:
            legacy.pop(key)

    restored = _make_term(_FakeEnv())
    restored.load_state_dict(legacy)

    torch.testing.assert_close(restored.type_survival_counts, term.type_success_counts)
    torch.testing.assert_close(
        restored.type_failure_counts,
        term.type_episode_counts - term.type_success_counts,
    )
    assert restored.type_crossing_counts.sum().item() == 0
    assert restored.type_level_episode_counts.sum().item() == 0


def test_version_two_radial_progress_migrates_to_zero_signed_progress():
    env = _FakeEnv()
    term = _make_term(env)
    term.type_crossing_counts.copy_(torch.tensor([2, 1, 0, 0]))
    term.type_level_crossing_counts[0, 1] = 2
    legacy = term.state_dict()
    legacy["version"] = 2
    legacy["max_episode_progress_m"] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    for key in (
        "progress_semantics",
        "episode_target_heading_w",
        "max_episode_forward_progress_m",
    ):
        legacy.pop(key)

    restored_env = _FakeEnv()
    restored = _make_term(restored_env)
    restored.load_state_dict(legacy)

    assert restored.type_crossing_counts.sum().item() == 0
    assert restored.type_level_crossing_counts.sum().item() == 0
    assert restored.max_episode_forward_progress_m.sum().item() == 0.0
    torch.testing.assert_close(restored.episode_target_heading_w, restored_env.target_heading_w)


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ({"success_min_episode_fraction": 0.0}, ValueError),
        ({"promote_success_streak": 0}, ValueError),
        ({"enabled": 1}, TypeError),
        ({"initial_level": 4}, ValueError),
        ({"crossing_distance_m": -0.1}, ValueError),
    ],
)
def test_config_validation(params: dict[str, object], error: type[Exception]):
    env = _FakeEnv()
    with pytest.raises(error):
        _make_term(env, **params)
