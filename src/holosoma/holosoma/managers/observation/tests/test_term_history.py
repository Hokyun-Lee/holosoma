from __future__ import annotations

from types import SimpleNamespace

import torch
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager


def _term(name: str, *, history_length: int | None = None) -> ObsTermCfg:
    return ObsTermCfg(
        func=f"holosoma.managers.observation.terms.wbt:{name}",
        history_length=history_length,
    )


class _FakeWbtEnv:
    def __init__(self):
        self.num_envs = 2
        self.device = "cpu"
        self.logger = None
        self.base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(self.num_envs, 1)
        self.default_dof_pos = torch.zeros(self.num_envs, 2)
        self.simulator = SimpleNamespace(
            robot_root_states=torch.zeros(self.num_envs, 13),
            dof_pos=torch.zeros(self.num_envs, 2),
            dof_vel=torch.zeros(self.num_envs, 2),
        )
        self.action_manager = SimpleNamespace(action=torch.zeros(self.num_envs, 2))

    def set_step(self, step: float) -> None:
        env_offset = torch.tensor([0.0, 10.0])
        value = step + env_offset
        self.simulator.robot_root_states[:, 10] = value
        self.simulator.robot_root_states[:, 11] = value + 0.1
        self.simulator.robot_root_states[:, 12] = value + 0.2
        self.simulator.dof_pos[:, 0] = value
        self.simulator.dof_pos[:, 1] = value + 0.5
        self.action_manager.action[:, 0] = 100.0 + value
        self.action_manager.action[:, 1] = 200.0 + value


def _selective_manager(env: _FakeWbtEnv) -> ObservationManager:
    cfg = ObservationManagerCfg(
        groups={
            "actor_obs": ObsGroupCfg(
                concatenate=True,
                history_length=1,
                terms={
                    "actions": _term("actions", history_length=1),
                    "base_ang_vel": _term("base_ang_vel", history_length=3),
                    "dof_pos": _term("dof_pos", history_length=3),
                    "tracker_projected_gravity": _term("projected_gravity", history_length=3),
                },
            )
        }
    )
    return ObservationManager(cfg, env, env.device)


def test_selective_history_keeps_current_prefix_and_appends_past_frames():
    env = _FakeWbtEnv()
    manager = _selective_manager(env)
    env.set_step(1.0)
    first_obs = manager.compute_group("actor_obs")
    torch.testing.assert_close(first_obs[:, 10:], torch.zeros(2, 16))
    for step in (2.0, 3.0):
        env.set_step(step)
        obs = manager.compute_group("actor_obs")

    assert obs.shape == (2, 26)
    assert manager.get_obs_dims()["actor_obs"] == 26
    # Current-frame prefix, sorted as actions/base_ang_vel/dof_pos/gravity.
    torch.testing.assert_close(obs[0, :2], torch.tensor([103.0, 203.0]))
    torch.testing.assert_close(obs[0, 2:5], torch.tensor([3.0, 3.1, 3.2]))
    torch.testing.assert_close(obs[0, 5:7], torch.tensor([3.0, 3.5]))
    torch.testing.assert_close(obs[0, 7:10], torch.tensor([0.0, 0.0, -1.0]))
    # History suffix contains only t-2,t-1, oldest to newest, per term.
    torch.testing.assert_close(obs[0, 10:16], torch.tensor([1.0, 1.1, 1.2, 2.0, 2.1, 2.2]))
    torch.testing.assert_close(obs[0, 16:20], torch.tensor([1.0, 1.5, 2.0, 2.5]))
    torch.testing.assert_close(obs[0, 20:26], torch.tensor([0.0, 0.0, -1.0, 0.0, 0.0, -1.0]))
    assert manager.get_normalizer_expansion_source_indices("actor_obs") == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        2,
        3,
        4,
        2,
        3,
        4,
        5,
        6,
        5,
        6,
        7,
        8,
        9,
        7,
        8,
        9,
    ]


def test_selective_history_subset_reset_does_not_mix_environments():
    env = _FakeWbtEnv()
    manager = _selective_manager(env)
    for step in (1.0, 2.0, 3.0):
        env.set_step(step)
        manager.compute_group("actor_obs")

    manager.reset(torch.tensor([0]))
    env.set_step(4.0)
    obs = manager.compute_group("actor_obs")

    torch.testing.assert_close(obs[0, 10:], torch.zeros(16))
    torch.testing.assert_close(obs[1, 10:16], torch.tensor([12.0, 12.1, 12.2, 13.0, 13.1, 13.2]))
    torch.testing.assert_close(obs[1, 16:20], torch.tensor([12.0, 12.5, 13.0, 13.5]))


def test_final_observation_does_not_advance_history():
    env = _FakeWbtEnv()
    manager = _selective_manager(env)
    for step in (1.0, 2.0, 3.0):
        env.set_step(step)
        manager.compute_group("actor_obs")

    env.set_step(99.0)
    final_obs = manager.compute_group("actor_obs", modify_history=False)
    env.set_step(4.0)
    next_obs = manager.compute_group("actor_obs")

    torch.testing.assert_close(final_obs[0, 10:16], torch.tensor([2.0, 2.1, 2.2, 3.0, 3.1, 3.2]))
    torch.testing.assert_close(next_obs[0, 10:16], torch.tensor([2.0, 2.1, 2.2, 3.0, 3.1, 3.2]))


def test_group_history_without_term_override_retains_legacy_layout():
    env = _FakeWbtEnv()
    cfg = ObservationManagerCfg(
        groups={
            "actor_obs": ObsGroupCfg(
                concatenate=True,
                history_length=2,
                terms={"actions": _term("actions"), "dof_pos": _term("dof_pos")},
            )
        }
    )
    manager = ObservationManager(cfg, env, env.device)
    env.set_step(1.0)
    manager.compute_group("actor_obs")
    env.set_step(2.0)
    obs = manager.compute_group("actor_obs")

    # Legacy layout is per-term [past,current], then next term [past,current].
    torch.testing.assert_close(obs[0], torch.tensor([101.0, 201.0, 102.0, 202.0, 1.0, 1.5, 2.0, 2.5]))
