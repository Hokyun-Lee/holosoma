from __future__ import annotations

from types import SimpleNamespace

import pytest
from holosoma.agents.base_algo.base_algo import BaseAlgo


class _FakeEnv:
    def __init__(self) -> None:
        self.loaded_states: list[dict[str, float]] = []

    def load_checkpoint_state(self, state: dict[str, float]) -> None:
        self.loaded_states.append(state)


def _make_algo() -> tuple[BaseAlgo, _FakeEnv]:
    env = _FakeEnv()
    algo = BaseAlgo(env=env, config=SimpleNamespace(), device="cpu")  # type: ignore[arg-type]
    return algo, env


def test_checkpoint_environment_state_restore_defaults_to_training_semantics() -> None:
    algo, env = _make_algo()

    algo._restore_env_state({"curriculum": 1.0})

    assert env.loaded_states == [{"curriculum": 1.0}]


def test_checkpoint_environment_state_can_be_skipped_for_eval_num_env_change() -> None:
    algo, env = _make_algo()
    algo.set_checkpoint_env_state_restore(False)

    algo._restore_env_state({"curriculum": 1.0})

    assert env.loaded_states == []


def test_checkpoint_environment_state_restore_flag_requires_bool() -> None:
    algo, _ = _make_algo()

    with pytest.raises(TypeError, match="bool"):
        algo.set_checkpoint_env_state_restore(1)  # type: ignore[arg-type]
