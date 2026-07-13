from __future__ import annotations

from types import SimpleNamespace

import torch
from holosoma.agents.ppo.ppo import PPO


def test_evaluate_policy_honors_callback_stop_after_current_step() -> None:
    step_count = 0
    post_called = False

    def observations():
        return {
            "actor_obs": torch.zeros(1, 2),
            "critic_obs": torch.zeros(1, 3),
        }

    def env_step(_actor_state):
        nonlocal step_count
        step_count += 1
        return observations(), torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}

    algo = PPO.__new__(PPO)
    algo.device = torch.device("cpu")
    algo.num_act = 1
    algo.actor_obs_keys = ["actor_obs"]
    algo.critic_obs_keys = ["critic_obs"]
    algo.eval_callbacks = []
    algo.env = SimpleNamespace(num_envs=1, reset_all=observations, step=env_step)
    algo._create_eval_callbacks = lambda: None
    algo._pre_evaluate_policy = lambda: None
    algo.get_inference_policy = lambda: (lambda _obs: torch.zeros(1, 1))

    def stop_after_step(actor_state):
        actor_state["stop"] = True
        return actor_state

    def post_evaluate():
        nonlocal post_called
        post_called = True

    algo._post_eval_env_step = stop_after_step
    algo._post_evaluate_policy = post_evaluate
    algo.evaluate_policy(max_eval_steps=10)

    assert step_count == 1
    assert post_called
