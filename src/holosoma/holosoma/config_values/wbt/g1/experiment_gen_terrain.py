"""Terrain variant of the closed-loop generated-motion WBT experiment.

This preset intentionally stays separate from ``g1_29dof_wbt_gen`` so the
Stage-5 flat baseline remains unchanged. Stage 6 introduced simulator height
scans; Stage 7 opts this preset into the balanced flat/box/stair/hurdle
curriculum while keeping the preset name and checkpoint migration path stable.
"""

from dataclasses import replace

from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_values import terrain
from holosoma.config_values.wbt.g1.command_gen_terrain import g1_29dof_wbt_gen_terrain_command
from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_gen_terrain_curriculum
from holosoma.config_values.wbt.g1.experiment_gen import g1_29dof_wbt_gen
from holosoma.config_values.wbt.g1.observation_gen_terrain import g1_29dof_wbt_gen_terrain_observation

_terrain_reward_terms = dict(g1_29dof_wbt_gen.reward.terms)
_terrain_reward_terms["motion_heading_alignment"] = RewardTermCfg(
    func="holosoma.managers.reward.terms.wbt:motion_heading_alignment",
    weight=1.0,
    tags=["tracking", "heading"],
)
_terrain_reward = replace(g1_29dof_wbt_gen.reward, terms=_terrain_reward_terms)

g1_29dof_wbt_gen_terrain = replace(
    g1_29dof_wbt_gen,
    # Evaluation implementation choice: four environments are the minimum
    # needed for the fixed round-robin flat/box/stair/hurdle assignment. Keep
    # the training episode duration so each reset samples a new random heading
    # and closed-loop motion instead of extending one episode to 100000 s.
    eval_overrides=replace(
        g1_29dof_wbt_gen.eval_overrides,
        num_envs=4,
        max_episode_length_s=10.0,
    ),
    training=replace(
        g1_29dof_wbt_gen.training,
        name="g1_29dof_wbt_gen_terrain_manager",
    ),
    terrain=terrain.terrain_locomotion_curriculum,
    command=g1_29dof_wbt_gen_terrain_command,
    curriculum=g1_29dof_wbt_gen_terrain_curriculum,
    observation=g1_29dof_wbt_gen_terrain_observation,
    reward=_terrain_reward,
    algo=replace(
        g1_29dof_wbt_gen.algo,
        config=replace(g1_29dof_wbt_gen.algo.config, checkpoint_load_mode="expand_input"),
    ),
)


# Tracker-from-scratch counterpart of the canonical terrain closed-loop
# experiment.  The diffusion generator is still a pretrained frozen model;
# "scratch" refers specifically to the PPO policy/value networks, observation
# normalizers, and optimizer state.  Keeping this as a distinct preset prevents
# a Stage-4/5 tracker checkpoint from being mistaken for fresh initialization.
# The 1,024-env setting is an implementation choice for a single RTX 4090, not
# a value reported by the paper.
g1_29dof_wbt_gen_terrain_scratch = replace(
    g1_29dof_wbt_gen_terrain,
    training=replace(
        g1_29dof_wbt_gen_terrain.training,
        name="g1_29dof_wbt_gen_terrain_scratch_manager",
        num_envs=1024,
        checkpoint=None,
    ),
    algo=replace(
        g1_29dof_wbt_gen_terrain.algo,
        config=replace(
            g1_29dof_wbt_gen_terrain.algo.config,
            checkpoint_load_mode="strict",
            load_optimizer=False,
            num_learning_iterations=30000,
            save_interval=500,
        ),
    ),
)


__all__ = ["g1_29dof_wbt_gen_terrain", "g1_29dof_wbt_gen_terrain_scratch"]
