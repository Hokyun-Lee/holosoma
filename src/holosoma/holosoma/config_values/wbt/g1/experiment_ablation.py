"""Stage-10 A--F terrain/heading ablation presets.

The presets encode architecture and update intent; checkpoint paths stay CLI
arguments. BASE means the Stage-5 ``model_12000.pt`` tracker. Presets B and C
set PPO iterations to zero, while A/D/E/F are terrain fine-tuning variants.

A has a fixed reference and follows the source clip's original heading. It does
not apply random whole-trajectory yaw augmentation; see ``command_ablation``.
"""

from dataclasses import replace

from holosoma.config_values.wbt.g1.command_ablation import g1_29dof_wbt_fixed_reference_heading_command
from holosoma.config_values.wbt.g1.command_gen import g1_29dof_wbt_gen_command
from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_gen_terrain_curriculum
from holosoma.config_values.wbt.g1.experiment_gen import g1_29dof_wbt_gen
from holosoma.config_values.wbt.g1.experiment_gen_terrain import g1_29dof_wbt_gen_terrain
from holosoma.config_values.wbt.g1.observation_ablation import (
    g1_29dof_wbt_gen_history_no_tracker_terrain_observation,
)


def _with_heading_weight(experiment, weight: float):
    terms = dict(experiment.reward.terms)
    terms["motion_heading_alignment"] = replace(terms["motion_heading_alignment"], weight=weight)
    return replace(experiment.reward, terms=terms)


def _with_terrain_curriculum_enabled(enabled: bool):
    setup_terms = dict(g1_29dof_wbt_gen_terrain_curriculum.setup_terms)
    terrain_term = setup_terms["terrain_curriculum"]
    setup_terms["terrain_curriculum"] = replace(
        terrain_term,
        params={**terrain_term.params, "enabled": enabled},
    )
    return replace(g1_29dof_wbt_gen_terrain_curriculum, setup_terms=setup_terms)


def _with_training_name(experiment, name: str):
    return replace(experiment, training=replace(experiment.training, name=name))


def _with_generator_past_noise(command, past_noise_std: float):
    setup_terms = dict(command.setup_terms)
    motion_term = setup_terms["motion_command"]
    params = dict(motion_term.params)
    params["motion_config"] = replace(params["motion_config"], past_noise_std=past_noise_std)
    setup_terms["motion_command"] = replace(motion_term, params=params)
    return replace(command, setup_terms=setup_terms)


# A: fixed reference + physical curriculum + tracker scan/history + heading
# reward. BASE inputs are expanded, then the tracker is terrain fine-tuned.
g1_29dof_wbt_ablation_a_fixed_reference = replace(
    _with_training_name(
        g1_29dof_wbt_gen_terrain,
        "g1_29dof_wbt_ablation_a_fixed_reference_terrain_ft",
    ),
    command=g1_29dof_wbt_fixed_reference_heading_command,
)


# B: online generator and tracker are terrain blind. The physical balanced
# terrain remains present for evaluation, but curriculum adaptation and PPO
# updates are disabled by default. Condition noise is not a requested ablation
# axis: training/evaluation comparisons use clean measured past state, while B
# retains the canonical Stage-5 command's zero terrain scan.
g1_29dof_wbt_ablation_b_generator_blind = replace(
    _with_training_name(
        g1_29dof_wbt_gen_terrain,
        "g1_29dof_wbt_ablation_b_generator_blind_no_terrain_ft",
    ),
    command=_with_generator_past_noise(g1_29dof_wbt_gen_command, 0.0),
    observation=g1_29dof_wbt_gen.observation,
    curriculum=_with_terrain_curriculum_enabled(False),
    reward=_with_heading_weight(g1_29dof_wbt_gen_terrain, 0.0),
    algo=replace(
        g1_29dof_wbt_gen.algo,
        config=replace(
            g1_29dof_wbt_gen.algo.config,
            num_learning_iterations=0,
            load_optimizer=False,
            checkpoint_load_mode="strict",
        ),
    ),
)


# C: full terrain-conditioned architecture loaded from BASE with appended
# inputs, but no PPO update. This is a config alias of the full preset with an
# explicit zero-update intent and distinct run metadata.
g1_29dof_wbt_ablation_c_full_no_finetune = replace(
    _with_training_name(
        g1_29dof_wbt_gen_terrain,
        "g1_29dof_wbt_ablation_c_full_terrain_update0",
    ),
    algo=replace(
        g1_29dof_wbt_gen_terrain.algo,
        config=replace(
            g1_29dof_wbt_gen_terrain.algo.config,
            num_learning_iterations=0,
            load_optimizer=False,
        ),
    ),
)


# D: current full terrain closed-loop PPO, separated only by run metadata so
# its outputs cannot be confused with the canonical Stage-9 preset.
g1_29dof_wbt_ablation_d_full_finetune = _with_training_name(
    g1_29dof_wbt_gen_terrain,
    "g1_29dof_wbt_ablation_d_full_terrain_ft",
)


# E: generator sees the simulator scan, tracker does not. Five-frame
# proprioception and heading reward remain enabled.
g1_29dof_wbt_ablation_e_generator_terrain_only = replace(
    _with_training_name(
        g1_29dof_wbt_gen_terrain,
        "g1_29dof_wbt_ablation_e_generator_terrain_only_ft",
    ),
    observation=g1_29dof_wbt_gen_history_no_tracker_terrain_observation,
)


# F: full terrain architecture and fine-tuning with only heading reward off.
g1_29dof_wbt_ablation_f_no_heading_reward = replace(
    _with_training_name(
        g1_29dof_wbt_gen_terrain,
        "g1_29dof_wbt_ablation_f_no_heading_reward_ft",
    ),
    reward=_with_heading_weight(g1_29dof_wbt_gen_terrain, 0.0),
)


__all__ = [
    "g1_29dof_wbt_ablation_a_fixed_reference",
    "g1_29dof_wbt_ablation_b_generator_blind",
    "g1_29dof_wbt_ablation_c_full_no_finetune",
    "g1_29dof_wbt_ablation_d_full_finetune",
    "g1_29dof_wbt_ablation_e_generator_terrain_only",
    "g1_29dof_wbt_ablation_f_no_heading_reward",
]
