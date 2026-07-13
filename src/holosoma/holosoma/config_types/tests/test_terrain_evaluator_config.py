from __future__ import annotations

import dataclasses

import pytest
import torch
from holosoma.config_types.command import GeneratedMotionConfig
from holosoma.config_types.eval_callback import TerrainMetricsConfig
from holosoma.config_values.wbt.g1.experiment_ablation import (
    g1_29dof_wbt_ablation_a_fixed_reference,
    g1_29dof_wbt_ablation_b_generator_blind,
    g1_29dof_wbt_ablation_c_full_no_finetune,
)
from holosoma.evaluate_wbt_terrain import (
    TerrainEvaluationRunConfig,
    build_metrics_callbacks,
    prepare_terrain_evaluation_config,
    select_base_experiment,
)
from holosoma.managers.curriculum.terms.terrain import target_heading_forward_progress_m


def _with_saved_motion_config(motion_config: object):
    command = g1_29dof_wbt_ablation_b_generator_blind.command
    setup_terms = dict(command.setup_terms)
    motion_term = setup_terms["motion_command"]
    setup_terms["motion_command"] = dataclasses.replace(
        motion_term,
        params={**motion_term.params, "motion_config": motion_config},
    )
    return dataclasses.replace(
        g1_29dof_wbt_ablation_b_generator_blind,
        command=dataclasses.replace(command, setup_terms=setup_terms),
    )


def test_explicit_current_preset_can_override_saved_checkpoint_config() -> None:
    selected, key = select_base_experiment(
        g1_29dof_wbt_ablation_b_generator_blind,
        "exp:g1-29dof-wbt-ablation-c-full-no-finetune",
    )
    assert key == "g1_29dof_wbt_ablation_c_full_no_finetune"
    assert selected is g1_29dof_wbt_ablation_c_full_no_finetune
    assert selected.algo.config.checkpoint_load_mode == "expand_input"


def test_prepare_config_freezes_level_and_generator_sampling() -> None:
    args = TerrainEvaluationRunConfig(
        checkpoint="model.pt",
        episode_count=8,
        seed=17,
        fixed_terrain_level=3,
        deterministic_generator=True,
        generator_sampling_seed=23,
        generator_checkpoint="robust_generator.pt",
    )
    prepared = prepare_terrain_evaluation_config(
        g1_29dof_wbt_ablation_c_full_no_finetune.get_eval_config(),
        saved_config=g1_29dof_wbt_ablation_b_generator_blind,
        args=args,
    )
    motion_config = prepared.command.setup_terms["motion_command"].params["motion_config"]
    assert motion_config.deterministic_sampling
    assert motion_config.sampling_seed == 23
    assert motion_config.generator_checkpoint == "robust_generator.pt"
    terrain_params = prepared.curriculum.setup_terms["terrain_curriculum"].params
    assert terrain_params["enabled"] is True
    assert terrain_params["initial_level"] == 3
    assert terrain_params["min_level"] == 3
    assert terrain_params["max_level"] == 3
    assert terrain_params["skip_first_episode"] is False
    assert prepared.training.seed == 17
    assert prepared.training.export_onnx is False
    assert prepared.training.max_eval_steps == 8 * 500


@pytest.mark.parametrize("serialized", [False, True], ids=["dataclass", "checkpoint-dict"])
def test_generator_checkpoint_inherits_from_saved_motion_config(serialized: bool) -> None:
    saved_motion_config = g1_29dof_wbt_ablation_b_generator_blind.command.setup_terms[
        "motion_command"
    ].params["motion_config"]
    saved_motion_config = dataclasses.replace(
        saved_motion_config,
        generator_checkpoint="saved_generator.pt",
    )
    if serialized:
        saved_motion_config = dataclasses.asdict(saved_motion_config)

    prepared = prepare_terrain_evaluation_config(
        g1_29dof_wbt_ablation_c_full_no_finetune.get_eval_config(),
        saved_config=_with_saved_motion_config(saved_motion_config),
        args=TerrainEvaluationRunConfig(checkpoint="model.pt", episode_count=8),
    )

    motion_config = prepared.command.setup_terms["motion_command"].params["motion_config"]
    assert motion_config.generator_checkpoint == "saved_generator.pt"


def test_explicit_generator_checkpoint_overrides_saved_value() -> None:
    prepared = prepare_terrain_evaluation_config(
        g1_29dof_wbt_ablation_c_full_no_finetune.get_eval_config(),
        saved_config=_with_saved_motion_config({"generator_checkpoint": "saved_generator.pt"}),
        args=TerrainEvaluationRunConfig(
            checkpoint="model.pt",
            episode_count=8,
            generator_checkpoint="explicit_generator.pt",
        ),
    )

    motion_config = prepared.command.setup_terms["motion_command"].params["motion_config"]
    assert motion_config.generator_checkpoint == "explicit_generator.pt"


def test_fixed_reference_rejects_explicit_generator_checkpoint() -> None:
    with pytest.raises(ValueError, match="supplied for a fixed-reference command"):
        prepare_terrain_evaluation_config(
            g1_29dof_wbt_ablation_a_fixed_reference.get_eval_config(),
            saved_config=_with_saved_motion_config({"generator_checkpoint": "saved_generator.pt"}),
            args=TerrainEvaluationRunConfig(
                checkpoint="model.pt",
                episode_count=8,
                generator_checkpoint="explicit_generator.pt",
            ),
        )


def test_metrics_callback_collection_is_opt_in_and_carries_runtime_controls() -> None:
    callbacks = build_metrics_callbacks(
        TerrainEvaluationRunConfig(
            checkpoint="model.pt",
            episode_count=12,
            fixed_terrain_level=4,
            deterministic_generator=True,
            generator_sampling_seed=9,
            body_origin_correction_min_improvement_m=0.015,
        ),
        variant="D",
    )
    active = callbacks.collect_active_callbacks()
    assert list(active) == ["terrain_metrics"]
    config = active["terrain_metrics"].config
    assert config.variant == "D"
    assert config.episode_count == 12
    assert config.fixed_terrain_level == 4
    assert config.deterministic_generator
    assert config.generator_sampling_seed == 9
    assert config.body_origin_correction_min_improvement_m == pytest.approx(0.015)


def test_correction_improvement_default_matches_generated_motion_diagnostic() -> None:
    assert (
        TerrainMetricsConfig().body_origin_correction_min_improvement_m
        == GeneratedMotionConfig(
            motion_file="motion.npz",
            body_name_ref=["torso"],
            body_names_to_track=["torso"],
        ).body_origin_correction_min_improvement_m
        == 0.01
    )


def test_episode_count_must_split_evenly_across_curriculum_types() -> None:
    args = TerrainEvaluationRunConfig(
        checkpoint="model.pt",
        episode_count=10,
        generator_checkpoint="generator.pt",
    )
    with pytest.raises(ValueError, match="divisible by active terrain type count"):
        prepare_terrain_evaluation_config(
            g1_29dof_wbt_ablation_c_full_no_finetune.get_eval_config(),
            saved_config=g1_29dof_wbt_ablation_b_generator_blind,
            args=args,
        )


def test_shared_forward_progress_helper_rejects_lateral_and_backward_motion() -> None:
    root_xy = torch.tensor([[0.0, 3.0], [-1.0, 0.0], [2.0, 0.0]])
    starts = torch.zeros_like(root_xy)
    headings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    progress = target_heading_forward_progress_m(root_xy, starts, headings)
    torch.testing.assert_close(progress, torch.tensor([0.0, -1.0, 2.0]))
