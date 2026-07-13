"""Deterministic common evaluator for Stage-10 WBT terrain ablations.

Unlike ``eval_agent.py``, this entry point can apply a current experiment
preset to an older checkpoint.  That permits policy-only ``expand_input``
evaluation of the zero-update C ablation without materializing a misleading
training checkpoint.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from pathlib import Path

import tyro
from pydantic.dataclasses import dataclass

from holosoma.config_types.command import GeneratedMotionConfig, MotionConfig
from holosoma.config_types.eval_callback import (
    EvalCallbacksConfig,
    TerrainMetricsCallbackConfig,
    TerrainMetricsConfig,
)
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import DEFAULTS as EXPERIMENT_DEFAULTS
from holosoma.eval_agent import run_eval_with_tyro
from holosoma.utils.eval_utils import CheckpointConfig, init_eval_logging, load_saved_experiment_config
from holosoma.utils.tyro_utils import TYRO_CONIFG


@dataclass(frozen=True)
class TerrainEvaluationRunConfig:
    """Stage-10 evaluator-only options parsed before ExperimentConfig overrides."""

    checkpoint: str | None = None
    """Tracker checkpoint path or W&B URI."""

    experiment_preset: str | None = None
    """Optional current preset, e.g. ``g1_29dof_wbt_ablation_c_full_no_finetune``.

    When omitted, use the experiment config embedded in the checkpoint.  Both
    registry keys and ``exp:g1-29dof-...`` spellings are accepted.
    """

    variant: str = "unspecified"
    scenario_label: str = ""
    episode_count: int = 100
    output_prefix: str = "terrain_evaluation"
    seed: int = 42
    fixed_terrain_level: int = 0
    evaluation_phase_mode: str = "uniform"
    deterministic_generator: bool = True
    deterministic_per_env_sampling: bool = True
    generator_sampling_seed: int = 0
    generator_checkpoint: str | None = None
    motion_file: str | None = None
    success_distance_m: float = 1.5
    fall_root_height_m: float = 0.45
    fall_upright_cosine: float = 0.5
    body_origin_penetration_threshold_m: float = 0.02
    body_origin_correction_min_improvement_m: float = 0.01
    heading_speed_threshold_mps: float = 0.05
    torch_deterministic: bool = False
    fail_on_incomplete: bool = True


def _preset_key(value: str) -> str:
    key = value.strip()
    if key.startswith("exp:"):
        key = key[4:]
    return key.replace("-", "_")


def select_base_experiment(
    saved_config: ExperimentConfig,
    experiment_preset: str | None,
) -> tuple[ExperimentConfig, str | None]:
    """Select a saved or current preset config before CLI field overrides."""
    if experiment_preset is None:
        return saved_config, None
    key = _preset_key(experiment_preset)
    if key not in EXPERIMENT_DEFAULTS:
        available = ", ".join(sorted(name for name in EXPERIMENT_DEFAULTS if "wbt" in name))
        raise ValueError(f"Unknown experiment_preset {experiment_preset!r}. Available WBT presets: {available}")
    return EXPERIMENT_DEFAULTS[key], key


def _saved_generator_checkpoint(motion_config: object | None) -> str:
    """Read a generator path from typed or checkpoint-deserialized config.

    ``CommandTermCfg.params`` is intentionally typed as ``dict[str, object]``.
    Consequently, reconstructing an :class:`ExperimentConfig` from checkpoint
    metadata preserves its nested motion config as a plain mapping, while
    current presets carry a :class:`GeneratedMotionConfig` dataclass.
    """
    if motion_config is None:
        return ""
    if isinstance(motion_config, Mapping):
        value = motion_config.get("generator_checkpoint", "")
    else:
        value = getattr(motion_config, "generator_checkpoint", "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(
            "Saved motion_config.generator_checkpoint must be a string, "
            f"got {type(value).__name__}"
        )
    return value


def _evaluation_horizon_steps(config: ExperimentConfig) -> int:
    """Return policy steps before timeout for the effective simulator config."""
    sim = config.simulator.config.sim
    horizon = math.ceil(sim.max_episode_length_s * sim.fps / sim.control_decimation)
    if horizon < 1:
        raise ValueError("Effective simulator config must provide at least one evaluation step")
    return horizon


def _replace_motion_config(
    config: ExperimentConfig,
    *,
    saved_config: ExperimentConfig,
    args: TerrainEvaluationRunConfig,
) -> ExperimentConfig:
    setup_terms = dict(config.command.setup_terms)
    command_term = setup_terms.get("motion_command")
    if command_term is None:
        raise ValueError("Terrain WBT evaluation requires command.setup_terms['motion_command']")
    params = dict(command_term.params)
    motion_config = params.get("motion_config")
    if motion_config is None:
        raise ValueError("motion_command is missing motion_config")

    saved_term = saved_config.command.setup_terms.get("motion_command")
    saved_motion_config = None if saved_term is None else saved_term.params.get("motion_config")
    updates: dict[str, object] = {
        "evaluation_phase_mode": args.evaluation_phase_mode,
        "reanchor_motion_xy_on_reset": True,
        "phase_horizon_steps": _evaluation_horizon_steps(config),
    }
    if args.motion_file is not None:
        updates["motion_file"] = args.motion_file
    if isinstance(motion_config, GeneratedMotionConfig):
        inherited_generator = _saved_generator_checkpoint(saved_motion_config)
        generator_checkpoint = args.generator_checkpoint or motion_config.generator_checkpoint or inherited_generator
        if not generator_checkpoint:
            raise ValueError(
                "Generated-motion evaluation requires --generator-checkpoint or a "
                "checkpoint/preset config containing one"
            )
        updates.update(
            {
                "generator_checkpoint": generator_checkpoint,
                "deterministic_sampling": args.deterministic_generator,
                "deterministic_per_env_sampling": args.deterministic_per_env_sampling,
                "sampling_seed": args.generator_sampling_seed,
            }
        )
    elif args.generator_checkpoint is not None:
        raise ValueError("--generator-checkpoint was supplied for a fixed-reference command")

    params["motion_config"] = dataclasses.replace(motion_config, **updates)
    setup_terms["motion_command"] = dataclasses.replace(command_term, params=params)
    return dataclasses.replace(config, command=dataclasses.replace(config.command, setup_terms=setup_terms))


def _freeze_terrain_curriculum(config: ExperimentConfig, fixed_level: int) -> ExperimentConfig:
    """Keep curriculum bookkeeping while clamping every env to one row."""
    if fixed_level < 0:
        raise ValueError("fixed_terrain_level must be non-negative")
    layout = config.terrain.terrain_term.curriculum_layout
    if layout.enabled and fixed_level >= config.terrain.terrain_term.num_rows:
        raise ValueError(
            f"fixed_terrain_level {fixed_level} outside [0, {config.terrain.terrain_term.num_rows})"
        )
    setup_terms = dict(config.curriculum.setup_terms)
    terrain_term = setup_terms.get("terrain_curriculum")
    if terrain_term is None:
        return config
    params = {
        **terrain_term.params,
        "enabled": True,
        "initial_level": fixed_level,
        "min_level": fixed_level,
        "max_level": fixed_level,
        "skip_first_episode": False,
    }
    setup_terms["terrain_curriculum"] = dataclasses.replace(terrain_term, params=params)
    return dataclasses.replace(config, curriculum=dataclasses.replace(config.curriculum, setup_terms=setup_terms))


def prepare_terrain_evaluation_config(
    config: ExperimentConfig,
    *,
    saved_config: ExperimentConfig,
    args: TerrainEvaluationRunConfig,
) -> ExperimentConfig:
    """Apply evaluator invariants after all general ExperimentConfig overrides."""
    if args.episode_count < 1:
        raise ValueError("episode_count must be >= 1")
    if not 0 <= args.generator_sampling_seed <= (1 << 63) - 1:
        raise ValueError("generator_sampling_seed must be in [0, 2^63 - 1]")
    if args.evaluation_phase_mode not in {"zero", "uniform"}:
        raise ValueError("evaluation_phase_mode must be 'zero' or 'uniform'")
    if args.deterministic_per_env_sampling and not args.deterministic_generator:
        raise ValueError(
            "deterministic_per_env_sampling requires deterministic_generator=True"
        )
    layout = config.terrain.terrain_term.curriculum_layout
    if layout.enabled:
        terrain_type_count = len(layout.terrain_types)
    elif args.scenario_label:
        terrain_type_count = 1
    else:
        configured_types = [name for name, weight in config.terrain.terrain_term.terrain_config.items() if weight > 0]
        if len(configured_types) > 1:
            raise ValueError(
                "Non-curriculum mixed terrain evaluation requires --scenario-label or per-environment type IDs"
            )
        terrain_type_count = 1
    if args.episode_count % terrain_type_count != 0:
        raise ValueError(
            f"episode_count ({args.episode_count}) must be divisible by active terrain type count "
            f"({terrain_type_count})"
        )
    if config.training.num_envs < terrain_type_count:
        raise ValueError(
            f"num_envs ({config.training.num_envs}) must cover all active terrain types ({terrain_type_count})"
        )
    config = _replace_motion_config(config, saved_config=saved_config, args=args)
    config = _freeze_terrain_curriculum(config, args.fixed_terrain_level)

    steps_per_episode = _evaluation_horizon_steps(config)
    max_eval_steps = config.training.max_eval_steps
    if max_eval_steps is None:
        # A one-env upper bound. Vectorized evaluation generally stops much
        # earlier through the callback's exact episode-count stop signal.
        max_eval_steps = args.episode_count * max(1, steps_per_episode)
    training = dataclasses.replace(
        config.training,
        seed=args.seed,
        torch_deterministic=args.torch_deterministic,
        export_onnx=False,
        max_eval_steps=max_eval_steps,
    )
    return dataclasses.replace(config, training=training)


def build_metrics_callbacks(
    args: TerrainEvaluationRunConfig,
    *,
    variant: str,
    effective_config: ExperimentConfig,
) -> EvalCallbacksConfig:
    motion_term = effective_config.command.setup_terms.get("motion_command")
    motion_config = None if motion_term is None else motion_term.params.get("motion_config")
    if not isinstance(motion_config, MotionConfig):
        raise TypeError("Effective terrain evaluation config must contain a typed MotionConfig")
    metrics = TerrainMetricsConfig(
        enabled=True,
        output_prefix=args.output_prefix,
        variant=variant,
        scenario_label=args.scenario_label,
        episode_count=args.episode_count,
        success_distance_m=args.success_distance_m,
        fall_root_height_m=args.fall_root_height_m,
        fall_upright_cosine=args.fall_upright_cosine,
        body_origin_penetration_threshold_m=args.body_origin_penetration_threshold_m,
        body_origin_correction_min_improvement_m=args.body_origin_correction_min_improvement_m,
        heading_speed_threshold_mps=args.heading_speed_threshold_mps,
        evaluation_seed=args.seed,
        fixed_terrain_level=args.fixed_terrain_level,
        evaluation_phase_mode=args.evaluation_phase_mode,
        reanchor_motion_xy_on_reset=motion_config.reanchor_motion_xy_on_reset,
        phase_horizon_steps=motion_config.phase_horizon_steps,
        deterministic_generator=args.deterministic_generator,
        deterministic_per_env_sampling=args.deterministic_per_env_sampling,
        generator_sampling_seed=args.generator_sampling_seed,
        fail_on_incomplete=args.fail_on_incomplete,
    )
    return EvalCallbacksConfig(terrain_metrics=TerrainMetricsCallbackConfig(config=metrics))


def main() -> None:
    init_eval_logging()
    args, remaining_args = tyro.cli(
        TerrainEvaluationRunConfig,
        return_unknown_args=True,
        add_help=False,
    )
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")

    # Common evaluation never restores checkpoint curriculum buffers.  The
    # selected fixed level and current num_envs are authoritative.
    checkpoint_cfg = CheckpointConfig(checkpoint=args.checkpoint, restore_env_state=False)
    saved_config, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    base_config, selected_preset = select_base_experiment(saved_config, args.experiment_preset)
    eval_config = base_config.get_eval_config()
    overridden_config = tyro.cli(
        ExperimentConfig,
        default=eval_config,
        args=remaining_args,
        description="Override the saved/current terrain evaluation config.",
        config=TYRO_CONIFG,
    )
    effective_config = prepare_terrain_evaluation_config(
        overridden_config,
        saved_config=saved_config,
        args=args,
    )
    variant = args.variant
    if variant == "unspecified":
        variant = selected_preset or Path(args.checkpoint).stem
    callbacks = build_metrics_callbacks(args, variant=variant, effective_config=effective_config)
    run_eval_with_tyro(
        effective_config,
        checkpoint_cfg,
        saved_config,
        saved_wandb_path,
        eval_cbs_cfg=callbacks,
    )


if __name__ == "__main__":
    main()
