"""Terrain-conditioned variant of the closed-loop generated-motion command."""

from dataclasses import replace

from holosoma.config_values.wbt.g1.command_gen import (
    g1_29dof_wbt_gen_command,
    gen_motion_config,
)

terrain_gen_motion_config = replace(
    gen_motion_config,
    # Terrain tiles are centered on each env origin.  Keep the sampled motion
    # pose/velocity, but remove the source clip's accumulated world XY so a
    # random phase cannot spawn in an adjacent tile.  The 500-step horizon is
    # an implementation choice for the default 10 s, 50 Hz evaluation; the
    # Stage-10 evaluator derives and overwrites it from the effective sim config.
    reanchor_motion_xy_on_reset=True,
    phase_horizon_steps=500,
    use_sim_terrain_scan=True,
    require_fully_measured_history=True,
    # Stage-8 structured condition noise is applied while training the
    # generator.  Do not stack the legacy all-feature inference perturbation
    # on the measured terrain closed-loop history.
    past_noise_std=0.0,
    # Stage-9 scan-derived diagnostics.  These unpublished thresholds are
    # implementation choices and remain CLI-overridable.
    body_origin_penetration_threshold_m=0.02,
    body_origin_correction_min_improvement_m=0.01,
)

_setup_terms = dict(g1_29dof_wbt_gen_command.setup_terms)
_setup_terms["motion_command"] = replace(
    _setup_terms["motion_command"],
    params={"motion_config": terrain_gen_motion_config},
)

g1_29dof_wbt_gen_terrain_command = replace(
    g1_29dof_wbt_gen_command,
    setup_terms=_setup_terms,
)


__all__ = ["g1_29dof_wbt_gen_terrain_command", "terrain_gen_motion_config"]
