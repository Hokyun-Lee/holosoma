"""Terrain-conditioned variant of the closed-loop generated-motion command."""

from dataclasses import replace

from holosoma.config_values.wbt.g1.command_gen import (
    g1_29dof_wbt_gen_command,
    gen_motion_config,
)

terrain_gen_motion_config = replace(gen_motion_config, use_sim_terrain_scan=True)

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
