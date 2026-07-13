"""Terrain variant of the closed-loop generated-motion WBT experiment.

This preset intentionally stays separate from ``g1_29dof_wbt_gen`` so the
Stage-5 flat baseline remains unchanged.  Stage 6 starts from HoloSoma's
existing procedural terrain mix; the generator height scan is connected in
the following tasks.
"""

from dataclasses import replace

from holosoma.config_values import terrain
from holosoma.config_values.wbt.g1.experiment_gen import g1_29dof_wbt_gen


g1_29dof_wbt_gen_terrain = replace(
    g1_29dof_wbt_gen,
    training=replace(
        g1_29dof_wbt_gen.training,
        name="g1_29dof_wbt_gen_terrain_manager",
    ),
    terrain=terrain.terrain_locomotion_mix,
)


__all__ = ["g1_29dof_wbt_gen_terrain"]
