"""Tracker-observation variants for Stage-10 terrain ablations."""

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg
from holosoma.config_values.wbt.g1.observation_gen_terrain import g1_29dof_wbt_gen_terrain_observation


def _remove_terrain_scan(cfg: ObservationManagerCfg) -> ObservationManagerCfg:
    groups = {
        group_name: replace(
            group,
            terms={name: term for name, term in group.terms.items() if name != "terrain_height_scan"},
        )
        for group_name, group in cfg.groups.items()
    }
    return replace(cfg, groups=groups)


# E: generator still receives the simulator scan through its command term,
# while actor/critic keep five-frame proprioception but receive no scan tensor.
g1_29dof_wbt_gen_history_no_tracker_terrain_observation = _remove_terrain_scan(
    g1_29dof_wbt_gen_terrain_observation
)


__all__ = ["g1_29dof_wbt_gen_history_no_tracker_terrain_observation"]
