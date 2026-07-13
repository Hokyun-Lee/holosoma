"""Current-frame terrain observation for generated-motion tracking."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_values.wbt.g1.observation import actor_obs_shared, critic_obs_shared_terms

_terrain_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:terrain_height_scan",
    scale=1.0,
    noise=0.0,
    history_length=1,
)

_HISTORY_TERMS = {"base_ang_vel", "dof_pos", "dof_vel"}


def _with_selective_history(terms: dict[str, ObsTermCfg]) -> dict[str, ObsTermCfg]:
    return {
        name: replace(term, history_length=5 if name in _HISTORY_TERMS else 1)
        for name, term in terms.items()
    }


_projected_gravity_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:projected_gravity",
    scale=1.0,
    noise=0.0,
    history_length=5,
)

_actor_terms = _with_selective_history(actor_obs_shared.terms)
_actor_terms["terrain_height_scan"] = _terrain_term
_actor_terms["tracker_projected_gravity"] = _projected_gravity_term

_critic_terms = _with_selective_history(critic_obs_shared_terms)
_critic_terms["terrain_height_scan"] = _terrain_term
_critic_terms["tracker_projected_gravity"] = _projected_gravity_term

g1_29dof_wbt_gen_terrain_observation = ObservationManagerCfg(
    groups={
        "actor_obs": replace(actor_obs_shared, terms=_actor_terms),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=_critic_terms,
        ),
    },
)


__all__ = ["g1_29dof_wbt_gen_terrain_observation"]
