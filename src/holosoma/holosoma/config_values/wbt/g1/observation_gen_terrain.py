"""Current-frame terrain observation for generated-motion tracking."""

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_values.wbt.g1.observation import actor_obs_shared, critic_obs_shared_terms

_terrain_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:terrain_height_scan",
    scale=1.0,
    noise=0.0,
)

_actor_terms = dict(actor_obs_shared.terms)
_actor_terms["terrain_height_scan"] = _terrain_term

_critic_terms = dict(critic_obs_shared_terms)
_critic_terms["terrain_height_scan"] = _terrain_term

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
