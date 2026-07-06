"""Default reward manager configurations."""

from holosoma.config_values.loco.g1.reward import g1_29dof_loco, g1_29dof_loco_fast_sac
from holosoma.config_values.loco.tocabi.reward import tocabi_33dof_loco, tocabi_33dof_loco_fast_sac
from holosoma.config_values.loco.t1.reward import t1_29dof_loco, t1_29dof_loco_fast_sac
from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_fast_sac_reward,
    g1_29dof_wbt_reward,
    g1_29dof_wbt_reward_w_object,
)
from holosoma.config_values.wbt.tocabi.reward import tocabi_33dof_wbt_fast_sac_reward, tocabi_33dof_wbt_reward

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof_loco": t1_29dof_loco,
    "t1_29dof_loco_fast_sac": t1_29dof_loco_fast_sac,
    "g1_29dof_loco": g1_29dof_loco,
    "g1_29dof_loco_fast_sac": g1_29dof_loco_fast_sac,
    "tocabi_33dof_loco": tocabi_33dof_loco,
    "tocabi_33dof_loco_fast_sac": tocabi_33dof_loco_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt_reward,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_reward_w_object,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac_reward,
    "tocabi_33dof_wbt": tocabi_33dof_wbt_reward,
    "tocabi_33dof_wbt_fast_sac": tocabi_33dof_wbt_fast_sac_reward,
}
