"""Default curriculum manager configurations."""

from holosoma.config_values.loco.g1.curriculum import g1_29dof_curriculum, g1_29dof_curriculum_fast_sac
from holosoma.config_values.loco.tocabi.curriculum import tocabi_33dof_curriculum, tocabi_33dof_curriculum_fast_sac
from holosoma.config_values.loco.t1.curriculum import t1_29dof_curriculum, t1_29dof_curriculum_fast_sac
from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_curriculum
from holosoma.config_values.wbt.tocabi.curriculum import tocabi_33dof_wbt_curriculum

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_curriculum,
    "g1_29dof": g1_29dof_curriculum,
    "t1_29dof_fast_sac": t1_29dof_curriculum_fast_sac,
    "g1_29dof_fast_sac": g1_29dof_curriculum_fast_sac,
    "tocabi_33dof": tocabi_33dof_curriculum,
    "tocabi_33dof_fast_sac": tocabi_33dof_curriculum_fast_sac,
    "g1_29dof_wbt_curriculum": g1_29dof_wbt_curriculum,
    "tocabi_33dof_wbt_curriculum": tocabi_33dof_wbt_curriculum,
}
