import tyro
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.loco.t1.experiment import t1_29dof, t1_29dof_fast_sac
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt,
    g1_29dof_wbt_fast_sac,
    g1_29dof_wbt_fast_sac_w_object,
    g1_29dof_wbt_w_object,
)
from holosoma.config_values.wbt.g1.experiment_ablation import (
    g1_29dof_wbt_ablation_a_fixed_reference,
    g1_29dof_wbt_ablation_b_generator_blind,
    g1_29dof_wbt_ablation_c_full_no_finetune,
    g1_29dof_wbt_ablation_d_full_finetune,
    g1_29dof_wbt_ablation_e_generator_terrain_only,
    g1_29dof_wbt_ablation_f_no_heading_reward,
)
from holosoma.config_values.wbt.g1.experiment_gen import g1_29dof_wbt_gen
from holosoma.config_values.wbt.g1.experiment_gen_terrain import (
    g1_29dof_wbt_gen_terrain,
    g1_29dof_wbt_gen_terrain_scratch,
)
from typing_extensions import Annotated

DEFAULTS = {
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_w_object,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac,
    "g1_29dof_wbt_fast_sac_w_object": g1_29dof_wbt_fast_sac_w_object,
    "g1_29dof_wbt_gen": g1_29dof_wbt_gen,
    "g1_29dof_wbt_gen_terrain": g1_29dof_wbt_gen_terrain,
    "g1_29dof_wbt_gen_terrain_scratch": g1_29dof_wbt_gen_terrain_scratch,
    "g1_29dof_wbt_ablation_a_fixed_reference": g1_29dof_wbt_ablation_a_fixed_reference,
    "g1_29dof_wbt_ablation_b_generator_blind": g1_29dof_wbt_ablation_b_generator_blind,
    "g1_29dof_wbt_ablation_c_full_no_finetune": g1_29dof_wbt_ablation_c_full_no_finetune,
    "g1_29dof_wbt_ablation_d_full_finetune": g1_29dof_wbt_ablation_d_full_finetune,
    "g1_29dof_wbt_ablation_e_generator_terrain_only": g1_29dof_wbt_ablation_e_generator_terrain_only,
    "g1_29dof_wbt_ablation_f_no_heading_reward": g1_29dof_wbt_ablation_f_no_heading_reward,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{k.replace('_', '-')}": v for k, v in DEFAULTS.items()}
        )
    ),
]
