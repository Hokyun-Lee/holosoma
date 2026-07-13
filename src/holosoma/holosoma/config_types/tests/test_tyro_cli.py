import tyro
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.config_values.wbt.g1.experiment_gen import g1_29dof_wbt_gen
from holosoma.config_values.wbt.g1.experiment_gen_terrain import g1_29dof_wbt_gen_terrain
from holosoma.utils.tyro_utils import TYRO_CONIFG


def test_experiment_config():
    assert isinstance(tyro.cli(ExperimentConfig, args=(), config=TYRO_CONIFG), ExperimentConfig)


def test_generated_motion_terrain_experiment_is_separate_from_flat():
    parsed = tyro.cli(
        AnnotatedExperimentConfig,
        args=("exp:g1-29dof-wbt-gen-terrain",),
        config=TYRO_CONIFG,
    )

    assert parsed.training.name == "g1_29dof_wbt_gen_terrain_manager"
    assert parsed.terrain.terrain_term.mesh_type == "trimesh"
    assert g1_29dof_wbt_gen.terrain.terrain_term.mesh_type == "plane"
    flat_motion_cfg = g1_29dof_wbt_gen.command.setup_terms["motion_command"].params["motion_config"]
    terrain_motion_cfg = g1_29dof_wbt_gen_terrain.command.setup_terms["motion_command"].params[
        "motion_config"
    ]
    assert not flat_motion_cfg.use_sim_terrain_scan
    assert terrain_motion_cfg.use_sim_terrain_scan
    assert g1_29dof_wbt_gen.algo.config.checkpoint_load_mode == "strict"
    assert g1_29dof_wbt_gen_terrain.algo.config.checkpoint_load_mode == "expand_input"

    for group_name in ("actor_obs", "critic_obs"):
        flat_terms = sorted(g1_29dof_wbt_gen.observation.groups[group_name].terms)
        terrain_terms = sorted(g1_29dof_wbt_gen_terrain.observation.groups[group_name].terms)
        assert terrain_terms == flat_terms + ["terrain_height_scan"]
