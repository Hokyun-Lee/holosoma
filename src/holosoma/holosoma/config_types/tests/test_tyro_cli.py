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
    assert g1_29dof_wbt_gen_terrain.command == g1_29dof_wbt_gen.command
