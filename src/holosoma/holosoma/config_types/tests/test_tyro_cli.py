import tyro
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.config_values.wbt.g1.experiment_gen import g1_29dof_wbt_gen
from holosoma.config_values.wbt.g1.experiment_gen_terrain import (
    g1_29dof_wbt_gen_terrain,
    g1_29dof_wbt_gen_terrain_scratch,
)
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
    terrain_motion_cfg = g1_29dof_wbt_gen_terrain.command.setup_terms["motion_command"].params["motion_config"]
    assert not flat_motion_cfg.use_sim_terrain_scan
    assert terrain_motion_cfg.use_sim_terrain_scan
    assert flat_motion_cfg.past_noise_std == 0.01
    assert terrain_motion_cfg.past_noise_std == 0.0
    assert flat_motion_cfg.denoise_steps == 2
    assert terrain_motion_cfg.denoise_steps == 2
    assert g1_29dof_wbt_gen.algo.config.checkpoint_load_mode == "strict"
    assert g1_29dof_wbt_gen_terrain.algo.config.checkpoint_load_mode == "expand_input"
    assert "motion_heading_alignment" not in g1_29dof_wbt_gen.reward.terms
    assert parsed.reward.terms["motion_heading_alignment"].weight == 1.0
    layout_cfg = parsed.terrain.terrain_term.curriculum_layout
    assert layout_cfg.enabled
    assert layout_cfg.terrain_types == ["flat", "box", "stair", "hurdle"]
    assert parsed.terrain.terrain_term.num_rows == 10
    assert parsed.terrain.terrain_term.num_cols == 20
    assert not g1_29dof_wbt_gen.terrain.terrain_term.curriculum_layout.enabled

    curriculum_term = parsed.curriculum.setup_terms["terrain_curriculum"]
    assert curriculum_term.params["enabled"] is True
    assert curriculum_term.params["initial_level"] == 0
    assert curriculum_term.params["success_min_episode_fraction"] == 0.9
    assert curriculum_term.params["skip_first_episode"] is True
    assert "terrain_curriculum" not in g1_29dof_wbt_gen.curriculum.setup_terms

    for group_name in ("actor_obs", "critic_obs"):
        flat_terms = sorted(g1_29dof_wbt_gen.observation.groups[group_name].terms)
        terrain_terms = sorted(g1_29dof_wbt_gen_terrain.observation.groups[group_name].terms)
        assert terrain_terms == flat_terms + ["terrain_height_scan", "tracker_projected_gravity"]
        terrain_cfgs = g1_29dof_wbt_gen_terrain.observation.groups[group_name].terms
        for term_name in ("base_ang_vel", "dof_pos", "dof_vel", "tracker_projected_gravity"):
            assert terrain_cfgs[term_name].history_length == 5
        for term_name in set(terrain_cfgs) - {
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "tracker_projected_gravity",
        }:
            assert terrain_cfgs[term_name].history_length == 1


def test_generated_motion_heading_reward_cli_off_switch():
    parsed = tyro.cli(
        AnnotatedExperimentConfig,
        args=(
            "exp:g1-29dof-wbt-gen-terrain",
            "--reward.terms.motion-heading-alignment.weight=0.0",
        ),
        config=TYRO_CONIFG,
    )
    assert parsed.reward.terms["motion_heading_alignment"].weight == 0.0


def test_generated_motion_terrain_scratch_has_fresh_tracker_initialization() -> None:
    parsed = tyro.cli(
        AnnotatedExperimentConfig,
        args=("exp:g1-29dof-wbt-gen-terrain-scratch",),
        config=TYRO_CONIFG,
    )

    assert parsed is not g1_29dof_wbt_gen_terrain
    assert parsed.training.name == "g1_29dof_wbt_gen_terrain_scratch_manager"
    assert parsed.training.num_envs == 1024
    assert parsed.training.checkpoint is None
    assert parsed.algo.config.checkpoint_load_mode == "strict"
    assert parsed.algo.config.load_optimizer is False
    assert parsed.algo.config.num_learning_iterations == 30000
    assert parsed.algo.config.save_interval == 500

    motion_cfg = parsed.command.setup_terms["motion_command"].params["motion_config"]
    assert motion_cfg.use_sim_terrain_scan is True
    assert motion_cfg.require_fully_measured_history is True
    assert motion_cfg.denoise_steps == 2
    assert "terrain_height_scan" in parsed.observation.groups["actor_obs"].terms
    assert parsed.reward.terms["motion_heading_alignment"].weight == 1.0
    assert parsed.curriculum.setup_terms["terrain_curriculum"].params["enabled"] is True

    # The scratch alias must not change the warm-start migration contract.
    assert g1_29dof_wbt_gen_terrain.algo.config.checkpoint_load_mode == "expand_input"
    assert g1_29dof_wbt_gen_terrain.algo.config.load_optimizer is True
    assert g1_29dof_wbt_gen_terrain_scratch.training.checkpoint is None
