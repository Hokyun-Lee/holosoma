from __future__ import annotations

import tyro
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _parse(name: str):
    return tyro.cli(AnnotatedExperimentConfig, args=(f"exp:{name}",), config=TYRO_CONIFG)


def _motion_cfg(experiment):
    return experiment.command.setup_terms["motion_command"].params["motion_config"]


def _has_five_frame_proprio(experiment) -> bool:
    terms = experiment.observation.groups["actor_obs"].terms
    return all(terms[name].history_length == 5 for name in ("base_ang_vel", "dof_pos", "dof_vel"))


def test_ablation_a_is_fixed_reference_terrain_finetune() -> None:
    cfg = _parse("g1-29dof-wbt-ablation-a-fixed-reference")

    assert cfg.command.setup_terms["motion_command"].func.endswith("wbt:MotionCommand")
    assert _motion_cfg(cfg).motion_file.endswith("lafan1_walk4_subject1_s500_rollout83x12_gen_mj.npz")
    assert _motion_cfg(cfg).reference_heading_source == "velocity_then_lookahead"
    assert _motion_cfg(cfg).reference_heading_stationary_fallback == "root_yaw"
    assert _motion_cfg(cfg).configure_local_terrain_scan is True
    assert _motion_cfg(cfg).local_terrain_scan_spacing == 0.1
    assert "terrain_height_scan" in cfg.observation.groups["actor_obs"].terms
    assert _has_five_frame_proprio(cfg)
    assert cfg.reward.terms["motion_heading_alignment"].weight == 1.0
    assert cfg.curriculum.setup_terms["terrain_curriculum"].params["enabled"] is True
    assert cfg.algo.config.checkpoint_load_mode == "expand_input"
    assert cfg.algo.config.num_learning_iterations > 0


def test_ablation_a_fixed_heading_fallback_is_cli_configurable() -> None:
    cfg = tyro.cli(
        AnnotatedExperimentConfig,
        args=(
            "exp:g1-29dof-wbt-ablation-a-fixed-reference",
            "--command.setup-terms.motion-command.params.motion-config.reference-heading-stationary-fallback=world_x",
        ),
        config=TYRO_CONIFG,
    )

    assert _motion_cfg(cfg).reference_heading_stationary_fallback == "world_x"


def test_ablation_b_is_generator_and_tracker_terrain_blind_update0() -> None:
    cfg = _parse("g1-29dof-wbt-ablation-b-generator-blind")

    assert cfg.terrain.terrain_term.mesh_type == "trimesh"
    assert cfg.command.setup_terms["motion_command"].func.endswith("wbt_gen:GeneratedMotionCommand")
    assert _motion_cfg(cfg).use_sim_terrain_scan is False
    assert "terrain_height_scan" not in cfg.observation.groups["actor_obs"].terms
    assert not _has_five_frame_proprio(cfg)
    assert cfg.reward.terms["motion_heading_alignment"].weight == 0.0
    assert cfg.curriculum.setup_terms["terrain_curriculum"].params["enabled"] is False
    assert cfg.algo.config.checkpoint_load_mode == "strict"
    assert cfg.algo.config.num_learning_iterations == 0


def test_ablation_c_and_d_share_full_architecture_but_d_updates() -> None:
    cfg_c = _parse("g1-29dof-wbt-ablation-c-full-no-finetune")
    cfg_d = _parse("g1-29dof-wbt-ablation-d-full-finetune")

    for cfg in (cfg_c, cfg_d):
        assert _motion_cfg(cfg).use_sim_terrain_scan is True
        assert "terrain_height_scan" in cfg.observation.groups["actor_obs"].terms
        assert _has_five_frame_proprio(cfg)
        assert cfg.reward.terms["motion_heading_alignment"].weight == 1.0
        assert cfg.algo.config.checkpoint_load_mode == "expand_input"
    assert cfg_c.algo.config.num_learning_iterations == 0
    assert cfg_c.algo.config.load_optimizer is False
    assert cfg_d.algo.config.num_learning_iterations > 0
    assert cfg_c.training.name != cfg_d.training.name


def test_ablation_e_only_hides_tracker_terrain() -> None:
    cfg = _parse("g1-29dof-wbt-ablation-e-generator-terrain-only")

    assert _motion_cfg(cfg).use_sim_terrain_scan is True
    assert "terrain_height_scan" not in cfg.observation.groups["actor_obs"].terms
    assert "terrain_height_scan" not in cfg.observation.groups["critic_obs"].terms
    assert _has_five_frame_proprio(cfg)
    assert cfg.observation.groups["actor_obs"].terms["tracker_projected_gravity"].history_length == 5
    assert cfg.reward.terms["motion_heading_alignment"].weight == 1.0
    assert cfg.algo.config.checkpoint_load_mode == "expand_input"


def test_ablation_f_only_disables_heading_reward() -> None:
    cfg = _parse("g1-29dof-wbt-ablation-f-no-heading-reward")

    assert _motion_cfg(cfg).use_sim_terrain_scan is True
    assert "terrain_height_scan" in cfg.observation.groups["actor_obs"].terms
    assert _has_five_frame_proprio(cfg)
    assert cfg.reward.terms["motion_heading_alignment"].weight == 0.0
    assert cfg.algo.config.num_learning_iterations > 0
