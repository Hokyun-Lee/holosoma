from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.wbt.g1.experiment_gen import g1_29dof_wbt_gen
from holosoma.config_values.wbt.g1.experiment_gen_terrain import g1_29dof_wbt_gen_terrain
from holosoma.utils.eval_utils import CheckpointConfig


def test_flat_generated_motion_eval_defaults_remain_unchanged() -> None:
    eval_cfg = g1_29dof_wbt_gen.get_eval_config()

    assert eval_cfg.training.num_envs == 1
    assert eval_cfg.simulator.config.sim.max_episode_length_s == 100000.0


def test_terrain_eval_covers_all_types_and_keeps_episode_resets() -> None:
    eval_cfg = g1_29dof_wbt_gen_terrain.get_eval_config()

    assert eval_cfg.training.num_envs == 4
    assert eval_cfg.simulator.config.sim.max_episode_length_s == 10.0


def test_legacy_config_and_eval_checkpoint_state_defaults_are_safe() -> None:
    # Existing checkpoint metadata predates the terrain-specific override. Its
    # generic defaults remain parseable and can be overridden by eval CLI args.
    serialized = g1_29dof_wbt_gen_terrain.to_serializable_dict()
    legacy_cfg = ExperimentConfig(**{key: value for key, value in serialized.items() if key != "eval_overrides"})

    assert legacy_cfg.get_eval_config().training.num_envs == 1
    assert CheckpointConfig().restore_env_state is False
