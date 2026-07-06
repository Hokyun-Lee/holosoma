"""Whole Body Tracking experiment presets for the Tocabi robot."""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import algo, simulator, terrain
from holosoma.config_values import robot as robot_values
from holosoma.config_values.loco.tocabi.action import tocabi_33dof_joint_pos
from holosoma.config_values.wbt.tocabi.command import tocabi_33dof_wbt_command
from holosoma.config_values.wbt.tocabi.curriculum import tocabi_33dof_wbt_curriculum
from holosoma.config_values.wbt.tocabi.observation import tocabi_33dof_wbt_observation
from holosoma.config_values.wbt.tocabi.randomization import tocabi_33dof_wbt_randomization
from holosoma.config_values.wbt.tocabi.reward import tocabi_33dof_wbt_fast_sac_reward, tocabi_33dof_wbt_reward
from holosoma.config_values.wbt.tocabi.termination import tocabi_33dof_wbt_termination

tocabi_33dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="tocabi_33dof_wbt_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=30000,
            num_learning_epochs=5,
            save_interval=4000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            init_at_random_ep_len=True,
            empirical_normalization=True,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
            virtual_gantry=replace(
                simulator.isaacsim.config.virtual_gantry,
                attachment_body_names=[
                    "Upperbody_Link",
                    "Pelvis_Link",
                    "Waist2_Link",
                    "Waist1_Link",
                ],
            ),
        ),
    ),
    robot=replace(
        robot_values.tocabi_33dof,
        init_state=replace(robot_values.tocabi_33dof.init_state, pos=[0.0, 0.0, 0.93]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=tocabi_33dof_wbt_observation,
    action=tocabi_33dof_joint_pos,
    termination=tocabi_33dof_wbt_termination,
    randomization=tocabi_33dof_wbt_randomization,
    command=tocabi_33dof_wbt_command,
    curriculum=tocabi_33dof_wbt_curriculum,
    reward=tocabi_33dof_wbt_reward,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.3, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.4, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.85, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.7, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.60, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.45, "inf"],
        },
    ),
)

tocabi_33dof_wbt_fast_sac = replace(
    tocabi_33dof_wbt,
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="tocabi_33dof_wbt_fast_sac_manager",
        num_envs=4096,
    ),
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=400000,
            v_max=20.0,
            v_min=-20.0,
            gamma=0.99,
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
        ),
    ),
    reward=tocabi_33dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

__all__ = ["tocabi_33dof_wbt", "tocabi_33dof_wbt_fast_sac"]
