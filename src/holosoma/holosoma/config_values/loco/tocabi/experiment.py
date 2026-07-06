from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import algo, simulator, terrain
from holosoma.config_values import robot as robot_values
from holosoma.config_values.loco.tocabi.action import TOCABI_LOCO_CONTROLLED_JOINT_NAMES, tocabi_33dof_joint_pos
from holosoma.config_values.loco.tocabi.command import tocabi_33dof_command
from holosoma.config_values.loco.tocabi.curriculum import tocabi_33dof_curriculum, tocabi_33dof_curriculum_fast_sac
from holosoma.config_values.loco.tocabi.observation import tocabi_33dof_loco_single_wolinvel
from holosoma.config_values.loco.tocabi.randomization import tocabi_33dof_randomization
from holosoma.config_values.loco.tocabi.reward import tocabi_33dof_loco, tocabi_33dof_loco_fast_sac
from holosoma.config_values.loco.tocabi.termination import tocabi_33dof_termination

tocabi_33dof = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-tocabi-manager", name="tocabi_33dof_manager"),
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=25000,
            use_symmetry=False,
            init_noise_std=0.8,
            entropy_coef=0.005,
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
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
        actions_dim=len(TOCABI_LOCO_CONTROLLED_JOINT_NAMES),
        control=replace(
            robot_values.tocabi_33dof.control,
            action_scale=0.12,
            action_clip_value=1.0,
            action_scales_by_effort_limit_over_p_gain=False,
        ),
    ),
    terrain=terrain.terrain_locomotion_mix,
    observation=tocabi_33dof_loco_single_wolinvel,
    action=tocabi_33dof_joint_pos,
    termination=tocabi_33dof_termination,
    randomization=tocabi_33dof_randomization,
    command=tocabi_33dof_command,
    curriculum=tocabi_33dof_curriculum,
    reward=tocabi_33dof_loco,
    nightly=NightlyConfig(
        iterations=10000,
        metrics={"Episode/rew_tracking_ang_vel": [0.5, "inf"], "Episode/rew_tracking_lin_vel": [0.4, "inf"]},
    ),
)

tocabi_33dof_fast_sac = replace(
    tocabi_33dof,
    training=TrainingConfig(project="hv-tocabi-manager", name="tocabi_33dof_fast_sac_manager"),
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=100000,
            use_symmetry=False,
        ),
    ),
    curriculum=tocabi_33dof_curriculum_fast_sac,
    reward=tocabi_33dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.6, "inf"], "Episode/rew_tracking_lin_vel": [0.7, "inf"]},
    ),
)

__all__ = ["tocabi_33dof", "tocabi_33dof_fast_sac"]
