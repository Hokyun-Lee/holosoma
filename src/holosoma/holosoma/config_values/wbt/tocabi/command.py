"""Whole Body Tracking command presets for the Tocabi robot."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

TOCABI_WBT_BODY_NAMES_TO_TRACK = [
    "Pelvis_Link",
    "L_Thigh_Link",
    "L_Knee_Link",
    "L_AnkleRoll_Link",
    "R_Thigh_Link",
    "R_Knee_Link",
    "R_AnkleRoll_Link",
    "Upperbody_Link",
    "L_Shoulder2_Link",
    "L_Elbow_Link",
    "L_Wrist2_Link",
    "R_Shoulder2_Link",
    "R_Elbow_Link",
    "R_Wrist2_Link",
]

init_pose_config = NoiseToInitialPoseConfig(
    # Keep resets deterministic while bringing Tocabi up. Randomization can be
    # re-enabled after the spawned pose is physically stable.
    overall_noise_scale=0.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.5, 0.5, 0.2],
    root_ang_vel=[0.52, 0.52, 0.78],
    object_pos=[0.05, 0.05, 0.0],
)

motion_config = MotionConfig(
    motion_file="holosoma/data/motions/tocabi_33dof/whole_body_tracking/sub3_largebox_003_mj.npz",
    body_names_to_track=TOCABI_WBT_BODY_NAMES_TO_TRACK,
    body_name_ref=["Upperbody_Link"],
    use_adaptive_timesteps_sampler=True,
    start_at_timestep_zero_prob=1.0,
    enable_default_pose_prepend=True,
    default_pose_prepend_duration_s=2.0,
    noise_to_initial_pose=init_pose_config,
)

tocabi_33dof_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt_tocabi:MotionCommand",
            params={
                "motion_config": motion_config,
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt_tocabi:MotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt_tocabi:MotionCommand",
        )
    },
)

__all__ = ["TOCABI_WBT_BODY_NAMES_TO_TRACK", "tocabi_33dof_wbt_command"]
