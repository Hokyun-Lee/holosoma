"""Closed-loop generated-motion command preset for the G1 robot.

The frozen diffusion generator (holosoma.motion_gen) is queried inside the
training loop with the robot's measured state; ``motion_file`` is only the
seed used for episode initialization. Point ``generator_checkpoint`` at a
trained generator via CLI, e.g.:

    --command.setup_terms.motion_command.params.motion_config.generator-checkpoint=<ckpt.pt>
"""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, GeneratedMotionConfig
from holosoma.config_values.wbt.g1.command import init_pose_config, motion_config

gen_motion_config = GeneratedMotionConfig(
    # Seed motion for reset-state sampling (any WBT-format clip works).
    motion_file=motion_config.motion_file,
    body_names_to_track=list(motion_config.body_names_to_track),
    body_name_ref=list(motion_config.body_name_ref),
    use_adaptive_timesteps_sampler=False,  # failure bins are meaningless on seed frames
    noise_to_initial_pose=init_pose_config,
    generator_checkpoint="",  # required via CLI override
    replan_interval_s=0.5,
    denoise_steps=2,
    heading_mode="random",
    past_noise_std=0.01,
)

g1_29dof_wbt_gen_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt_gen:GeneratedMotionCommand",
            params={"motion_config": gen_motion_config},
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt_gen:GeneratedMotionCommand")
    },
    step_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt_gen:GeneratedMotionCommand")
    },
)

__all__ = ["g1_29dof_wbt_gen_command", "gen_motion_config"]
