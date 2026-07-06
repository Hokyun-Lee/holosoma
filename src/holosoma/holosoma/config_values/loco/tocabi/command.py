"""Locomotion command presets for the Tocabi robot."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg

tocabi_33dof_command = CommandManagerCfg(
    params={
        "locomotion_command_resampling_time": 10.0,
    },
    setup_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
            params={
                "gait_period": 1.0,
                "gait_period_randomization_width": 0.2,
            },
        ),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
            params={
                "command_ranges": {
                    "lin_vel_x": [-0.4, 0.6],
                    "lin_vel_y": [-0.3, 0.3],
                    "ang_vel_yaw": [-0.5, 0.5],
                    # "heading": [-3.14, 3.14],
                    # "lin_vel_x": [0.0, 0.0],
                    # "lin_vel_y": [-0.0, 0.0],
                    # "ang_vel_yaw": [-0.0, 0.0],
                    "heading": [-0.0, 0.0],
                },
                "stand_prob": 0.3,
            },
        ),
    },
    reset_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionCommand"),
    },
    step_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionCommand"),
    },
)

__all__ = ["tocabi_33dof_command"]
