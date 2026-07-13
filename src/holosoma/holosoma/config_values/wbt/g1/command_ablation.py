"""Fixed-reference command used by the Stage-10 A ablation.

The fixed clip keeps its original world heading. Random whole-trajectory yaw
augmentation is intentionally not approximated by rotating only the heading:
doing it fairly requires rotating every root/body pose, orientation, and
velocity together. This is a known comparability limitation versus generated
references conditioned on a random heading. The implemented reference heading
source, lookahead, thresholds, and stationary fallback remain configurable.
"""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg
from holosoma.config_values.wbt.g1.command import motion_config

fixed_reference_heading_motion_config = replace(
    motion_config,
    # Stage-4 fixed generated-reference rollout; callers may override this
    # with another WBT-schema NPZ.
    motion_file=(
        "logs/motion_gen/terrain_4090/samples/manual/"
        "lafan1_walk4_subject1_s500_rollout83x12_gen_mj.npz"
    ),
    use_adaptive_timesteps_sampler=False,
    reference_heading_source="velocity_then_lookahead",
    reference_heading_lookahead_s=0.5,
    reference_heading_speed_threshold=0.05,
    reference_heading_displacement_threshold_m=0.02,
    reference_heading_stationary_fallback="root_yaw",
    configure_local_terrain_scan=True,
    local_terrain_scan_x_min=-0.3,
    local_terrain_scan_x_max=1.3,
    local_terrain_scan_y_min=-0.8,
    local_terrain_scan_y_max=0.8,
    local_terrain_scan_spacing=0.1,
)

g1_29dof_wbt_fixed_reference_heading_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": fixed_reference_heading_motion_config},
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand"),
    },
    step_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand"),
    },
)


__all__ = [
    "fixed_reference_heading_motion_config",
    "g1_29dof_wbt_fixed_reference_heading_command",
]
