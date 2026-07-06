"""Locomotion reward presets for the Tocabi robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

TOCABI_LOCO_POSE_WEIGHTS = [
    # Legs: keep hip yaw/roll relaxed, prefer nominal hip pitch/knee/ankle shape.
    0.5,
    2.0,
    5.0,
    3.0,
    5.0,
    4.0,
    0.5,
    2.0,
    5.0,
    3.0,
    5.0,
    4.0,
    # Waist/head: strongly discourage upper-body drift for early walking.
    20.0,
    20.0,
    20.0,
    # Left arm.
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    1.0,
    1.0,
    # Neck/head.
    5.0,
    5.0,
    # Right arm.
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    1.0,
    1.0,
]

tocabi_33dof_loco = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.08, "tracking_sigma": 0.008},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={"pose_weights": TOCABI_LOCO_POSE_WEIGHTS},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.18},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=1.0,
            params={},
        ),
    },
)

tocabi_33dof_loco_fast_sac = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        **tocabi_33dof_loco.terms,
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=10.0,
            params={},
        ),
    },
)

__all__ = ["TOCABI_LOCO_POSE_WEIGHTS", "tocabi_33dof_loco", "tocabi_33dof_loco_fast_sac"]
