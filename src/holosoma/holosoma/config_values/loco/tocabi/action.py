"""Joint-position action preset for the Tocabi robot."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg

TOCABI_LOCO_CONTROLLED_JOINT_NAMES = [
    "L_HipYaw_Joint",
    "L_HipRoll_Joint",
    "L_HipPitch_Joint",
    "L_Knee_Joint",
    "L_AnklePitch_Joint",
    "L_AnkleRoll_Joint",
    "R_HipYaw_Joint",
    "R_HipRoll_Joint",
    "R_HipPitch_Joint",
    "R_Knee_Joint",
    "R_AnklePitch_Joint",
    "R_AnkleRoll_Joint",
    "L_Shoulder1_Joint",
    "L_Shoulder2_Joint",
    "L_Shoulder3_Joint",
    "R_Shoulder1_Joint",
    "R_Shoulder2_Joint",
    "R_Shoulder3_Joint",
]

tocabi_33dof_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:SelectiveJointPositionActionTerm",
            params={
                "controlled_joint_names": TOCABI_LOCO_CONTROLLED_JOINT_NAMES,
            },
            scale=1.0,
            clip=None,
        ),
    }
)

__all__ = ["TOCABI_LOCO_CONTROLLED_JOINT_NAMES", "tocabi_33dof_joint_pos"]
