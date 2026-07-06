"""Whole Body Tracking termination presets for the Tocabi robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.config_values.wbt.tocabi.command import TOCABI_WBT_BODY_NAMES_TO_TRACK

tocabi_33dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                # Tocabi starts from a retargeted motion with larger early
                # waist/leg errors than G1. Keep termination loose while the
                # policy learns the basic recovery instead of resetting every
                # few frames.
                "bad_ref_pos_threshold": 0.8,
                "bad_ref_ori_threshold": 1.0,
                "bad_motion_body_pos_threshold": 0.5,
                "body_names_to_track": TOCABI_WBT_BODY_NAMES_TO_TRACK,
                "bad_motion_body_pos_body_names": [
                    "L_AnkleRoll_Link",
                    "R_AnkleRoll_Link",
                ],
                "bad_object_pos_threshold": 0.5,
                "bad_object_ori_threshold": 1.0,
            },
        ),
    }
)

__all__ = ["tocabi_33dof_wbt_termination"]
