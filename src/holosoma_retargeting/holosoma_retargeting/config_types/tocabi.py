"""Tocabi-only configs for duplicated retargeting/conversion entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from holosoma_retargeting.config_types.data_type import (
    DATA_FORMAT_CONSTANTS,
    DEMO_JOINTS_REGISTRY,
    TOE_NAMES_BY_FORMAT,
)
from holosoma_retargeting.config_types.retargeter import RetargeterConfig, SelfCollisionConfig
from holosoma_retargeting.config_types.task import TaskConfig


TOCABI_SMPLH_JOINTS_MAPPING = {
    "Pelvis": "Pelvis_Link",
    "L_Hip": "L_Thigh_Link",
    "R_Hip": "R_Thigh_Link",
    "L_Knee": "L_Knee_Link",
    "R_Knee": "R_Knee_Link",
    "L_Shoulder": "L_Shoulder2_Link",
    "R_Shoulder": "R_Shoulder2_Link",
    "L_Elbow": "L_Elbow_Link",
    "R_Elbow": "R_Elbow_Link",
    "L_Ankle": "L_AnkleCenter_Link",
    "R_Ankle": "R_AnkleCenter_Link",
    "L_Toe": "L_Foot_Link",
    "R_Toe": "R_Foot_Link",
    "L_Wrist": "L_Wrist2_Link",
    "R_Wrist": "R_Wrist2_Link",
}

TOCABI_JOINT_NAMES = [
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
    "Waist1_Joint",
    "Waist2_Joint",
    "Upperbody_Joint",
    "L_Shoulder1_Joint",
    "L_Shoulder2_Joint",
    "L_Shoulder3_Joint",
    "L_Armlink_Joint",
    "L_Elbow_Joint",
    "L_Forearm_Joint",
    "L_Wrist1_Joint",
    "L_Wrist2_Joint",
    "Neck_Joint",
    "Head_Joint",
    "R_Shoulder1_Joint",
    "R_Shoulder2_Joint",
    "R_Shoulder3_Joint",
    "R_Armlink_Joint",
    "R_Elbow_Joint",
    "R_Forearm_Joint",
    "R_Wrist1_Joint",
    "R_Wrist2_Joint",
]


@dataclass(frozen=True)
class TocabiRobotConfig:
    robot_type: str = "tocabi"
    robot_dof: int = 33
    robot_height: float = 1.8
    robot_name: str = "tocabi_33dof"
    robot_urdf_file: str = "models/tocabi/tocabi_33dof.urdf"

    @property
    def ROBOT_DOF(self) -> int:
        return self.robot_dof

    @property
    def ROBOT_HEIGHT(self) -> float:
        return self.robot_height

    @property
    def ROBOT_NAME(self) -> str:
        return self.robot_name

    @property
    def ROBOT_URDF_FILE(self) -> str:
        return self.robot_urdf_file

    @property
    def FOOT_STICKING_LINKS(self) -> list[str]:
        return ["L_Foot_Link", "R_Foot_Link"]

    @property
    def MANUAL_LB(self) -> dict[str, float]:
        return {
            "3": -1.0,
            "4": -1.0,
            "5": -1.0,
            "6": -1.0,
            "7": -0.3,
            "8": -0.5,
            "9": -1.0,
            "10": -0.3,
            "11": -0.8,
            "12": -0.6,
            "13": -0.3,
            "14": -0.5,
            "15": -1.0,
            "16": -0.3,
            "17": -0.8,
            "18": -0.6,
            "19": -0.2, # waist yaw
            "20": -1.0, # waist pitch
            "21": -0.2, # waist roll
            "28": -0.2, # left wrist roll
            "29": -0.2, # left wrist pitch
            "38": -0.2, # right wrist roll
            "39": -0.2, # right wrist pitch
            "26": 0.3, # left elbow
            "36": 0.3, # right elbow
        }

    @property
    def MANUAL_UB(self) -> dict[str, float]:
        return {
            "3": 1.0,
            "4": 1.0,
            "5": 1.0,
            "6": 1.0,
            "7": 0.3,
            "8": 0.5,
            "9": 0.5,
            "10": 1.2,
            "11": 0.5,
            "12": 0.6,
            "13": 0.3,
            "14": 0.5,
            "15": 0.5,
            "16": 1.2,
            "17": 0.5,
            "18": 0.6,
            "19": 0.2, # waist yaw
            "20": 1.0, # waist pitch
            "21": 0.2, # waist roll
            "28": 0.2, # left wrist roll
            "29": 0.2, # left wrist pitch
            "38": 0.2, # right wrist roll
            "39": 0.2, # right wrist pitch
            "26": 1.4, # left elbow
            "36": 1.4, # right elbow
        }

    @property
    def MANUAL_COST(self) -> dict[str, float]:
        return {"19": 0.2, "20": 0.2, "21": 0.2,
                "12": 0.5,  # L_AnkleRoll_Joint
                "18": 0.5,  # R_AnkleRoll_Joint
                "22": 0.5,  # L_Shoulder1_Joint
                "32": 0.5,  # R_Shoulder1_Joint
    }

    @property
    def NOMINAL_TRACKING_INDICES(self) -> np.ndarray:
        return np.arange(22)


@dataclass(frozen=True)
class TocabiMotionDataConfig:
    data_format: str = "smplh"
    robot_type: str = "tocabi"

    @property
    def resolved_demo_joints(self) -> list[str]:
        if self.data_format not in DEMO_JOINTS_REGISTRY:
            raise ValueError(f"Unknown data_format: {self.data_format}")
        return DEMO_JOINTS_REGISTRY[self.data_format]

    @property
    def resolved_joints_mapping(self) -> dict[str, str]:
        if self.data_format != "smplh":
            raise ValueError("Tocabi duplicated retargeting currently supports data_format='smplh'.")
        return TOCABI_SMPLH_JOINTS_MAPPING

    @property
    def toe_names(self) -> list[str]:
        return TOE_NAMES_BY_FORMAT[self.data_format]

    @property
    def default_scale_factor(self) -> float | None:
        return DATA_FORMAT_CONSTANTS.get(self.data_format, {}).get("default_scale_factor")

    @property
    def default_human_height(self) -> float | None:
        return DATA_FORMAT_CONSTANTS.get(self.data_format, {}).get("default_human_height")

    def legacy_constants(self) -> dict[str, object]:
        return {
            "DEMO_JOINTS": self.resolved_demo_joints,
            "JOINTS_MAPPING": self.resolved_joints_mapping,
            "TOE_NAMES": self.toe_names,
            "DEFAULT_SCALE_FACTOR": self.default_scale_factor,
            "DEFAULT_HUMAN_HEIGHT": self.default_human_height,
        }


@dataclass
class TocabiRetargetingConfig:
    task_type: Literal["robot_only", "object_interaction", "climbing"] = "robot_only"
    robot: str = "tocabi"
    data_format: str | None = "smplh"
    task_name: str = "sub3_largebox_003"
    data_path: Path = Path("demo_data/OMOMO_new")
    save_dir: Path | None = None
    augmentation: bool = False
    robot_config: TocabiRobotConfig = field(default_factory=TocabiRobotConfig)
    motion_data_config: TocabiMotionDataConfig = field(default_factory=TocabiMotionDataConfig)
    task_config: TaskConfig = field(default_factory=TaskConfig)
    retargeter: RetargeterConfig = field(
        default_factory=lambda: RetargeterConfig(
            self_collision=SelfCollisionConfig(
                enable=True,
                pairs=[
                    ("L_Thigh_Link", "R_Thigh_Link"),
                    ("L_Knee_Link", "R_Knee_Link"),
                    ("L_AnkleCenter_Link", "R_AnkleCenter_Link"),
                    ("L_AnkleRoll_Link", "R_AnkleRoll_Link"),
                ],
                tolerance=0.04,
            ),
        )
    )


@dataclass(frozen=True)
class TocabiDataConversionConfig:
    input_file: str
    robot: str = "tocabi"
    data_format: str = "smplh"
    object_name: str | None = None
    input_fps: int = 30
    output_fps: int = 50
    line_range: tuple[int, int] | None = None
    has_dynamic_object: bool = False
    output_name: str | None = None
    once: bool = False
    use_omniretarget_data: bool = False
    robot_config: TocabiRobotConfig = field(default_factory=TocabiRobotConfig)
    motion_data_config: TocabiMotionDataConfig = field(default_factory=TocabiMotionDataConfig)
    joint_names: list[str] | None = None

    @property
    def JOINT_NAMES(self) -> list[str]:
        return self.joint_names if self.joint_names is not None else TOCABI_JOINT_NAMES
