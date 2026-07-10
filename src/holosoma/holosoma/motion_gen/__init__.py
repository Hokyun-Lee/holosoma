"""Diffusion-based future motion generator for whole-body humanoid locomotion.

Minimal reproduction of the motion generator from "Learning Whole-Body Humanoid
Locomotion via Motion Generation and Motion Tracking" (arXiv:2604.17335),
adapted to the HoloSoma G1 29-DoF motion format.

Public inference API:
    from holosoma.motion_gen import MotionGenerator, MotionGeneratorInput
"""

from holosoma.motion_gen.features import FeatureLayout
from holosoma.motion_gen.sampling import MotionGenerator, MotionGeneratorInput, MotionGeneratorOutput

__all__ = [
    "FeatureLayout",
    "MotionGenerator",
    "MotionGeneratorInput",
    "MotionGeneratorOutput",
]
