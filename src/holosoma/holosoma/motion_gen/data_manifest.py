"""Manifest of the ~11 motions used for the first motion-generator training.

Selection criteria (see docs/motion_generator_implementation_notes.md):
    - flat locomotion coverage: straight walk, turns, side/backward steps,
      omnidirectional dance, run, sprint  (LAFAN1 G1 retarget, CC BY-NC-ND 4.0)
    - terrain motions: 3 OmniRetarget climb clips (MIT)
    - obstacle interaction: 1 OmniRetarget chair scene (MIT)
    - whole-body arm coordination: OMOMO large-box demo already shipped with
      HoloSoma (upstream OMOMO data is CC BY-NC 4.0)
LAFAN1 clips are trimmed to ~60 s each to keep the total dataset within the
3-10 minute scope of this first reproduction.

Data is downloaded locally and never committed to the repository.
"""

from __future__ import annotations

from dataclasses import dataclass

LAFAN1_BASE_URL = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1"
OMNIRETARGET_BASE_URL = "https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset/resolve/main"

# sha256 of the OmniRetarget zips as downloaded on 2026-07-10; a mismatch
# means the upstream dataset changed and the manifest should be re-checked.
OMNIRETARGET_ZIPS = {
    "robot-terrain.zip": "f30bdc915287547dbb2225bdb4efe323dbfcc2237e2a4e82c7482dd05402aea6",
    "robot-object-terrain.zip": "c3e2a7a3827182b23f5e6b72f496cda0448ee58c4ebe6f316bee46adc21c5505",
}


@dataclass(frozen=True)
class ClipSpec:
    stem: str  # processed file stem (data/motion_gen/processed/<stem>.npz)
    source: str  # lafan1 | omniretarget | holosoma_demo
    license: str
    origin: str  # download URL / zip member / repo path
    flat_terrain: bool
    description: str
    frame_range: tuple[int, int] | None = None  # input-fps frames, [start, end)
    input_fps: int = 30


MANIFEST: list[ClipSpec] = [
    # ---- LAFAN1 retargeted to G1 (30 fps CSV; root pos xyz, quat xyzw, 29 joints)
    ClipSpec(
        "lafan1_walk1_subject1", "lafan1", "CC BY-NC-ND 4.0",
        f"{LAFAN1_BASE_URL}/walk1_subject1.csv", True,
        "steady straight walking", frame_range=(600, 2400),
    ),
    ClipSpec(
        "lafan1_walk3_subject2", "lafan1", "CC BY-NC-ND 4.0",
        f"{LAFAN1_BASE_URL}/walk3_subject2.csv", True,
        "walking with turns and direction changes", frame_range=(600, 2400),
    ),
    ClipSpec(
        "lafan1_walk4_subject1", "lafan1", "CC BY-NC-ND 4.0",
        f"{LAFAN1_BASE_URL}/walk4_subject1.csv", True,
        "walking variations incl. side/backward stepping (val)", frame_range=(600, 2400),
    ),
    ClipSpec(
        "lafan1_dance1_subject2", "lafan1", "CC BY-NC-ND 4.0",
        f"{LAFAN1_BASE_URL}/dance1_subject2.csv", True,
        "omnidirectional stepping and crouch-like poses", frame_range=(122, 1922),
    ),
    ClipSpec(
        "lafan1_run2_subject1", "lafan1", "CC BY-NC-ND 4.0",
        f"{LAFAN1_BASE_URL}/run2_subject1.csv", True,
        "jogging / running", frame_range=(600, 2400),
    ),
    ClipSpec(
        "lafan1_sprint1_subject2", "lafan1", "CC BY-NC-ND 4.0",
        f"{LAFAN1_BASE_URL}/sprint1_subject2.csv", True,
        "dynamic sprint", frame_range=(600, 1800),
    ),
    # ---- OmniRetarget (30 fps npz; qpos = [root_quat wxyz, root_pos, 29 joints])
    ClipSpec(
        "omni_climb_02", "omniretarget", "MIT",
        "robot-terrain.zip:robot-terrain/climb_02_z_scale_1.0.npz", False,
        "terrain climb (stair/platform-like)",
    ),
    ClipSpec(
        "omni_climb_14", "omniretarget", "MIT",
        "robot-terrain.zip:robot-terrain/climb_14_z_scale_1.0.npz", False,
        "terrain climb",
    ),
    ClipSpec(
        "omni_climb_09", "omniretarget", "MIT",
        "robot-terrain.zip:robot-terrain/climb_09_z_scale_1.0.npz", False,
        "terrain climb (val)",
    ),
    ClipSpec(
        "omni_chair_scene_06", "omniretarget", "MIT",
        "robot-object-terrain.zip:robot-object-terrain/scene_06_original.npz", False,
        "chair scene: obstacle interaction (robot columns only)",
    ),
    # ---- Already shipped with HoloSoma (converted, 50 fps WBT format)
    ClipSpec(
        "omomo_largebox_003", "holosoma_demo", "OMOMO upstream: CC BY-NC 4.0",
        "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz",
        True,
        "whole-body large-box interaction (arm coordination)",
        input_fps=50,
    ),
]

DEFAULT_SPLITS = {
    "train": [
        "lafan1_walk1_subject1",
        "lafan1_walk3_subject2",
        "lafan1_dance1_subject2",
        "lafan1_run2_subject1",
        "lafan1_sprint1_subject2",
        "omni_climb_02",
        "omni_climb_14",
        "omni_chair_scene_06",
        "omomo_largebox_003",
    ],
    "val": [
        "lafan1_walk4_subject1",
        "omni_climb_09",
    ],
}
