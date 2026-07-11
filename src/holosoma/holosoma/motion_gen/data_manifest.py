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

# ---------------------------------------------------------------------------
# "paperscale" profile: match (exceed) the paper's ~1 h of training data.
#
# - All 40 LAFAN1 G1 clips except the 6 fallAndGetUp ones (ground-recovery
#   poses are out of this locomotion scope) -> 34 full-length clips ~2.2 h.
#   The first 2 s of each clip are trimmed (neutral-stance lead-in).
# - All OmniRetarget robot-terrain climb clips with every z-scale variant
#   (0.8-1.2) -> the dataset's own terrain augmentation, mirroring the
#   paper's terrain-height augmentation idea (~150 clips, ~30 min).
# - All robot-object-terrain clips (chair scenes + takes, ~15 clips).
# - Validation holds out *whole sequence groups* to avoid choreography /
#   z-scale-variant leakage: walk4 (only subject1 exists), run1 (both
#   subjects), climb_09 (all z-scales), scene_04.
# ---------------------------------------------------------------------------

LAFAN1_ALL_CLIPS = [
    # name, description  (fallAndGetUp* intentionally excluded)
    ("dance1_subject1", "dance / omnidirectional stepping"),
    ("dance1_subject2", "dance / omnidirectional stepping"),
    ("dance1_subject3", "dance / omnidirectional stepping"),
    ("dance2_subject1", "dance"),
    ("dance2_subject2", "dance"),
    ("dance2_subject3", "dance"),
    ("dance2_subject4", "dance"),
    ("dance2_subject5", "dance"),
    ("fight1_subject2", "fight motions"),
    ("fight1_subject3", "fight motions"),
    ("fight1_subject5", "fight motions"),
    ("fightAndSports1_subject1", "fight and sports"),
    ("fightAndSports1_subject4", "fight and sports"),
    ("jumps1_subject1", "jumps"),
    ("jumps1_subject2", "jumps"),
    ("jumps1_subject5", "jumps"),
    ("run1_subject2", "running (val)"),
    ("run1_subject5", "running (val)"),
    ("run2_subject1", "running"),
    ("run2_subject4", "running"),
    ("sprint1_subject2", "sprint"),
    ("sprint1_subject4", "sprint"),
    ("walk1_subject1", "walking"),
    ("walk1_subject2", "walking"),
    ("walk1_subject5", "walking"),
    ("walk2_subject1", "walking"),
    ("walk2_subject3", "walking"),
    ("walk2_subject4", "walking"),
    ("walk3_subject1", "walking with turns"),
    ("walk3_subject2", "walking with turns"),
    ("walk3_subject3", "walking with turns"),
    ("walk3_subject4", "walking with turns"),
    ("walk3_subject5", "walking with turns"),
    ("walk4_subject1", "side/backward stepping (val)"),
]

OMNI_CLIMB_IDS = list(range(29))  # climb_00 .. climb_28, each with 5 z-scales
OMNI_Z_SCALES = ["0.8", "0.9", "1.0", "1.1", "1.2"]

OMNI_TAKE_FILES = {
    # zip member basename -> stem suffix
    "Take 2025-09-02 08.48.33 PM_Skeleton 001_joint_positions_f0-2618_chair_2_stage.npz": "take_084833_chair",
    "Take 2025-09-02 08.48.33 PM_Skeleton 001_joint_positions_f0-2618_climb.npz": "take_084833_climb",
    "Take 2025-09-02 08.49.06 PM_Skeleton 001_joint_positions_f0-2994_chair_2_stage.npz": "take_084906_chair",
    "Take 2025-09-02 08.49.06 PM_Skeleton 001_joint_positions_f0-2994_climb.npz": "take_084906_climb",
    "Take 2025-09-02 08.51.15 PM_Skeleton 001_joint_positions_f0-2916_chair_2_stage.npz": "take_085115_chair",
    "Take 2025-09-02 08.51.15 PM_Skeleton 001_joint_positions_f0-2916_climb.npz": "take_085115_climb",
    "Take 2025-09-02 09.02.22 PM_Skeleton 001_joint_positions_f0-2613_chair_2_stage.npz": "take_090222_chair",
    "Take 2025-09-02 09.02.22 PM_Skeleton 001_joint_positions_f0-2613_climb.npz": "take_090222_climb",
    "Take 2025-09-02 09.04.06 PM_Skeleton 001_joint_positions_f0-2248_chair_2_stage.npz": "take_090406_chair",
    "Take 2025-09-02 09.04.06 PM_Skeleton 001_joint_positions_f0-2248_climb.npz": "take_090406_climb",
}
OMNI_SCENE_IDS = ["00", "01", "02", "04", "06"]

_LAFAN1_TRIM_HEAD_FRAMES = 60  # skip ~2 s of neutral-stance lead-in (choice)
_NO_END = 10**9  # sentinel: use the whole clip


def paperscale_manifest() -> list[ClipSpec]:
    clips: list[ClipSpec] = []
    for name, desc in LAFAN1_ALL_CLIPS:
        clips.append(
            ClipSpec(
                f"lafan1_{name}", "lafan1", "CC BY-NC-ND 4.0",
                f"{LAFAN1_BASE_URL}/{name}.csv", True, desc,
                frame_range=(_LAFAN1_TRIM_HEAD_FRAMES, _NO_END),
            )
        )
    for cid in OMNI_CLIMB_IDS:
        for z in OMNI_Z_SCALES:
            member = f"robot-terrain/climb_{cid:02d}_z_scale_{z}.npz"
            clips.append(
                ClipSpec(
                    f"omni_climb_{cid:02d}_z{z.replace('.', '_')}", "omniretarget", "MIT",
                    f"robot-terrain.zip:{member}", False,
                    f"terrain climb {cid:02d} (z-scale {z})",
                )
            )
    for sid in OMNI_SCENE_IDS:
        clips.append(
            ClipSpec(
                f"omni_scene_{sid}", "omniretarget", "MIT",
                f"robot-object-terrain.zip:robot-object-terrain/scene_{sid}_original.npz",
                False, f"chair scene {sid} (obstacle interaction)",
            )
        )
    for member, suffix in OMNI_TAKE_FILES.items():
        clips.append(
            ClipSpec(
                f"omni_{suffix}", "omniretarget", "MIT",
                f"robot-object-terrain.zip:robot-object-terrain/{member}",
                False, "take: chair/climb interaction",
            )
        )
    clips.append(MANIFEST[-1])  # omomo_largebox_003 (already in WBT format)
    return clips


def paperscale_splits() -> dict[str, list[str]]:
    manifest = paperscale_manifest()
    val_stems = {
        "lafan1_walk4_subject1",
        "lafan1_run1_subject2",
        "lafan1_run1_subject5",
        "omni_scene_04",
        *{f"omni_climb_09_z{z.replace('.', '_')}" for z in OMNI_Z_SCALES},
    }
    all_stems = [c.stem for c in manifest]
    missing = val_stems - set(all_stems)
    if missing:
        raise ValueError(f"val stems not in manifest: {missing}")
    return {
        "train": [s for s in all_stems if s not in val_stems],
        "val": sorted(val_stems),
    }


def get_manifest(profile: str) -> list[ClipSpec]:
    if profile == "small":
        return MANIFEST
    if profile == "paperscale":
        return paperscale_manifest()
    raise ValueError(f"Unknown profile '{profile}' (use 'small' or 'paperscale')")


def get_default_splits(profile: str) -> dict[str, list[str]]:
    if profile == "small":
        return DEFAULT_SPLITS
    if profile == "paperscale":
        return paperscale_splits()
    raise ValueError(f"Unknown profile '{profile}' (use 'small' or 'paperscale')")
