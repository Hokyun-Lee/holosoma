"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CommandTermCfg:
    """Configuration for a single command or curriculum hook."""

    func: str
    """Import path for the command hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class CommandManagerCfg:
    """Configuration for the command manager."""

    params: dict[str, Any] = field(default_factory=dict)
    """Global parameters shared across command hooks."""

    setup_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, CommandTermCfg] = field(default_factory=dict)


########################################################################################################################
# Motion command configuration
########################################################################################################################
@dataclass(frozen=True)
class NoiseToInitialPoseConfig:
    """Initial pose of the robot and object to those in the motion file."""

    overall_noise_scale: float = 0.0
    """Overall noise scale for the initial pose."""

    dof_pos: float = 0.0
    """Noise scale for the initial dof position."""

    root_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root position x, y, z."""

    root_rot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root rotation roll, pitch, yaw."""

    root_lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root linear velocity vx, vy, vz."""

    root_ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root angular velocity wx, wy, wz."""

    object_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for object position x, y, z."""


@dataclass(frozen=True)
class MotionConfig:
    """Motion related configuration for Whole Body Tracking.

    NOTE:
    - Motion file is assumed to be in the format of:
      - joint_pos: (T, J)
      - joint_vel: (T, J)

      - body_pos_w: (T, B, 3)
      - body_quat_w: (T, B, 4) # wxyz -> xyzw
      - body_lin_vel_w: (T, B, 3)
      - body_ang_vel_w: (T, B, 3)

      If object is present in the motion file, it is assumed to be in the format of:
      - object_pos_w: (T, 3)
      - object_quat_w: (T, 4)
      - object_lin_vel_w: (T, 3)
      - object_ang_vel_w: (T, 3)

      If the motion clip assumes a terrain, the terrain has to be specified in holosoma/config/terrain/terrain_wbt.yaml
    """

    motion_file: str
    """Motion file (.npz) that contains motion_clips to track. """

    body_name_ref: list[str]
    """Body name of the reference frame (in general, torso_link). """
    body_names_to_track: list[str]
    """Key body names to track, used for reward/termination computation."""

    motion_dir: str = ""
    """Directory (or comma-separated directories) of .npz motion files.
    When non-empty, takes precedence over motion_file."""

    # motion sampling related
    use_adaptive_timesteps_sampler: bool = False
    """During training, whether to prioritize training on motion segments where the robot fails often."""

    start_at_timestep_zero_prob: float = 0.0
    """Probability of starting at timestep zero."""

    freeze_at_timestep_zero_prob: float = 0.0
    """When starting at timestep 0, probability of freezing motion counter at 0 (not advancing).
    This makes the robot practice holding the initial pose. Only applies when episode starts at timestep 0.
    Sampled independently each policy step; expected wait is roughly 1 / (1 - p) steps before unfreezing."""

    enable_default_pose_prepend: bool = False
    """If True, pre-append interpolated frames from default pose to the motion's first pose.
    This provides a smooth transition trajectory that the policy can track."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration in seconds of the pre-appended interpolation phase.
    Only used if enable_default_pose_prepend is True."""

    enable_default_pose_append: bool = False
    """If True, post-append interpolated frames from the motion's last pose back to default pose.
    This provides a smooth return trajectory that the policy can track."""

    default_pose_append_duration_s: float = 2.0
    """Duration in seconds of the post-appended interpolation phase.
    Only used if enable_default_pose_append is True."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)

    heading_reward_epsilon: float = 1.0e-6
    """Positive denominator epsilon for the optional heading reward.

    The paper does not publish this value; it is an implementation choice
    exposed for both fixed and generated reference commands.
    """

    heading_error_speed_threshold: float = 0.05
    """Measured speed below which logged heading error is reported as pi/2."""

    reference_heading_source: str = "velocity_then_lookahead"
    """Heading source for fixed references.

    ``velocity_then_lookahead`` uses reference root XY velocity first and a
    future root displacement second. ``lookahead_then_velocity`` reverses that
    priority. Generated commands provide their conditioned heading directly.
    """

    reference_heading_lookahead_s: float = 0.5
    """Lookahead duration used to infer a fixed reference's travel direction."""

    reference_heading_speed_threshold: float = 0.05
    """Minimum reference root XY speed considered a valid heading source."""

    reference_heading_displacement_threshold_m: float = 0.02
    """Minimum lookahead XY displacement considered a valid heading source."""

    reference_heading_stationary_fallback: str = "root_yaw"
    """Fixed-reference fallback when velocity and lookahead are stationary.

    Supported choices are ``root_yaw`` and ``world_x``.
    """

    configure_local_terrain_scan: bool = False
    """Configure a simulator scan for tracker observations in fixed-reference tasks."""

    local_terrain_scan_x_min: float = -0.3
    local_terrain_scan_x_max: float = 1.3
    local_terrain_scan_y_min: float = -0.8
    local_terrain_scan_y_max: float = 0.8
    local_terrain_scan_spacing: float = 0.1
    """Local scan grid in metres, flattened x-major with y changing fastest.

    These defaults match the generator dataset contract (17 x 17 = 289 raw
    absolute world-Z heights). The paper does not publish these values.
    """


@dataclass(frozen=True)
class GeneratedMotionConfig(MotionConfig):
    """Closed-loop generated-motion tracking (paper's fine-tuning stage).

    ``motion_file`` acts as the *seed* motion: episode resets initialize the
    robot from its frames (+noise) and seed the generator's conditioning
    history; all references afterwards come from the frozen diffusion
    generator, re-planned from the robot's measured state.
    """

    generator_checkpoint: str = ""
    """Path to a trained holosoma.motion_gen checkpoint (frozen at load)."""

    replan_interval_s: float = 0.5
    """How often the generator is queried with the measured robot state
    (paper: every 0.5 s = 2 Hz during fine-tuning)."""

    denoise_steps: int = 2
    """DDIM denoising steps per query (paper deployment/fine-tuning: 2)."""

    use_ema: bool = True
    """Use the EMA weights from the generator checkpoint."""

    deterministic_sampling: bool = False
    """Use fixed-seed DDIM initial noise for reproducible evaluation.

    Disabled by default to preserve the stochastic Stage-5--9 behavior.  The
    common Stage-10 evaluator enables it explicitly.
    """

    sampling_seed: int = 0
    """Initial-noise seed used when ``deterministic_sampling`` is enabled."""

    heading_mode: str = "random"
    """Target-heading conditioning per episode: "random" world direction or
    "current" (keep the facing direction at reset)."""

    past_noise_std: float = 0.01
    """Gaussian noise added to the measured past-state conditioning
    (paper perturbs generator conditions during training; scale is an
    implementation choice)."""

    use_sim_terrain_scan: bool = False
    """Condition the generator on the simulator's current local height scan.

    Disabled by default to preserve the Stage-5 flat experiment exactly;
    the Stage-6 terrain preset enables it and requires a scan-trained
    generator checkpoint.
    """

    require_fully_measured_history: bool = False
    """Require every past-state slot to be measured before a non-bootstrap
    replan.

    This is a strict runtime guard, not a different sampling clock: replans
    still occur every ``replan_interval_s``.  It defaults off to preserve the
    Stage-5 flat scheduler and is enabled by the terrain preset.  The paper
    does not specify this assertion; exposing it is an implementation choice.
    """

    body_origin_penetration_threshold_m: float = 0.02
    """Minimum generated-reference body-origin penetration for reporting a
    tracker correction case.

    This is only a scan-derived diagnostic proxy, not a collision test.  The
    paper does not publish such a threshold; the default is an implementation
    choice exposed for evaluation and ablation.
    """

    body_origin_correction_min_improvement_m: float = 0.01
    """Minimum reduction from generated-reference to measured-robot
    body-origin penetration for reporting a tracker correction case.

    This scan-derived diagnostic cannot establish that the policy deliberately
    corrected the reference.  The unpublished default is an implementation
    choice exposed for evaluation and ablation.
    """
