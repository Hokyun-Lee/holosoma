"""Common Stage-10 metrics for whole-body terrain evaluation."""

from __future__ import annotations

import dataclasses
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.command import GeneratedMotionConfig, MotionConfig
from holosoma.config_types.eval_callback import TerrainMetricsConfig
from holosoma.managers.curriculum.terms.terrain import target_heading_forward_progress_m
from holosoma.managers.observation.terms.wbt import get_projected_gravity
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.terrain_evaluation import (
    TerrainEvaluationAccumulator,
    checkpoint_sha256,
    write_terrain_evaluation_outputs,
)
from loguru import logger


class TerrainMetricsCallback(RLEvalCallback):
    """Collect exact pre-reset WBT metrics and stop at a fixed episode count.

    ``BaseTask`` resets terminated environments before algorithm-level
    callbacks run.  To preserve the terminal robot pose and termination masks,
    this callback wraps the WBT ``_update_log_dict`` hook during evaluation.
    The hook runs after termination/reward computation and before reset; it is
    restored when evaluation ends.
    """

    _COMMON_TRACKING_METRICS = (
        "motion/error_ref_pos",
        "motion/error_ref_rot",
        "motion/error_ref_lin_vel",
        "motion/error_body_pos",
        "motion/error_body_lin_vel",
        "motion/error_joint_pos",
        "motion/error_joint_vel",
    )

    def __init__(self, config: TerrainMetricsConfig, training_loop: Any = None):
        super().__init__(config, training_loop)
        self._validate_config()
        self._env: Any = None
        self._motion_command: Any = None
        self._terrain_state: Any = None
        self._terrain_curriculum: Any = None
        self._original_update_log_dict: Any = None
        self._armed = False
        self._outputs_written = False
        self._fallback_start_root_xy: torch.Tensor | None = None
        self._fallback_episode_heading_w: torch.Tensor | None = None
        self._original_motion_config: MotionConfig | None = None
        self._original_curriculum_runtime: dict[str, Any] | None = None
        self._original_terrain_levels: torch.Tensor | None = None
        self._current_actions: torch.Tensor | None = None
        self._current_eval_step = -1
        self._correction_exemplar: dict[str, np.ndarray] | None = None
        self.accumulator: TerrainEvaluationAccumulator | None = None
        self.output_paths: dict[str, Path] = {}

    def _validate_config(self) -> None:
        if self.config.episode_count < 1:
            raise ValueError("episode_count must be >= 1")
        if self.config.success_distance_m < 0.0:
            raise ValueError("success_distance_m must be non-negative")
        if self.config.fall_root_height_m < 0.0:
            raise ValueError("fall_root_height_m must be non-negative")
        if not 0.0 <= self.config.fall_upright_cosine <= 1.0:
            raise ValueError("fall_upright_cosine must be in [0, 1]")
        if (
            not math.isfinite(self.config.body_origin_penetration_threshold_m)
            or self.config.body_origin_penetration_threshold_m < 0.0
        ):
            raise ValueError("body_origin_penetration_threshold_m must be finite and non-negative")
        if (
            not math.isfinite(self.config.body_origin_correction_min_improvement_m)
            or self.config.body_origin_correction_min_improvement_m < 0.0
        ):
            raise ValueError("body_origin_correction_min_improvement_m must be finite and non-negative")
        if self.config.heading_speed_threshold_mps < 0.0:
            raise ValueError("heading_speed_threshold_mps must be non-negative")
        if self.config.fixed_terrain_level < 0:
            raise ValueError("fixed_terrain_level must be non-negative")
        if self.config.evaluation_phase_mode not in {"zero", "uniform"}:
            raise ValueError("evaluation_phase_mode must be 'zero' or 'uniform'")
        if self.config.deterministic_per_env_sampling and not self.config.deterministic_generator:
            raise ValueError(
                "deterministic_per_env_sampling requires deterministic_generator=True"
            )

    def _get_env(self) -> Any:
        return self.training_loop._unwrap_env()

    def on_pre_evaluate_policy(self) -> None:
        self._env = self._get_env()
        if not hasattr(self._env, "_update_log_dict"):
            raise TypeError("TerrainMetricsCallback requires an environment with _update_log_dict")
        self._motion_command = self._env.command_manager.get_state("motion_command")
        if self._motion_command is None or not hasattr(self._motion_command, "target_heading_w"):
            raise TypeError("Terrain evaluation requires motion_command.target_heading_w")
        self._terrain_state = self._env.terrain_manager.get_state("locomotion_terrain")
        self._terrain_curriculum = self._env.curriculum_manager.get_term("terrain_curriculum")
        self._enforce_runtime_controls()
        # PPO performs a setup reset before callbacks and a second reset after
        # them. Re-seed here so the evaluated episode phases/headings are not
        # affected by model construction or that setup-only reset.
        random.seed(self.config.evaluation_seed)
        np.random.seed(self.config.evaluation_seed % (2**32))
        torch.manual_seed(self.config.evaluation_seed)
        reset_sampling_counters = getattr(
            self._motion_command, "reset_deterministic_sampling_counters", None
        )
        if reset_sampling_counters is not None:
            reset_sampling_counters()
        self._fallback_start_root_xy = torch.zeros(
            self._env.num_envs,
            2,
            # Isaac Sim exposes a RootStatesProxy (without tensor metadata),
            # while Isaac Gym exposes a tensor directly.  Slicing either
            # boundary returns the canonical xyzw tensor.
            dtype=self._env.simulator.robot_root_states[:].dtype,
            device=self._env.device,
        )
        self._fallback_episode_heading_w = torch.zeros_like(self._fallback_start_root_xy)
        self.accumulator = TerrainEvaluationAccumulator(
            num_envs=self._env.num_envs,
            target_episodes=self.config.episode_count,
            success_distance_m=self.config.success_distance_m,
            expected_terrain_types=self._expected_terrain_types(),
        )

        self._original_update_log_dict = self._env._update_log_dict

        def _capture_after_update() -> None:
            self._original_update_log_dict()
            if self._armed and not self._require_accumulator().complete:
                self._capture_pre_reset_step()

        self._env._update_log_dict = _capture_after_update
        logger.info(
            "TerrainMetricsCallback: variant={}, episodes={}, success_distance={}m, fixed_level={}",
            self.config.variant,
            self.config.episode_count,
            self.config.success_distance_m,
            self.config.fixed_terrain_level,
        )

    def on_pre_eval_env_step(self, actor_state: dict) -> dict:
        accumulator = self._require_accumulator()
        if accumulator.complete:
            actor_state["stop"] = True
            return actor_state

        actions = actor_state.get("actions")
        if not torch.is_tensor(actions) or actions.ndim != 2 or actions.shape[0] != self._env.num_envs:
            actual_shape = None if not torch.is_tensor(actions) else tuple(actions.shape)
            raise RuntimeError(
                "Terrain correction exemplars require current actions shaped "
                f"({self._env.num_envs}, A), got {actual_shape}"
            )
        self._current_actions = actions.detach().clone()
        eval_step = actor_state.get("step", -1)
        if isinstance(eval_step, bool) or not isinstance(eval_step, int):
            raise TypeError(f"actor_state['step'] must be an int, got {type(eval_step).__name__}")
        self._current_eval_step = eval_step

        inactive_ids = np.flatnonzero(~accumulator.active_mask)
        if inactive_ids.size:
            self._start_episodes(inactive_ids)
        # PPO performs an initialization reset after on_pre_evaluate_policy.
        # Arming here deliberately ignores that setup-only simulator step.
        self._armed = True
        return actor_state

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        if self._require_accumulator().complete:
            actor_state["stop"] = True
        return actor_state

    def on_post_evaluate_policy(self) -> None:
        if self._original_update_log_dict is not None:
            self._env._update_log_dict = self._original_update_log_dict
        try:
            self._write_outputs()
            accumulator = self._require_accumulator()
            if not accumulator.complete:
                message = (
                    f"Terrain evaluation stopped after {len(accumulator.records)}/"
                    f"{accumulator.target_episodes} completed episodes"
                )
                if self.config.fail_on_incomplete:
                    raise RuntimeError(message)
                logger.warning(message)
        finally:
            self._restore_runtime_controls()

    def _enforce_runtime_controls(self) -> None:
        motion_config = getattr(self._motion_command, "motion_cfg", None)
        if isinstance(motion_config, MotionConfig):
            self._original_motion_config = motion_config
            motion_config = dataclasses.replace(
                motion_config,
                evaluation_phase_mode=self.config.evaluation_phase_mode,
            )
            self._motion_command.motion_cfg = motion_config
        if isinstance(motion_config, GeneratedMotionConfig):
            deterministic_config = dataclasses.replace(
                motion_config,
                deterministic_sampling=self.config.deterministic_generator,
                deterministic_per_env_sampling=self.config.deterministic_per_env_sampling,
                sampling_seed=self.config.generator_sampling_seed,
            )
            self._motion_command.motion_cfg = deterministic_config
            self._motion_command.gen_cfg = deterministic_config

        if not self._curriculum_layout_enabled():
            if self.config.fixed_terrain_level != 0:
                raise ValueError("A non-zero fixed_terrain_level requires a curriculum terrain layout")
            return

        num_levels = int(self._terrain_state.num_curriculum_levels)
        if self.config.fixed_terrain_level >= num_levels:
            raise ValueError(
                f"fixed_terrain_level {self.config.fixed_terrain_level} outside [0, {num_levels})"
            )
        env_ids = torch.arange(self._env.num_envs, dtype=torch.long, device=self._env.device)
        self._original_terrain_levels = self._terrain_state.terrain_levels.detach().clone()
        self._terrain_state.set_curriculum_origins(env_ids, self.config.fixed_terrain_level)

        curriculum = self._terrain_curriculum
        if curriculum is not None:
            self._original_curriculum_runtime = {
                "enabled": curriculum.enabled,
                "initial_level": curriculum.initial_level,
                "min_level": curriculum.min_level,
                "max_level": curriculum.max_level,
            }
            curriculum.enabled = True
            curriculum.initial_level = self.config.fixed_terrain_level
            curriculum.min_level = self.config.fixed_terrain_level
            curriculum.max_level = self.config.fixed_terrain_level

    def _restore_runtime_controls(self) -> None:
        if self._original_motion_config is not None:
            self._motion_command.motion_cfg = self._original_motion_config
            if isinstance(self._original_motion_config, GeneratedMotionConfig):
                self._motion_command.gen_cfg = self._original_motion_config
        if self._original_terrain_levels is not None:
            env_ids = torch.arange(self._env.num_envs, dtype=torch.long, device=self._env.device)
            self._terrain_state.set_curriculum_origins(env_ids, self._original_terrain_levels)
        if self._original_curriculum_runtime is not None:
            for name, value in self._original_curriculum_runtime.items():
                setattr(self._terrain_curriculum, name, value)

    def _curriculum_layout_enabled(self) -> bool:
        terrain = getattr(self._terrain_state, "terrain", None)
        return bool(getattr(terrain, "curriculum_enabled", False))

    def _expected_terrain_types(self) -> tuple[str, ...]:
        if self._curriculum_layout_enabled():
            return tuple(str(name) for name in self._terrain_state.terrain_type_names)
        if self.config.scenario_label:
            return (self.config.scenario_label,)
        terrain = getattr(self._terrain_state, "terrain", None)
        generated_types = tuple(str(name) for name in getattr(terrain, "_terrain_types", ()))
        if len(generated_types) == 1:
            return generated_types
        mesh_type = getattr(getattr(self._terrain_state, "_cfg", None), "mesh_type", "unknown")
        mesh_label = str(getattr(mesh_type, "value", mesh_type))
        if generated_types:
            raise ValueError(
                "Non-curriculum mixed terrain evaluation requires an explicit scenario_label "
                "or per-environment terrain type IDs"
            )
        return (mesh_label,)

    def _require_accumulator(self) -> TerrainEvaluationAccumulator:
        if self.accumulator is None:
            raise RuntimeError("TerrainMetricsCallback has not been initialized")
        return self.accumulator

    def _start_episodes(self, env_ids_np: np.ndarray) -> None:
        env_ids = torch.as_tensor(env_ids_np, dtype=torch.long, device=self._env.device)
        root_xy = self._env.simulator.robot_root_states[env_ids, :2]
        headings = self._episode_headings(env_ids)
        self._fallback_start_root_xy[env_ids] = root_xy
        self._fallback_episode_heading_w[env_ids] = headings
        terrain_types, terrain_levels = self._terrain_labels(env_ids)
        self._require_accumulator().start_episodes(
            env_ids_np,
            target_headings=self._to_numpy(headings),
            terrain_types=terrain_types,
            terrain_levels=self._to_numpy(terrain_levels),
        )

    def _episode_headings(self, env_ids: torch.Tensor) -> torch.Tensor:
        curriculum = self._terrain_curriculum
        if curriculum is not None and getattr(curriculum, "enabled", False):
            headings = curriculum.episode_target_heading_w[env_ids]
        else:
            headings = self._motion_command.target_heading_w[env_ids]
        norms = torch.linalg.vector_norm(headings, dim=-1, keepdim=True)
        if not bool(torch.isfinite(headings).all()) or bool(torch.any(norms <= 1.0e-8)):
            raise RuntimeError("Episode target headings must be finite non-zero vectors")
        return headings / norms

    def _terrain_labels(self, env_ids: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        if self._curriculum_layout_enabled():
            type_ids = self._terrain_state.terrain_type_ids[env_ids].detach().cpu().tolist()
            names = list(self._terrain_state.terrain_type_names)
            terrain_types = [str(names[type_id]) for type_id in type_ids]
            return terrain_types, self._terrain_state.terrain_levels[env_ids]

        mesh_type = getattr(getattr(self._terrain_state, "_cfg", None), "mesh_type", "unknown")
        terrain_type = self.config.scenario_label or str(getattr(mesh_type, "value", mesh_type))
        terrain = getattr(self._terrain_state, "terrain", None)
        generated_types = tuple(str(name) for name in getattr(terrain, "_terrain_types", ()))
        if not self.config.scenario_label and len(generated_types) == 1:
            terrain_type = generated_types[0]
        return [terrain_type] * env_ids.numel(), torch.full_like(env_ids, -1)

    def _capture_pre_reset_step(self) -> None:
        root_states = self._env.simulator.robot_root_states
        root_xy = root_states[:, :2]
        headings = self._active_episode_headings()
        forward_progress = self._forward_progress(root_xy, headings)
        terrain_height_at_root = self._terrain_state.query_terrain_heights(root_xy)
        root_height = root_states[:, 2] - terrain_height_at_root
        projected_gravity = get_projected_gravity(self._env)
        upright_cosine = -projected_gravity[:, 2]
        falls = (root_height < self.config.fall_root_height_m) | (
            upright_cosine < self.config.fall_upright_cosine
        )

        metrics = self._collect_step_metrics(
            headings=headings,
            root_height=root_height,
            upright_cosine=upright_cosine,
        )
        termination_manager = self._env.termination_manager
        self._require_accumulator().observe(
            forward_progress_m=self._to_numpy(forward_progress),
            falls=self._to_numpy(falls),
            terminated=self._to_numpy(termination_manager.terminated),
            timeouts=self._to_numpy(termination_manager.time_outs),
            metrics={name: self._to_numpy(value) for name, value in metrics.items()},
        )

    def _active_episode_headings(self) -> torch.Tensor:
        curriculum = self._terrain_curriculum
        if curriculum is not None and getattr(curriculum, "enabled", False):
            return curriculum.episode_target_heading_w
        return self._fallback_episode_heading_w

    def _forward_progress(self, root_xy: torch.Tensor, headings: torch.Tensor) -> torch.Tensor:
        curriculum = self._terrain_curriculum
        if curriculum is not None and getattr(curriculum, "enabled", False):
            starts = curriculum.episode_start_root_xy
        else:
            starts = self._fallback_start_root_xy
        return target_heading_forward_progress_m(root_xy, starts, headings)

    def _collect_step_metrics(
        self,
        *,
        headings: torch.Tensor,
        root_height: torch.Tensor,
        upright_cosine: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        velocity_xy = self._env.simulator.robot_root_states[:, 7:9]
        speed = torch.linalg.vector_norm(velocity_xy, dim=-1)
        velocity_direction = velocity_xy / speed.unsqueeze(-1).clamp_min(1.0e-8)
        cosine = torch.sum(velocity_direction * headings, dim=-1).clamp(-1.0, 1.0)
        heading_error = torch.acos(cosine)
        heading_error = torch.where(
            speed < self.config.heading_speed_threshold_mps,
            torch.full_like(heading_error, math.pi / 2.0),
            heading_error,
        )
        moving = speed >= self.config.heading_speed_threshold_mps
        metrics: dict[str, torch.Tensor] = {
            "motion/heading_error_rad": heading_error,
            "motion/heading_error_moving_rad": torch.where(
                moving,
                heading_error,
                torch.full_like(heading_error, torch.nan),
            ),
            "motion/heading_low_speed_fraction": (~moving).to(dtype=torch.float32),
            "motion/heading_speed_mps": speed,
            "fall/root_height_above_terrain_m": root_height,
            "fall/upright_cosine": upright_cosine,
        }

        command_metrics = getattr(self._motion_command, "metrics", {})
        for name in self._COMMON_TRACKING_METRICS:
            value = command_metrics.get(name)
            if torch.is_tensor(value) and value.shape == (self._env.num_envs,):
                metrics[name] = value

        proxy_metrics, proxy_frame = self._body_origin_penetration_proxy_metrics()
        metrics.update(proxy_metrics)
        self._capture_first_correction_exemplar(proxy_frame, headings)
        contact_count = self._undesired_contact_count()
        if contact_count is not None:
            metrics["terrain/undesired_contact_body_count"] = contact_count
            metrics["terrain/undesired_contact_any"] = (contact_count > 0).to(dtype=torch.float32)
        return metrics

    def _body_origin_penetration_proxy_metrics(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute scan-independent body-origin diagnostics for every variant.

        These values compare point origins with terrain height.  They are not
        collision-shape or mesh penetration measurements.
        """

        robot_body_pos = self._motion_command.robot_body_pos_w
        reference_body_pos = self._motion_command.body_pos_relative_w
        if robot_body_pos.shape != reference_body_pos.shape:
            raise RuntimeError(
                "Robot/reference body arrays must match for the correction proxy, got "
                f"{tuple(robot_body_pos.shape)} and {tuple(reference_body_pos.shape)}"
            )
        robot_height, robot_penetration = self._origin_penetration_frame("robot", robot_body_pos)
        reference_height, reference_penetration = self._origin_penetration_frame(
            "reference", reference_body_pos
        )
        robot_max = robot_penetration.max(dim=-1).values
        reference_max = reference_penetration.max(dim=-1).values
        signed_improvement = reference_max - robot_max
        correction_case = (
            (reference_max >= self.config.body_origin_penetration_threshold_m)
            & (signed_improvement >= self.config.body_origin_correction_min_improvement_m)
        )
        metrics = {
            **self._origin_penetration_metrics("robot", robot_penetration),
            **self._origin_penetration_metrics("reference", reference_penetration),
            "terrain/tracker_body_origin_penetration_improvement_m": signed_improvement,
            "terrain/tracker_body_origin_correction_proxy_m": signed_improvement.clamp_min(0.0),
            "terrain/tracker_body_origin_correction_case": correction_case.to(dtype=torch.float32),
        }
        frame = {
            "robot_body_pos_w": robot_body_pos,
            "reference_body_pos_w": reference_body_pos,
            "robot_body_terrain_height_w": robot_height,
            "reference_body_terrain_height_w": reference_height,
            "robot_body_origin_penetration_m": robot_penetration,
            "reference_body_origin_penetration_m": reference_penetration,
            "correction_case": correction_case,
        }
        return metrics, frame

    def _origin_penetration_frame(self, prefix: str, body_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if body_pos.ndim != 3 or body_pos.shape[0] != self._env.num_envs or body_pos.shape[-1] != 3:
            raise RuntimeError(f"Expected {prefix} body positions shaped (N, B, 3), got {tuple(body_pos.shape)}")
        terrain_height = self._terrain_state.query_terrain_heights(body_pos[..., :2].reshape(-1, 2)).reshape(
            body_pos.shape[:2]
        )
        penetration = (terrain_height - body_pos[..., 2]).clamp_min(0.0)
        return terrain_height, penetration

    def _origin_penetration_metrics(self, prefix: str, penetration: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = f"terrain/{prefix}_body_origin_penetration"
        return {
            f"{stem}_mean_m": penetration.mean(dim=-1),
            f"{stem}_max_m": penetration.max(dim=-1).values,
            f"{stem}_rate": (
                penetration > self.config.body_origin_penetration_threshold_m
            ).to(dtype=torch.float32).mean(dim=-1),
        }

    def _capture_first_correction_exemplar(
        self,
        frame: dict[str, torch.Tensor],
        headings: torch.Tensor,
    ) -> None:
        """Capture one exact action/pre-reset body-origin proxy improvement frame."""
        if self._correction_exemplar is not None:
            return
        active = torch.as_tensor(
            self._require_accumulator().active_mask,
            dtype=torch.bool,
            device=self._env.device,
        )
        candidate_ids = torch.nonzero(frame["correction_case"] & active, as_tuple=False).flatten()
        if candidate_ids.numel() == 0:
            return
        if self._current_actions is None:
            raise RuntimeError("Current evaluation actions were not captured before the simulator step")
        env_id = int(candidate_ids[0].item())
        env_id_tensor = candidate_ids[:1]
        terrain_types, terrain_levels = self._terrain_labels(env_id_tensor)
        episode_length_buf = getattr(self._env, "episode_length_buf", None)
        episode_step = -1 if episode_length_buf is None else int(episode_length_buf[env_id].item())
        motion_cfg = self._motion_command.motion_cfg
        body_names = tuple(getattr(motion_cfg, "body_names_to_track", ()))
        if len(body_names) != frame["robot_body_pos_w"].shape[1]:
            body_names = tuple(f"body_{index}" for index in range(frame["robot_body_pos_w"].shape[1]))
        robot_penetration = frame["robot_body_origin_penetration_m"][env_id]
        reference_penetration = frame["reference_body_origin_penetration_m"][env_id]
        robot_max = robot_penetration.max()
        reference_max = reference_penetration.max()

        exemplar: dict[str, np.ndarray] = {
            "env_id": np.asarray(env_id, dtype=np.int64),
            "terrain_type": np.asarray(terrain_types[0]),
            "terrain_level": np.asarray(int(terrain_levels[0].item()), dtype=np.int64),
            "evaluation_step": np.asarray(self._current_eval_step, dtype=np.int64),
            "episode_step": np.asarray(episode_step, dtype=np.int64),
            "target_heading_w": self._to_numpy(headings[env_id]),
            "action": self._to_numpy(self._current_actions[env_id]),
            "root_state_w": self._to_numpy(self._env.simulator.robot_root_states[env_id]),
            "body_names": np.asarray(body_names),
            "robot_body_pos_w": self._to_numpy(frame["robot_body_pos_w"][env_id]),
            "reference_body_pos_w": self._to_numpy(frame["reference_body_pos_w"][env_id]),
            "robot_body_terrain_height_w": self._to_numpy(frame["robot_body_terrain_height_w"][env_id]),
            "reference_body_terrain_height_w": self._to_numpy(
                frame["reference_body_terrain_height_w"][env_id]
            ),
            "robot_body_origin_penetration_m": self._to_numpy(
                frame["robot_body_origin_penetration_m"][env_id]
            ),
            "reference_body_origin_penetration_m": self._to_numpy(
                reference_penetration
            ),
            "robot_max_body_origin_penetration_m": self._to_numpy(robot_max),
            "reference_max_body_origin_penetration_m": self._to_numpy(reference_max),
            "reference_minus_robot_max_body_origin_penetration_m": self._to_numpy(
                reference_max - robot_max
            ),
            "correction_proxy_case": np.asarray(True),
            "reference_penetration_threshold_m": np.asarray(
                self.config.body_origin_penetration_threshold_m,
                dtype=np.float64,
            ),
            "minimum_improvement_threshold_m": np.asarray(
                self.config.body_origin_correction_min_improvement_m,
                dtype=np.float64,
            ),
            "proxy_limitation": np.asarray(
                "Body-origin terrain-height proxy only; not collision-shape penetration or proof of policy intent."
            ),
            "root_state_layout": np.asarray("position_xyz,quaternion_xyzw,linear_velocity_xyz,angular_velocity_xyz"),
            "body_position_frame_units": np.asarray("world_xyz_metres"),
            "action_semantics": np.asarray("raw_policy_action_passed_to_environment_step"),
        }
        exemplar.update(self._local_scan_exemplar_arrays(env_id))
        self._validate_correction_exemplar(exemplar)
        self._correction_exemplar = exemplar

    def _local_scan_exemplar_arrays(self, env_id: int) -> dict[str, np.ndarray]:
        configured = bool(getattr(self._terrain_state, "local_height_scan_configured", False))
        empty = {
            "local_scan_configured": np.asarray(False),
            "local_scan_valid": np.asarray(False),
            "local_scan_height_w": np.empty((0,), dtype=np.float32),
            "local_scan_root_xy_w": np.empty((0,), dtype=np.float32),
            "local_scan_root_yaw_w": np.asarray(np.nan, dtype=np.float32),
            "local_scan_local_xy": np.empty((0, 2), dtype=np.float32),
            "local_scan_world_xy": np.empty((0, 2), dtype=np.float32),
            "local_scan_grid": np.empty((0,), dtype=np.float32),
            "local_scan_flatten_order": np.asarray("x-major,y-fastest"),
            "local_scan_height_units": np.asarray("absolute-world-z-metres"),
        }
        if not configured:
            return empty
        grid = getattr(self._terrain_state, "_local_scan_grid", None)
        local_xy = np.empty((0, 2), dtype=np.float32) if grid is None else grid.offsets()
        grid_array = np.empty((0,), dtype=np.float32) if grid is None else grid.to_array()
        world_xy = getattr(self._terrain_state, "_local_scan_world_xy", None)
        world_xy_array = (
            np.empty((0, 2), dtype=np.float32)
            if world_xy is None
            else self._to_numpy(world_xy[env_id])
        )
        return {
            "local_scan_configured": np.asarray(True),
            "local_scan_valid": np.asarray(bool(self._terrain_state.local_height_scan_valid[env_id])),
            "local_scan_height_w": self._to_numpy(self._terrain_state.local_height_scan[env_id]),
            "local_scan_root_xy_w": self._to_numpy(self._terrain_state.local_height_scan_root_xy[env_id]),
            "local_scan_root_yaw_w": self._to_numpy(self._terrain_state.local_height_scan_root_yaw[env_id]),
            "local_scan_local_xy": local_xy,
            "local_scan_world_xy": world_xy_array,
            "local_scan_grid": grid_array,
            "local_scan_flatten_order": np.asarray("x-major,y-fastest"),
            "local_scan_height_units": np.asarray("absolute-world-z-metres"),
        }

    @staticmethod
    def _validate_correction_exemplar(exemplar: dict[str, np.ndarray]) -> None:
        robot_pos = exemplar["robot_body_pos_w"]
        reference_pos = exemplar["reference_body_pos_w"]
        if robot_pos.ndim != 2 or robot_pos.shape[-1] != 3 or reference_pos.shape != robot_pos.shape:
            raise RuntimeError(
                "Correction exemplar body positions must share shape (B, 3), got "
                f"{robot_pos.shape} and {reference_pos.shape}"
            )
        body_count = robot_pos.shape[0]
        if exemplar["body_names"].shape != (body_count,):
            raise RuntimeError(f"Correction exemplar body_names must have shape ({body_count},)")
        for name in (
            "robot_body_terrain_height_w",
            "reference_body_terrain_height_w",
            "robot_body_origin_penetration_m",
            "reference_body_origin_penetration_m",
        ):
            if exemplar[name].shape != (body_count,):
                raise RuntimeError(f"Correction exemplar {name} must have shape ({body_count},)")
        if exemplar["target_heading_w"].shape != (2,):
            raise RuntimeError("Correction exemplar target_heading_w must have shape (2,)")
        if exemplar["action"].ndim != 1 or exemplar["root_state_w"].ndim != 1:
            raise RuntimeError("Correction exemplar action and root_state_w must be one-dimensional")

    def _undesired_contact_count(self) -> torch.Tensor | None:
        reward_manager = getattr(self._env, "reward_manager", None)
        if reward_manager is None or "undesired_contacts" not in reward_manager.active_terms:
            return None
        term = reward_manager.get_term("undesired_contacts")
        count = term(self._env).to(dtype=torch.float32)
        if count.shape != (self._env.num_envs,):
            raise RuntimeError(
                f"undesired_contacts must return ({self._env.num_envs},), got {tuple(count.shape)}"
            )
        return count

    def _write_outputs(self) -> None:
        if self._outputs_written:
            return
        accumulator = self._require_accumulator()
        checkpoint_path = getattr(self.training_loop, "_evaluation_checkpoint_path", None)
        evaluation_config = getattr(self.training_loop, "_evaluation_config", None)
        if checkpoint_path is None:
            raise RuntimeError("Resolved evaluation checkpoint path is missing from the training loop")
        if evaluation_config is None:
            raise RuntimeError("Evaluation config metadata is missing from the training loop")
        serialized_config = (
            evaluation_config.to_serializable_dict()
            if hasattr(evaluation_config, "to_serializable_dict")
            else evaluation_config
        )
        metadata = {
            "variant": self.config.variant,
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
            "evaluation_seed": self.config.evaluation_seed,
            "fixed_terrain_level": self.config.fixed_terrain_level,
            "evaluation_phase_mode": self.config.evaluation_phase_mode,
            "deterministic_generator": self.config.deterministic_generator,
            "deterministic_per_env_sampling": self.config.deterministic_per_env_sampling,
            "generator_sampling_seed": self.config.generator_sampling_seed,
            "metrics_config": dataclasses.asdict(self.config),
            "evaluation_config": serialized_config,
        }
        generator_checkpoint = getattr(self._motion_command.motion_cfg, "generator_checkpoint", "")
        if generator_checkpoint:
            resolved_generator = Path(generator_checkpoint).expanduser().resolve()
            metadata["generator_checkpoint_path"] = str(resolved_generator)
            metadata["generator_checkpoint_sha256"] = checkpoint_sha256(resolved_generator)
        output_prefix = Path(self.config.output_prefix)
        if not output_prefix.is_absolute():
            output_prefix = Path(self.training_loop.log_dir) / output_prefix
        exemplar_path = output_prefix.parent / f"{output_prefix.name}_first_correction_exemplar.npz"
        exemplar_metadata = {
            "found": self._correction_exemplar is not None,
            "path": None,
            "selection_condition": {
                "reference_max_body_origin_penetration_m_gte": (
                    self.config.body_origin_penetration_threshold_m
                ),
                "reference_minus_robot_max_body_origin_penetration_m_gte": (
                    self.config.body_origin_correction_min_improvement_m
                ),
            },
            "simultaneous_match_tie_break": "lowest env_id on the first qualifying timestep",
            "limitation": (
                "This is a body-origin terrain-height proxy exemplar, not collision-shape or mesh "
                "penetration and not proof that the policy intentionally corrected the reference."
            ),
        }
        if self._correction_exemplar is not None:
            exemplar_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(exemplar_path, **self._correction_exemplar)
            exemplar_metadata["path"] = str(exemplar_path.resolve())
        metadata["first_correction_exemplar"] = exemplar_metadata
        self.output_paths = write_terrain_evaluation_outputs(
            accumulator=accumulator,
            output_prefix=output_prefix,
            metadata=metadata,
        )
        if self._correction_exemplar is not None:
            self.output_paths["correction_exemplar"] = exemplar_path
        self._outputs_written = True
        logger.info(
            "TerrainMetricsCallback: saved {} episodes to {}",
            len(accumulator.records),
            ", ".join(str(path) for path in self.output_paths.values()),
        )

    @staticmethod
    def _to_numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy().copy()
