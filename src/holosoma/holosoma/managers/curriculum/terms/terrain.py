"""Terrain-difficulty curriculum for closed-loop locomotion."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from holosoma.managers.curriculum.base import CurriculumTermBase
from holosoma.utils.safe_torch_import import torch


class TerrainCurriculum(CurriculumTermBase):
    """Promote or demote per-environment terrain difficulty at episode ends.

    Terrain type assignment is owned by the terrain term and remains fixed for
    each environment.  This curriculum only changes ``terrain_levels`` and asks
    the terrain term to apply the corresponding origins before command reset.
    """

    _STATE_VERSION = 2

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}

        self.enabled = self._bool_param(params, "enabled", True)
        self.skip_first_episode = self._bool_param(params, "skip_first_episode", True)
        self.terrain_state_name = str(params.get("terrain_state_name", "locomotion_terrain"))
        if not self.terrain_state_name:
            raise ValueError("terrain_state_name must be non-empty")

        self.initial_level = self._int_param(params, "initial_level", 0, minimum=0)
        self.min_level = self._int_param(params, "min_level", 0, minimum=0)
        self._configured_max_level = self._optional_int_param(params, "max_level", minimum=0)
        self.promote_success_streak = self._int_param(params, "promote_success_streak", 5, minimum=1)
        self.demote_failure_streak = self._int_param(params, "demote_failure_streak", 2, minimum=1)

        self.success_min_episode_fraction = self._float_param(params, "success_min_episode_fraction", 0.9)
        if not 0.0 < self.success_min_episode_fraction <= 1.0:
            raise ValueError("success_min_episode_fraction must be in (0, 1]")
        self.crossing_distance_m = self._float_param(params, "crossing_distance_m", 0.0)
        if self.crossing_distance_m < 0.0:
            raise ValueError("crossing_distance_m must be >= 0")

        self._terrain_state: Any | None = None
        self._terrain_type_names: tuple[str, ...] = ()
        self._num_curriculum_levels = 0
        self.max_level = -1
        self._success_min_steps = 0
        self._is_setup = False

    def _robot_root_xy(self) -> Any:
        """Return measured root XY; crossing is radial for concentric terrain."""
        simulator = getattr(self.env, "simulator", None)
        root_states = getattr(simulator, "robot_root_states", None)
        try:
            root_xy = root_states[:, :2]
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "TerrainCurriculum crossing metrics require indexable "
                "simulator.robot_root_states"
            ) from exc
        if not torch.is_tensor(root_xy) or root_xy.shape != (self.env.num_envs, 2):
            raise RuntimeError(
                "TerrainCurriculum crossing metrics require simulator.robot_root_states "
                f"yielding shape ({self.env.num_envs}, 2) for [:, :2]"
            )
        if not root_xy.is_floating_point():
            raise RuntimeError("simulator.robot_root_states must be floating point")
        return root_xy

    @staticmethod
    def _bool_param(params: Mapping[str, Any], name: str, default: bool) -> bool:
        value = params.get(name, default)
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
        return value

    @staticmethod
    def _int_param(params: Mapping[str, Any], name: str, default: int, *, minimum: int) -> int:
        value = params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value}")
        return value

    @classmethod
    def _optional_int_param(cls, params: Mapping[str, Any], name: str, *, minimum: int) -> int | None:
        value = params.get(name)
        if value is None:
            return None
        return cls._int_param(params, name, value, minimum=minimum)

    @staticmethod
    def _float_param(params: Mapping[str, Any], name: str, default: float) -> float:
        value = params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite float, got {type(value).__name__}")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite, got {result}")
        return result

    def setup(self) -> None:
        """Validate the terrain interface and initialize curriculum state."""
        terrain_manager = getattr(self.env, "terrain_manager", None)
        if terrain_manager is None:
            raise RuntimeError("TerrainCurriculum requires env.terrain_manager")
        terrain_state = terrain_manager.get_state(self.terrain_state_name)
        if terrain_state is None:
            raise RuntimeError(f"Terrain state {self.terrain_state_name!r} was not found")

        required = (
            "terrain_levels",
            "terrain_type_ids",
            "terrain_type_names",
            "set_curriculum_origins",
        )
        missing = [name for name in required if not hasattr(terrain_state, name)]
        if missing:
            raise RuntimeError(f"Terrain state is missing curriculum interface fields: {missing}")

        levels = terrain_state.terrain_levels
        type_ids = terrain_state.terrain_type_ids
        expected_shape = (self.env.num_envs,)
        self._validate_runtime_tensor("terrain_levels", levels, expected_shape, torch.long)
        self._validate_runtime_tensor("terrain_type_ids", type_ids, expected_shape, torch.long)

        type_names = terrain_state.terrain_type_names
        if not isinstance(type_names, Sequence) or isinstance(type_names, (str, bytes)):
            raise TypeError("terrain_type_names must be a sequence of strings")
        self._terrain_type_names = tuple(type_names)
        if not self._terrain_type_names or not all(isinstance(name, str) and name for name in type_names):
            raise ValueError("terrain_type_names must contain at least one non-empty string")
        if torch.any(type_ids < 0) or torch.any(type_ids >= len(self._terrain_type_names)):
            raise ValueError("terrain_type_ids contains an index outside terrain_type_names")

        state_num_levels = getattr(terrain_state, "num_curriculum_levels", None)
        if state_num_levels is None:
            raise RuntimeError("Terrain state must expose num_curriculum_levels so curriculum levels can be clamped")
        if isinstance(state_num_levels, bool) or not isinstance(state_num_levels, int):
            raise TypeError("terrain_state.num_curriculum_levels must be an int")
        if state_num_levels < 1:
            raise ValueError("terrain_state.num_curriculum_levels must be >= 1")
        self._num_curriculum_levels = state_num_levels

        state_max_level = state_num_levels - 1
        self.max_level = (
            state_max_level if self._configured_max_level is None else min(self._configured_max_level, state_max_level)
        )
        if self.min_level > self.max_level:
            raise ValueError(f"min_level ({self.min_level}) exceeds available max_level ({self.max_level})")
        if not self.min_level <= self.initial_level <= self.max_level:
            raise ValueError(f"initial_level must be in [{self.min_level}, {self.max_level}], got {self.initial_level}")

        max_episode_length = int(self.env.max_episode_length)
        if max_episode_length < 1:
            raise ValueError("env.max_episode_length must be >= 1")
        self._success_min_steps = max(1, math.ceil(self.success_min_episode_fraction * max_episode_length))

        device = self.env.device
        num_envs = self.env.num_envs
        num_types = len(self._terrain_type_names)
        self.success_streaks = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.failure_streaks = torch.zeros_like(self.success_streaks)
        self.actual_episode_steps = torch.zeros_like(self.success_streaks)
        self.success_eligible = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.outcome_eligible = torch.full(
            (num_envs,),
            not self.skip_first_episode,
            dtype=torch.bool,
            device=device,
        )
        self._skip_next_step_increment = torch.zeros_like(self.success_eligible)
        self.type_success_counts = torch.zeros(num_types, dtype=torch.long, device=device)
        self.type_episode_counts = torch.zeros_like(self.type_success_counts)
        self.type_survival_counts = torch.zeros_like(self.type_success_counts)
        self.type_crossing_counts = torch.zeros_like(self.type_success_counts)
        self.type_failure_counts = torch.zeros_like(self.type_success_counts)
        type_level_shape = (num_types, self._num_curriculum_levels)
        self.type_level_episode_counts = torch.zeros(type_level_shape, dtype=torch.long, device=device)
        self.type_level_success_counts = torch.zeros_like(self.type_level_episode_counts)
        self.type_level_survival_counts = torch.zeros_like(self.type_level_episode_counts)
        self.type_level_crossing_counts = torch.zeros_like(self.type_level_episode_counts)
        self.type_level_failure_counts = torch.zeros_like(self.type_level_episode_counts)
        root_xy = self._robot_root_xy()
        self.episode_start_root_xy = root_xy.detach().clone()
        self.max_episode_progress_m = torch.zeros(num_envs, dtype=root_xy.dtype, device=device)

        self._terrain_state = terrain_state
        self._is_setup = True
        if self.enabled:
            env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
            initial_levels = torch.full_like(env_ids, self.initial_level)
            self._set_origins(env_ids, initial_levels)
            self._write_metrics()

    def before_reset(self, env_ids) -> None:
        """Consume previous outcomes and apply new origins before command reset."""
        if not self.enabled:
            return
        self._require_setup()
        ids = self._as_env_ids(env_ids)
        if ids.numel() == 0:
            return

        termination_manager = getattr(self.env, "termination_manager", None)
        if termination_manager is None:
            raise RuntimeError("TerrainCurriculum requires env.termination_manager")

        failures = termination_manager.terminated.index_select(0, ids).to(dtype=torch.bool)
        time_outs = termination_manager.time_outs.index_select(0, ids).to(dtype=torch.bool)
        raw_outcomes = failures | time_outs
        can_evaluate_outcome = self.outcome_eligible.index_select(0, ids)

        effective_steps = self.actual_episode_steps.index_select(0, ids).clone()
        updates_before_termination = bool(getattr(self.env, "_update_tasks_before_termination", False))
        if not updates_before_termination:
            # On Isaac Sim the curriculum step hook runs after reset, so the
            # terminal physics step has not yet reached this counter.
            effective_steps[raw_outcomes] += 1

        eligible = self.success_eligible.index_select(0, ids) | (effective_steps >= self._success_min_steps)
        evaluated_failures = failures & can_evaluate_outcome
        survival_successes = (~failures) & time_outs & eligible & can_evaluate_outcome
        if self.crossing_distance_m > 0.0:
            crossing_achieved = self.max_episode_progress_m.index_select(0, ids) >= self.crossing_distance_m
            successes = survival_successes & crossing_achieved
        else:
            crossing_achieved = torch.zeros_like(survival_successes)
            successes = survival_successes
        # Timeouts that fail the configured progress gate remain evaluated
        # failures for curriculum streaks and denominators. This prevents a
        # stationary policy from being promoted merely for surviving.
        progress_failures = survival_successes & (~successes)
        evaluated_failures |= progress_failures
        evaluated = evaluated_failures | successes
        self._accumulate_type_outcomes(
            ids,
            evaluated,
            successes,
            survival_successes,
            crossing_achieved & evaluated,
            evaluated_failures,
        )

        # The first reset after rollout initialization may only be the tail of
        # an episode because PPO randomizes episode_length_buf.  It is never a
        # comparable full outcome, even if that tail happens to be long.
        first_fragments = raw_outcomes & (~can_evaluate_outcome)
        self.outcome_eligible[ids[first_fragments]] = True

        success_ids = ids[successes]
        failure_ids = ids[evaluated_failures]
        if success_ids.numel() > 0:
            self.success_streaks[success_ids] += 1
            self.failure_streaks[success_ids] = 0
        if failure_ids.numel() > 0:
            self.failure_streaks[failure_ids] += 1
            self.success_streaks[failure_ids] = 0

        current_levels = self._terrain_state.terrain_levels
        changed_ids: list[Any] = []
        changed_levels: list[Any] = []

        promote_ids = success_ids[self.success_streaks.index_select(0, success_ids) >= self.promote_success_streak]
        if promote_ids.numel() > 0:
            promoted = torch.clamp(
                current_levels.index_select(0, promote_ids) + 1,
                min=self.min_level,
                max=self.max_level,
            )
            self.success_streaks[promote_ids] = 0
            changed_ids.append(promote_ids)
            changed_levels.append(promoted)

        demote_ids = failure_ids[self.failure_streaks.index_select(0, failure_ids) >= self.demote_failure_streak]
        if demote_ids.numel() > 0:
            demoted = torch.clamp(
                current_levels.index_select(0, demote_ids) - 1,
                min=self.min_level,
                max=self.max_level,
            )
            self.failure_streaks[demote_ids] = 0
            changed_ids.append(demote_ids)
            changed_levels.append(demoted)

        if changed_ids:
            self._set_origins(torch.cat(changed_ids), torch.cat(changed_levels))

        self.actual_episode_steps[ids] = 0
        self.success_eligible[ids] = False
        if not updates_before_termination:
            # The post-reset step hook corresponds to the terminal physics
            # step, not to a step in the newly reset episode.
            self._skip_next_step_increment[ids[raw_outcomes]] = True
        self._write_metrics()

    def reset(self, env_ids) -> None:
        """Capture the new measured episode origin after command reset."""
        if not self.enabled:
            return
        self._require_setup()
        ids = self._as_env_ids(env_ids)
        if ids.numel() == 0:
            return
        self.episode_start_root_xy[ids] = self._robot_root_xy().index_select(0, ids)
        self.max_episode_progress_m[ids] = 0.0

    def step(self) -> None:
        """Advance actual-step guards and publish cumulative metrics."""
        if not self.enabled:
            return
        self._require_setup()
        root_xy = self._robot_root_xy()
        radial_progress = torch.linalg.vector_norm(root_xy - self.episode_start_root_xy, dim=-1)
        self.max_episode_progress_m = torch.maximum(self.max_episode_progress_m, radial_progress)
        increment = ~self._skip_next_step_increment
        self.actual_episode_steps[increment] += 1
        self._skip_next_step_increment.zero_()
        self.success_eligible |= self.actual_episode_steps >= self._success_min_steps
        self._write_metrics()

    def state_dict(self) -> dict[str, Any]:
        """Return strict, device-independent curriculum state."""
        self._require_setup()
        return {
            "version": self._STATE_VERSION,
            "num_envs": self.env.num_envs,
            "terrain_type_names": self._terrain_type_names,
            "terrain_type_ids": self._terrain_state.terrain_type_ids.detach().cpu().clone(),
            "terrain_levels": self._terrain_state.terrain_levels.detach().cpu().clone(),
            "success_streaks": self.success_streaks.detach().cpu().clone(),
            "failure_streaks": self.failure_streaks.detach().cpu().clone(),
            "actual_episode_steps": self.actual_episode_steps.detach().cpu().clone(),
            "success_eligible": self.success_eligible.detach().cpu().clone(),
            "outcome_eligible": self.outcome_eligible.detach().cpu().clone(),
            "skip_next_step_increment": self._skip_next_step_increment.detach().cpu().clone(),
            "type_success_counts": self.type_success_counts.detach().cpu().clone(),
            "type_episode_counts": self.type_episode_counts.detach().cpu().clone(),
            "type_survival_counts": self.type_survival_counts.detach().cpu().clone(),
            "type_crossing_counts": self.type_crossing_counts.detach().cpu().clone(),
            "type_failure_counts": self.type_failure_counts.detach().cpu().clone(),
            "type_level_episode_counts": self.type_level_episode_counts.detach().cpu().clone(),
            "type_level_success_counts": self.type_level_success_counts.detach().cpu().clone(),
            "type_level_survival_counts": self.type_level_survival_counts.detach().cpu().clone(),
            "type_level_crossing_counts": self.type_level_crossing_counts.detach().cpu().clone(),
            "type_level_failure_counts": self.type_level_failure_counts.detach().cpu().clone(),
            "episode_start_root_xy": self.episode_start_root_xy.detach().cpu().clone(),
            "max_episode_progress_m": self.max_episode_progress_m.detach().cpu().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state and immediately apply restored terrain origins."""
        self._require_setup()
        if not isinstance(state, Mapping):
            raise TypeError("Terrain curriculum state must be a mapping")
        version = state.get("version")
        if version not in (1, self._STATE_VERSION):
            raise ValueError(
                f"Unsupported terrain curriculum state version {version!r}; expected 1 or {self._STATE_VERSION}"
            )
        if state.get("num_envs") != self.env.num_envs:
            raise ValueError(f"Terrain curriculum num_envs mismatch: {state.get('num_envs')!r} != {self.env.num_envs}")
        if tuple(state.get("terrain_type_names", ())) != self._terrain_type_names:
            raise ValueError("Terrain curriculum terrain_type_names mismatch")

        num_envs_shape = (self.env.num_envs,)
        num_types_shape = (len(self._terrain_type_names),)
        type_level_shape = (len(self._terrain_type_names), self._num_curriculum_levels)
        type_ids = self._state_tensor(state, "terrain_type_ids", torch.long, num_envs_shape)
        levels = self._state_tensor(state, "terrain_levels", torch.long, num_envs_shape)
        success_streaks = self._state_tensor(state, "success_streaks", torch.long, num_envs_shape)
        failure_streaks = self._state_tensor(state, "failure_streaks", torch.long, num_envs_shape)
        actual_steps = self._state_tensor(state, "actual_episode_steps", torch.long, num_envs_shape)
        eligible = self._state_tensor(state, "success_eligible", torch.bool, num_envs_shape)
        outcome_eligible = self._state_tensor(state, "outcome_eligible", torch.bool, num_envs_shape)
        skip_increment = self._state_tensor(state, "skip_next_step_increment", torch.bool, num_envs_shape)
        success_counts = self._state_tensor(state, "type_success_counts", torch.long, num_types_shape)
        episode_counts = self._state_tensor(state, "type_episode_counts", torch.long, num_types_shape)
        if version == 1:
            # Stage-7 success meant a full timeout without a progress gate.
            # Preserve those aggregate statistics while starting the new
            # crossing/type-level channels from zero because they cannot be
            # reconstructed from the legacy checkpoint.
            survival_counts = success_counts.clone()
            crossing_counts = torch.zeros_like(success_counts)
            failure_counts = episode_counts - success_counts
            type_level_episode_counts = torch.zeros(type_level_shape, dtype=torch.long, device=self.env.device)
            type_level_success_counts = torch.zeros_like(type_level_episode_counts)
            type_level_survival_counts = torch.zeros_like(type_level_episode_counts)
            type_level_crossing_counts = torch.zeros_like(type_level_episode_counts)
            type_level_failure_counts = torch.zeros_like(type_level_episode_counts)
            episode_start_root_xy = self._robot_root_xy().detach().clone()
            max_episode_progress_m = torch.zeros(
                self.env.num_envs,
                dtype=episode_start_root_xy.dtype,
                device=self.env.device,
            )
        else:
            survival_counts = self._state_tensor(state, "type_survival_counts", torch.long, num_types_shape)
            crossing_counts = self._state_tensor(state, "type_crossing_counts", torch.long, num_types_shape)
            failure_counts = self._state_tensor(state, "type_failure_counts", torch.long, num_types_shape)
            type_level_episode_counts = self._state_tensor(
                state, "type_level_episode_counts", torch.long, type_level_shape
            )
            type_level_success_counts = self._state_tensor(
                state, "type_level_success_counts", torch.long, type_level_shape
            )
            type_level_survival_counts = self._state_tensor(
                state, "type_level_survival_counts", torch.long, type_level_shape
            )
            type_level_crossing_counts = self._state_tensor(
                state, "type_level_crossing_counts", torch.long, type_level_shape
            )
            type_level_failure_counts = self._state_tensor(
                state, "type_level_failure_counts", torch.long, type_level_shape
            )
            episode_start_root_xy = self._state_tensor(
                state,
                "episode_start_root_xy",
                self.episode_start_root_xy.dtype,
                (self.env.num_envs, 2),
            )
            max_episode_progress_m = self._state_tensor(
                state,
                "max_episode_progress_m",
                self.max_episode_progress_m.dtype,
                num_envs_shape,
            )

        if torch.any(levels < self.min_level) or torch.any(levels > self.max_level):
            raise ValueError(f"terrain_levels must be in [{self.min_level}, {self.max_level}]")
        if not torch.equal(type_ids, self._terrain_state.terrain_type_ids):
            raise ValueError("Terrain curriculum per-environment terrain_type_ids mismatch")
        nonnegative = {
            "success_streaks": success_streaks,
            "failure_streaks": failure_streaks,
            "actual_episode_steps": actual_steps,
            "type_success_counts": success_counts,
            "type_episode_counts": episode_counts,
            "type_survival_counts": survival_counts,
            "type_crossing_counts": crossing_counts,
            "type_failure_counts": failure_counts,
            "type_level_episode_counts": type_level_episode_counts,
            "type_level_success_counts": type_level_success_counts,
            "type_level_survival_counts": type_level_survival_counts,
            "type_level_crossing_counts": type_level_crossing_counts,
            "type_level_failure_counts": type_level_failure_counts,
        }
        for name, tensor in nonnegative.items():
            if torch.any(tensor < 0):
                raise ValueError(f"{name} must be non-negative")
        if torch.any(success_counts > episode_counts):
            raise ValueError("type_success_counts cannot exceed type_episode_counts")
        for name, counts in (
            ("type_survival_counts", survival_counts),
            ("type_crossing_counts", crossing_counts),
            ("type_failure_counts", failure_counts),
        ):
            if torch.any(counts > episode_counts):
                raise ValueError(f"{name} cannot exceed type_episode_counts")
        for name, counts in (
            ("type_level_success_counts", type_level_success_counts),
            ("type_level_survival_counts", type_level_survival_counts),
            ("type_level_crossing_counts", type_level_crossing_counts),
            ("type_level_failure_counts", type_level_failure_counts),
        ):
            if torch.any(counts > type_level_episode_counts):
                raise ValueError(f"{name} cannot exceed type_level_episode_counts")
        if not torch.isfinite(episode_start_root_xy).all():
            raise ValueError("episode_start_root_xy must be finite")
        if not torch.isfinite(max_episode_progress_m).all() or torch.any(max_episode_progress_m < 0):
            raise ValueError("max_episode_progress_m must be finite and non-negative")
        if torch.any(eligible & (actual_steps < self._success_min_steps)):
            raise ValueError("success_eligible is inconsistent with actual_episode_steps")

        # All validation happens before mutation so malformed checkpoints cannot
        # leave partially restored curriculum state.
        self.success_streaks.copy_(success_streaks)
        self.failure_streaks.copy_(failure_streaks)
        self.actual_episode_steps.copy_(actual_steps)
        self.success_eligible.copy_(eligible)
        self.outcome_eligible.copy_(outcome_eligible)
        self._skip_next_step_increment.copy_(skip_increment)
        self.type_success_counts.copy_(success_counts)
        self.type_episode_counts.copy_(episode_counts)
        self.type_survival_counts.copy_(survival_counts)
        self.type_crossing_counts.copy_(crossing_counts)
        self.type_failure_counts.copy_(failure_counts)
        self.type_level_episode_counts.copy_(type_level_episode_counts)
        self.type_level_success_counts.copy_(type_level_success_counts)
        self.type_level_survival_counts.copy_(type_level_survival_counts)
        self.type_level_crossing_counts.copy_(type_level_crossing_counts)
        self.type_level_failure_counts.copy_(type_level_failure_counts)
        self.episode_start_root_xy.copy_(episode_start_root_xy)
        self.max_episode_progress_m.copy_(max_episode_progress_m)
        all_ids = torch.arange(self.env.num_envs, dtype=torch.long, device=self.env.device)
        self._set_origins(all_ids, levels)
        self._write_metrics()

    def _accumulate_type_outcomes(
        self,
        env_ids: Any,
        evaluated: Any,
        successes: Any,
        survivals: Any,
        crossings: Any,
        failures: Any,
    ) -> None:
        if not torch.any(evaluated):
            return
        type_ids = self._terrain_state.terrain_type_ids.index_select(0, env_ids)
        levels = self._terrain_state.terrain_levels.index_select(0, env_ids)
        num_types = len(self._terrain_type_names)
        self.type_episode_counts += torch.bincount(type_ids[evaluated], minlength=num_types)
        channels = (
            (self.type_success_counts, successes),
            (self.type_survival_counts, survivals),
            (self.type_crossing_counts, crossings),
            (self.type_failure_counts, failures),
        )
        for counts, mask in channels:
            if torch.any(mask):
                counts.add_(torch.bincount(type_ids[mask], minlength=num_types))

        flat_bins = type_ids * self._num_curriculum_levels + levels
        total_bins = num_types * self._num_curriculum_levels
        self.type_level_episode_counts += torch.bincount(flat_bins[evaluated], minlength=total_bins).view_as(
            self.type_level_episode_counts
        )
        level_channels = (
            (self.type_level_success_counts, successes),
            (self.type_level_survival_counts, survivals),
            (self.type_level_crossing_counts, crossings),
            (self.type_level_failure_counts, failures),
        )
        for counts, mask in level_channels:
            if torch.any(mask):
                counts.add_(torch.bincount(flat_bins[mask], minlength=total_bins).view_as(counts))

    def _write_metrics(self) -> None:
        if not hasattr(self.env, "log_dict"):
            return
        levels = self._terrain_state.terrain_levels
        type_ids = self._terrain_state.terrain_type_ids
        device = levels.device
        float_dtype = torch.float32
        self.env.log_dict["terrain_curriculum/mean_level"] = levels.to(float_dtype).mean()
        self.env.log_dict["terrain_curriculum/crossing_distance_m"] = torch.tensor(
            self.crossing_distance_m,
            dtype=float_dtype,
            device=device,
        )
        self.env.log_dict["terrain_curriculum/current_mean_max_progress_m"] = self.max_episode_progress_m.mean()

        for level in range(self._num_curriculum_levels):
            self.env.log_dict[f"terrain_curriculum/level/{level}/env_fraction"] = (
                (levels == level).to(float_dtype).mean()
            )

        type_fractions = torch.bincount(type_ids, minlength=len(self._terrain_type_names)).to(float_dtype) / float(
            self.env.num_envs
        )
        self.env.log_dict["terrain_curriculum/type_env_fraction_min"] = type_fractions.min()
        self.env.log_dict["terrain_curriculum/type_env_fraction_max"] = type_fractions.max()
        self.env.log_dict["terrain_curriculum/type_env_fraction_range"] = type_fractions.max() - type_fractions.min()

        for type_id, type_name in enumerate(self._terrain_type_names):
            key = f"{type_id}_{self._metric_name(type_name)}"
            episodes = self.type_episode_counts[type_id]
            successes = self.type_success_counts[type_id]
            survivals = self.type_survival_counts[type_id]
            crossings = self.type_crossing_counts[type_id]
            failures = self.type_failure_counts[type_id]
            zero = torch.zeros((), dtype=float_dtype, device=device)
            prefix = f"terrain_curriculum/type/{key}"
            self.env.log_dict[f"{prefix}/episode_count"] = episodes.to(float_dtype)
            self.env.log_dict[f"{prefix}/env_fraction"] = type_fractions[type_id]
            for channel, count in (
                ("success", successes),
                ("survival", survivals),
                ("crossing", crossings),
                ("failure", failures),
            ):
                rate = torch.where(
                    episodes > 0,
                    count.to(float_dtype) / episodes.to(float_dtype),
                    zero,
                )
                self.env.log_dict[f"{prefix}/{channel}_count"] = count.to(float_dtype)
                self.env.log_dict[f"{prefix}/{channel}_rate"] = rate

            for level in range(self._num_curriculum_levels):
                level_prefix = f"{prefix}/level/{level}"
                level_episodes = self.type_level_episode_counts[type_id, level]
                self.env.log_dict[f"{level_prefix}/episode_count"] = level_episodes.to(float_dtype)
                for channel, matrix in (
                    ("success", self.type_level_success_counts),
                    ("survival", self.type_level_survival_counts),
                    ("crossing", self.type_level_crossing_counts),
                    ("failure", self.type_level_failure_counts),
                ):
                    count = matrix[type_id, level]
                    rate = torch.where(
                        level_episodes > 0,
                        count.to(float_dtype) / level_episodes.to(float_dtype),
                        zero,
                    )
                    self.env.log_dict[f"{level_prefix}/{channel}_count"] = count.to(float_dtype)
                    self.env.log_dict[f"{level_prefix}/{channel}_rate"] = rate

    @staticmethod
    def _metric_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "unnamed"

    def _set_origins(self, env_ids: Any, levels: Any) -> None:
        env_ids = env_ids.to(device=self.env.device, dtype=torch.long)
        levels = levels.to(device=self.env.device, dtype=torch.long)
        self._terrain_state.set_curriculum_origins(env_ids, levels)
        actual = self._terrain_state.terrain_levels.index_select(0, env_ids)
        if not torch.equal(actual, levels):
            raise RuntimeError("set_curriculum_origins did not apply the requested terrain levels")

    def _as_env_ids(self, env_ids: Any) -> Any:
        if env_ids is None:
            return torch.arange(self.env.num_envs, dtype=torch.long, device=self.env.device)
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.env.device).flatten()
        if ids.numel() > 0 and (torch.any(ids < 0) or torch.any(ids >= self.env.num_envs)):
            raise IndexError("env_ids contains an out-of-range environment index")
        if ids.unique().numel() != ids.numel():
            raise ValueError("env_ids must not contain duplicates")
        return ids

    def _validate_runtime_tensor(self, name: str, value: Any, shape: tuple[int, ...], dtype: Any) -> None:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        if value.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
        expected_device = torch.device(self.env.device)
        wrong_device = value.device.type != expected_device.type or (
            expected_device.index is not None and value.device.index != expected_device.index
        )
        if wrong_device:
            raise ValueError(f"{name} must be on {self.env.device}, got {value.device}")

    def _state_tensor(self, state: Mapping[str, Any], name: str, dtype: Any, shape: tuple[int, ...]) -> Any:
        if name not in state:
            raise KeyError(f"Terrain curriculum state is missing {name!r}")
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError(f"Terrain curriculum state {name!r} must be a tensor")
        if value.dtype != dtype:
            raise TypeError(f"Terrain curriculum state {name!r} must have dtype {dtype}, got {value.dtype}")
        if tuple(value.shape) != shape:
            raise ValueError(f"Terrain curriculum state {name!r} must have shape {shape}, got {tuple(value.shape)}")
        return value.to(device=self.env.device).clone()

    def _require_setup(self) -> None:
        if not self._is_setup:
            raise RuntimeError("TerrainCurriculum.setup() must be called first")
