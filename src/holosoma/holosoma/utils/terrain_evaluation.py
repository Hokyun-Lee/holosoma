"""Pure accumulation and serialization helpers for WBT terrain evaluation.

The helpers in this module intentionally do not depend on Isaac Sim.  The
runtime callback converts simulator tensors to NumPy arrays and feeds them to
``TerrainEvaluationAccumulator``; unit tests can therefore exercise episode
accounting without importing a simulator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass
class _ActiveEpisode:
    env_id: int
    env_episode_index: int
    terrain_type: str
    terrain_level: int
    target_heading_x: float
    target_heading_y: float
    max_forward_progress_m: float = 0.0
    step_count: int = 0
    fall: bool = False
    undesired_contact: bool = False
    metric_sums: dict[str, float] = field(default_factory=dict)
    metric_counts: dict[str, int] = field(default_factory=dict)
    metric_maxima: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TerrainEpisodeRecord:
    """One completed evaluation episode and its raw aggregations."""

    episode_index: int
    env_id: int
    env_episode_index: int
    terrain_type: str
    terrain_level: int
    target_heading_x: float
    target_heading_y: float
    max_forward_progress_m: float
    step_count: int
    success: bool
    fall: bool
    timeout: bool
    bad_tracking: bool
    survival: bool
    undesired_contact: bool
    metric_sums: dict[str, float]
    metric_counts: dict[str, int]
    metric_maxima: dict[str, float]

    def metric_mean(self, name: str) -> float | None:
        count = self.metric_counts.get(name, 0)
        if count == 0:
            return None
        return self.metric_sums[name] / count


class TerrainEvaluationAccumulator:
    """Accumulate exactly ``target_episodes`` from a vectorized environment.

    ``forward_progress_m`` must use the same episode-start target-heading
    projection as the terrain curriculum.  The runtime callback obtains it
    through ``target_heading_forward_progress_m`` from the curriculum module.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        target_episodes: int,
        success_distance_m: float,
        expected_terrain_types: Sequence[str] | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if target_episodes < 1:
            raise ValueError("target_episodes must be >= 1")
        if not math.isfinite(success_distance_m) or success_distance_m < 0.0:
            raise ValueError("success_distance_m must be finite and non-negative")

        self.num_envs = num_envs
        self.target_episodes = target_episodes
        self.success_distance_m = float(success_distance_m)
        self.records: list[TerrainEpisodeRecord] = []
        self._active: list[_ActiveEpisode | None] = [None] * num_envs
        self._env_episode_counts = [0] * num_envs
        self._expected_terrain_types: tuple[str, ...] | None = None
        self._per_type_quota: int | None = None
        if expected_terrain_types is not None:
            self._set_expected_terrain_types(expected_terrain_types)

    @property
    def complete(self) -> bool:
        if self._expected_terrain_types is None or self._per_type_quota is None:
            return False
        counts = self.completed_per_terrain_type
        return all(counts.get(name, 0) >= self._per_type_quota for name in self._expected_terrain_types)

    @property
    def completed_per_terrain_type(self) -> dict[str, int]:
        counts = dict.fromkeys(self._expected_terrain_types or (), 0)
        for record in self.records:
            counts[record.terrain_type] = counts.get(record.terrain_type, 0) + 1
        return counts

    @property
    def requested_per_terrain_type(self) -> dict[str, int]:
        if self._expected_terrain_types is None or self._per_type_quota is None:
            return {}
        return dict.fromkeys(self._expected_terrain_types, self._per_type_quota)

    @property
    def active_mask(self) -> np.ndarray:
        return np.asarray([episode is not None for episode in self._active], dtype=np.bool_)

    def start_episodes(
        self,
        env_ids: Sequence[int] | np.ndarray,
        *,
        target_headings: np.ndarray,
        terrain_types: Sequence[str],
        terrain_levels: Sequence[int] | np.ndarray,
    ) -> None:
        """Start inactive environments from their current reset state."""
        ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        headings = np.asarray(target_headings, dtype=np.float64)
        levels = np.asarray(terrain_levels, dtype=np.int64).reshape(-1)
        if headings.shape != (ids.size, 2):
            raise ValueError(f"target_headings must have shape ({ids.size}, 2), got {headings.shape}")
        if len(terrain_types) != ids.size or levels.shape != (ids.size,):
            raise ValueError("terrain labels and levels must match env_ids")
        if self._expected_terrain_types is None:
            self._set_expected_terrain_types(terrain_types)
        if self.complete:
            return

        norms = np.linalg.norm(headings, axis=-1)
        if not np.isfinite(headings).all() or np.any(norms <= 1.0e-8):
            raise ValueError("target_headings must contain finite non-zero vectors")
        headings = headings / norms[:, None]

        for row, env_id_value in enumerate(ids.tolist()):
            if not 0 <= env_id_value < self.num_envs:
                raise IndexError(f"env_id {env_id_value} outside [0, {self.num_envs})")
            if self._active[env_id_value] is not None:
                continue
            terrain_type = str(terrain_types[row])
            if not terrain_type:
                raise ValueError("terrain_type must be non-empty")
            if terrain_type not in self.requested_per_terrain_type:
                raise ValueError(
                    f"terrain_type {terrain_type!r} not in expected types {self._expected_terrain_types}"
                )
            if (
                self.completed_per_terrain_type.get(terrain_type, 0)
                >= self.requested_per_terrain_type[terrain_type]
            ):
                continue
            env_episode_index = self._env_episode_counts[env_id_value]
            self._env_episode_counts[env_id_value] += 1
            self._active[env_id_value] = _ActiveEpisode(
                env_id=env_id_value,
                env_episode_index=env_episode_index,
                terrain_type=terrain_type,
                terrain_level=int(levels[row]),
                target_heading_x=float(headings[row, 0]),
                target_heading_y=float(headings[row, 1]),
            )

    def observe(
        self,
        *,
        forward_progress_m: np.ndarray,
        falls: np.ndarray,
        terminated: np.ndarray,
        timeouts: np.ndarray,
        metrics: Mapping[str, np.ndarray],
    ) -> None:
        """Consume one exact pre-reset simulator step for every active env."""
        progress = self._vector(forward_progress_m, "forward_progress_m", np.float64)
        fall_values = self._vector(falls, "falls", np.bool_)
        terminated_values = self._vector(terminated, "terminated", np.bool_)
        timeout_values = self._vector(timeouts, "timeouts", np.bool_)
        metric_values = {
            name: self._vector(value, f"metrics[{name!r}]", np.float64) for name, value in metrics.items()
        }

        for env_id, episode in enumerate(self._active):
            if episode is None or self.complete:
                continue
            current_progress = float(progress[env_id])
            if math.isfinite(current_progress):
                episode.max_forward_progress_m = max(episode.max_forward_progress_m, current_progress, 0.0)
            episode.step_count += 1
            episode.fall |= bool(fall_values[env_id])

            for name, values in metric_values.items():
                value = float(values[env_id])
                if not math.isfinite(value):
                    continue
                episode.metric_sums[name] = episode.metric_sums.get(name, 0.0) + value
                episode.metric_counts[name] = episode.metric_counts.get(name, 0) + 1
                episode.metric_maxima[name] = max(episode.metric_maxima.get(name, -math.inf), value)
                if name == "terrain/undesired_contact_any":
                    episode.undesired_contact |= value > 0.0

            timeout = bool(timeout_values[env_id])
            bad_tracking = bool(terminated_values[env_id])
            if not (timeout or bad_tracking):
                continue

            if (
                self.completed_per_terrain_type.get(episode.terrain_type, 0)
                >= self.requested_per_terrain_type[episode.terrain_type]
            ):
                # Another env of the same type may have filled the quota on
                # this vectorized step. The completed extra episode is
                # intentionally excluded from all raw denominators.
                self._active[env_id] = None
                continue

            survival = timeout and not bad_tracking and not episode.fall
            success = (
                episode.max_forward_progress_m >= self.success_distance_m
                and not episode.fall
                and not bad_tracking
            )
            self.records.append(
                TerrainEpisodeRecord(
                    episode_index=len(self.records),
                    env_id=episode.env_id,
                    env_episode_index=episode.env_episode_index,
                    terrain_type=episode.terrain_type,
                    terrain_level=episode.terrain_level,
                    target_heading_x=episode.target_heading_x,
                    target_heading_y=episode.target_heading_y,
                    max_forward_progress_m=episode.max_forward_progress_m,
                    step_count=episode.step_count,
                    success=success,
                    fall=episode.fall,
                    timeout=timeout,
                    bad_tracking=bad_tracking,
                    survival=survival,
                    undesired_contact=episode.undesired_contact,
                    metric_sums=dict(episode.metric_sums),
                    metric_counts=dict(episode.metric_counts),
                    metric_maxima=dict(episode.metric_maxima),
                )
            )
            self._active[env_id] = None

    def summary(self) -> dict[str, Any]:
        """Return overall and terrain-stratified raw numerator/denominator data."""
        by_type: dict[str, list[TerrainEpisodeRecord]] = {}
        by_type_level: dict[str, list[TerrainEpisodeRecord]] = {}
        for record in self.records:
            by_type.setdefault(record.terrain_type, []).append(record)
            key = f"{record.terrain_type}/level_{record.terrain_level}"
            by_type_level.setdefault(key, []).append(record)
        return {
            "requested_episode_count": self.target_episodes,
            "completed_episode_count": len(self.records),
            "complete": self.complete,
            "requested_per_terrain_type": self.requested_per_terrain_type,
            "completed_per_terrain_type": self.completed_per_terrain_type,
            "success_distance_m": self.success_distance_m,
            "overall": _summarize_records(self.records),
            "by_terrain_type": {name: _summarize_records(records) for name, records in sorted(by_type.items())},
            "by_terrain_type_and_level": {
                name: _summarize_records(records) for name, records in sorted(by_type_level.items())
            },
        }

    def _set_expected_terrain_types(self, terrain_types: Sequence[str]) -> None:
        names = tuple(dict.fromkeys(str(name) for name in terrain_types))
        if not names or any(not name for name in names):
            raise ValueError("expected_terrain_types must contain non-empty names")
        if self.target_episodes % len(names) != 0:
            raise ValueError(
                f"target_episodes ({self.target_episodes}) must be divisible by terrain type count ({len(names)})"
            )
        self._expected_terrain_types = names
        self._per_type_quota = self.target_episodes // len(names)

    def _vector(self, value: np.ndarray, name: str, dtype: Any) -> np.ndarray:
        array = np.asarray(value, dtype=dtype).reshape(-1)
        if array.shape != (self.num_envs,):
            raise ValueError(f"{name} must have shape ({self.num_envs},), got {array.shape}")
        return array


def _ratio(numerator: float, denominator: int) -> dict[str, float | int | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else float(numerator) / denominator,
    }


def _summarize_records(records: Sequence[TerrainEpisodeRecord]) -> dict[str, Any]:
    episode_count = len(records)
    rates = {
        "episode/success_rate": _ratio(sum(record.success for record in records), episode_count),
        "episode/fall_rate": _ratio(sum(record.fall for record in records), episode_count),
        "episode/timeout_rate": _ratio(sum(record.timeout for record in records), episode_count),
        "episode/bad_tracking_rate": _ratio(sum(record.bad_tracking for record in records), episode_count),
        "episode/survival_rate": _ratio(sum(record.survival for record in records), episode_count),
        "episode/undesired_contact_rate": _ratio(sum(record.undesired_contact for record in records), episode_count),
        "episode/mean_length_steps": _ratio(sum(record.step_count for record in records), episode_count),
        "motion/mean_max_episode_forward_progress_m": _ratio(
            sum(record.max_forward_progress_m for record in records), episode_count
        ),
    }

    metric_names = sorted({name for record in records for name in record.metric_sums})
    metrics: dict[str, dict[str, float | int | None]] = {}
    for name in metric_names:
        numerator = sum(record.metric_sums.get(name, 0.0) for record in records)
        denominator = sum(record.metric_counts.get(name, 0) for record in records)
        maxima = [record.metric_maxima[name] for record in records if name in record.metric_maxima]
        metric = _ratio(numerator, denominator)
        metric["max"] = max(maxima) if maxima else None
        metrics[name] = metric
    return {"episode_count": episode_count, "rates": rates, "step_metrics": metrics}


def checkpoint_sha256(path: str | Path) -> str:
    """Hash a resolved local checkpoint without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_terrain_evaluation_outputs(
    *,
    accumulator: TerrainEvaluationAccumulator,
    output_prefix: str | Path,
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    """Write JSON, summary CSV, and per-episode CSV artifacts."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = accumulator.summary()
    payload = {
        "schema_version": 1,
        "metadata": dict(metadata),
        "metric_definition": {
            "success": (
                "max target-heading signed forward progress >= success_distance_m, "
                "with neither fall nor bad-tracking termination"
            ),
            "survival": "timeout with neither fall nor bad-tracking termination",
            "body_origin_penetration_proxy": (
                "terrain height minus rigid-body origin Z, clamped at zero; "
                "this is not collision-shape or mesh penetration"
            ),
            "tracker_correction_exemplar": (
                "first thresholded reduction in the body-origin penetration proxy; "
                "not proof of collision resolution or intentional policy correction"
            ),
        },
        "summary": summary,
        "episodes": [_record_to_json(record) for record in accumulator.records],
    }

    json_path = prefix.with_suffix(".json")
    summary_csv_path = prefix.parent / f"{prefix.name}_summary.csv"
    episodes_csv_path = prefix.parent / f"{prefix.name}_episodes.csv"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
    _write_summary_csv(summary_csv_path, summary, metadata)
    _write_episode_csv(episodes_csv_path, accumulator.records)
    return {"json": json_path, "summary_csv": summary_csv_path, "episodes_csv": episodes_csv_path}


def _record_to_json(record: TerrainEpisodeRecord) -> dict[str, Any]:
    return {
        "episode_index": record.episode_index,
        "env_id": record.env_id,
        "env_episode_index": record.env_episode_index,
        "terrain_type": record.terrain_type,
        "terrain_level": record.terrain_level,
        "target_heading": [record.target_heading_x, record.target_heading_y],
        "max_forward_progress_m": record.max_forward_progress_m,
        "step_count": record.step_count,
        "success": record.success,
        "fall": record.fall,
        "timeout": record.timeout,
        "bad_tracking": record.bad_tracking,
        "survival": record.survival,
        "undesired_contact": record.undesired_contact,
        "metric_sums": record.metric_sums,
        "metric_counts": record.metric_counts,
        "metric_maxima": record.metric_maxima,
        "metric_means": {name: record.metric_mean(name) for name in sorted(record.metric_sums)},
    }


def _iter_summary_groups(summary: Mapping[str, Any]) -> Iterable[tuple[str, str, int | str, Mapping[str, Any]]]:
    yield "overall", "all", "all", summary["overall"]
    for terrain_type, values in summary["by_terrain_type"].items():
        yield "terrain_type", terrain_type, "all", values
    for key, values in summary["by_terrain_type_and_level"].items():
        terrain_type, level_text = key.rsplit("/level_", 1)
        yield "terrain_type_and_level", terrain_type, int(level_text), values


def _write_summary_csv(path: Path, summary: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    fields = [
        "variant",
        "checkpoint_sha256",
        "generator_checkpoint_sha256",
        "evaluation_seed",
        "fixed_terrain_level",
        "config_json",
        "group",
        "terrain_type",
        "terrain_level",
        "metric",
        "numerator",
        "denominator",
        "value",
        "max",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        common = {
            "variant": metadata.get("variant", "unspecified"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256", ""),
            "generator_checkpoint_sha256": metadata.get("generator_checkpoint_sha256", ""),
            "evaluation_seed": metadata.get("evaluation_seed", ""),
            "fixed_terrain_level": metadata.get("fixed_terrain_level", ""),
            "config_json": json.dumps(
                {
                    "metrics_config": metadata.get("metrics_config", {}),
                    "evaluation_config": metadata.get("evaluation_config", {}),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        requested_total = int(summary["requested_episode_count"])
        completed_total = int(summary["completed_episode_count"])
        writer.writerow(
            {
                **common,
                "group": "overall",
                "terrain_type": "all",
                "terrain_level": "all",
                "metric": "episode/completion_quota",
                "numerator": completed_total,
                "denominator": requested_total,
                "value": None if requested_total == 0 else completed_total / requested_total,
                "max": "",
            }
        )
        requested_by_type = summary["requested_per_terrain_type"]
        completed_by_type = summary["completed_per_terrain_type"]
        for terrain_type, requested in requested_by_type.items():
            completed = int(completed_by_type.get(terrain_type, 0))
            writer.writerow(
                {
                    **common,
                    "group": "terrain_type",
                    "terrain_type": terrain_type,
                    "terrain_level": "all",
                    "metric": "episode/completion_quota",
                    "numerator": completed,
                    "denominator": requested,
                    "value": None if requested == 0 else completed / requested,
                    "max": "",
                }
            )
        for group, terrain_type, terrain_level, values in _iter_summary_groups(summary):
            for namespace in ("rates", "step_metrics"):
                for metric_name, metric in values[namespace].items():
                    writer.writerow(
                        {
                            **common,
                            "group": group,
                            "terrain_type": terrain_type,
                            "terrain_level": terrain_level,
                            "metric": metric_name,
                            "numerator": metric["numerator"],
                            "denominator": metric["denominator"],
                            "value": metric["value"],
                            "max": metric.get("max", ""),
                        }
                    )


def _write_episode_csv(path: Path, records: Sequence[TerrainEpisodeRecord]) -> None:
    metric_names = sorted({name for record in records for name in record.metric_sums})
    fields = [
        "episode_index",
        "env_id",
        "env_episode_index",
        "terrain_type",
        "terrain_level",
        "target_heading_x",
        "target_heading_y",
        "max_forward_progress_m",
        "step_count",
        "success",
        "fall",
        "timeout",
        "bad_tracking",
        "survival",
        "undesired_contact",
        *[f"mean:{name}" for name in metric_names],
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {
                "episode_index": record.episode_index,
                "env_id": record.env_id,
                "env_episode_index": record.env_episode_index,
                "terrain_type": record.terrain_type,
                "terrain_level": record.terrain_level,
                "target_heading_x": record.target_heading_x,
                "target_heading_y": record.target_heading_y,
                "max_forward_progress_m": record.max_forward_progress_m,
                "step_count": record.step_count,
                "success": record.success,
                "fall": record.fall,
                "timeout": record.timeout,
                "bad_tracking": record.bad_tracking,
                "survival": record.survival,
                "undesired_contact": record.undesired_contact,
            }
            row.update({f"mean:{name}": record.metric_mean(name) for name in metric_names})
            writer.writerow(row)
