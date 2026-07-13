"""Aggregate Stage-10 terrain evaluator JSONs without simulator dependencies.

The common evaluator intentionally writes raw numerator/denominator pairs.
This module validates that contract before producing a compact comparison; it
never reconstructs rates from rounded values and never silently combines
incompatible evaluation protocols.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

INPUT_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
PROTOCOL_FIELDS = (
    "evaluation_seed",
    "fixed_terrain_level",
    "success_distance_m",
    "episode_quotas",
    "terrain_types",
    "evaluation_phase_mode",
    "deterministic_generator",
    "deterministic_per_env_sampling",
    "generator_sampling_seed",
    "fall_root_height_m",
    "fall_upright_cosine",
    "body_origin_penetration_threshold_m",
    "body_origin_correction_min_improvement_m",
    "heading_speed_threshold_mps",
    "max_episode_length_s",
    "control_dt_s",
    "terrain_geometry_layout",
)

_RATE_METRICS = (
    "episode/success_rate",
    "episode/fall_rate",
    "episode/survival_rate",
    "episode/undesired_contact_rate",
    "episode/mean_length_steps",
    "motion/mean_max_episode_forward_progress_m",
)
_STEP_METRICS = (
    "motion/heading_error_rad",
    "motion/error_ref_pos",
    "motion/error_ref_rot",
    "motion/error_ref_lin_vel",
    "motion/error_body_pos",
    "motion/error_body_lin_vel",
    "motion/error_joint_pos",
    "motion/error_joint_vel",
    "terrain/robot_body_origin_penetration_mean_m",
    "terrain/robot_body_origin_penetration_max_m",
    "terrain/robot_body_origin_penetration_rate",
    "terrain/reference_body_origin_penetration_mean_m",
    "terrain/reference_body_origin_penetration_max_m",
    "terrain/reference_body_origin_penetration_rate",
    "terrain/tracker_body_origin_penetration_improvement_m",
    "terrain/undesired_contact_body_count",
    "terrain/undesired_contact_any",
)
_OPTIONAL_STEP_METRICS = (
    "motion/heading_error_moving_rad",
    "motion/heading_low_speed_fraction",
    "motion/heading_speed_mps",
    "terrain/tracker_body_origin_correction_proxy_m",
    "terrain/tracker_body_origin_correction_case",
)
_ALL_SELECTED_METRICS = (*_RATE_METRICS, *_STEP_METRICS, *_OPTIONAL_STEP_METRICS)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_BOUNDED_RATE_METRICS = _RATE_METRICS[:4]
_TERRAIN_GEOMETRY_KEYS = (
    "mesh_type",
    "static_friction",
    "dynamic_friction",
    "restitution",
    "horizontal_scale",
    "vertical_scale",
    "border_size",
    "terrain_move_down_ratio",
    "terrain_move_up_ratio",
    "terrain_length",
    "terrain_width",
    "num_rows",
    "num_cols",
    "spawn",
    "curriculum_layout",
    "terrain_config",
    "max_slope",
    "platform_size",
    "step_width_range",
    "amplitude_range",
    "slope_treshold",
    "obj_file_path",
    "scale_factor",
)


class TerrainReportError(ValueError):
    """Raised when an evaluator artifact cannot be compared safely."""


@dataclass(frozen=True)
class _LoadedEvaluation:
    source_path: str
    source_sha256: str
    variant: str
    scenario_label: str
    checkpoint_path: str | None
    checkpoint_sha256: str
    generator_checkpoint_path: str | None
    generator_checkpoint_sha256: str | None
    evaluation_phase_mode: str
    deterministic_generator: bool | None
    deterministic_per_env_sampling: bool
    generator_sampling_seed: int | None
    protocol: dict[str, Any]
    quota: dict[str, Any]
    overall_metrics: dict[str, dict[str, Any]]
    by_terrain_type: dict[str, dict[str, dict[str, Any]]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TerrainReportError(f"{context} must be a JSON object")
    return value


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TerrainReportError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise TerrainReportError(f"{context} must be >= {minimum}")
    return value


def _number(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TerrainReportError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TerrainReportError(f"{context} must be a finite number")
    if minimum is not None and result < minimum:
        raise TerrainReportError(f"{context} must be >= {minimum}")
    return result


def _sha256_value(value: Any, context: str, *, optional: bool) -> str | None:
    if value in (None, "") and optional:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TerrainReportError(f"{context} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12)
    except (TypeError, ValueError):
        return False


def _metric_record(value: Any, context: str) -> dict[str, Any]:
    record = _mapping(value, context)
    numerator = _number(record.get("numerator"), f"{context}.numerator")
    denominator = _integer(record.get("denominator"), f"{context}.denominator", minimum=0)
    reported_value = record.get("value")
    if denominator == 0:
        if reported_value is not None:
            raise TerrainReportError(f"{context}.value must be null when denominator is zero")
    else:
        numeric_value = _number(reported_value, f"{context}.value")
        if not _same_number(numeric_value, numerator / denominator):
            raise TerrainReportError(
                f"{context}.value does not equal its raw numerator/denominator ({numerator}/{denominator})"
            )
    result: dict[str, Any] = {
        "numerator": record["numerator"],
        "denominator": denominator,
        "value": reported_value,
    }
    if "max" in record:
        maximum = record["max"]
        if maximum is not None:
            _number(maximum, f"{context}.max")
        result["max"] = maximum
    return result


def _selected_metrics(summary_group: Any, context: str) -> dict[str, dict[str, Any]]:
    group = _mapping(summary_group, context)
    rates = _mapping(group.get("rates"), f"{context}.rates")
    steps = _mapping(group.get("step_metrics"), f"{context}.step_metrics")
    missing_rates = [name for name in _RATE_METRICS if name not in rates]
    missing_steps = [name for name in _STEP_METRICS if name not in steps]
    if missing_rates or missing_steps:
        missing = ", ".join((*missing_rates, *missing_steps))
        raise TerrainReportError(f"{context} is missing required common evaluator metrics: {missing}")

    selected: dict[str, dict[str, Any]] = {}
    for name in _RATE_METRICS:
        selected[name] = _metric_record(rates[name], f"{context}.rates[{name!r}]")
    for name in (*_STEP_METRICS, *_OPTIONAL_STEP_METRICS):
        if name in steps:
            selected[name] = _metric_record(steps[name], f"{context}.step_metrics[{name!r}]")
    return selected


def _validate_metric_episode_denominators(
    metrics: Mapping[str, Mapping[str, Any]],
    episode_count: int,
    context: str,
) -> None:
    for name in _RATE_METRICS:
        denominator = metrics[name]["denominator"]
        if denominator != episode_count:
            raise TerrainReportError(
                f"{context} metric {name!r} denominator {denominator} does not match episode_count {episode_count}"
            )
    for name in _BOUNDED_RATE_METRICS:
        numerator = float(metrics[name]["numerator"])
        denominator = metrics[name]["denominator"]
        if not 0.0 <= numerator <= denominator:
            raise TerrainReportError(f"{context} metric {name!r} numerator {numerator} is outside [0, {denominator}]")


def _quota_mapping(value: Any, context: str) -> dict[str, int]:
    mapping = _mapping(value, context)
    if not mapping:
        raise TerrainReportError(f"{context} must not be empty")
    quotas: dict[str, int] = {}
    for terrain_type, quota in mapping.items():
        if not isinstance(terrain_type, str) or not terrain_type:
            raise TerrainReportError(f"{context} terrain type names must be non-empty strings")
        quotas[terrain_type] = _integer(quota, f"{context}[{terrain_type!r}]", minimum=1)
    return dict(sorted(quotas.items()))


def _nested_mapping(value: Mapping[str, Any], keys: Sequence[str], context: str) -> Mapping[str, Any]:
    current: Mapping[str, Any] = value
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        current = _mapping(current.get(key), f"{context}.{'.'.join(traversed)}")
    return current


def _terrain_geometry_layout(evaluation_config: Mapping[str, Any], context: str) -> dict[str, Any]:
    terrain_term = _nested_mapping(
        evaluation_config,
        ("terrain", "terrain_term"),
        f"{context}.metadata.evaluation_config",
    )
    missing = [name for name in ("mesh_type", "curriculum_layout", "terrain_config") if name not in terrain_term]
    if missing:
        raise TerrainReportError(
            f"{context}.metadata.evaluation_config terrain term is missing geometry fields: {missing!r}"
        )
    selected = {name: terrain_term[name] for name in _TERRAIN_GEOMETRY_KEYS if name in terrain_term}
    try:
        canonical_json = _canonical(selected)
    except (TypeError, ValueError) as error:
        raise TerrainReportError(
            f"{context}.metadata.evaluation_config terrain geometry is not canonical JSON: {error}"
        ) from error
    return {
        "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
        "config": json.loads(canonical_json),
    }


def _simulator_protocol(evaluation_config: Mapping[str, Any], context: str) -> tuple[float, float]:
    sim = _nested_mapping(
        evaluation_config,
        ("simulator", "config", "sim"),
        f"{context}.metadata.evaluation_config",
    )
    fps = _number(sim.get("fps"), f"{context}.metadata.evaluation_config.simulator.config.sim.fps", minimum=1.0)
    decimation = _integer(
        sim.get("control_decimation"),
        f"{context}.metadata.evaluation_config.simulator.config.sim.control_decimation",
        minimum=1,
    )
    episode_length = _number(
        sim.get("max_episode_length_s"),
        f"{context}.metadata.evaluation_config.simulator.config.sim.max_episode_length_s",
        minimum=0.0,
    )
    return episode_length, decimation / fps


def _metadata_protocol(metadata: Mapping[str, Any], summary: Mapping[str, Any], context: str) -> dict[str, Any]:
    metrics_config = _mapping(metadata.get("metrics_config"), f"{context}.metadata.metrics_config")
    evaluation_config = _mapping(
        metadata.get("evaluation_config"),
        f"{context}.metadata.evaluation_config",
    )
    evaluation_seed = _integer(metadata.get("evaluation_seed"), f"{context}.metadata.evaluation_seed")
    fixed_level = _integer(
        metadata.get("fixed_terrain_level"),
        f"{context}.metadata.fixed_terrain_level",
        minimum=0,
    )
    success_distance = _number(
        summary.get("success_distance_m"),
        f"{context}.summary.success_distance_m",
        minimum=0.0,
    )
    requested_total = _integer(
        summary.get("requested_episode_count"),
        f"{context}.summary.requested_episode_count",
        minimum=1,
    )
    requested_by_type = _quota_mapping(
        summary.get("requested_per_terrain_type"),
        f"{context}.summary.requested_per_terrain_type",
    )
    evaluation_phase_mode = metadata.get("evaluation_phase_mode")
    if evaluation_phase_mode not in {"zero", "uniform"}:
        raise TerrainReportError(
            f"{context}.metadata.evaluation_phase_mode must be 'zero' or 'uniform'"
        )
    deterministic_generator = metadata.get("deterministic_generator")
    if not isinstance(deterministic_generator, bool):
        raise TerrainReportError(f"{context}.metadata.deterministic_generator must be boolean")
    deterministic_per_env_sampling = metadata.get("deterministic_per_env_sampling")
    if not isinstance(deterministic_per_env_sampling, bool):
        raise TerrainReportError(
            f"{context}.metadata.deterministic_per_env_sampling must be boolean"
        )
    generator_sampling_seed = _integer(
        metadata.get("generator_sampling_seed"),
        f"{context}.metadata.generator_sampling_seed",
        minimum=0,
    )
    fall_root_height = _number(
        metrics_config.get("fall_root_height_m"),
        f"{context}.metadata.metrics_config.fall_root_height_m",
        minimum=0.0,
    )
    fall_upright_cosine = _number(
        metrics_config.get("fall_upright_cosine"),
        f"{context}.metadata.metrics_config.fall_upright_cosine",
    )
    if not 0.0 <= fall_upright_cosine <= 1.0:
        raise TerrainReportError(f"{context}.metadata.metrics_config.fall_upright_cosine must be in [0, 1]")
    penetration_threshold = _number(
        metrics_config.get("body_origin_penetration_threshold_m"),
        f"{context}.metadata.metrics_config.body_origin_penetration_threshold_m",
        minimum=0.0,
    )
    correction_threshold = _number(
        metrics_config.get("body_origin_correction_min_improvement_m"),
        f"{context}.metadata.metrics_config.body_origin_correction_min_improvement_m",
        minimum=0.0,
    )
    heading_speed_threshold = _number(
        metrics_config.get("heading_speed_threshold_mps"),
        f"{context}.metadata.metrics_config.heading_speed_threshold_mps",
        minimum=0.0,
    )
    max_episode_length_s, control_dt_s = _simulator_protocol(evaluation_config, context)
    terrain_geometry_layout = _terrain_geometry_layout(evaluation_config, context)
    training = _nested_mapping(
        evaluation_config,
        ("training",),
        f"{context}.metadata.evaluation_config",
    )
    config_seed = _integer(
        training.get("seed"),
        f"{context}.metadata.evaluation_config.training.seed",
    )
    if config_seed != evaluation_seed:
        raise TerrainReportError(
            f"{context} metadata evaluation_seed {evaluation_seed} does not match effective config seed {config_seed}"
        )

    consistency_fields = (
        ("evaluation_seed", evaluation_seed),
        ("fixed_terrain_level", fixed_level),
        ("episode_count", requested_total),
        ("success_distance_m", success_distance),
        ("evaluation_phase_mode", evaluation_phase_mode),
        ("deterministic_generator", deterministic_generator),
        ("deterministic_per_env_sampling", deterministic_per_env_sampling),
        ("generator_sampling_seed", generator_sampling_seed),
    )
    for name, expected in consistency_fields:
        if name not in metrics_config:
            raise TerrainReportError(f"{context}.metadata.metrics_config is missing {name!r}")
        actual = metrics_config[name]
        matches = _same_number(actual, expected) if isinstance(expected, float) else actual == expected
        if not matches:
            raise TerrainReportError(f"{context} metadata/summary mismatch for {name}: {actual!r} != {expected!r}")

    return {
        "evaluation_seed": evaluation_seed,
        "fixed_terrain_level": fixed_level,
        "success_distance_m": success_distance,
        "episode_quotas": {
            "total": requested_total,
            "per_terrain_type": requested_by_type,
        },
        "terrain_types": list(requested_by_type),
        "evaluation_phase_mode": evaluation_phase_mode,
        "deterministic_generator": deterministic_generator,
        "deterministic_per_env_sampling": deterministic_per_env_sampling,
        "generator_sampling_seed": generator_sampling_seed,
        "fall_root_height_m": fall_root_height,
        "fall_upright_cosine": fall_upright_cosine,
        "body_origin_penetration_threshold_m": penetration_threshold,
        "body_origin_correction_min_improvement_m": correction_threshold,
        "heading_speed_threshold_mps": heading_speed_threshold,
        "max_episode_length_s": max_episode_length_s,
        "control_dt_s": control_dt_s,
        "terrain_geometry_layout": terrain_geometry_layout,
    }


def _validate_complete_quota(
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    protocol: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    if summary.get("complete") is not True:
        raise TerrainReportError(f"{context} has an incomplete episode quota (summary.complete is not true)")
    requested = protocol["episode_quotas"]
    requested_total = requested["total"]
    requested_by_type = requested["per_terrain_type"]
    completed_total = _integer(
        summary.get("completed_episode_count"),
        f"{context}.summary.completed_episode_count",
        minimum=0,
    )
    completed_by_type = _quota_mapping(
        summary.get("completed_per_terrain_type"),
        f"{context}.summary.completed_per_terrain_type",
    )
    if sum(requested_by_type.values()) != requested_total:
        raise TerrainReportError(f"{context} requested per-terrain quotas do not sum to the total quota")
    if completed_total != requested_total or completed_by_type != requested_by_type:
        raise TerrainReportError(
            f"{context} has incomplete quotas: completed={completed_total}/{requested_total}, "
            f"per_type={completed_by_type!r}/{requested_by_type!r}"
        )

    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != completed_total:
        count = "not a list" if not isinstance(episodes, list) else len(episodes)
        raise TerrainReportError(f"{context}.episodes count {count} does not match completed quota {completed_total}")
    episode_type_counts = dict.fromkeys(requested_by_type, 0)
    episode_indices: set[int] = set()
    for row, episode_value in enumerate(episodes):
        episode = _mapping(episode_value, f"{context}.episodes[{row}]")
        episode_index = _integer(episode.get("episode_index"), f"{context}.episodes[{row}].episode_index")
        if episode_index in episode_indices:
            raise TerrainReportError(f"{context} contains duplicate episode_index {episode_index}")
        episode_indices.add(episode_index)
        terrain_type = episode.get("terrain_type")
        if terrain_type not in episode_type_counts:
            raise TerrainReportError(f"{context}.episodes[{row}] has unexpected terrain_type {terrain_type!r}")
        episode_type_counts[terrain_type] += 1
    if episode_type_counts != requested_by_type:
        raise TerrainReportError(
            f"{context} episode terrain counts {episode_type_counts!r} do not match quota {requested_by_type!r}"
        )
    return {
        "requested_episode_count": requested_total,
        "completed_episode_count": completed_total,
        "requested_per_terrain_type": requested_by_type,
        "completed_per_terrain_type": completed_by_type,
    }


def load_terrain_evaluation(path: str | Path) -> _LoadedEvaluation:
    """Load and strictly validate one common terrain evaluator JSON."""
    source = Path(path).expanduser().resolve()
    try:
        with source.open(encoding="utf-8") as input_file:
            payload_value = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise TerrainReportError(f"Cannot read evaluator JSON {source}: {error}") from error
    context = str(source)
    payload = _mapping(payload_value, context)
    if payload.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise TerrainReportError(
            f"{context} has unsupported schema_version {payload.get('schema_version')!r}; "
            f"expected {INPUT_SCHEMA_VERSION}"
        )
    metadata = _mapping(payload.get("metadata"), f"{context}.metadata")
    summary = _mapping(payload.get("summary"), f"{context}.summary")
    variant = metadata.get("variant")
    if not isinstance(variant, str) or not variant.strip() or variant == "unspecified":
        raise TerrainReportError(f"{context}.metadata.variant must be an explicit non-empty label")
    variant = variant.strip()
    metrics_config = _mapping(metadata.get("metrics_config"), f"{context}.metadata.metrics_config")
    configured_variant = metrics_config.get("variant")
    if configured_variant is not None and configured_variant != variant:
        raise TerrainReportError(
            f"{context} metadata.variant {variant!r} does not match metrics_config.variant {configured_variant!r}"
        )
    scenario_label = metrics_config.get("scenario_label", "")
    if not isinstance(scenario_label, str):
        raise TerrainReportError(f"{context}.metadata.metrics_config.scenario_label must be a string")

    protocol = _metadata_protocol(metadata, summary, context)
    quota = _validate_complete_quota(payload, summary, protocol, context)
    overall = _mapping(summary.get("overall"), f"{context}.summary.overall")
    overall_count = _integer(
        overall.get("episode_count"),
        f"{context}.summary.overall.episode_count",
        minimum=0,
    )
    if overall_count != quota["completed_episode_count"]:
        raise TerrainReportError(f"{context} overall episode_count does not match the completed quota")
    overall_metrics = _selected_metrics(overall, f"{context}.summary.overall")
    _validate_metric_episode_denominators(overall_metrics, overall_count, f"{context}.summary.overall")

    by_type_value = _mapping(summary.get("by_terrain_type"), f"{context}.summary.by_terrain_type")
    if set(by_type_value) != set(protocol["terrain_types"]):
        raise TerrainReportError(f"{context}.summary.by_terrain_type keys do not match the protocol terrain types")
    by_type: dict[str, dict[str, dict[str, Any]]] = {}
    for terrain_type in protocol["terrain_types"]:
        terrain_summary = _mapping(
            by_type_value[terrain_type],
            f"{context}.summary.by_terrain_type[{terrain_type!r}]",
        )
        terrain_count = _integer(
            terrain_summary.get("episode_count"),
            f"{context}.summary.by_terrain_type[{terrain_type!r}].episode_count",
            minimum=0,
        )
        expected_count = protocol["episode_quotas"]["per_terrain_type"][terrain_type]
        if terrain_count != expected_count:
            raise TerrainReportError(
                f"{context} terrain {terrain_type!r} episode_count {terrain_count} != quota {expected_count}"
            )
        terrain_metrics = _selected_metrics(
            terrain_summary,
            f"{context}.summary.by_terrain_type[{terrain_type!r}]",
        )
        _validate_metric_episode_denominators(
            terrain_metrics,
            terrain_count,
            f"{context}.summary.by_terrain_type[{terrain_type!r}]",
        )
        by_type[terrain_type] = terrain_metrics

    checkpoint_sha = _sha256_value(
        metadata.get("checkpoint_sha256"),
        f"{context}.metadata.checkpoint_sha256",
        optional=False,
    )
    generator_sha = _sha256_value(
        metadata.get("generator_checkpoint_sha256"),
        f"{context}.metadata.generator_checkpoint_sha256",
        optional=True,
    )
    deterministic = metadata.get("deterministic_generator")
    if deterministic is not None and not isinstance(deterministic, bool):
        raise TerrainReportError(f"{context}.metadata.deterministic_generator must be boolean")
    evaluation_phase_mode = metadata.get("evaluation_phase_mode")
    if evaluation_phase_mode not in {"zero", "uniform"}:
        raise TerrainReportError(
            f"{context}.metadata.evaluation_phase_mode must be 'zero' or 'uniform'"
        )
    deterministic_per_env_sampling = metadata.get("deterministic_per_env_sampling")
    if not isinstance(deterministic_per_env_sampling, bool):
        raise TerrainReportError(
            f"{context}.metadata.deterministic_per_env_sampling must be boolean"
        )
    sampling_seed = metadata.get("generator_sampling_seed")
    if sampling_seed is not None:
        sampling_seed = _integer(sampling_seed, f"{context}.metadata.generator_sampling_seed", minimum=0)

    return _LoadedEvaluation(
        source_path=str(source),
        source_sha256=_sha256(source),
        variant=variant,
        scenario_label=scenario_label,
        checkpoint_path=metadata.get("checkpoint_path"),
        checkpoint_sha256=str(checkpoint_sha),
        generator_checkpoint_path=metadata.get("generator_checkpoint_path"),
        generator_checkpoint_sha256=generator_sha,
        evaluation_phase_mode=evaluation_phase_mode,
        deterministic_generator=deterministic,
        deterministic_per_env_sampling=deterministic_per_env_sampling,
        generator_sampling_seed=sampling_seed,
        protocol=protocol,
        quota=quota,
        overall_metrics=overall_metrics,
        by_terrain_type=by_type,
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _variant_sort_key(variant: str) -> tuple[int, str, str]:
    match = re.match(r"^([A-F])(?:$|[-_])", variant, flags=re.IGNORECASE)
    if match is None:
        return 99, variant.casefold(), variant
    return ord(match.group(1).upper()) - ord("A"), variant.casefold(), variant


def _protocol_for_group(
    group_name: str,
    evaluations: Sequence[_LoadedEvaluation],
    allowed_mismatches: set[str],
) -> dict[str, Any]:
    protocol: dict[str, Any] = {}
    for field in PROTOCOL_FIELDS:
        values_by_variant = [(item.variant, item.protocol[field]) for item in evaluations]
        distinct = {_canonical(value) for _, value in values_by_variant}
        if len(distinct) > 1 and field not in allowed_mismatches:
            rendered = ", ".join(
                f"{variant}={value['sha256'] if field == 'terrain_geometry_layout' else value!r}"
                for variant, value in values_by_variant
            )
            raise TerrainReportError(
                f"Protocol mismatch in group {group_name!r} for {field}: {rendered}. "
                "Split the inputs with repeated --group NAME=JSON arguments or explicitly allow "
                f"this field with --allow-mismatch {field}."
            )
        if len(distinct) == 1:
            protocol[field] = values_by_variant[0][1]
        else:
            protocol[field] = {"varies": [{"variant": variant, "value": value} for variant, value in values_by_variant]}
    return protocol


def _evaluation_json(item: _LoadedEvaluation) -> dict[str, Any]:
    return {
        "variant": item.variant,
        "scenario_label": item.scenario_label,
        "source_json": item.source_path,
        "source_json_sha256": item.source_sha256,
        "checkpoint_path": item.checkpoint_path,
        "checkpoint_sha256": item.checkpoint_sha256,
        "generator_checkpoint_path": item.generator_checkpoint_path,
        "generator_checkpoint_sha256": item.generator_checkpoint_sha256,
        "evaluation_phase_mode": item.evaluation_phase_mode,
        "deterministic_generator": item.deterministic_generator,
        "deterministic_per_env_sampling": item.deterministic_per_env_sampling,
        "generator_sampling_seed": item.generator_sampling_seed,
        "protocol": item.protocol,
        "quota": item.quota,
        "overall_metrics": item.overall_metrics,
        "by_terrain_type": item.by_terrain_type,
    }


def build_ablation_report(
    grouped_paths: Mapping[str, Sequence[str | Path]],
    *,
    allowed_protocol_mismatches: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate grouped evaluator JSONs and build a serializable report."""
    if not grouped_paths:
        raise TerrainReportError("At least one input group is required")
    allowed = set(allowed_protocol_mismatches)
    unknown = allowed.difference(PROTOCOL_FIELDS)
    if unknown:
        raise TerrainReportError(
            f"Unknown allowed protocol mismatch fields: {sorted(unknown)!r}; choices are {PROTOCOL_FIELDS!r}"
        )

    groups: list[dict[str, Any]] = []
    seen_variants: dict[str, str] = {}
    seen_paths: set[str] = set()
    for group_name in sorted(grouped_paths):
        if not isinstance(group_name, str) or not group_name.strip():
            raise TerrainReportError("Group names must be non-empty strings")
        paths = grouped_paths[group_name]
        if not paths:
            raise TerrainReportError(f"Group {group_name!r} has no inputs")
        evaluations: list[_LoadedEvaluation] = []
        for path in paths:
            item = load_terrain_evaluation(path)
            if item.source_path in seen_paths:
                raise TerrainReportError(f"Evaluator JSON {item.source_path} was assigned more than once")
            seen_paths.add(item.source_path)
            if item.variant in seen_variants:
                raise TerrainReportError(
                    f"Duplicate variant {item.variant!r} in {seen_variants[item.variant]!r} and {group_name!r}. "
                    "Use unique labels such as D-final, D-30cm, or D-unseen-rough."
                )
            seen_variants[item.variant] = group_name
            evaluations.append(item)
        evaluations.sort(key=lambda item: _variant_sort_key(item.variant))
        groups.append(
            {
                "name": group_name,
                "protocol": _protocol_for_group(group_name, evaluations, allowed),
                "allowed_protocol_mismatches": sorted(allowed),
                "results": [_evaluation_json(item) for item in evaluations],
            }
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "source_count": sum(len(group["results"]) for group in groups),
        "metric_contract": {
            "raw_pairs_preserved": True,
            "required_rate_metrics": list(_RATE_METRICS),
            "required_step_metrics": list(_STEP_METRICS),
            "optional_step_metrics": list(_OPTIONAL_STEP_METRICS),
            "body_origin_proxy_limitation": (
                "Terrain height minus rigid-body origin Z; not collision-shape or mesh penetration."
            ),
        },
        "groups": groups,
    }


def _iter_report_rows(report: Mapping[str, Any]):
    for group in report["groups"]:
        for result in group["results"]:
            common = {
                "group": group["name"],
                "variant": result["variant"],
                "scenario_label": result["scenario_label"],
                "source_json": result["source_json"],
                "source_json_sha256": result["source_json_sha256"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "generator_checkpoint_sha256": result["generator_checkpoint_sha256"] or "",
                "evaluation_seed": result["protocol"]["evaluation_seed"],
                "fixed_terrain_level": result["protocol"]["fixed_terrain_level"],
                "success_distance_m": result["protocol"]["success_distance_m"],
                "evaluation_phase_mode": result["protocol"]["evaluation_phase_mode"],
                "deterministic_generator": result["protocol"]["deterministic_generator"],
                "deterministic_per_env_sampling": result["protocol"][
                    "deterministic_per_env_sampling"
                ],
                "generator_sampling_seed": result["protocol"]["generator_sampling_seed"],
                "fall_root_height_m": result["protocol"]["fall_root_height_m"],
                "fall_upright_cosine": result["protocol"]["fall_upright_cosine"],
                "body_origin_penetration_threshold_m": result["protocol"]["body_origin_penetration_threshold_m"],
                "body_origin_correction_min_improvement_m": result["protocol"][
                    "body_origin_correction_min_improvement_m"
                ],
                "heading_speed_threshold_mps": result["protocol"]["heading_speed_threshold_mps"],
                "max_episode_length_s": result["protocol"]["max_episode_length_s"],
                "control_dt_s": result["protocol"]["control_dt_s"],
                "terrain_geometry_layout_sha256": result["protocol"]["terrain_geometry_layout"]["sha256"],
                "requested_episode_count": result["quota"]["requested_episode_count"],
                "completed_episode_count": result["quota"]["completed_episode_count"],
                "terrain_types_json": _canonical(result["protocol"]["terrain_types"]),
                "requested_per_terrain_type_json": _canonical(result["quota"]["requested_per_terrain_type"]),
            }
            for metric_name in _ALL_SELECTED_METRICS:
                metric = result["overall_metrics"].get(metric_name)
                if metric is not None:
                    yield {**common, "scope": "overall", "terrain_type": "all", "metric": metric_name, **metric}
            for terrain_type, metrics in result["by_terrain_type"].items():
                for metric_name in _ALL_SELECTED_METRICS:
                    metric = metrics.get(metric_name)
                    if metric is not None:
                        yield {
                            **common,
                            "scope": "terrain_type",
                            "terrain_type": terrain_type,
                            "metric": metric_name,
                            **metric,
                        }


def _format_rate(metric: Mapping[str, Any]) -> str:
    value = metric["value"]
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}% ({metric['numerator']}/{metric['denominator']})"


def _format_mean(metric: Mapping[str, Any], digits: int = 3) -> str:
    value = metric["value"]
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Terrain ablation report",
        "",
        "Rates show raw numerator/denominator. Body-origin penetration is a height proxy, not mesh collision.",
        "",
    ]
    for group in report["groups"]:
        lines.extend(
            (
                f"## {group['name']}",
                "",
                "| Variant | Episodes | Success | Fall | Survival | Contact | Heading rad | "
                "Body pos err | Robot origin proxy | Length | Progress m | Tracker SHA |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            )
        )
        for result in group["results"]:
            metrics = result["overall_metrics"]
            quota = result["quota"]
            values = (
                result["variant"].replace("|", "\\|"),
                f"{quota['completed_episode_count']}/{quota['requested_episode_count']}",
                _format_rate(metrics["episode/success_rate"]),
                _format_rate(metrics["episode/fall_rate"]),
                _format_rate(metrics["episode/survival_rate"]),
                _format_rate(metrics["episode/undesired_contact_rate"]),
                _format_mean(metrics["motion/heading_error_rad"]),
                _format_mean(metrics["motion/error_body_pos"]),
                _format_mean(metrics["terrain/robot_body_origin_penetration_mean_m"], digits=4),
                _format_mean(metrics["episode/mean_length_steps"], digits=1),
                _format_mean(metrics["motion/mean_max_episode_forward_progress_m"]),
                result["checkpoint_sha256"][:12],
            )
            lines.append("| " + " | ".join(values) + " |")
        if group["allowed_protocol_mismatches"]:
            allowed_text = ", ".join(group["allowed_protocol_mismatches"])
            lines.extend(("", f"Explicitly allowed protocol mismatches: `{allowed_text}`."))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_ablation_report(report: Mapping[str, Any], output_prefix: str | Path) -> dict[str, Path]:
    """Write report JSON, long-form CSV, and compact Markdown table."""
    prefix = Path(output_prefix).expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    markdown_path = prefix.with_suffix(".md")
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")

    fields = [
        "group",
        "variant",
        "scenario_label",
        "source_json",
        "source_json_sha256",
        "checkpoint_sha256",
        "generator_checkpoint_sha256",
        "evaluation_seed",
        "fixed_terrain_level",
        "success_distance_m",
        "evaluation_phase_mode",
        "deterministic_generator",
        "deterministic_per_env_sampling",
        "generator_sampling_seed",
        "fall_root_height_m",
        "fall_upright_cosine",
        "body_origin_penetration_threshold_m",
        "body_origin_correction_min_improvement_m",
        "heading_speed_threshold_mps",
        "max_episode_length_s",
        "control_dt_s",
        "terrain_geometry_layout_sha256",
        "requested_episode_count",
        "completed_episode_count",
        "terrain_types_json",
        "requested_per_terrain_type_json",
        "scope",
        "terrain_type",
        "metric",
        "numerator",
        "denominator",
        "value",
        "max",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row in _iter_report_rows(report):
            writer.writerow(row)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def _parse_group_specs(specs: Sequence[str]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for spec in specs:
        name, separator, path_text = spec.partition("=")
        if not separator or not name.strip() or not path_text.strip():
            raise TerrainReportError(f"Invalid --group value {spec!r}; expected NAME=EVALUATOR.json")
        grouped.setdefault(name.strip(), []).append(Path(path_text.strip()))
    return grouped


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and aggregate Stage-10 common terrain evaluator JSONs. Positional inputs form one "
            "strict comparison group; use repeated --group NAME=JSON for distinct height/unseen protocols."
        )
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="Compatible evaluator JSONs (one default group)")
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="Assign one JSON to a named protocol group; repeat for multiple inputs/groups",
    )
    parser.add_argument("--output-prefix", type=Path, required=True, help="Output prefix for .json/.csv/.md")
    parser.add_argument(
        "--allow-mismatch",
        action="append",
        default=[],
        choices=PROTOCOL_FIELDS,
        help="Explicit protocol field allowed to vary inside a group; repeat as needed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the standalone command-line aggregator."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if bool(args.inputs) == bool(args.group):
        parser.error("provide either positional JSONs or repeated --group NAME=JSON, but not both")
    grouped_paths = {"default": args.inputs} if args.inputs else _parse_group_specs(args.group)
    try:
        report = build_ablation_report(
            grouped_paths,
            allowed_protocol_mismatches=args.allow_mismatch,
        )
        outputs = write_ablation_report(report, args.output_prefix)
    except TerrainReportError as error:
        parser.error(str(error))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
