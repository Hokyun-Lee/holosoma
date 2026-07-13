"""Aggregate Stage-10 training/evaluation-seed grids without a simulator.

The manifest is authoritative for policy identities and training seeds.  Each
source is still passed through the strict common terrain-evaluator loader, so
schema, exact quotas, raw metric pairs, and the effective simulator protocol
are validated before anything is pooled.
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

from holosoma.aggregate_terrain_evaluations import (
    PROTOCOL_FIELDS,
    TerrainReportError,
    load_terrain_evaluation,
)

MANIFEST_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
SUCCESS_METRIC = "episode/success_rate"
EPISODE_DENOMINATOR_METRICS = (
    SUCCESS_METRIC,
    "episode/fall_rate",
    "episode/survival_rate",
    "episode/undesired_contact_rate",
    "episode/mean_length_steps",
    "motion/mean_max_episode_forward_progress_m",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMON_PROTOCOL_FIELDS = tuple(field for field in PROTOCOL_FIELDS if field != "evaluation_seed") + (
    "torch_deterministic",
)


@dataclass(frozen=True)
class _ManifestEntry:
    policy_id: str
    training_seed: int
    evaluation_seed: int
    source_json: str
    source_json_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    source_training_run_id: str
    checkpoint_capture_kind: str
    update_budget: int
    source_variant: str
    scenario_label: str
    generator_checkpoint_path: str | None
    generator_checkpoint_sha256: str | None
    torch_deterministic: bool
    protocol: dict[str, Any]
    quota: dict[str, Any]
    overall_metrics: dict[str, dict[str, Any]]
    by_terrain_type: dict[str, dict[str, dict[str, Any]]]

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.policy_id, self.training_seed, self.evaluation_seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise TerrainReportError(f"Cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _cached_artifact_sha256(path: Path, cache: dict[str, str]) -> str:
    key = str(path)
    if key not in cache:
        cache[key] = _sha256(path)
    return cache[key]


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TerrainReportError(f"Value is not canonical JSON: {error}") from error


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TerrainReportError(f"{context} must be a JSON object")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TerrainReportError(f"{context} must be a non-empty string")
    return value.strip()


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TerrainReportError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise TerrainReportError(f"{context} must be >= {minimum}")
    return value


def _digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TerrainReportError(f"{context} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _resolve_path(value: Any, context: str, base_directory: Path) -> Path:
    text = _string(value, context)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise TerrainReportError(f"Cannot read {context} {path}: {error}") from error
    return _mapping(payload, str(path))


def _source_torch_deterministic(source_path: Path) -> bool:
    payload = _read_json(source_path, "evaluator JSON")
    metadata = _mapping(payload.get("metadata"), f"{source_path}.metadata")
    evaluation_config = _mapping(
        metadata.get("evaluation_config"),
        f"{source_path}.metadata.evaluation_config",
    )
    training = _mapping(
        evaluation_config.get("training"),
        f"{source_path}.metadata.evaluation_config.training",
    )
    value = training.get("torch_deterministic")
    if not isinstance(value, bool):
        raise TerrainReportError(
            f"{source_path}.metadata.evaluation_config.training.torch_deterministic must be boolean"
        )
    return value


def _load_entry(
    value: Any,
    *,
    index: int,
    manifest_directory: Path,
    artifact_hash_cache: dict[str, str],
) -> _ManifestEntry:
    context = f"manifest.entries[{index}]"
    entry = _mapping(value, context)
    policy_id = _string(entry.get("policy_id"), f"{context}.policy_id")
    training_seed = _integer(entry.get("training_seed"), f"{context}.training_seed")
    evaluation_seed = _integer(entry.get("evaluation_seed"), f"{context}.evaluation_seed")
    source_path = _resolve_path(entry.get("source_json"), f"{context}.source_json", manifest_directory)
    expected_source_sha = _digest(entry.get("source_json_sha256"), f"{context}.source_json_sha256")
    actual_source_sha = _sha256(source_path)
    if actual_source_sha != expected_source_sha:
        raise TerrainReportError(
            f"{context} source JSON SHA-256 mismatch: manifest={expected_source_sha}, actual={actual_source_sha}"
        )

    checkpoint_path = _resolve_path(
        entry.get("checkpoint_path"),
        f"{context}.checkpoint_path",
        manifest_directory,
    )
    expected_checkpoint_sha = _digest(
        entry.get("checkpoint_sha256"),
        f"{context}.checkpoint_sha256",
    )
    actual_checkpoint_sha = _cached_artifact_sha256(checkpoint_path, artifact_hash_cache)
    if actual_checkpoint_sha != expected_checkpoint_sha:
        raise TerrainReportError(
            f"{context} checkpoint SHA-256 mismatch: manifest={expected_checkpoint_sha}, actual={actual_checkpoint_sha}"
        )

    source_training_run_id = _string(
        entry.get("source_training_run_id"),
        f"{context}.source_training_run_id",
    )
    checkpoint_capture_kind = _string(
        entry.get("checkpoint_capture_kind"),
        f"{context}.checkpoint_capture_kind",
    )
    update_budget = _integer(entry.get("update_budget"), f"{context}.update_budget", minimum=0)

    loaded = load_terrain_evaluation(source_path)
    if loaded.source_sha256 != expected_source_sha:
        raise TerrainReportError(f"{context} evaluator JSON changed while it was being validated")
    if loaded.checkpoint_sha256 != expected_checkpoint_sha:
        raise TerrainReportError(
            f"{context} checkpoint SHA-256 disagrees with evaluator metadata: "
            f"manifest={expected_checkpoint_sha}, evaluator={loaded.checkpoint_sha256}"
        )
    if not isinstance(loaded.checkpoint_path, str) or not loaded.checkpoint_path.strip():
        raise TerrainReportError(f"{context} evaluator metadata checkpoint_path must be a non-empty string")
    evaluator_checkpoint = Path(loaded.checkpoint_path).expanduser()
    if not evaluator_checkpoint.is_absolute():
        evaluator_checkpoint = source_path.parent / evaluator_checkpoint
    evaluator_checkpoint = evaluator_checkpoint.resolve()
    if evaluator_checkpoint != checkpoint_path:
        raise TerrainReportError(
            f"{context} checkpoint path disagrees with evaluator metadata: "
            f"manifest={checkpoint_path}, evaluator={evaluator_checkpoint}"
        )
    if loaded.protocol["evaluation_seed"] != evaluation_seed:
        raise TerrainReportError(
            f"{context} evaluation_seed {evaluation_seed} disagrees with evaluator protocol "
            f"{loaded.protocol['evaluation_seed']}"
        )

    generator_path_value = loaded.generator_checkpoint_path
    generator_sha_value = loaded.generator_checkpoint_sha256
    if (generator_path_value in (None, "")) != (generator_sha_value is None):
        raise TerrainReportError(
            f"{context} evaluator generator checkpoint path and SHA-256 must either both be present or both be null"
        )
    generator_path: str | None = None
    generator_sha: str | None = None
    if generator_path_value not in (None, ""):
        if not isinstance(generator_path_value, str):
            raise TerrainReportError(f"{context} evaluator generator checkpoint path must be a string")
        resolved_generator = Path(generator_path_value).expanduser()
        if not resolved_generator.is_absolute():
            resolved_generator = source_path.parent / resolved_generator
        resolved_generator = resolved_generator.resolve()
        actual_generator_sha = _cached_artifact_sha256(resolved_generator, artifact_hash_cache)
        if actual_generator_sha != generator_sha_value:
            raise TerrainReportError(
                f"{context} generator checkpoint SHA-256 mismatch: evaluator={generator_sha_value}, "
                f"actual={actual_generator_sha}"
            )
        generator_path = str(resolved_generator)
        generator_sha = generator_sha_value

    torch_deterministic = _source_torch_deterministic(source_path)
    if _sha256(source_path) != expected_source_sha:
        raise TerrainReportError(f"{context} evaluator JSON changed while it was being validated")
    protocol = {**loaded.protocol, "torch_deterministic": torch_deterministic}
    return _ManifestEntry(
        policy_id=policy_id,
        training_seed=training_seed,
        evaluation_seed=evaluation_seed,
        source_json=loaded.source_path,
        source_json_sha256=expected_source_sha,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=expected_checkpoint_sha,
        source_training_run_id=source_training_run_id,
        checkpoint_capture_kind=checkpoint_capture_kind,
        update_budget=update_budget,
        source_variant=loaded.variant,
        scenario_label=loaded.scenario_label,
        generator_checkpoint_path=generator_path,
        generator_checkpoint_sha256=generator_sha,
        torch_deterministic=torch_deterministic,
        protocol=protocol,
        quota=loaded.quota,
        overall_metrics=loaded.overall_metrics,
        by_terrain_type=loaded.by_terrain_type,
    )


def _validate_common_protocol(entries: Sequence[_ManifestEntry]) -> dict[str, Any]:
    reference = entries[0]
    common = {field: reference.protocol[field] for field in _COMMON_PROTOCOL_FIELDS}
    for item in entries[1:]:
        for field in _COMMON_PROTOCOL_FIELDS:
            if _canonical(item.protocol[field]) != _canonical(common[field]):
                raise TerrainReportError(
                    "Protocol mismatch for "
                    f"{field}: {reference.identity!r}={common[field]!r}, "
                    f"{item.identity!r}={item.protocol[field]!r}. Only evaluation_seed may vary."
                )
    return common


def _validate_group_provenance(entries: Sequence[_ManifestEntry]) -> None:
    grouped: dict[tuple[str, int], list[_ManifestEntry]] = {}
    for item in entries:
        grouped.setdefault((item.policy_id, item.training_seed), []).append(item)
    fields = (
        "checkpoint_path",
        "checkpoint_sha256",
        "generator_checkpoint_path",
        "generator_checkpoint_sha256",
        "source_training_run_id",
        "checkpoint_capture_kind",
        "update_budget",
    )
    for key, group in grouped.items():
        for field in fields:
            values = {_canonical(getattr(item, field)) for item in group}
            if len(values) != 1:
                raise TerrainReportError(
                    f"Manifest provenance mismatch for policy/training seed {key!r}: {field} varies across evaluations"
                )

    by_policy: dict[str, list[_ManifestEntry]] = {}
    for item in entries:
        by_policy.setdefault(item.policy_id, []).append(item)
    for policy_id, group in by_policy.items():
        for field in (
            "generator_checkpoint_path",
            "generator_checkpoint_sha256",
            "update_budget",
        ):
            values = {_canonical(getattr(item, field)) for item in group}
            if len(values) != 1:
                raise TerrainReportError(
                    f"Policy {policy_id!r} combines different {field} values; use distinct policy_id labels"
                )
        representatives = {item.training_seed: item for item in group}
        for field in ("checkpoint_sha256", "source_training_run_id"):
            values = [getattr(item, field) for item in representatives.values()]
            if len(set(values)) != len(values):
                raise TerrainReportError(
                    f"Policy {policy_id!r} reuses {field} across different training seeds; "
                    "this would create pseudoreplication"
                )
        evaluation_grids: dict[int, set[int]] = {}
        for item in group:
            evaluation_grids.setdefault(item.training_seed, set()).add(item.evaluation_seed)
        distinct_grids = {tuple(sorted(seeds)) for seeds in evaluation_grids.values()}
        if len(distinct_grids) != 1:
            rendered = {seed: sorted(eval_seeds) for seed, eval_seeds in sorted(evaluation_grids.items())}
            raise TerrainReportError(
                f"Policy {policy_id!r} has an incomplete evaluation-seed grid across training seeds: {rendered!r}"
            )


def _pool_metric_records(records: Sequence[Mapping[str, Any]], context: str) -> dict[str, Any]:
    if not records:
        raise TerrainReportError(f"Cannot pool empty metric records for {context}")
    numerator = math.fsum(float(record["numerator"]) for record in records)
    denominator = sum(int(record["denominator"]) for record in records)
    result: dict[str, Any] = {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
    }
    if any("max" in record for record in records):
        maxima = [float(record["max"]) for record in records if record.get("max") is not None]
        result["max"] = max(maxima) if maxima else None
    return result


def _pool_metric_maps(
    metric_maps: Sequence[Mapping[str, Mapping[str, Any]]],
    context: str,
) -> dict[str, dict[str, Any]]:
    if not metric_maps:
        raise TerrainReportError(f"Cannot pool empty metric maps for {context}")
    metric_names = set(metric_maps[0])
    for index, metrics in enumerate(metric_maps[1:], start=1):
        if set(metrics) != metric_names:
            raise TerrainReportError(
                f"Metric schema mismatch for {context}: input 0 and input {index} have different metric names"
            )
    return {
        metric_name: _pool_metric_records(
            [metrics[metric_name] for metrics in metric_maps],
            f"{context}.{metric_name}",
        )
        for metric_name in sorted(metric_names)
    }


def _pool_scopes(
    overall_maps: Sequence[Mapping[str, Mapping[str, Any]]],
    terrain_maps: Sequence[Mapping[str, Mapping[str, Mapping[str, Any]]]],
    context: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    terrain_types = set(terrain_maps[0])
    for index, by_type in enumerate(terrain_maps[1:], start=1):
        if set(by_type) != terrain_types:
            raise TerrainReportError(f"Terrain metric schema mismatch for {context}: input 0 and input {index} differ")
    overall = _pool_metric_maps(overall_maps, f"{context}.overall")
    by_terrain_type = {
        terrain_type: {
            "episode_count": sum(int(metrics[terrain_type][SUCCESS_METRIC]["denominator"]) for metrics in terrain_maps),
            "metrics": _pool_metric_maps(
                [metrics[terrain_type] for metrics in terrain_maps],
                f"{context}.by_terrain_type.{terrain_type}",
            ),
        }
        for terrain_type in sorted(terrain_types)
    }
    return overall, by_terrain_type


def _entry_provenance(item: _ManifestEntry) -> dict[str, Any]:
    return {
        "policy_id": item.policy_id,
        "training_seed": item.training_seed,
        "evaluation_seed": item.evaluation_seed,
        "identity": [item.policy_id, item.training_seed, item.evaluation_seed],
        "source_json": item.source_json,
        "source_json_sha256": item.source_json_sha256,
        "source_evaluator_variant": item.source_variant,
        "scenario_label": item.scenario_label,
        "checkpoint_path": item.checkpoint_path,
        "checkpoint_sha256": item.checkpoint_sha256,
        "generator_checkpoint_path": item.generator_checkpoint_path,
        "generator_checkpoint_sha256": item.generator_checkpoint_sha256,
        "source_training_run_id": item.source_training_run_id,
        "checkpoint_capture_kind": item.checkpoint_capture_kind,
        "update_budget": item.update_budget,
        "torch_deterministic": item.torch_deterministic,
        "protocol": item.protocol,
        "quota": item.quota,
    }


def _pool_policy_training_seed(
    policy_id: str,
    training_seed: int,
    entries: Sequence[_ManifestEntry],
) -> dict[str, Any]:
    ordered = sorted(entries, key=lambda item: item.evaluation_seed)
    overall, by_terrain_type = _pool_scopes(
        [item.overall_metrics for item in ordered],
        [item.by_terrain_type for item in ordered],
        f"{policy_id}.training_seed_{training_seed}",
    )
    first = ordered[0]
    return {
        "policy_id": policy_id,
        "training_seed": training_seed,
        "evaluation_seeds": [item.evaluation_seed for item in ordered],
        "source_count": len(ordered),
        "checkpoint_path": first.checkpoint_path,
        "checkpoint_sha256": first.checkpoint_sha256,
        "source_training_run_id": first.source_training_run_id,
        "checkpoint_capture_kind": first.checkpoint_capture_kind,
        "update_budget": first.update_budget,
        "overall_episode_count": overall[SUCCESS_METRIC]["denominator"],
        "overall_metrics": overall,
        "by_terrain_type": by_terrain_type,
    }


def _pool_policy(
    policy_id: str,
    training_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(training_results, key=lambda item: int(item["training_seed"]))
    overall, by_terrain_type = _pool_scopes(
        [item["overall_metrics"] for item in ordered],
        [
            {
                terrain_type: terrain_result["metrics"]
                for terrain_type, terrain_result in item["by_terrain_type"].items()
            }
            for item in ordered
        ],
        f"{policy_id}.all_training_seeds",
    )
    capture_kinds_by_training_seed = {str(item["training_seed"]): item["checkpoint_capture_kind"] for item in ordered}
    distinct_capture_kinds = set(capture_kinds_by_training_seed.values())
    capture_kind_summary: str | dict[str, Any]
    if len(distinct_capture_kinds) == 1:
        capture_kind_summary = next(iter(distinct_capture_kinds))
    else:
        capture_kind_summary = {"varies": capture_kinds_by_training_seed}
    return {
        "policy_id": policy_id,
        "training_seeds": [int(item["training_seed"]) for item in ordered],
        "evaluation_seeds_by_training_seed": {str(item["training_seed"]): item["evaluation_seeds"] for item in ordered},
        "training_seed_count": len(ordered),
        "source_count": sum(int(item["source_count"]) for item in ordered),
        "checkpoint_capture_kind": capture_kind_summary,
        "checkpoint_capture_kinds_by_training_seed": capture_kinds_by_training_seed,
        "update_budget": ordered[0]["update_budget"],
        "checkpoints": [
            {
                "training_seed": item["training_seed"],
                "path": item["checkpoint_path"],
                "sha256": item["checkpoint_sha256"],
                "source_training_run_id": item["source_training_run_id"],
                "checkpoint_capture_kind": item["checkpoint_capture_kind"],
                "update_budget": item["update_budget"],
            }
            for item in ordered
        ],
        "overall_episode_count": overall[SUCCESS_METRIC]["denominator"],
        "overall_metrics": overall,
        "by_terrain_type": by_terrain_type,
    }


def _sign(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def _build_contrasts(
    specs: Sequence[Any],
    training_results: Sequence[Mapping[str, Any]],
    policy_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_policy_training = {(str(item["policy_id"]), int(item["training_seed"])): item for item in training_results}
    policy_ids = {str(item["policy_id"]) for item in policy_results}
    policy_result_by_id = {str(item["policy_id"]): item for item in policy_results}
    seen_names: set[str] = set()
    results: list[dict[str, Any]] = []
    for index, value in enumerate(specs):
        context = f"manifest.contrasts[{index}]"
        spec = _mapping(value, context)
        name = _string(spec.get("name"), f"{context}.name")
        left = _string(spec.get("left_policy_id"), f"{context}.left_policy_id")
        right = _string(spec.get("right_policy_id"), f"{context}.right_policy_id")
        if name in seen_names:
            raise TerrainReportError(f"Duplicate contrast name {name!r}")
        seen_names.add(name)
        if left == right:
            raise TerrainReportError(f"{context} must compare two different policy IDs")
        missing = {left, right}.difference(policy_ids)
        if missing:
            raise TerrainReportError(f"{context} references unknown policies: {sorted(missing)!r}")

        left_seeds = {seed for policy, seed in by_policy_training if policy == left}
        right_seeds = {seed for policy, seed in by_policy_training if policy == right}
        if left_seeds != right_seeds:
            raise TerrainReportError(
                f"{context} requires paired training seeds: {left}={sorted(left_seeds)}, {right}={sorted(right_seeds)}"
            )
        paired: list[dict[str, Any]] = []
        for training_seed in sorted(left_seeds):
            left_result = by_policy_training[(left, training_seed)]
            right_result = by_policy_training[(right, training_seed)]
            if left_result["evaluation_seeds"] != right_result["evaluation_seeds"]:
                raise TerrainReportError(
                    f"{context} training seed {training_seed} has unpaired evaluation seeds: "
                    f"{left}={left_result['evaluation_seeds']}, {right}={right_result['evaluation_seeds']}"
                )
            left_rate = float(left_result["overall_metrics"][SUCCESS_METRIC]["value"])
            right_rate = float(right_result["overall_metrics"][SUCCESS_METRIC]["value"])
            delta = left_rate - right_rate
            paired.append(
                {
                    "training_seed": training_seed,
                    "evaluation_seeds": left_result["evaluation_seeds"],
                    "left_success_rate": left_rate,
                    "right_success_rate": right_rate,
                    "delta": delta,
                    "sign": _sign(delta),
                }
            )
        deltas = [float(item["delta"]) for item in paired]
        mean = math.fsum(deltas) / len(deltas)
        sample_sd = None
        if len(deltas) >= 2:
            sample_sd = math.sqrt(math.fsum((delta - mean) ** 2 for delta in deltas) / (len(deltas) - 1))
        signs = {sign: sum(item["sign"] == sign for item in paired) for sign in ("positive", "zero", "negative")}
        pooled_left = float(policy_result_by_id[left]["overall_metrics"][SUCCESS_METRIC]["value"])
        pooled_right = float(policy_result_by_id[right]["overall_metrics"][SUCCESS_METRIC]["value"])
        results.append(
            {
                "name": name,
                "metric": SUCCESS_METRIC,
                "definition": f"{left} - {right}",
                "left_policy_id": left,
                "right_policy_id": right,
                "training_seed_deltas": paired,
                "descriptive_statistics": {
                    "n_training_seeds": len(deltas),
                    "minimum": min(deltas),
                    "maximum": max(deltas),
                    "range": max(deltas) - min(deltas),
                    "mean": mean,
                    "sample_standard_deviation": sample_sd,
                    "sign_counts": signs,
                },
                "raw_pooled_success_rate_difference": pooled_left - pooled_right,
            }
        )
    return results


def build_training_seed_report(manifest_path: str | Path) -> dict[str, Any]:
    """Validate a manifest and build a two-stage raw-pooled report."""
    manifest = Path(manifest_path).expanduser().resolve()
    manifest_sha = _sha256(manifest)
    payload = _read_json(manifest, "manifest")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise TerrainReportError(
            f"Manifest has unsupported schema_version {payload.get('schema_version')!r}; "
            f"expected {MANIFEST_SCHEMA_VERSION}"
        )
    metadata_value = payload.get("metadata", {})
    metadata = dict(_mapping(metadata_value, "manifest.metadata"))
    _canonical(metadata)
    entries_value = payload.get("entries")
    if not isinstance(entries_value, list) or not entries_value:
        raise TerrainReportError("manifest.entries must be a non-empty JSON array")
    contrast_specs = payload.get("contrasts", [])
    if not isinstance(contrast_specs, list):
        raise TerrainReportError("manifest.contrasts must be a JSON array")

    entries: list[_ManifestEntry] = []
    artifact_hash_cache: dict[str, str] = {}
    identities: dict[tuple[str, int, int], int] = {}
    source_paths: dict[str, int] = {}
    for index, value in enumerate(entries_value):
        item = _load_entry(
            value,
            index=index,
            manifest_directory=manifest.parent,
            artifact_hash_cache=artifact_hash_cache,
        )
        if item.identity in identities:
            raise TerrainReportError(
                f"Duplicate (policy_id, training_seed, evaluation_seed) identity {item.identity!r} "
                f"in manifest entries {identities[item.identity]} and {index}"
            )
        identities[item.identity] = index
        if item.source_json in source_paths:
            raise TerrainReportError(
                f"Evaluator JSON {item.source_json} is reused by manifest entries "
                f"{source_paths[item.source_json]} and {index}"
            )
        source_paths[item.source_json] = index
        entries.append(item)
    entries.sort(key=lambda item: item.identity)
    _validate_group_provenance(entries)
    common_protocol = _validate_common_protocol(entries)

    grouped: dict[tuple[str, int], list[_ManifestEntry]] = {}
    for item in entries:
        grouped.setdefault((item.policy_id, item.training_seed), []).append(item)
    training_results = [
        _pool_policy_training_seed(policy_id, training_seed, group)
        for (policy_id, training_seed), group in sorted(grouped.items())
    ]
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for item in training_results:
        by_policy.setdefault(str(item["policy_id"]), []).append(item)
    policy_results = [_pool_policy(policy_id, group) for policy_id, group in sorted(by_policy.items())]
    contrasts = _build_contrasts(contrast_specs, training_results, policy_results)
    common_protocol_sha = hashlib.sha256(_canonical(common_protocol).encode("utf-8")).hexdigest()
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_provenance": {
            "path": str(manifest),
            "sha256": manifest_sha,
            "metadata": metadata,
        },
        "source_count": len(entries),
        "policy_count": len(policy_results),
        "policy_training_seed_count": len(training_results),
        "identity_contract": {
            "axes": ["policy_id", "training_seed", "evaluation_seed"],
            "policy_id_authority": "manifest",
            "source_evaluator_variant_is_provenance_only": True,
            "unique": True,
        },
        "provenance_contract": {
            "checkpoint_capture_kind_scope": "(policy_id, training_seed)",
            "checkpoint_capture_kind_may_vary_across_training_seeds": True,
            "policy_results_preserve_capture_kind_by_training_seed": True,
        },
        "pooling_contract": {
            "stages": [
                "sum evaluator raw numerator/denominator pairs within (policy_id, training_seed)",
                "sum those raw pairs across training seeds within policy_id",
            ],
            "numerator_summation": "math.fsum",
            "denominator_summation": "integer sum",
            "rounded_rate_averaging": False,
            "episode_and_step_denominators_preserved": True,
            "overall_and_terrain_stratified_metrics_preserved": True,
        },
        "metric_contract": {
            "episode_denominator_metrics": list(EPISODE_DENOMINATOR_METRICS),
            "step_denominator_metrics": sorted(set(entries[0].overall_metrics).difference(EPISODE_DENOMINATOR_METRICS)),
            "raw_numerator_denominator_pairs_preserved": True,
            "source_maxima_reduced_with_max": True,
        },
        "contrast_inference_contract": {
            "unit_of_replication": "training_seed",
            "descriptive_only": True,
            "episode_level_confidence_interval": False,
            "significance_test": False,
            "sample_standard_deviation_basis": "paired training-seed success-rate deltas",
            "warning": (
                "Training-seed n is not an episode sample size; contrast summaries are descriptive and "
                "must not be interpreted as episode-level confidence intervals."
            ),
        },
        "protocol_provenance": {
            "common_fields": list(_COMMON_PROTOCOL_FIELDS),
            "only_allowed_source_variation": ["evaluation_seed"],
            "common_protocol": common_protocol,
            "common_protocol_sha256": common_protocol_sha,
            "evaluation_seeds": sorted({item.evaluation_seed for item in entries}),
            "torch_deterministic": common_protocol["torch_deterministic"],
        },
        "sources": [_entry_provenance(item) for item in entries],
        "policy_training_seed_results": training_results,
        "policy_results": policy_results,
        "contrasts": contrasts,
    }


def _format_rate(metric: Mapping[str, Any]) -> str:
    if metric["value"] is None:
        return "n/a"
    return f"{100.0 * float(metric['value']):.1f}% ({metric['numerator']:g}/{metric['denominator']})"


def _format_mean(metric: Mapping[str, Any], digits: int = 3) -> str:
    return "n/a" if metric["value"] is None else f"{float(metric['value']):.{digits}f}"


def _markdown(report: Mapping[str, Any]) -> str:
    protocol = report["protocol_provenance"]
    lines = [
        "# Training-seed evaluation aggregation",
        "",
        (
            "Raw numerators and denominators are pooled first within each policy/training seed and then "
            "across training seeds. Rounded evaluator rates are never averaged."
        ),
        "",
        (
            "Contrast statistics use training seed as the replication unit and are descriptive only; "
            "they are not episode-level confidence intervals or significance tests."
        ),
        "",
        f"- Manifest SHA-256: `{report['manifest_provenance']['sha256']}`",
        f"- Common protocol SHA-256: `{protocol['common_protocol_sha256']}`",
        f"- Evaluation seeds: `{protocol['evaluation_seeds']}`",
        f"- `torch_deterministic`: `{str(protocol['torch_deterministic']).lower()}`",
        "",
        "## Policy raw-pooled results",
        "",
        "| Policy | Training seeds | Sources | Episodes | Success | Fall | Episode any-contact | "
        "Contact-step fraction | Heading rad | Body pos err |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["policy_results"]:
        metrics = result["overall_metrics"]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(result["policy_id"]).replace("|", "\\|"),
                    str(result["training_seeds"]),
                    str(result["source_count"]),
                    str(result["overall_episode_count"]),
                    _format_rate(metrics[SUCCESS_METRIC]),
                    _format_rate(metrics["episode/fall_rate"]),
                    _format_rate(metrics["episode/undesired_contact_rate"]),
                    _format_rate(metrics["terrain/undesired_contact_any"]),
                    _format_mean(metrics["motion/heading_error_rad"]),
                    _format_mean(metrics["motion/error_body_pos"]),
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## Per-training-seed success",
            "",
            "| Policy | Training seed | Evaluation seeds | Episodes | Success | Checkpoint SHA | Run ID | "
            "Capture | Update budget |",
            "|---|---:|---|---:|---:|---|---|---|---:|",
        )
    )
    lines.extend(
        (
            "| "
            + " | ".join(
                (
                    str(result["policy_id"]).replace("|", "\\|"),
                    str(result["training_seed"]),
                    str(result["evaluation_seeds"]),
                    str(result["overall_episode_count"]),
                    _format_rate(result["overall_metrics"][SUCCESS_METRIC]),
                    str(result["checkpoint_sha256"])[:12],
                    str(result["source_training_run_id"]).replace("|", "\\|"),
                    str(result["checkpoint_capture_kind"]).replace("|", "\\|"),
                    str(result["update_budget"]),
                )
            )
            + " |"
        )
        for result in report["policy_training_seed_results"]
    )

    if report["contrasts"]:
        lines.extend(
            (
                "",
                "## Descriptive paired success contrasts",
                "",
                "| Contrast | n training seeds | Mean paired delta | Raw-pooled delta | Sample SD | Range | Min | "
                "Max | Signs (+/0/-) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            )
        )
        for contrast in report["contrasts"]:
            stats = contrast["descriptive_statistics"]
            sample_sd = stats["sample_standard_deviation"]
            signs = stats["sign_counts"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(contrast["definition"]).replace("|", "\\|"),
                        str(stats["n_training_seeds"]),
                        f"{stats['mean']:.4f}",
                        f"{contrast['raw_pooled_success_rate_difference']:.4f}",
                        "n/a" if sample_sd is None else f"{sample_sd:.4f}",
                        f"{stats['range']:.4f}",
                        f"{stats['minimum']:.4f}",
                        f"{stats['maximum']:.4f}",
                        f"{signs['positive']}/{signs['zero']}/{signs['negative']}",
                    )
                )
                + " |"
            )
            lines.extend(("", f"Per-training-seed deltas for `{contrast['name']}`:"))
            lines.extend(
                (
                    f"- seed {item['training_seed']}: {item['delta']:+.4f} ({item['sign']}), "
                    f"evaluation seeds {item['evaluation_seeds']}"
                )
                for item in contrast["training_seed_deltas"]
            )

    lines.extend(
        (
            "",
            "## Source provenance",
            "",
            "Manifest `policy_id` is authoritative; evaluator variants below are provenance labels only.",
            "",
            "| Policy | Train seed | Eval seed | Evaluator variant | Source JSON | Source SHA-256 | "
            "Tracker checkpoint | Tracker SHA-256 | Generator checkpoint | Generator SHA-256 | Run ID | "
            "Capture | Budget |",
            "|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|",
        )
    )
    lines.extend(
        (
            "| "
            + " | ".join(
                (
                    str(source["policy_id"]).replace("|", "\\|"),
                    str(source["training_seed"]),
                    str(source["evaluation_seed"]),
                    str(source["source_evaluator_variant"]).replace("|", "\\|"),
                    str(source["source_json"]).replace("|", "\\|"),
                    str(source["source_json_sha256"]),
                    str(source["checkpoint_path"]).replace("|", "\\|"),
                    str(source["checkpoint_sha256"]),
                    str(source["generator_checkpoint_path"] or "").replace("|", "\\|"),
                    str(source["generator_checkpoint_sha256"] or ""),
                    str(source["source_training_run_id"]).replace("|", "\\|"),
                    str(source["checkpoint_capture_kind"]).replace("|", "\\|"),
                    str(source["update_budget"]),
                )
            )
            + " |"
        )
        for source in report["sources"]
    )
    return "\n".join(lines).rstrip() + "\n"


def _metric_rows(report: Mapping[str, Any]):
    manifest = report["manifest_provenance"]
    protocol = report["protocol_provenance"]
    common = {
        "manifest_path": manifest["path"],
        "manifest_sha256": manifest["sha256"],
        "common_protocol_sha256": protocol["common_protocol_sha256"],
        "torch_deterministic": protocol["torch_deterministic"],
    }
    for level, key in (
        ("policy_training_seed", "policy_training_seed_results"),
        ("policy", "policy_results"),
    ):
        for result in report[key]:
            result_common = {
                **common,
                "record_type": "pooled_metric",
                "aggregation_level": level,
                "policy_id": result["policy_id"],
                "training_seed": result.get("training_seed", ""),
                "training_seeds_json": _canonical(
                    [result["training_seed"]] if "training_seed" in result else result["training_seeds"]
                ),
                "evaluation_seeds_json": _canonical(
                    result["evaluation_seeds"]
                    if "evaluation_seeds" in result
                    else result["evaluation_seeds_by_training_seed"]
                ),
                "source_count": result["source_count"],
                "checkpoint_sha256s_json": _canonical(
                    [result["checkpoint_sha256"]]
                    if "checkpoint_sha256" in result
                    else [checkpoint["sha256"] for checkpoint in result["checkpoints"]]
                ),
                "checkpoint_capture_kinds_json": _canonical(
                    {str(result["training_seed"]): result["checkpoint_capture_kind"]}
                    if "training_seed" in result
                    else result["checkpoint_capture_kinds_by_training_seed"]
                ),
            }
            for metric_name, metric in result["overall_metrics"].items():
                yield {
                    **result_common,
                    "scope": "overall",
                    "terrain_type": "all",
                    "metric": metric_name,
                    **metric,
                }
            for terrain_type, terrain_result in result["by_terrain_type"].items():
                for metric_name, metric in terrain_result["metrics"].items():
                    yield {
                        **result_common,
                        "scope": "terrain_type",
                        "terrain_type": terrain_type,
                        "metric": metric_name,
                        **metric,
                    }


def _contrast_rows(report: Mapping[str, Any]):
    common = {
        "manifest_path": report["manifest_provenance"]["path"],
        "manifest_sha256": report["manifest_provenance"]["sha256"],
        "common_protocol_sha256": report["protocol_provenance"]["common_protocol_sha256"],
        "torch_deterministic": report["protocol_provenance"]["torch_deterministic"],
    }
    for contrast in report["contrasts"]:
        for item in contrast["training_seed_deltas"]:
            yield {
                **common,
                "record_type": "contrast_training_seed",
                "aggregation_level": "training_seed_contrast",
                "training_seed": item["training_seed"],
                "evaluation_seeds_json": _canonical(item["evaluation_seeds"]),
                "metric": SUCCESS_METRIC,
                "contrast_name": contrast["name"],
                "left_policy_id": contrast["left_policy_id"],
                "right_policy_id": contrast["right_policy_id"],
                "delta": item["delta"],
                "sign": item["sign"],
            }
        stats = contrast["descriptive_statistics"]
        yield {
            **common,
            "record_type": "contrast_summary",
            "aggregation_level": "training_seed_contrast_summary",
            "metric": SUCCESS_METRIC,
            "contrast_name": contrast["name"],
            "left_policy_id": contrast["left_policy_id"],
            "right_policy_id": contrast["right_policy_id"],
            "n_training_seeds": stats["n_training_seeds"],
            "minimum": stats["minimum"],
            "maximum": stats["maximum"],
            "range": stats["range"],
            "mean": stats["mean"],
            "sample_standard_deviation": stats["sample_standard_deviation"],
            "sign_counts_json": _canonical(stats["sign_counts"]),
            "delta": contrast["raw_pooled_success_rate_difference"],
        }


def _provenance_rows(report: Mapping[str, Any]):
    manifest = report["manifest_provenance"]
    protocol = report["protocol_provenance"]
    common = {
        "manifest_path": manifest["path"],
        "manifest_sha256": manifest["sha256"],
        "common_protocol_sha256": protocol["common_protocol_sha256"],
        "torch_deterministic": protocol["torch_deterministic"],
    }
    yield {
        **common,
        "record_type": "protocol_provenance",
        "aggregation_level": "common_protocol",
        "evaluation_seeds_json": _canonical(protocol["evaluation_seeds"]),
        "protocol_json": _canonical(protocol["common_protocol"]),
    }
    for source in report["sources"]:
        yield {
            **common,
            "record_type": "source_provenance",
            "aggregation_level": "source",
            "policy_id": source["policy_id"],
            "training_seed": source["training_seed"],
            "evaluation_seed": source["evaluation_seed"],
            "source_json": source["source_json"],
            "source_json_sha256": source["source_json_sha256"],
            "source_evaluator_variant": source["source_evaluator_variant"],
            "checkpoint_path": source["checkpoint_path"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "generator_checkpoint_path": source["generator_checkpoint_path"],
            "generator_checkpoint_sha256": source["generator_checkpoint_sha256"],
            "source_training_run_id": source["source_training_run_id"],
            "checkpoint_capture_kind": source["checkpoint_capture_kind"],
            "update_budget": source["update_budget"],
            "protocol_json": _canonical(source["protocol"]),
        }


def write_training_seed_report(report: Mapping[str, Any], output_prefix: str | Path) -> dict[str, Path]:
    """Write JSON, long-form CSV, and Markdown training-seed reports."""
    prefix = Path(output_prefix).expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": prefix.with_suffix(".json"),
        "csv": prefix.with_suffix(".csv"),
        "markdown": prefix.with_suffix(".md"),
    }
    with paths["json"].open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")

    rows = [*_metric_rows(report), *_contrast_rows(report), *_provenance_rows(report)]
    fields = [
        "record_type",
        "aggregation_level",
        "manifest_path",
        "manifest_sha256",
        "common_protocol_sha256",
        "torch_deterministic",
        "policy_id",
        "training_seed",
        "evaluation_seed",
        "training_seeds_json",
        "evaluation_seeds_json",
        "source_count",
        "checkpoint_sha256s_json",
        "checkpoint_capture_kinds_json",
        "scope",
        "terrain_type",
        "metric",
        "numerator",
        "denominator",
        "value",
        "max",
        "contrast_name",
        "left_policy_id",
        "right_policy_id",
        "delta",
        "sign",
        "n_training_seeds",
        "minimum",
        "maximum",
        "range",
        "mean",
        "sample_standard_deviation",
        "sign_counts_json",
        "source_json",
        "source_json_sha256",
        "source_evaluator_variant",
        "checkpoint_path",
        "checkpoint_sha256",
        "generator_checkpoint_path",
        "generator_checkpoint_sha256",
        "source_training_run_id",
        "checkpoint_capture_kind",
        "update_budget",
        "protocol_json",
    ]
    with paths["csv"].open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    return paths


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and raw-pool a Stage-10 policy/training-seed/evaluation-seed manifest. "
            "Only evaluation_seed may vary in the common evaluator protocol."
        ),
        epilog=(
            "Each manifest entry requires policy_id, training_seed, evaluation_seed, source_json, "
            "source_json_sha256, checkpoint_path, checkpoint_sha256, source_training_run_id, "
            "checkpoint_capture_kind, and non-negative integer update_budget. Optional contrasts use "
            "{name, left_policy_id, right_policy_id}; delta is left minus right success rate. Relative "
            "paths are resolved from the manifest directory. Capture kind must be fixed across evaluation "
            "seeds for one policy/training seed, but may record backfill/direct differences across training seeds."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSON seed-grid manifest")
    parser.add_argument("--output-prefix", type=Path, required=True, help="Output prefix for .json/.csv/.md")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the simulator-free training-seed grid aggregator."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        report = build_training_seed_report(args.manifest)
        outputs = write_training_seed_report(report, args.output_prefix)
    except TerrainReportError as error:
        parser.error(str(error))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
