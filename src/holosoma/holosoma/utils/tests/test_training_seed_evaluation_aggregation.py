from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
from holosoma.aggregate_terrain_evaluations import TerrainReportError
from holosoma.aggregate_training_seed_evaluations import (
    SUCCESS_METRIC,
    build_training_seed_report,
    main,
    write_training_seed_report,
)

RATE_METRICS = (
    "episode/success_rate",
    "episode/fall_rate",
    "episode/survival_rate",
    "episode/undesired_contact_rate",
    "episode/mean_length_steps",
    "motion/mean_max_episode_forward_progress_m",
)
STEP_METRICS = (
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(numerator: float, denominator: int, *, maximum: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
    }
    if maximum is not None:
        result["max"] = maximum
    return result


def _summary_group(
    episode_count: int,
    *,
    success_count: int,
    contact_step_numerator: float,
    contact_step_denominator: int,
) -> dict[str, Any]:
    rate_numerators = {
        SUCCESS_METRIC: success_count,
        "episode/fall_rate": episode_count - success_count,
        "episode/survival_rate": success_count,
        "episode/undesired_contact_rate": min(1, episode_count),
        "episode/mean_length_steps": 100 * episode_count,
        "motion/mean_max_episode_forward_progress_m": 1.5 * episode_count,
    }
    step_metrics = {
        name: _ratio(0.01 * (index + 1) * contact_step_denominator, contact_step_denominator)
        for index, name in enumerate(STEP_METRICS)
    }
    step_metrics["terrain/undesired_contact_any"] = _ratio(
        contact_step_numerator,
        contact_step_denominator,
        maximum=1.0,
    )
    step_metrics["terrain/undesired_contact_body_count"] = _ratio(
        2.0 * contact_step_numerator,
        contact_step_denominator,
        maximum=2.0,
    )
    return {
        "episode_count": episode_count,
        "rates": {name: _ratio(rate_numerators[name], episode_count) for name in RATE_METRICS},
        "step_metrics": step_metrics,
    }


def _payload(
    variant: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    generator_checkpoint_path: Path,
    generator_checkpoint_sha256: str,
    *,
    evaluation_seed: int,
    success_count: int,
    contact_step_numerator: float,
    contact_step_denominator: int,
    fixed_level: int = 1,
    torch_deterministic: bool = False,
    num_envs: int = 64,
) -> dict[str, Any]:
    terrain_types = ("box", "stair")
    per_type_quota = 2
    total = len(terrain_types) * per_type_quota
    requested_by_type = dict.fromkeys(terrain_types, per_type_quota)
    episodes = [
        {"episode_index": index, "terrain_type": terrain_type}
        for index, terrain_type in enumerate(("box", "box", "stair", "stair"))
    ]
    box_success = min(success_count, per_type_quota)
    stair_success = max(0, success_count - per_type_quota)
    box_denominator = contact_step_denominator // 2
    stair_denominator = contact_step_denominator - box_denominator
    box_numerator = contact_step_numerator / 2.0
    stair_numerator = contact_step_numerator - box_numerator
    metrics_config = {
        "variant": variant,
        "scenario_label": "common-l1",
        "evaluation_seed": evaluation_seed,
        "fixed_terrain_level": fixed_level,
        "evaluation_phase_mode": "uniform",
        "reanchor_motion_xy_on_reset": True,
        "phase_horizon_steps": 500,
        "episode_count": total,
        "success_distance_m": 1.5,
        "deterministic_generator": True,
        "deterministic_per_env_sampling": True,
        "generator_sampling_seed": 0,
        "fall_root_height_m": 0.45,
        "fall_upright_cosine": 0.5,
        "body_origin_penetration_threshold_m": 0.02,
        "body_origin_correction_min_improvement_m": 0.01,
        "heading_speed_threshold_mps": 0.05,
    }
    return {
        "schema_version": 1,
        "metadata": {
            "variant": variant,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "generator_checkpoint_path": str(generator_checkpoint_path.resolve()),
            "generator_checkpoint_sha256": generator_checkpoint_sha256,
            "evaluation_seed": evaluation_seed,
            "fixed_terrain_level": fixed_level,
            "evaluation_phase_mode": "uniform",
            "reanchor_motion_xy_on_reset": True,
            "phase_horizon_steps": 500,
            "deterministic_generator": True,
            "deterministic_per_env_sampling": True,
            "generator_sampling_seed": 0,
            "metrics_config": metrics_config,
            "evaluation_config": {
                "training": {
                    "seed": evaluation_seed,
                    "torch_deterministic": torch_deterministic,
                    "num_envs": num_envs,
                },
                "simulator": {
                    "config": {
                        "sim": {
                            "fps": 200,
                            "control_decimation": 4,
                            "max_episode_length_s": 10.0,
                        }
                    }
                },
                "terrain": {
                    "terrain_term": {
                        "mesh_type": "trimesh",
                        "horizontal_scale": 0.1,
                        "vertical_scale": 0.005,
                        "num_rows": 10,
                        "num_cols": 20,
                        "spawn": {"randomize_tiles": False, "xy_offset_range": 0.0},
                        "curriculum_layout": {
                            "enabled": True,
                            "terrain_types": list(terrain_types),
                            "box_height_range": [0.05, 0.3],
                            "stair_height_range": [0.05, 0.35],
                            "hurdle_height_range": [0.05, 0.35],
                        },
                        "terrain_config": {"box": 0.5, "stair": 0.5},
                    }
                },
            },
        },
        "summary": {
            "requested_episode_count": total,
            "completed_episode_count": total,
            "complete": True,
            "requested_per_terrain_type": requested_by_type,
            "completed_per_terrain_type": requested_by_type,
            "success_distance_m": 1.5,
            "overall": _summary_group(
                total,
                success_count=success_count,
                contact_step_numerator=contact_step_numerator,
                contact_step_denominator=contact_step_denominator,
            ),
            "by_terrain_type": {
                "box": _summary_group(
                    per_type_quota,
                    success_count=box_success,
                    contact_step_numerator=box_numerator,
                    contact_step_denominator=box_denominator,
                ),
                "stair": _summary_group(
                    per_type_quota,
                    success_count=stair_success,
                    contact_step_numerator=stair_numerator,
                    contact_step_denominator=stair_denominator,
                ),
            },
        },
        "episodes": episodes,
    }


def _entry(
    tmp_path: Path,
    policy_id: str,
    training_seed: int,
    evaluation_seed: int,
    *,
    success_count: int = 2,
    contact_step_numerator: float = 2.0,
    contact_step_denominator: int = 20,
    fixed_level: int = 1,
    torch_deterministic: bool = False,
    num_envs: int = 64,
    checkpoint_capture_kind: str = "fair-501-logged-updates",
) -> dict[str, Any]:
    checkpoint = tmp_path / "checkpoints" / f"{policy_id}_{training_seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint:{policy_id}:{training_seed}".encode())
    checkpoint_sha = _sha256(checkpoint)
    generator_checkpoint = checkpoint.parent / "frozen-generator.pt"
    generator_checkpoint.write_bytes(b"shared frozen generator")
    generator_checkpoint_sha = _sha256(generator_checkpoint)
    variant = f"opaque-source-label-{policy_id}-{training_seed}-{evaluation_seed}"
    source = tmp_path / "evaluations" / f"{variant}.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps(
            _payload(
                variant,
                checkpoint,
                checkpoint_sha,
                generator_checkpoint,
                generator_checkpoint_sha,
                evaluation_seed=evaluation_seed,
                success_count=success_count,
                contact_step_numerator=contact_step_numerator,
                contact_step_denominator=contact_step_denominator,
                fixed_level=fixed_level,
                torch_deterministic=torch_deterministic,
                num_envs=num_envs,
            )
        ),
        encoding="utf-8",
    )
    return {
        "policy_id": policy_id,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "source_json": str(source.relative_to(tmp_path)),
        "source_json_sha256": _sha256(source),
        "checkpoint_path": str(checkpoint.relative_to(tmp_path)),
        "checkpoint_sha256": checkpoint_sha,
        "source_training_run_id": f"run-{policy_id}-{training_seed}",
        "checkpoint_capture_kind": checkpoint_capture_kind,
        "update_budget": 501,
    }


def _manifest(
    tmp_path: Path,
    entries: list[dict[str, Any]],
    *,
    contrasts: list[dict[str, Any]] | None = None,
) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata": {"protocol": "synthetic common L1"},
                "entries": entries,
                "contrasts": contrasts or [],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_two_stage_raw_pooling_weighted_steps_contrast_and_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    successes = {
        ("D_fair", 1, 10): 1,
        ("D_fair", 1, 11): 3,
        ("D_fair", 2, 10): 2,
        ("D_fair", 2, 11): 4,
        ("E_fair", 1, 10): 3,
        ("E_fair", 1, 11): 4,
        ("E_fair", 2, 10): 1,
        ("E_fair", 2, 11): 4,
    }
    contact_pairs = {
        ("D_fair", 1, 10): (1.0, 10),
        ("D_fair", 1, 11): (90.0, 100),
        ("D_fair", 2, 10): (2.0, 20),
        ("D_fair", 2, 11): (3.0, 30),
    }
    entries = []
    for identity, success_count in successes.items():
        numerator, denominator = contact_pairs.get(identity, (2.0, 20))
        entries.append(
            _entry(
                tmp_path,
                *identity,
                success_count=success_count,
                contact_step_numerator=numerator,
                contact_step_denominator=denominator,
            )
        )
    manifest = _manifest(
        tmp_path,
        entries,
        contrasts=[
            {
                "name": "E_fair-D_fair",
                "left_policy_id": "E_fair",
                "right_policy_id": "D_fair",
            }
        ],
    )

    report = build_training_seed_report(manifest)
    assert report["source_count"] == 8
    assert report["protocol_provenance"]["torch_deterministic"] is False
    assert report["protocol_provenance"]["num_envs"] == 64
    assert report["identity_contract"]["policy_id_authority"] == "manifest"
    assert report["pooling_contract"]["numerator_summation"] == "math.fsum"
    assert report["contrast_inference_contract"]["episode_level_confidence_interval"] is False

    by_policy_training = {
        (result["policy_id"], result["training_seed"]): result for result in report["policy_training_seed_results"]
    }
    d_seed_1 = by_policy_training[("D_fair", 1)]
    assert d_seed_1["overall_metrics"][SUCCESS_METRIC] == {
        "numerator": 4.0,
        "denominator": 8,
        "value": 0.5,
    }
    assert d_seed_1["overall_metrics"]["terrain/undesired_contact_any"] == {
        "numerator": 91.0,
        "denominator": 110,
        "value": 91 / 110,
        "max": 1.0,
    }
    assert d_seed_1["by_terrain_type"]["box"]["episode_count"] == 4
    assert d_seed_1["by_terrain_type"]["box"]["metrics"][SUCCESS_METRIC]["denominator"] == 4

    by_policy = {result["policy_id"]: result for result in report["policy_results"]}
    assert by_policy["D_fair"]["overall_metrics"][SUCCESS_METRIC] == {
        "numerator": 10.0,
        "denominator": 16,
        "value": 0.625,
    }
    assert by_policy["E_fair"]["overall_metrics"][SUCCESS_METRIC]["numerator"] == 12.0
    assert by_policy["D_fair"]["overall_metrics"]["terrain/undesired_contact_any"] == {
        "numerator": 96.0,
        "denominator": 160,
        "value": 0.6,
        "max": 1.0,
    }

    contrast = report["contrasts"][0]
    assert [item["delta"] for item in contrast["training_seed_deltas"]] == [0.375, -0.125]
    assert [item["sign"] for item in contrast["training_seed_deltas"]] == ["positive", "negative"]
    stats = contrast["descriptive_statistics"]
    assert stats["minimum"] == -0.125
    assert stats["maximum"] == 0.375
    assert stats["range"] == 0.5
    assert stats["mean"] == 0.125
    assert stats["sample_standard_deviation"] == pytest.approx(math.sqrt(0.125))
    assert stats["sign_counts"] == {"positive": 1, "zero": 0, "negative": 1}
    assert contrast["raw_pooled_success_rate_difference"] == 0.125
    assert all(source["policy_id"] in {"D_fair", "E_fair"} for source in report["sources"])
    assert all(source["source_evaluator_variant"].startswith("opaque-source-label") for source in report["sources"])

    outputs = write_training_seed_report(report, tmp_path / "direct_report")
    saved = json.loads(outputs["json"].read_text())
    assert saved["policy_results"][0]["overall_metrics"][SUCCESS_METRIC]["denominator"] == 16
    with outputs["csv"].open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    weighted_row = next(
        row
        for row in rows
        if row["record_type"] == "pooled_metric"
        and row["aggregation_level"] == "policy_training_seed"
        and row["policy_id"] == "D_fair"
        and row["training_seed"] == "1"
        and row["scope"] == "overall"
        and row["metric"] == "terrain/undesired_contact_any"
    )
    assert weighted_row["numerator"] == "91.0"
    assert weighted_row["denominator"] == "110"
    contrast_row = next(row for row in rows if row["record_type"] == "contrast_summary")
    assert float(contrast_row["sample_standard_deviation"]) == pytest.approx(math.sqrt(0.125))
    source_row = next(row for row in rows if row["record_type"] == "source_provenance")
    assert len(source_row["source_json_sha256"]) == 64
    assert len(source_row["checkpoint_sha256"]) == 64
    assert json.loads(source_row["protocol_json"])["torch_deterministic"] is False
    assert source_row["num_envs"] == "64"
    assert json.loads(source_row["protocol_json"])["num_envs"] == 64
    protocol_row = next(row for row in rows if row["record_type"] == "protocol_provenance")
    assert json.loads(protocol_row["protocol_json"])["fixed_terrain_level"] == 1
    markdown = outputs["markdown"].read_text()
    assert "not episode-level confidence intervals" in markdown
    assert "Effective evaluation `num_envs`: `64`" in markdown
    assert "E_fair - D_fair" in markdown
    assert "opaque-source-label" in markdown
    assert source_row["source_json_sha256"] in markdown

    main(["--manifest", str(manifest), "--output-prefix", str(tmp_path / "cli_report")])
    assert "json:" in capsys.readouterr().out
    assert (tmp_path / "cli_report.json").is_file()


@pytest.mark.parametrize("failure", ["duplicate_identity", "source_sha", "checkpoint_sha", "evaluation_seed"])
def test_manifest_identity_and_hash_validation(tmp_path: Path, failure: str) -> None:
    entry = _entry(tmp_path, "D_fair", 42, 42)
    entries = [entry]
    expected = ""
    if failure == "duplicate_identity":
        entries.append(dict(entry))
        expected = "Duplicate .*identity"
    elif failure == "source_sha":
        entry["source_json_sha256"] = "0" * 64
        expected = "source JSON SHA-256 mismatch"
    elif failure == "checkpoint_sha":
        entry["checkpoint_sha256"] = "0" * 64
        expected = "checkpoint SHA-256 mismatch"
    else:
        entry["evaluation_seed"] = 43
        expected = "evaluation_seed .* disagrees"
    manifest = _manifest(tmp_path, entries)
    with pytest.raises(TerrainReportError, match=expected):
        build_training_seed_report(manifest)


@pytest.mark.parametrize(
    ("second_kwargs", "expected"),
    [
        ({"fixed_level": 2}, "fixed_terrain_level"),
        ({"torch_deterministic": True}, "torch_deterministic"),
        ({"num_envs": 32}, "num_envs"),
    ],
)
def test_only_evaluation_seed_may_vary_in_protocol(
    tmp_path: Path,
    second_kwargs: dict[str, Any],
    expected: str,
) -> None:
    entries = [
        _entry(tmp_path, "D_fair", 42, 42),
        _entry(tmp_path, "D_fair", 42, 43, **second_kwargs),
    ]
    manifest = _manifest(tmp_path, entries)
    with pytest.raises(TerrainReportError, match=expected):
        build_training_seed_report(manifest)


def test_strict_evaluator_loader_rejects_incomplete_quota(tmp_path: Path) -> None:
    entry = _entry(tmp_path, "D_fair", 42, 42)
    source = tmp_path / entry["source_json"]
    payload = json.loads(source.read_text())
    payload["summary"]["complete"] = False
    source.write_text(json.dumps(payload), encoding="utf-8")
    entry["source_json_sha256"] = _sha256(source)
    manifest = _manifest(tmp_path, [entry])

    with pytest.raises(TerrainReportError, match="incomplete episode quota"):
        build_training_seed_report(manifest)


def test_contrast_requires_paired_training_and_evaluation_seeds(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, "D_fair", 42, 42),
        _entry(tmp_path, "D_fair", 42, 43),
        _entry(tmp_path, "E_fair", 42, 42),
    ]
    manifest = _manifest(
        tmp_path,
        entries,
        contrasts=[
            {
                "name": "E_fair-D_fair",
                "left_policy_id": "E_fair",
                "right_policy_id": "D_fair",
            }
        ],
    )

    with pytest.raises(TerrainReportError, match="unpaired evaluation seeds"):
        build_training_seed_report(manifest)


def test_policy_requires_rectangular_training_evaluation_seed_grid(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, "D_fair", 42, 42),
        _entry(tmp_path, "D_fair", 42, 43),
        _entry(tmp_path, "D_fair", 43, 42),
    ]
    manifest = _manifest(tmp_path, entries)

    with pytest.raises(TerrainReportError, match="incomplete evaluation-seed grid"):
        build_training_seed_report(manifest)


@pytest.mark.parametrize("duplicate", ["checkpoint_sha256", "source_training_run_id"])
def test_training_seed_replicates_must_have_independent_checkpoint_and_run(
    tmp_path: Path,
    duplicate: str,
) -> None:
    first = _entry(tmp_path, "D_fair", 42, 42)
    second = _entry(tmp_path, "D_fair", 43, 42)
    if duplicate == "source_training_run_id":
        second["source_training_run_id"] = first["source_training_run_id"]
    else:
        source = tmp_path / second["source_json"]
        payload = json.loads(source.read_text())
        first_checkpoint = (tmp_path / first["checkpoint_path"]).resolve()
        payload["metadata"]["checkpoint_path"] = str(first_checkpoint)
        payload["metadata"]["checkpoint_sha256"] = first["checkpoint_sha256"]
        source.write_text(json.dumps(payload), encoding="utf-8")
        second["source_json_sha256"] = _sha256(source)
        second["checkpoint_path"] = first["checkpoint_path"]
        second["checkpoint_sha256"] = first["checkpoint_sha256"]
    manifest = _manifest(tmp_path, [first, second])

    with pytest.raises(TerrainReportError, match=f"reuses {duplicate}.*pseudoreplication"):
        build_training_seed_report(manifest)


def test_generator_checkpoint_must_be_fixed_within_policy(tmp_path: Path) -> None:
    first = _entry(tmp_path, "D_fair", 42, 42)
    second = _entry(tmp_path, "D_fair", 42, 43)
    source = tmp_path / second["source_json"]
    payload = json.loads(source.read_text())
    payload["metadata"]["generator_checkpoint_sha256"] = "c" * 64
    source.write_text(json.dumps(payload), encoding="utf-8")
    second["source_json_sha256"] = _sha256(source)
    manifest = _manifest(tmp_path, [first, second])

    with pytest.raises(TerrainReportError, match="generator checkpoint SHA-256 mismatch"):
        build_training_seed_report(manifest)


@pytest.mark.parametrize("missing_field", ["generator_checkpoint_path", "generator_checkpoint_sha256"])
def test_generator_checkpoint_path_and_sha_must_form_a_complete_pair(
    tmp_path: Path,
    missing_field: str,
) -> None:
    entry = _entry(tmp_path, "D_fair", 42, 42)
    source = tmp_path / entry["source_json"]
    payload = json.loads(source.read_text())
    payload["metadata"].pop(missing_field)
    source.write_text(json.dumps(payload), encoding="utf-8")
    entry["source_json_sha256"] = _sha256(source)
    manifest = _manifest(tmp_path, [entry])

    with pytest.raises(TerrainReportError, match="path and SHA-256 must either both be present"):
        build_training_seed_report(manifest)


def test_episode_quota_is_part_of_common_protocol(tmp_path: Path) -> None:
    first = _entry(tmp_path, "D_fair", 42, 42)
    second = _entry(tmp_path, "D_fair", 42, 43)
    source = tmp_path / second["source_json"]
    payload = json.loads(source.read_text())
    quota = {"box": 3, "stair": 1}
    payload["summary"]["requested_per_terrain_type"] = quota
    payload["summary"]["completed_per_terrain_type"] = quota
    payload["episodes"] = [
        {"episode_index": index, "terrain_type": terrain_type}
        for index, terrain_type in enumerate(("box", "box", "box", "stair"))
    ]
    payload["summary"]["by_terrain_type"] = {
        "box": _summary_group(
            3,
            success_count=2,
            contact_step_numerator=1.5,
            contact_step_denominator=15,
        ),
        "stair": _summary_group(
            1,
            success_count=0,
            contact_step_numerator=0.5,
            contact_step_denominator=5,
        ),
    }
    source.write_text(json.dumps(payload), encoding="utf-8")
    second["source_json_sha256"] = _sha256(source)
    manifest = _manifest(tmp_path, [first, second])

    with pytest.raises(TerrainReportError, match="episode_quotas"):
        build_training_seed_report(manifest)


def test_capture_kind_may_record_backfill_vs_direct_across_training_seeds(tmp_path: Path) -> None:
    entries = [
        _entry(
            tmp_path,
            "D_fair",
            training_seed,
            evaluation_seed,
            checkpoint_capture_kind=capture_kind,
        )
        for training_seed, capture_kind in (
            (42, "post-hoc-wandb-backfill"),
            (43, "direct-online-wandb"),
        )
        for evaluation_seed in (42, 43)
    ]
    report = build_training_seed_report(_manifest(tmp_path, entries))

    policy = report["policy_results"][0]
    expected = {"42": "post-hoc-wandb-backfill", "43": "direct-online-wandb"}
    assert policy["checkpoint_capture_kind"] == {"varies": expected}
    assert policy["checkpoint_capture_kinds_by_training_seed"] == expected
    assert {
        checkpoint["training_seed"]: checkpoint["checkpoint_capture_kind"] for checkpoint in policy["checkpoints"]
    } == {42: "post-hoc-wandb-backfill", 43: "direct-online-wandb"}
    assert report["provenance_contract"]["checkpoint_capture_kind_may_vary_across_training_seeds"] is True

    outputs = write_training_seed_report(report, tmp_path / "capture_report")
    with outputs["csv"].open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    policy_metric = next(
        row
        for row in rows
        if row["record_type"] == "pooled_metric"
        and row["aggregation_level"] == "policy"
        and row["metric"] == SUCCESS_METRIC
        and row["scope"] == "overall"
    )
    assert json.loads(policy_metric["checkpoint_capture_kinds_json"]) == expected
    markdown = outputs["markdown"].read_text()
    assert "post-hoc-wandb-backfill" in markdown
    assert "direct-online-wandb" in markdown


def test_capture_kind_must_be_fixed_across_evaluations_of_one_training_seed(tmp_path: Path) -> None:
    entries = [
        _entry(
            tmp_path,
            "D_fair",
            42,
            42,
            checkpoint_capture_kind="post-hoc-wandb-backfill",
        ),
        _entry(
            tmp_path,
            "D_fair",
            42,
            43,
            checkpoint_capture_kind="direct-online-wandb",
        ),
    ]

    with pytest.raises(TerrainReportError, match="checkpoint_capture_kind varies across evaluations"):
        build_training_seed_report(_manifest(tmp_path, entries))
