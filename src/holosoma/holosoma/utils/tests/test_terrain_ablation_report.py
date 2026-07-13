from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from holosoma.aggregate_terrain_evaluations import (
    TerrainReportError,
    build_ablation_report,
    main,
    write_ablation_report,
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


def _ratio(numerator: float, denominator: int, *, maximum: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
    }
    if maximum is not None:
        result["max"] = maximum
    return result


def _summary_group(episode_count: int, *, success_count: int) -> dict[str, Any]:
    rate_numerators = {
        "episode/success_rate": success_count,
        "episode/fall_rate": episode_count - success_count,
        "episode/survival_rate": success_count,
        "episode/undesired_contact_rate": 1,
        "episode/mean_length_steps": 100 * episode_count,
        "motion/mean_max_episode_forward_progress_m": 1.75 * episode_count,
    }
    step_count = 10 * episode_count
    return {
        "episode_count": episode_count,
        "rates": {name: _ratio(rate_numerators[name], episode_count) for name in RATE_METRICS},
        "step_metrics": {
            name: _ratio(0.1 * (index + 1) * step_count, step_count, maximum=float(index + 1))
            for index, name in enumerate(STEP_METRICS)
        },
    }


def _payload(
    variant: str,
    *,
    evaluation_seed: int = 42,
    fixed_level: int = 0,
    success_distance_m: float = 1.5,
    terrain_types: tuple[str, ...] = ("box", "stair"),
    per_type_quota: int = 2,
    success_count: int = 3,
    scenario_label: str = "",
    evaluation_phase_mode: str = "uniform",
    deterministic_per_env_sampling: bool = True,
) -> dict[str, Any]:
    requested_by_type = dict.fromkeys(terrain_types, per_type_quota)
    total = len(terrain_types) * per_type_quota
    success_count = min(success_count, total)
    episodes = []
    episode_index = 0
    for terrain_type in terrain_types:
        for _ in range(per_type_quota):
            episodes.append({"episode_index": episode_index, "terrain_type": terrain_type})
            episode_index += 1
    per_type_success = min(per_type_quota, success_count)
    return {
        "schema_version": 1,
        "metadata": {
            "variant": variant,
            "checkpoint_path": f"/checkpoints/{variant}.pt",
            "checkpoint_sha256": "a" * 64,
            "generator_checkpoint_path": "/checkpoints/generator.pt",
            "generator_checkpoint_sha256": "b" * 64,
            "evaluation_seed": evaluation_seed,
            "fixed_terrain_level": fixed_level,
            "evaluation_phase_mode": evaluation_phase_mode,
            "deterministic_generator": True,
            "deterministic_per_env_sampling": deterministic_per_env_sampling,
            "generator_sampling_seed": 7,
            "metrics_config": {
                "variant": variant,
                "scenario_label": scenario_label,
                "evaluation_seed": evaluation_seed,
                "fixed_terrain_level": fixed_level,
                "evaluation_phase_mode": evaluation_phase_mode,
                "episode_count": total,
                "success_distance_m": success_distance_m,
                "deterministic_generator": True,
                "deterministic_per_env_sampling": deterministic_per_env_sampling,
                "generator_sampling_seed": 7,
                "fall_root_height_m": 0.45,
                "fall_upright_cosine": 0.5,
                "body_origin_penetration_threshold_m": 0.02,
                "body_origin_correction_min_improvement_m": 0.01,
                "heading_speed_threshold_mps": 0.05,
            },
            "evaluation_config": {
                "training": {"seed": evaluation_seed},
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
                        "terrain_config": {terrain_type: 1.0 / len(terrain_types) for terrain_type in terrain_types},
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
            "success_distance_m": success_distance_m,
            "overall": _summary_group(total, success_count=success_count),
            "by_terrain_type": {
                terrain_type: _summary_group(per_type_quota, success_count=per_type_success)
                for terrain_type in terrain_types
            },
        },
        "episodes": episodes,
    }


def _write_payload(tmp_path: Path, variant: str, **kwargs: Any) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{variant}.json"
    path.write_text(json.dumps(_payload(variant, **kwargs)), encoding="utf-8")
    return path


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def test_a_to_f_report_preserves_raw_pairs_hashes_and_writes_all_formats(tmp_path: Path) -> None:
    paths = [_write_payload(tmp_path, variant) for variant in "ABCDEF"]
    report = build_ablation_report({"ablation": paths})

    assert report["source_count"] == 6
    group = report["groups"][0]
    assert [result["variant"] for result in group["results"]] == list("ABCDEF")
    first = group["results"][0]
    assert first["checkpoint_sha256"] == "a" * 64
    assert first["generator_checkpoint_sha256"] == "b" * 64
    assert first["overall_metrics"]["episode/success_rate"] == {
        "numerator": 3,
        "denominator": 4,
        "value": 0.75,
    }
    assert first["by_terrain_type"]["box"]["motion/heading_error_rad"]["denominator"] == 20

    outputs = write_ablation_report(report, tmp_path / "report")
    saved = json.loads(outputs["json"].read_text())
    assert saved["groups"][0]["results"][0]["overall_metrics"]["episode/success_rate"]["numerator"] == 3
    with outputs["csv"].open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    success = next(
        row
        for row in rows
        if row["variant"] == "A" and row["scope"] == "overall" and row["metric"] == "episode/success_rate"
    )
    assert success["numerator"] == "3"
    assert success["denominator"] == "4"
    assert success["checkpoint_sha256"] == "a" * 64
    markdown = outputs["markdown"].read_text()
    assert "75.0% (3/4)" in markdown
    assert "Robot origin proxy" in markdown


@pytest.mark.parametrize(
    ("payload_update", "expected_field"),
    [
        ({"evaluation_seed": 43}, "evaluation_seed"),
        ({"fixed_level": 4}, "fixed_terrain_level"),
        ({"success_distance_m": 2.0}, "success_distance_m"),
        ({"per_type_quota": 3}, "episode_quotas"),
        ({"terrain_types": ("rough",)}, "episode_quotas"),
    ],
)
def test_protocol_mismatch_fails_with_field_name(
    tmp_path: Path,
    payload_update: dict[str, Any],
    expected_field: str,
) -> None:
    first = _write_payload(tmp_path, "A")
    second = _write_payload(tmp_path, "B", **payload_update)
    with pytest.raises(TerrainReportError, match=expected_field):
        build_ablation_report({"ablation": [first, second]})


@pytest.mark.parametrize(
    ("updates", "expected_field"),
    [
        (
            [
                (("metadata", "evaluation_phase_mode"), "zero"),
                (("metadata", "metrics_config", "evaluation_phase_mode"), "zero"),
            ],
            "evaluation_phase_mode",
        ),
        (
            [
                (("metadata", "deterministic_generator"), False),
                (("metadata", "metrics_config", "deterministic_generator"), False),
            ],
            "deterministic_generator",
        ),
        (
            [
                (("metadata", "deterministic_per_env_sampling"), False),
                (("metadata", "metrics_config", "deterministic_per_env_sampling"), False),
            ],
            "deterministic_per_env_sampling",
        ),
        (
            [
                (("metadata", "generator_sampling_seed"), 8),
                (("metadata", "metrics_config", "generator_sampling_seed"), 8),
            ],
            "generator_sampling_seed",
        ),
        ([(("metadata", "metrics_config", "fall_root_height_m"), 0.4)], "fall_root_height_m"),
        ([(("metadata", "metrics_config", "fall_upright_cosine"), 0.6)], "fall_upright_cosine"),
        (
            [(("metadata", "metrics_config", "body_origin_penetration_threshold_m"), 0.03)],
            "body_origin_penetration_threshold_m",
        ),
        (
            [(("metadata", "metrics_config", "body_origin_correction_min_improvement_m"), 0.02)],
            "body_origin_correction_min_improvement_m",
        ),
        ([(("metadata", "metrics_config", "heading_speed_threshold_mps"), 0.1)], "heading_speed_threshold_mps"),
        (
            [(("metadata", "evaluation_config", "simulator", "config", "sim", "max_episode_length_s"), 12.0)],
            "max_episode_length_s",
        ),
        (
            [(("metadata", "evaluation_config", "simulator", "config", "sim", "control_decimation"), 8)],
            "control_dt_s",
        ),
        (
            [
                (
                    (
                        "metadata",
                        "evaluation_config",
                        "terrain",
                        "terrain_term",
                        "curriculum_layout",
                        "box_height_range",
                    ),
                    [0.6, 0.6],
                )
            ],
            "terrain_geometry_layout",
        ),
    ],
)
def test_extended_protocol_fields_and_geometry_fingerprint_cannot_mix_silently(
    tmp_path: Path,
    updates: list[tuple[tuple[str, ...], Any]],
    expected_field: str,
) -> None:
    first = _write_payload(tmp_path, "A")
    payload = _payload("B")
    for path, value in updates:
        _set_nested(payload, path, value)
    second = tmp_path / "B.json"
    second.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TerrainReportError, match=expected_field):
        build_ablation_report({"ablation": [first, second]})


def test_explicit_groups_support_final_height_and_unseen_protocols(tmp_path: Path) -> None:
    final = _write_payload(tmp_path, "D-final")
    height = _write_payload(tmp_path, "D-60cm", success_distance_m=2.0)
    unseen = _write_payload(
        tmp_path,
        "D-unseen-rough",
        terrain_types=("rough",),
        fixed_level=9,
        scenario_label="rough",
    )
    report = build_ablation_report(
        {
            "final": [final],
            "height-60cm": [height],
            "unseen": [unseen],
        }
    )
    assert [group["name"] for group in report["groups"]] == ["final", "height-60cm", "unseen"]
    assert report["groups"][2]["results"][0]["scenario_label"] == "rough"


def test_protocol_override_is_field_specific_and_auditable(tmp_path: Path) -> None:
    first = _write_payload(tmp_path, "D-30cm", fixed_level=0)
    second = _write_payload(tmp_path, "D-60cm", fixed_level=9)
    report = build_ablation_report(
        {"height-sweep": [first, second]},
        allowed_protocol_mismatches=("fixed_terrain_level",),
    )
    group = report["groups"][0]
    assert group["allowed_protocol_mismatches"] == ["fixed_terrain_level"]
    assert group["protocol"]["fixed_terrain_level"] == {
        "varies": [
            {"variant": "D-30cm", "value": 0},
            {"variant": "D-60cm", "value": 9},
        ]
    }
    with pytest.raises(TerrainReportError, match="Unknown allowed protocol mismatch"):
        build_ablation_report({"height-sweep": [first]}, allowed_protocol_mismatches=("everything",))


@pytest.mark.parametrize(
    "mutation",
    [
        "complete_false",
        "short_episodes",
        "per_type_quota",
    ],
)
def test_incomplete_quota_always_fails(tmp_path: Path, mutation: str) -> None:
    payload = _payload("D")
    if mutation == "complete_false":
        payload["summary"]["complete"] = False
    elif mutation == "short_episodes":
        payload["episodes"].pop()
    else:
        payload["summary"]["completed_per_terrain_type"]["box"] = 1
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TerrainReportError, match=r"incomplete|episodes count"):
        build_ablation_report({"ablation": [path]})


@pytest.mark.parametrize("mutation", ["schema", "missing_metric", "bad_ratio", "bad_hash"])
def test_schema_and_raw_metric_corruption_fail(tmp_path: Path, mutation: str) -> None:
    payload = _payload("D")
    if mutation == "schema":
        payload["schema_version"] = 2
    elif mutation == "missing_metric":
        del payload["summary"]["overall"]["step_metrics"]["motion/error_body_pos"]
    elif mutation == "bad_ratio":
        payload["summary"]["overall"]["rates"]["episode/success_rate"]["value"] = 0.1
    else:
        payload["metadata"]["checkpoint_sha256"] = "short"
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TerrainReportError):
        build_ablation_report({"ablation": [path]})


def test_duplicate_variant_fails_even_across_protocol_groups(tmp_path: Path) -> None:
    first = _write_payload(tmp_path / "one", "D")
    second = _write_payload(tmp_path / "two", "D", fixed_level=9)
    with pytest.raises(TerrainReportError, match="Duplicate variant"):
        build_ablation_report({"final": [first], "unseen": [second]})


def test_cli_repeated_group_syntax_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    final = _write_payload(tmp_path, "D-final")
    unseen = _write_payload(tmp_path, "D-unseen", terrain_types=("rough",), fixed_level=9)
    prefix = tmp_path / "cli_report"
    main(
        [
            "--group",
            f"final={final}",
            "--group",
            f"unseen={unseen}",
            "--output-prefix",
            str(prefix),
        ]
    )
    assert prefix.with_suffix(".json").is_file()
    assert "markdown:" in capsys.readouterr().out
