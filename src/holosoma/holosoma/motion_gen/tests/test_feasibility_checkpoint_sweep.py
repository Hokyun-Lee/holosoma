from __future__ import annotations

import csv
import json

import pytest

from holosoma.motion_gen.scripts.sweep_feasibility_checkpoints import (
    Args,
    build_aggregate,
    extract_production_metrics,
    rank_successful_records,
    write_aggregate_files,
)


def _comparison(
    *,
    gate_pass: bool = False,
    depth: float = 0.1,
    rate: float = 0.2,
    joint: float = 0.0,
    fk: float = 0.03,
    root_mean: float = 0.4,
    root_max: float = 0.7,
    root_final: float = 0.6,
    tolerance: float = 0.005,
    production_match: bool = True,
) -> dict:
    generated = {
        "kinematic_gate_pass": gate_pass,
        "checks": {
            "finite_qpos": True,
            "environment_penetration_within_tolerance": gate_pass,
        },
        "environment_collision": {
            "surface_tolerance_m": tolerance,
            "max_depth_m": depth,
            "over_tolerance_frame_rate": rate,
        },
        "joint_limits": {"max_violation_rad": joint},
        "fk_consistency": {"body_origin_error_m": {"max": fk}},
    }
    return {
        "production_contract": {
            "full_horizon_stride": production_match,
            "current_command_stride_match": production_match,
        },
        "variants": {
            "generated_feedback__keep_current": {
                "aggregate_source_alignment": {
                    "root_position_error_m_mean": root_mean,
                    "root_position_error_m_max": root_max,
                    "root_position_error_m_final": root_final,
                },
                "mujoco_feasibility": {
                    "verdict": {"kinematic_gate_pass": gate_pass},
                    "generated": generated,
                },
            }
        },
    }


def _success(checkpoint: str, step: int, **metrics) -> dict:
    extracted = extract_production_metrics(_comparison(**metrics))
    return {
        "sweep_index": step,
        "status": "success",
        "checkpoint": checkpoint,
        "checkpoint_step": step,
        "comparison_json": f"{checkpoint}.json",
        "duration_s": 1.0,
        "reused": False,
        "returncode": 0,
        "metrics": extracted,
    }


def test_extract_production_metrics_preserves_gate_and_requested_errors() -> None:
    metrics = extract_production_metrics(
        _comparison(
            gate_pass=True,
            depth=0.012,
            rate=0.125,
            joint=0.002,
            fk=0.021,
            root_mean=0.3,
            root_max=0.5,
            root_final=0.45,
        )
    )
    assert metrics == {
        "strict_kinematic_gate_pass": True,
        "production_contract_match": True,
        "penetration_tolerance_m": 0.005,
        "max_environment_penetration_m": 0.012,
        "penetration_over_5mm_frame_rate": 0.125,
        "joint_limit_max_violation_rad": 0.002,
        "fk_body_error_max_m": 0.021,
        "root_position_error_m_mean": 0.3,
        "root_position_error_m_max": 0.5,
        "root_position_error_m_final": 0.45,
        "kinematic_checks": {
            "finite_qpos": True,
            "environment_penetration_within_tolerance": True,
        },
    }


def test_extract_rejects_nonproduction_or_non_5mm_contract() -> None:
    with pytest.raises(ValueError, match="production stride"):
        extract_production_metrics(_comparison(production_match=False))
    with pytest.raises(ValueError, match="5 mm"):
        extract_production_metrics(_comparison(tolerance=0.01))
    with pytest.raises(ValueError, match="finite and non-negative"):
        extract_production_metrics(_comparison(depth=float("nan")))
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        extract_production_metrics(_comparison(rate=1.01))


def test_ranking_prioritizes_strict_pass_then_production_errors() -> None:
    lower_depth_failure = _success("failure.pt", 10, gate_pass=False, depth=0.001, rate=0.0)
    larger_pass = _success("pass_large.pt", 20, gate_pass=True, depth=0.02, rate=0.5)
    smaller_pass = _success("pass_small.pt", 30, gate_pass=True, depth=0.01, rate=0.9)
    error = {"status": "error", "checkpoint": "error.pt", "sweep_index": 40}
    ranked = rank_successful_records([lower_depth_failure, larger_pass, smaller_pass, error])
    assert [row["checkpoint"] for row in ranked] == ["pass_small.pt", "pass_large.pt", "failure.pt"]
    assert [row["rank"] for row in ranked] == [1, 2, 3]


def test_aggregate_json_csv_keep_error_checkpoint(tmp_path) -> None:
    success = _success("ok.pt", 25, gate_pass=False, depth=0.03)
    error = {
        "sweep_index": 1,
        "status": "error",
        "checkpoint": "bad.pt",
        "checkpoint_step": None,
        "comparison_json": "bad/comparison.json",
        "duration_s": 0.1,
        "reused": False,
        "returncode": 1,
        "error": "RuntimeError: broken checkpoint",
    }
    report = build_aggregate(Args(), [error, success])
    assert report["summary"] == {"requested": 2, "succeeded": 1, "failed": 1, "strict_passed": 0}
    assert report["selected"]["checkpoint"] == "ok.pt"

    json_path, csv_path = write_aggregate_files(report, tmp_path)
    assert json.loads(json_path.read_text())["checkpoints"][0]["checkpoint"] == "bad.pt"
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["checkpoint"] for row in rows] == ["bad.pt", "ok.pt"]
    assert rows[0]["error"] == "RuntimeError: broken checkpoint"
    assert rows[1]["max_environment_penetration_m"] == "0.03"
    assert rows[1]["rank"] == "1"
