"""Evaluate and rank motion-generator checkpoints with one feasibility contract.

Each checkpoint is evaluated by :mod:`evaluate_rollout_modes` with per-checkpoint
W&B logging disabled.  The production-like ``generated_feedback__keep_current``
variant is extracted from each comparison and ranked with hard kinematic-gate
success first, followed by penetration, joint/FK, and root-drift errors.

The default glob targets the from-scratch terrain-feasibility run::

    python -m holosoma.motion_gen.scripts.sweep_feasibility_checkpoints \
      --wandb-mode online

The sweep is resumable: a valid comparison already associated with the exact
checkpoint path is reused.  A failed checkpoint is recorded in the aggregate
JSON/CSV and does not prevent later checkpoints from being evaluated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tyro

PRODUCTION_VARIANT = "generated_feedback__keep_current"
PENETRATION_TOLERANCE_M = 0.005


@dataclass(frozen=True)
class Args:
    checkpoint_glob: str = "logs/motion_gen/terrain_feasibility_4090/checkpoints/ckpt_*.pt"
    """Glob evaluated in numeric checkpoint order."""
    checkpoints: tuple[str, ...] = ()
    """Additional explicit checkpoint paths."""
    include_final: bool = True
    """Also include final.pt next to the glob matches when it exists."""
    clip: str = "omni_climb_09_z1_0"
    terrain_urdf: str = "data/motion_gen/raw/omniretarget_terrain/climb_09/multi_boxes_z_scale_1.0.urdf"
    start: int = 100
    seed: int = 123
    device: str = "cuda:0"
    num_steps: int = 2
    num_cycles: int = 17
    replan_stride: int = 25
    guidance_scale: float | None = None
    output_dir: str = "logs/motion_gen/terrain_feasibility/scratch_checkpoint_sweep_production17x25_2step"
    evaluation_python: str | None = None
    """Python containing torch and MuJoCo; None uses the current interpreter."""
    evaluator_script: str = (
        "src/holosoma_retargeting/holosoma_retargeting/data_conversion/evaluate_motion_feasibility_mj.py"
    )
    robot_xml: str = "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
    reuse_existing: bool = True
    timeout_s: float | None = None
    allow_nonproduction_contract: bool = False
    """Allow ranking when the evaluator says the stride is not production parity."""
    fail_on_checkpoint_error: bool = False
    require_selected_pass: bool = False
    wandb_mode: str = "disabled"
    """Aggregate run only: disabled, offline, or online."""
    wandb_entity: str | None = None
    wandb_project: str = "HoloSomaMotionGenerator"
    wandb_group: str = "terrain_feasibility_checkpoint_sweep"
    wandb_name: str | None = None


_CSV_FIELDS = (
    "rank",
    "sweep_index",
    "status",
    "checkpoint",
    "checkpoint_step",
    "strict_kinematic_gate_pass",
    "max_environment_penetration_m",
    "penetration_over_5mm_frame_rate",
    "nonfoot_penetration_over_5mm_frame_rate",
    "reference_max_environment_penetration_m",
    "reference_penetration_over_5mm_frame_rate",
    "reference_nonfoot_penetration_over_5mm_frame_rate",
    "penetration_over_5mm_frame_rate_delta_vs_reference",
    "nonfoot_penetration_over_5mm_frame_rate_delta_vs_reference",
    "worst_robot_body",
    "worst_robot_body_is_foot",
    "worst_robot_body_max_penetration_m",
    "worst_robot_body_contact_frame_rate",
    "worst_robot_body_over_5mm_frame_rate",
    "worst_robot_body_deep_penetration_frame_rate",
    "reference_worst_robot_body",
    "reference_worst_robot_body_is_foot",
    "reference_worst_robot_body_max_penetration_m",
    "reference_worst_robot_body_contact_frame_rate",
    "reference_worst_robot_body_over_5mm_frame_rate",
    "reference_worst_robot_body_deep_penetration_frame_rate",
    "joint_limit_max_violation_rad",
    "fk_body_error_max_m",
    "root_position_error_m_mean",
    "root_position_error_m_max",
    "root_position_error_m_final",
    "comparison_json",
    "duration_s",
    "reused",
    "returncode",
    "error",
)


def _repo_root() -> Path:
    source = Path(__file__).resolve()
    for candidate in source.parents:
        if (candidate / "src/holosoma").is_dir() and (candidate / "src/holosoma_retargeting").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root above {source}")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (_repo_root() / candidate).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    numbers = re.findall(r"\d+", path.stem)
    if numbers:
        return (0, int(numbers[-1]), str(path))
    if path.name == "final.pt":
        return (1, sys.maxsize, str(path))
    return (2, sys.maxsize, str(path))


def resolve_checkpoints(args: Args) -> list[Path]:
    """Resolve, de-duplicate, and naturally sort requested checkpoints."""
    pattern = _resolve(args.checkpoint_glob)
    paths = [match.resolve() for match in pattern.parent.glob(pattern.name)]
    paths.extend(_resolve(path) for path in args.checkpoints)
    if args.include_final:
        parents = {path.parent for path in paths}
        glob_parent = pattern.parent
        parents.add(glob_parent)
        paths.extend(parent / "final.pt" for parent in parents if (parent / "final.pt").is_file())
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Explicit checkpoint does not exist: " + ", ".join(map(str, missing)))
    unique = {str(path): path for path in paths}
    return sorted(unique.values(), key=_checkpoint_sort_key)


def extract_production_metrics(report: dict[str, Any], *, allow_nonproduction_contract: bool = False) -> dict[str, Any]:
    """Extract the strict production-like metrics from one comparison report."""
    contract = report["production_contract"]
    production_match = bool(contract["full_horizon_stride"] and contract["current_command_stride_match"])
    if not production_match and not allow_nonproduction_contract:
        raise ValueError("comparison.json does not use the 25-frame full-horizon production stride")

    variant = report["variants"][PRODUCTION_VARIANT]
    feasibility = variant["mujoco_feasibility"]
    if feasibility is None:
        raise ValueError("comparison.json has no MuJoCo feasibility result")
    generated = feasibility["generated"]
    reference = feasibility.get("reference")
    if reference is None:
        raise ValueError("comparison.json has no aligned reference feasibility result")
    collision = generated["environment_collision"]
    reference_collision = reference["environment_collision"]
    tolerance = float(collision["surface_tolerance_m"])
    if not math.isclose(tolerance, PENETRATION_TOLERANCE_M, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"Expected a 5 mm penetration tolerance, found {tolerance:.9g} m")
    fk = generated["fk_consistency"]
    if fk is None:
        raise ValueError("comparison.json has no raw-motion FK consistency result")
    alignment = variant["aggregate_source_alignment"]
    strict_pass = bool(feasibility["verdict"]["kinematic_gate_pass"])
    if strict_pass != bool(generated["kinematic_gate_pass"]):
        raise ValueError("MuJoCo feasibility verdict and generated result disagree")

    def worst_body_metrics(source: dict[str, Any], prefix: str) -> dict[str, Any]:
        breakdown = source["robot_body_breakdown"]
        if not breakdown:
            if float(source["max_depth_m"]) != 0.0:
                raise ValueError(f"{prefix or 'generated'} collision has depth but no body breakdown")
            return {
                f"{prefix}worst_robot_body": None,
                f"{prefix}worst_robot_body_is_foot": None,
                f"{prefix}worst_robot_body_max_penetration_m": 0.0,
                f"{prefix}worst_robot_body_contact_frame_rate": 0.0,
                f"{prefix}worst_robot_body_over_5mm_frame_rate": 0.0,
                f"{prefix}worst_robot_body_deep_penetration_frame_rate": 0.0,
            }
        worst = breakdown[0]
        worst_depth = float(worst["max_depth_m"])
        global_depth = float(source["max_depth_m"])
        if (
            math.isfinite(worst_depth)
            and math.isfinite(global_depth)
            and not math.isclose(
                worst_depth,
                global_depth,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(f"{prefix or 'generated'} worst-body depth disagrees with global depth")
        return {
            f"{prefix}worst_robot_body": str(worst["robot_body"]),
            f"{prefix}worst_robot_body_is_foot": bool(worst["is_foot"]),
            f"{prefix}worst_robot_body_max_penetration_m": float(worst["max_depth_m"]),
            f"{prefix}worst_robot_body_contact_frame_rate": float(worst["contact_frame_rate"]),
            f"{prefix}worst_robot_body_over_5mm_frame_rate": float(worst["over_tolerance_frame_rate"]),
            f"{prefix}worst_robot_body_deep_penetration_frame_rate": float(worst["deep_penetration_frame_rate"]),
        }

    comparison = feasibility["comparison_to_reference"]
    metrics = {
        "strict_kinematic_gate_pass": strict_pass,
        "production_contract_match": production_match,
        "penetration_tolerance_m": tolerance,
        "max_environment_penetration_m": float(collision["max_depth_m"]),
        "penetration_over_5mm_frame_rate": float(collision["over_tolerance_frame_rate"]),
        "nonfoot_penetration_over_5mm_frame_rate": float(collision["nonfoot_over_tolerance_frame_rate"]),
        "reference_max_environment_penetration_m": float(reference_collision["max_depth_m"]),
        "reference_penetration_over_5mm_frame_rate": float(reference_collision["over_tolerance_frame_rate"]),
        "reference_nonfoot_penetration_over_5mm_frame_rate": float(
            reference_collision["nonfoot_over_tolerance_frame_rate"]
        ),
        "penetration_over_5mm_frame_rate_delta_vs_reference": float(comparison["over_tolerance_frame_rate_delta"]),
        "nonfoot_penetration_over_5mm_frame_rate_delta_vs_reference": float(
            comparison["nonfoot_over_tolerance_frame_rate_delta"]
        ),
        **worst_body_metrics(collision, ""),
        **worst_body_metrics(reference_collision, "reference_"),
        "joint_limit_max_violation_rad": float(generated["joint_limits"]["max_violation_rad"]),
        "fk_body_error_max_m": float(fk["body_origin_error_m"]["max"]),
        "root_position_error_m_mean": float(alignment["root_position_error_m_mean"]),
        "root_position_error_m_max": float(alignment["root_position_error_m_max"]),
        "root_position_error_m_final": float(alignment["root_position_error_m_final"]),
        "kinematic_checks": dict(generated["checks"]),
    }
    numeric_metrics = {key: value for key, value in metrics.items() if isinstance(value, float)}
    invalid = {
        key: value
        for key, value in numeric_metrics.items()
        if not math.isfinite(value) or (value < 0.0 and "_delta_" not in key)
    }
    if invalid:
        raise ValueError(f"Production metrics must be finite and non-negative, got {invalid}")
    rates = {key: value for key, value in numeric_metrics.items() if "frame_rate" in key and "_delta_" not in key}
    invalid_rates = {key: value for key, value in rates.items() if value > 1.0}
    if invalid_rates:
        raise ValueError(f"Penetration frame rates must be in [0, 1], got {invalid_rates}")
    deltas = {key: value for key, value in numeric_metrics.items() if "frame_rate_delta" in key}
    invalid_deltas = {key: value for key, value in deltas.items() if not -1.0 <= value <= 1.0}
    if invalid_deltas:
        raise ValueError(f"Penetration frame-rate deltas must be in [-1, 1], got {invalid_deltas}")
    return metrics


def _ranking_key(record: dict[str, Any]) -> tuple[Any, ...]:
    metrics = record["metrics"]
    step = record.get("checkpoint_step")
    return (
        not metrics["strict_kinematic_gate_pass"],
        metrics["max_environment_penetration_m"],
        metrics["penetration_over_5mm_frame_rate"],
        metrics["joint_limit_max_violation_rad"],
        metrics["fk_body_error_max_m"],
        metrics["root_position_error_m_final"],
        metrics["root_position_error_m_mean"],
        -(step if isinstance(step, int) else -1),
        record["checkpoint"],
    )


def rank_successful_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank successful records without mutating the evaluation-order records."""
    ranked = sorted((record for record in records if record["status"] == "success"), key=_ranking_key)
    return [{**record, "rank": rank} for rank, record in enumerate(ranked, start=1)]


def _comparison_matches_request(report: dict[str, Any], checkpoint: Path, args: Args) -> bool:
    """Return whether a cached report matches inputs and current source assets."""
    provenance = report.get("provenance", {})
    provenance_path = provenance.get("checkpoint")
    if provenance_path is None or Path(provenance_path).resolve() != checkpoint.resolve():
        return False

    expected_config = {
        "checkpoint": str(checkpoint),
        "clip": args.clip,
        "terrain_urdf": args.terrain_urdf,
        "start": args.start,
        "seed": args.seed,
        "device": args.device,
        "num_steps": args.num_steps,
        "num_cycles": args.num_cycles,
        "replan_stride": args.replan_stride,
        "guidance_scale": args.guidance_scale,
        "mujoco_python": str(_resolve(args.evaluation_python)) if args.evaluation_python else sys.executable,
        "evaluator_script": args.evaluator_script,
        "robot_xml": args.robot_xml,
        "run_mujoco_feasibility": True,
    }
    config = report.get("config", {})
    if any(config.get(key) != value for key, value in expected_config.items()):
        return False

    # A path-only cache key becomes stale when an input is replaced in place.
    # The evaluators already record hashes, so verify those files before reuse.
    paths_and_hashes: dict[Path, str] = {}
    top_level_files = {
        provenance.get("checkpoint"): provenance.get("checkpoint_sha256"),
        provenance.get("clip"): provenance.get("clip_sha256"),
        provenance.get("script"): provenance.get("script_sha256"),
        provenance.get("terrain_urdf"): provenance.get("terrain_urdf_sha256"),
    }
    for raw_path, digest in top_level_files.items():
        if raw_path is None or digest is None:
            return False
        paths_and_hashes[Path(raw_path).resolve()] = str(digest)

    try:
        feasibility_files = report["variants"][PRODUCTION_VARIANT]["mujoco_feasibility"]["provenance"]["files"]
    except (KeyError, TypeError):
        return False
    input_roles = {
        "robot_mjcf",
        "feasibility_evaluator_source",
        "reference_motion_hash_only",
        "terrain_urdf",
        "terrain_asset",
        "generator_checkpoint_hash_only",
    }
    for entry in feasibility_files:
        if entry.get("role") in input_roles:
            raw_path, digest = entry.get("path"), entry.get("sha256")
            if raw_path is None or digest is None:
                return False
            paths_and_hashes[Path(raw_path).resolve()] = str(digest)

    for path, expected_digest in paths_and_hashes.items():
        if not path.is_file() or _sha256(path) != expected_digest:
            return False
    return True


def _evaluation_command(args: Args, checkpoint: Path, checkpoint_output: Path) -> list[str]:
    python = str(_resolve(args.evaluation_python)) if args.evaluation_python else sys.executable
    command = [
        python,
        "-m",
        "holosoma.motion_gen.scripts.evaluate_rollout_modes",
        "--checkpoint",
        str(checkpoint),
        "--clip",
        args.clip,
        "--terrain-urdf",
        args.terrain_urdf,
        "--start",
        str(args.start),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num-steps",
        str(args.num_steps),
        "--num-cycles",
        str(args.num_cycles),
        "--replan-stride",
        str(args.replan_stride),
        "--output-dir",
        str(checkpoint_output),
        "--mujoco-python",
        python,
        "--evaluator-script",
        args.evaluator_script,
        "--robot-xml",
        args.robot_xml,
        "--wandb-mode",
        "disabled",
    ]
    if args.guidance_scale is not None:
        command.extend(("--guidance-scale", str(args.guidance_scale)))
    return command


def evaluate_checkpoint(args: Args, checkpoint: Path, sweep_index: int) -> dict[str, Any]:
    """Evaluate one checkpoint and return a success/error record."""
    checkpoint_output = _resolve(args.output_dir) / "checkpoints" / checkpoint.stem
    comparison_path = checkpoint_output / "comparison.json"
    started = time.monotonic()
    completed: subprocess.CompletedProcess[str] | None = None
    reused = False
    try:
        if args.reuse_existing and comparison_path.is_file():
            report = json.loads(comparison_path.read_text())
            reused = _comparison_matches_request(report, checkpoint, args)
        if not reused:
            checkpoint_output.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                _evaluation_command(args, checkpoint, checkpoint_output),
                cwd=_repo_root(),
                check=False,
                capture_output=True,
                text=True,
                timeout=args.timeout_s,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"rollout evaluator exited with status {completed.returncode}")
            report = json.loads(comparison_path.read_text())
            if not _comparison_matches_request(report, checkpoint, args):
                raise ValueError("evaluator provenance, contract, or input hashes do not match the request")
        metrics = extract_production_metrics(
            report,
            allow_nonproduction_contract=args.allow_nonproduction_contract,
        )
        provenance = report["provenance"]
        return {
            "sweep_index": sweep_index,
            "status": "success",
            "checkpoint": str(checkpoint),
            "checkpoint_step": provenance.get("checkpoint_step"),
            "checkpoint_sha256": provenance.get("checkpoint_sha256"),
            "comparison_json": str(comparison_path),
            "output_dir": str(checkpoint_output),
            "duration_s": time.monotonic() - started,
            "reused": reused,
            "returncode": 0 if completed is None else completed.returncode,
            "metrics": metrics,
            "evaluator_verdict": report["verdict"],
            "production_contract": report["production_contract"],
        }
    except Exception as exc:
        return {
            "sweep_index": sweep_index,
            "status": "error",
            "checkpoint": str(checkpoint),
            "checkpoint_step": None,
            "comparison_json": str(comparison_path),
            "output_dir": str(checkpoint_output),
            "duration_s": time.monotonic() - started,
            "reused": reused,
            "returncode": None if completed is None else completed.returncode,
            "error": f"{type(exc).__name__}: {exc}",
            "stdout_tail": "" if completed is None else completed.stdout[-4000:],
            "stderr_tail": "" if completed is None else completed.stderr[-4000:],
        }


def build_aggregate(args: Args, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the stable aggregate report and select the best checkpoint."""
    ranked = rank_successful_records(records)
    ranking = [
        {
            "rank": record["rank"],
            "checkpoint": record["checkpoint"],
            "checkpoint_step": record["checkpoint_step"],
            "comparison_json": record["comparison_json"],
            "metrics": record["metrics"],
        }
        for record in ranked
    ]
    return {
        "schema_version": 1,
        "kind": "motion-generator-feasibility-checkpoint-sweep",
        "config": asdict(args),
        "production_variant": PRODUCTION_VARIANT,
        "semantics": {
            "scope": (
                "generator-only kinematic counterfactual using generated feedback; production tracker uses measured "
                "simulator history, and dynamic feasibility is not evaluated"
            ),
            "ranking": (
                "strict kinematic pass first, then ascending max penetration, >5 mm penetration frame rate, "
                "joint-limit violation, FK max error, final root drift, and mean root drift"
            ),
            "checkpoint_errors": "recorded and excluded from ranking",
        },
        "summary": {
            "requested": len(records),
            "succeeded": len(ranked),
            "failed": sum(record["status"] == "error" for record in records),
            "strict_passed": sum(record["metrics"]["strict_kinematic_gate_pass"] for record in ranked),
        },
        "selected": ranking[0] if ranking else None,
        "ranking": ranking,
        "checkpoints": records,
    }


def _csv_row(record: dict[str, Any], rank_by_checkpoint: dict[str, int]) -> dict[str, Any]:
    row = {field: record.get(field, "") for field in _CSV_FIELDS}
    row["rank"] = rank_by_checkpoint.get(record["checkpoint"], "")
    for key, value in record.get("metrics", {}).items():
        if key in row:
            row[key] = value
    return row


def write_aggregate_files(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and flat CSV aggregate artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sweep.json"
    csv_path = output_dir / "sweep.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    rank_by_checkpoint = {row["checkpoint"]: row["rank"] for row in report["ranking"]}
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in report["checkpoints"]:
            writer.writerow(_csv_row(record, rank_by_checkpoint))
    return json_path, csv_path


def _log_wandb(args: Args, report: dict[str, Any], json_path: Path, csv_path: Path) -> str | None:
    if args.wandb_mode == "disabled":
        return None
    if args.wandb_mode not in {"offline", "online"}:
        raise ValueError("wandb_mode must be disabled, offline, or online")
    import wandb  # noqa: PLC0415

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name or f"feasibility-sweep-{Path(args.clip).stem}-seed{args.seed}",
        job_type="terrain-feasibility-checkpoint-sweep",
        mode=args.wandb_mode,
        config=asdict(args),
    )
    rows = []
    for record in report["checkpoints"]:
        payload: dict[str, Any] = {
            "sweep/status_success": float(record["status"] == "success"),
            "sweep/checkpoint_step": record.get("checkpoint_step"),
            "sweep/duration_s": record["duration_s"],
        }
        if record["status"] == "success":
            payload.update(
                {f"production/{key}": value for key, value in record["metrics"].items() if key != "kinematic_checks"}
            )
        run.log(payload, step=record["sweep_index"])
        rows.append(_csv_row(record, {row["checkpoint"]: row["rank"] for row in report["ranking"]}))
    if rows:
        table = wandb.Table(columns=list(_CSV_FIELDS), data=[[row[field] for field in _CSV_FIELDS] for row in rows])
        run.log({"checkpoint_ranking": table})
    if report["selected"] is not None:
        run.summary["selected/checkpoint"] = report["selected"]["checkpoint"]
        run.summary["selected/checkpoint_step"] = report["selected"]["checkpoint_step"]
        for key, value in report["selected"]["metrics"].items():
            if key != "kinematic_checks":
                run.summary[f"selected/{key}"] = value
    artifact = wandb.Artifact(f"{run.name}-aggregate", type="terrain-feasibility-checkpoint-sweep")
    artifact.add_file(str(json_path), name="sweep.json")
    artifact.add_file(str(csv_path), name="sweep.csv")
    artifact.add_file(str(Path(__file__).resolve()), name="source/sweep_feasibility_checkpoints.py")
    run.log_artifact(artifact)
    url = run.url
    run.finish()
    return url


def main(args: Args) -> None:
    if args.wandb_mode not in {"disabled", "offline", "online"}:
        raise ValueError("wandb_mode must be disabled, offline, or online")
    checkpoints = resolve_checkpoints(args)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints match {args.checkpoint_glob!r}")

    records = []
    for index, checkpoint in enumerate(checkpoints):
        print(f"[{index + 1}/{len(checkpoints)}] evaluating {checkpoint}")
        record = evaluate_checkpoint(args, checkpoint, index)
        records.append(record)
        if record["status"] == "error":
            print(f"  ERROR: {record['error']}")
        else:
            metrics = record["metrics"]
            print(
                f"  pass={metrics['strict_kinematic_gate_pass']} "
                f"depth={metrics['max_environment_penetration_m']:.6f} m "
                f">5mm={metrics['penetration_over_5mm_frame_rate']:.6f}"
            )

    report = build_aggregate(args, records)
    output_dir = _resolve(args.output_dir)
    json_path, csv_path = write_aggregate_files(report, output_dir)
    wandb_url = _log_wandb(args, report, json_path, csv_path)
    if wandb_url is not None:
        report["wandb_url"] = wandb_url
        json_path, csv_path = write_aggregate_files(report, output_dir)

    print(json.dumps({"summary": report["summary"], "selected": report["selected"]}, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    if wandb_url is not None:
        print(f"W&B: {wandb_url}")
    if args.fail_on_checkpoint_error and report["summary"]["failed"]:
        raise RuntimeError(f"{report['summary']['failed']} checkpoint evaluation(s) failed")
    if args.require_selected_pass and (
        report["selected"] is None or not report["selected"]["metrics"]["strict_kinematic_gate_pass"]
    ):
        raise AssertionError("No selected checkpoint passed the strict kinematic gate")


if __name__ == "__main__":
    main(tyro.cli(Args))
