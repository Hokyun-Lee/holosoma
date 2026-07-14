"""Compare long-rollout history and heading modes with identical DDIM seeds.

This diagnostic evaluates the Cartesian product

``history_mode = {generated_feedback, source_history}``
``heading_mode = {keep_current, oracle_cycle}``

for one source clip.  ``source_history`` is a teacher-forced counterfactual:
every re-plan receives the two aligned source frames, but the sampled future
frames are still generated.  ``oracle_cycle`` derives a world-frame heading
from the aligned source anchor to the final frame of that cycle's full
prediction horizon.  It is a diagnostic oracle, not a deployable command.

``keep_current`` matches ``GeneratedMotionCommand(heading_mode="current")``:
the reset anchor's facing direction is converted to one world-frame unit
vector and held fixed for the whole episode.  It does *not* pass
``target_heading=None`` at every cycle, because that would silently rotate the
command to each newly generated anchor's facing direction.  The production
terrain command consumes its full 25-frame horizon before re-planning, so the
default stride is 25.  Shorter strides remain available as an explicit
generator-only diagnostic, but are not production-parity rollouts.

All four variants use seed ``seed + cycle`` and the same terrain sampler.  The
script writes generator-native/raw and MuJoCo-qpos NPZ files plus one JSON
comparison.  By default it also invokes the existing MuJoCo feasibility
evaluator for each variant.  Use an environment containing both torch and
MuJoCo (``hsmujoco`` in this workspace), for example::

    python -m holosoma.motion_gen.scripts.evaluate_rollout_modes \
      --checkpoint logs/motion_gen/terrain_robust_fk_4090/checkpoints/final.pt \
      --clip omni_climb_09_z1_0 --start 100 \
      --terrain-urdf data/motion_gen/raw/omniretarget_terrain/climb_09/multi_boxes_z_scale_1.0.urdf
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
import tyro

from holosoma.motion_gen.dataset import MotionClip, load_wbt_motion
from holosoma.motion_gen.export import export_generated_qpos_npz, export_generated_raw_npz
from holosoma.motion_gen.features import FeatureLayout, quat_normalize, quat_yaw, unpack_features
from holosoma.motion_gen.sampling import MotionGenerator, MotionGeneratorInput
from holosoma.motion_gen.terrain import BoxTerrain

HistoryMode = Literal["generated_feedback", "source_history"]
HeadingMode = Literal["keep_current", "oracle_cycle"]
TerrainScanFn = Callable[[torch.Tensor], torch.Tensor]

_HISTORY_MODES: tuple[HistoryMode, ...] = ("generated_feedback", "source_history")
_HEADING_MODES: tuple[HeadingMode, ...] = ("keep_current", "oracle_cycle")


@dataclass(frozen=True)
class Args:
    checkpoint: str = "logs/motion_gen/terrain_robust_fk_4090/checkpoints/final.pt"
    clip: str = "omni_climb_09_z1_0"
    """Processed clip stem or an explicit WBT NPZ path."""
    terrain_urdf: str = "data/motion_gen/raw/omniretarget_terrain/climb_09/multi_boxes_z_scale_1.0.urdf"
    start: int = 100
    seed: int = 123
    device: str = "cuda:0"
    num_steps: int = 2
    num_cycles: int = 17
    """Seventeen 25-frame cycles fit the default clip range starting at frame 100."""
    replan_stride: int = 25
    """Generated frames consumed per query. Production uses the full 25-frame horizon."""
    guidance_scale: float | None = None
    output_dir: str = "logs/motion_gen/terrain_feasibility/climb09_rollout_modes_production_2step"
    run_mujoco_feasibility: bool = True
    """Invoke the existing collision/FK/joint feasibility evaluator for every variant."""
    mujoco_python: str | None = None
    """Python executable for MuJoCo evaluation; None uses this process' executable."""
    evaluator_script: str = (
        "src/holosoma_retargeting/holosoma_retargeting/data_conversion/evaluate_motion_feasibility_mj.py"
    )
    robot_xml: str = "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
    require_all_kinematic_gates: bool = False
    """Write all artifacts first, then fail if any of the four MuJoCo gates fails."""
    wandb_mode: str = "disabled"
    """disabled, offline, or online."""
    wandb_entity: str | None = None
    wandb_project: str = "HoloSomaMotionGenerator"
    wandb_group: str = "terrain_rollout_modes"
    wandb_name: str | None = None


@dataclass
class VariantResult:
    features: torch.Tensor
    report: dict[str, Any]


def _repo_root() -> Path:
    source = Path(__file__).resolve()
    for candidate in source.parents:
        if (candidate / "src/holosoma_retargeting").is_dir() and (candidate / "src/holosoma").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root above {source}")


def _resolve_repo_path(path: str | Path) -> Path:
    result = Path(path).expanduser()
    return result.resolve() if result.is_absolute() else (_repo_root() / result).resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quaternion_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dots = (quat_normalize(a) * quat_normalize(b)).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dots)


def _anchor_forward_heading(anchor_quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    yaw = quat_yaw(quat_normalize(anchor_quaternion_wxyz))
    return torch.stack((torch.cos(yaw), torch.sin(yaw)))


def oracle_cycle_heading(
    source_features: torch.Tensor,
    layout: FeatureLayout,
    *,
    cycle_start: int,
    past_frames: int,
    future_frames: int,
    min_displacement_m: float,
) -> torch.Tensor:
    """Return the source-derived world heading for one prediction cycle.

    A stationary source horizon falls back to the source anchor's facing
    direction, matching ``MotionGenerator.generate(..., target_heading=None)``
    at that source state.
    """
    anchor_index = cycle_start + past_frames - 1
    future_index = cycle_start + past_frames + future_frames - 1
    if cycle_start < 0 or future_index >= len(source_features):
        raise ValueError(f"Cycle [{cycle_start}, {future_index}] is outside source length {len(source_features)}")
    anchor = source_features[anchor_index]
    displacement = source_features[future_index, layout.root_pos_slice][:2] - anchor[layout.root_pos_slice][:2]
    norm = torch.linalg.vector_norm(displacement)
    if bool(norm > min_displacement_m):
        return displacement / norm
    return _anchor_forward_heading(anchor[layout.root_quat_slice])


def _alignment_metrics(
    generated: torch.Tensor,
    source: torch.Tensor,
    layout: FeatureLayout,
) -> dict[str, float]:
    if generated.shape != source.shape or generated.ndim != 2 or generated.shape[1] != layout.dim:
        raise ValueError(
            f"Expected matching (T,{layout.dim}) generated/source features, got "
            f"{tuple(generated.shape)} and {tuple(source.shape)}"
        )
    generated_parts = unpack_features(generated, layout)
    source_parts = unpack_features(source, layout)
    root_error = torch.linalg.vector_norm(generated_parts["root_pos"] - source_parts["root_pos"], dim=-1)
    root_xy_error = torch.linalg.vector_norm(
        generated_parts["root_pos"][..., :2] - source_parts["root_pos"][..., :2], dim=-1
    )
    root_angle = _quaternion_angle(generated_parts["root_quat"], source_parts["root_quat"])
    joint_l2 = torch.linalg.vector_norm(generated_parts["joint_pos"] - source_parts["joint_pos"], dim=-1)
    body_error = torch.linalg.vector_norm(generated_parts["body_pos"] - source_parts["body_pos"], dim=-1)

    def summary(values: torch.Tensor, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_max": float(values.max()),
            f"{prefix}_final": float(values[-1]),
        }

    return {
        **summary(root_error, "root_position_error_m"),
        **summary(root_xy_error, "root_xy_error_m"),
        **summary(root_angle, "root_orientation_error_rad"),
        **summary(joint_l2, "joint_l2_error_rad"),
        "body_mpjpe_m_mean": float(body_error.mean()),
        "body_mpjpe_m_max": float(body_error.max()),
        "body_mpjpe_m_final": float(body_error[-1].mean()),
    }


def _heading_tracking_error(
    anchor_xy: torch.Tensor,
    generated_root_xy: torch.Tensor,
    target_heading: torch.Tensor,
    min_displacement_m: float,
) -> float | None:
    displacement = generated_root_xy[-1] - anchor_xy
    norm = torch.linalg.vector_norm(displacement)
    if not bool(norm > min_displacement_m):
        return None
    direction = displacement / norm
    dot = torch.sum(direction * target_heading).clamp(-1.0, 1.0)
    return float(torch.acos(dot))


def generate_variant(
    generator: MotionGenerator,
    source_features: torch.Tensor,
    *,
    start: int,
    num_cycles: int,
    replan_stride: int,
    seed: int,
    num_steps: int,
    guidance_scale: float | None,
    history_mode: HistoryMode,
    heading_mode: HeadingMode,
    terrain_scan_fn: TerrainScanFn,
) -> VariantResult:
    """Generate one reproducible rollout-comparison variant."""
    if history_mode not in _HISTORY_MODES:
        raise ValueError(f"Unknown history mode {history_mode!r}")
    if heading_mode not in _HEADING_MODES:
        raise ValueError(f"Unknown heading mode {heading_mode!r}")
    if start < 0 or num_cycles < 1 or replan_stride < 1:
        raise ValueError("start must be non-negative; num_cycles and replan_stride must be positive")
    cfg = generator.cfg.data
    past_frames, future_frames = cfg.past_frames, cfg.future_frames
    if replan_stride > future_frames:
        raise ValueError(f"replan_stride {replan_stride} exceeds future horizon {future_frames}")
    final_source_frame = start + (num_cycles - 1) * replan_stride + past_frames + future_frames - 1
    aligned_stop = start + past_frames + num_cycles * replan_stride
    if max(final_source_frame + 1, aligned_stop) > len(source_features):
        raise ValueError(
            f"Source has {len(source_features)} frames, but this comparison requires frame "
            f"{max(final_source_frame, aligned_stop - 1)}"
        )

    source_features = source_features.to(generator.device)
    initial = source_features[start : start + past_frames].unsqueeze(0)
    # GeneratedMotionCommand samples/derives one world-frame heading at reset
    # and keeps it for the episode.  Passing target_heading=None on every call
    # would instead mean "follow this cycle's anchor facing" and is therefore a
    # different, generator-only feedback process.
    episode_heading = _anchor_forward_heading(initial[0, -1, generator.layout.root_quat_slice]).unsqueeze(0)
    feedback_past = initial
    chunks = [initial]
    cycles: list[dict[str, Any]] = []

    with torch.no_grad():
        for cycle in range(num_cycles):
            source_start = start + cycle * replan_stride
            source_past = source_features[source_start : source_start + past_frames].unsqueeze(0)
            past = feedback_past if history_mode == "generated_feedback" else source_past
            if heading_mode == "oracle_cycle":
                target_heading = oracle_cycle_heading(
                    source_features,
                    generator.layout,
                    cycle_start=source_start,
                    past_frames=past_frames,
                    future_frames=future_frames,
                    min_displacement_m=cfg.min_heading_disp,
                ).unsqueeze(0)
            else:
                target_heading = episode_heading

            terrain_scan = terrain_scan_fn(past)
            source_scan = terrain_scan_fn(source_past)
            if terrain_scan.shape != (1, cfg.terrain_dim):
                raise ValueError(f"terrain_scan_fn must return (1,{cfg.terrain_dim}), got {tuple(terrain_scan.shape)}")
            cycle_seed = seed + cycle
            output = generator.generate(
                MotionGeneratorInput(
                    past_motion=past,
                    target_heading=target_heading,
                    terrain_height=terrain_scan,
                ),
                num_steps=num_steps,
                deterministic=True,
                seed=cycle_seed,
                guidance_scale=guidance_scale,
            )
            new_frames = output.features[:, :replan_stride]
            chunks.append(new_frames)
            feedback_past = torch.cat((past, new_frames), dim=1)[:, -past_frames:]

            source_future = source_features[source_start + past_frames : source_start + past_frames + replan_stride]
            effective_heading = target_heading[0]
            anchor_source = source_past[0, -1]
            anchor_actual = past[0, -1]
            anchor_yaw_error = _quaternion_angle(
                anchor_actual[generator.layout.root_quat_slice].unsqueeze(0),
                anchor_source[generator.layout.root_quat_slice].unsqueeze(0),
            )[0]
            cycle_metrics = _alignment_metrics(new_frames[0], source_future, generator.layout)
            generated_root = new_frames[0, :, generator.layout.root_pos_slice]
            cycles.append(
                {
                    "cycle": cycle,
                    "seed": cycle_seed,
                    "source_start_frame": source_start,
                    "output_local_frames": [
                        past_frames + cycle * replan_stride,
                        past_frames + (cycle + 1) * replan_stride - 1,
                    ],
                    "target_heading_world_xy": [float(value) for value in effective_heading],
                    "heading_tracking_error_rad": _heading_tracking_error(
                        anchor_actual[generator.layout.root_pos_slice][:2],
                        generated_root[:, :2],
                        effective_heading,
                        cfg.min_heading_disp,
                    ),
                    "history_anchor_root_error_m": float(
                        torch.linalg.vector_norm(
                            anchor_actual[generator.layout.root_pos_slice]
                            - anchor_source[generator.layout.root_pos_slice]
                        )
                    ),
                    "history_anchor_yaw_error_rad": float(anchor_yaw_error),
                    "terrain_scan_nonzero_points": int(torch.count_nonzero(terrain_scan > 1.0e-6)),
                    "terrain_scan_max_m": float(terrain_scan.max()),
                    "terrain_scan_source_rmse_m": float(torch.sqrt(torch.mean((terrain_scan - source_scan) ** 2))),
                    "source_alignment": cycle_metrics,
                }
            )

    trajectory = torch.cat(chunks, dim=1)[0].cpu()
    aligned_source = source_features[start:aligned_stop].cpu()
    predicted = trajectory[past_frames:]
    predicted_source = aligned_source[past_frames:]
    heading_errors = [
        record["heading_tracking_error_rad"] for record in cycles if record["heading_tracking_error_rad"] is not None
    ]
    report = {
        "history_mode": history_mode,
        "heading_mode": heading_mode,
        "frames": len(trajectory),
        "predicted_frames": len(predicted),
        "source_frame_range": [start, aligned_stop - 1],
        "aggregate_source_alignment": _alignment_metrics(predicted, predicted_source, generator.layout),
        "heading_tracking_error_rad": {
            "mean": float(np.mean(heading_errors)) if heading_errors else None,
            "max": float(np.max(heading_errors)) if heading_errors else None,
            "valid_cycles": len(heading_errors),
        },
        "cycles": cycles,
    }
    return VariantResult(features=trajectory, report=report)


def _variant_id(history_mode: HistoryMode, heading_mode: HeadingMode) -> str:
    return f"{history_mode}__{heading_mode}"


def _comparison_snapshot(variant: dict[str, Any]) -> dict[str, float]:
    alignment = variant["aggregate_source_alignment"]
    result = {
        "root_position_error_m_mean": alignment["root_position_error_m_mean"],
        "root_position_error_m_final": alignment["root_position_error_m_final"],
        "joint_l2_error_rad_mean": alignment["joint_l2_error_rad_mean"],
        "body_mpjpe_m_mean": alignment["body_mpjpe_m_mean"],
    }
    feasibility = variant.get("mujoco_feasibility")
    if feasibility is not None:
        generated = feasibility["generated"]
        result.update(
            {
                "environment_max_depth_m": generated["environment_collision"]["max_depth_m"],
                "environment_over_tolerance_frame_rate": generated["environment_collision"][
                    "over_tolerance_frame_rate"
                ],
                "joint_limit_max_violation_rad": generated["joint_limits"]["max_violation_rad"],
                "fk_body_error_max_m": generated["fk_consistency"]["body_origin_error_m"]["max"],
            }
        )
    return result


def build_comparisons(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build signed metric deltas for the two controlled comparison axes."""

    def delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
        a, b = _comparison_snapshot(left), _comparison_snapshot(right)
        common = sorted(set(a) & set(b))
        return {name: float(a[name] - b[name]) for name in common}

    history_effect = {}
    for heading_mode in _HEADING_MODES:
        feedback = variants[_variant_id("generated_feedback", heading_mode)]
        source = variants[_variant_id("source_history", heading_mode)]
        history_effect[heading_mode] = {
            "semantics": "generated_feedback minus source_history; positive is worse for listed error/depth metrics",
            "metric_delta": delta(feedback, source),
        }
    heading_effect = {}
    for history_mode in _HISTORY_MODES:
        oracle = variants[_variant_id(history_mode, "oracle_cycle")]
        keep = variants[_variant_id(history_mode, "keep_current")]
        heading_effect[history_mode] = {
            "semantics": "oracle_cycle minus keep_current; negative is better for listed error/depth metrics",
            "metric_delta": delta(oracle, keep),
        }
    return {
        "generated_feedback_minus_source_history": history_effect,
        "oracle_cycle_minus_keep_current": heading_effect,
    }


def _terrain_scan_fn(generator: MotionGenerator, terrain: BoxTerrain) -> TerrainScanFn:
    grid = generator.cfg.data.scan_grid

    def sample(past: torch.Tensor) -> torch.Tensor:
        scans = []
        for row in past:
            anchor = row[-1]
            scans.append(
                terrain.sample_scan(
                    anchor[:2].detach().cpu().numpy(),
                    float(quat_yaw(anchor[3:7])),
                    grid,
                )
            )
        return torch.from_numpy(np.stack(scans)).to(device=generator.device, dtype=past.dtype)

    return sample


def _run_mujoco_evaluator(
    args: Args,
    *,
    qpos_path: Path,
    raw_path: Path,
    output_path: Path,
    clip_path: Path,
    variant_id: str,
    history_frames: int,
) -> dict[str, Any]:
    python = Path(args.mujoco_python).expanduser() if args.mujoco_python else Path(sys.executable)
    evaluator = _resolve_repo_path(args.evaluator_script)
    command = [
        str(python),
        str(evaluator),
        "--motion",
        str(qpos_path),
        "--raw-motion",
        str(raw_path),
        "--terrain-urdf",
        str(_resolve_repo_path(args.terrain_urdf)),
        "--reference",
        str(clip_path),
        "--reference-start-frame",
        str(args.start),
        "--generator-checkpoint",
        str(_resolve_repo_path(args.checkpoint)),
        "--generator-clip",
        f"{args.clip}:{variant_id}",
        "--generator-seed",
        str(args.seed),
        "--generator-num-steps",
        str(args.num_steps),
        "--generator-num-cycles",
        str(args.num_cycles),
        "--history-frames",
        str(history_frames),
        "--replan-stride",
        str(args.replan_stride),
        "--robot-xml",
        str(_resolve_repo_path(args.robot_xml)),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"MuJoCo evaluator failed for {variant_id} (exit {completed.returncode})\n"
            f"stdout tail:\n{completed.stdout[-4000:]}\nstderr tail:\n{completed.stderr[-4000:]}"
        )
    return json.loads(output_path.read_text())


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            result.update(_flatten_scalars(child, child_prefix))
    elif isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
        result[prefix] = float(value)
    return result


def _log_wandb(args: Args, report: dict[str, Any], report_path: Path, artifact_paths: list[Path]) -> str | None:
    if args.wandb_mode == "disabled":
        return None
    if args.wandb_mode not in {"offline", "online"}:
        raise ValueError("wandb_mode must be disabled, offline, or online")
    import wandb  # noqa: PLC0415

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name or f"rollout-modes-{Path(args.clip).stem}-seed{args.seed}",
        job_type="terrain-rollout-mode-eval",
        mode=args.wandb_mode,
        config=asdict(args),
    )
    aggregate = {
        variant_id: {
            "metrics": _comparison_snapshot(variant),
            "kinematic_gate_pass": (
                variant["mujoco_feasibility"]["verdict"]["kinematic_gate_pass"]
                if variant["mujoco_feasibility"] is not None
                else None
            ),
            "heading_tracking_error_rad": variant["heading_tracking_error_rad"],
        }
        for variant_id, variant in report["variants"].items()
    }
    run.log(_flatten_scalars({"variants": aggregate, "comparisons": report["comparisons"]}), step=0)
    for cycle in range(args.num_cycles):
        cycle_metrics = {variant_id: variant["cycles"][cycle] for variant_id, variant in report["variants"].items()}
        run.log(_flatten_scalars({"cycle": cycle_metrics}), step=cycle + 1)
    artifact = wandb.Artifact(f"{run.name}-artifacts", type="terrain-rollout-mode-comparison")
    artifact.add_file(str(report_path), name="comparison.json")
    artifact.add_file(str(Path(__file__).resolve()), name="source/evaluate_rollout_modes.py")
    for path in artifact_paths:
        artifact.add_file(str(path), name=str(path.relative_to(report_path.parent)))
    run.log_artifact(artifact)
    url = run.url
    run.finish()
    return url


def main(args: Args) -> None:
    if args.num_steps < 1 or args.num_cycles < 1 or args.replan_stride < 1:
        raise ValueError("num_steps, num_cycles, and replan_stride must be positive")
    if not 0 <= args.seed <= (1 << 63) - 1 - args.num_cycles:
        raise ValueError("seed + num_cycles must fit in torch's non-negative 63-bit seed range")
    if args.require_all_kinematic_gates and not args.run_mujoco_feasibility:
        raise ValueError("require_all_kinematic_gates requires run_mujoco_feasibility")
    if args.wandb_mode not in {"disabled", "offline", "online"}:
        raise ValueError("wandb_mode must be disabled, offline, or online")

    checkpoint = _resolve_repo_path(args.checkpoint)
    terrain_path = _resolve_repo_path(args.terrain_urdf)
    output_dir = _resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = MotionGenerator.from_checkpoint(str(checkpoint), device=args.device)
    cfg = generator.cfg
    clip_path = (
        _resolve_repo_path(args.clip)
        if args.clip.endswith(".npz")
        else _resolve_repo_path(Path(cfg.data.processed_dir) / f"{args.clip}.npz")
    )
    clip: MotionClip = load_wbt_motion(clip_path, generator.layout, expected_fps=cfg.data.fps)
    if not cfg.data.use_terrain_scan:
        raise ValueError("This comparison requires a terrain-conditioned checkpoint")
    terrain = BoxTerrain.from_urdf(terrain_path)
    terrain_scan_fn = _terrain_scan_fn(generator, terrain)

    variants: dict[str, dict[str, Any]] = {}
    artifact_paths: list[Path] = []
    for history_mode in _HISTORY_MODES:
        for heading_mode in _HEADING_MODES:
            variant_id = _variant_id(history_mode, heading_mode)
            variant_dir = output_dir / "variants" / variant_id
            variant_dir.mkdir(parents=True, exist_ok=True)
            result = generate_variant(
                generator,
                clip.features,
                start=args.start,
                num_cycles=args.num_cycles,
                replan_stride=args.replan_stride,
                seed=args.seed,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                history_mode=history_mode,
                heading_mode=heading_mode,
                terrain_scan_fn=terrain_scan_fn,
            )
            qpos_path = export_generated_qpos_npz(
                result.features,
                generator.layout,
                cfg.data.fps,
                variant_dir / "motion_qpos.npz",
            )
            raw_path = export_generated_raw_npz(
                result.features,
                generator.layout,
                cfg.data.fps,
                variant_dir / "motion_raw.npz",
            )
            result.report["files"] = {
                "qpos": str(qpos_path),
                "raw": str(raw_path),
            }
            if args.run_mujoco_feasibility:
                feasibility_path = variant_dir / "feasibility.json"
                feasibility = _run_mujoco_evaluator(
                    args,
                    qpos_path=qpos_path,
                    raw_path=raw_path,
                    output_path=feasibility_path,
                    clip_path=clip_path,
                    variant_id=variant_id,
                    history_frames=cfg.data.past_frames,
                )
                result.report["files"]["feasibility"] = str(feasibility_path)
                result.report["mujoco_feasibility"] = feasibility
                artifact_paths.append(feasibility_path)
            else:
                result.report["mujoco_feasibility"] = None
            artifact_paths.extend((qpos_path, raw_path))
            variants[variant_id] = result.report

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "motion-generator-rollout-mode-comparison",
        "config": asdict(args),
        "semantics": {
            "generated_feedback": "each cycle consumes the previous generated frames",
            "source_history": "each cycle is reset to its two aligned source frames (teacher forced)",
            "keep_current": (
                "reset anchor facing held as one episode-constant world-frame heading, matching "
                "GeneratedMotionCommand heading_mode=current"
            ),
            "oracle_cycle": (
                "diagnostic source heading from the cycle anchor to the last source frame of the "
                "full prediction horizon; unavailable at deployment"
            ),
            "source_history_continuity": (
                "source-history windows are independently sampled and concatenated; their re-plan "
                "boundary continuity check is diagnostic, not a deployable-rollout gate"
            ),
            "environment_collision": (
                "the reused evaluator includes intended hand/support contacts; inspect each full "
                "feasibility report before interpreting aggregate contact rates"
            ),
            "production_parity_scope": (
                f"this run uses replan_stride={args.replan_stride}; the current 0.5 s / 50 Hz "
                "GeneratedMotionCommand uses 25 and consumes the full prediction horizon. "
                "generated_feedback is still a perfect-reference counterfactual because production "
                "conditions on measured simulator states"
            ),
        },
        "production_contract": {
            "command_replan_interval_s": 0.5,
            "control_fps": cfg.data.fps,
            "command_replan_steps": 25,
            "generator_future_frames": cfg.data.future_frames,
            "evaluated_replan_stride": args.replan_stride,
            "full_horizon_stride": args.replan_stride == cfg.data.future_frames,
            "current_command_stride_match": args.replan_stride == 25,
            "history_source_match": False,
            "history_source_note": (
                "generated_feedback reuses generated features; GeneratedMotionCommand uses two measured "
                "simulator frames after bootstrap"
            ),
        },
        "provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_step": generator.checkpoint_step,
            "clip": str(clip_path),
            "clip_sha256": _sha256(clip_path),
            "terrain_urdf": str(terrain_path),
            "terrain_urdf_sha256": _sha256(terrain_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "torch_version": torch.__version__,
        },
        "variants": variants,
    }
    report["comparisons"] = build_comparisons(variants)
    generated_feedback_gate_pass = (
        all(
            variants[_variant_id("generated_feedback", heading_mode)]["mujoco_feasibility"]["verdict"][
                "kinematic_gate_pass"
            ]
            for heading_mode in _HEADING_MODES
        )
        if args.run_mujoco_feasibility
        else None
    )
    report["verdict"] = {
        "all_kinematic_gates_pass": (
            all(variant["mujoco_feasibility"]["verdict"]["kinematic_gate_pass"] for variant in variants.values())
            if args.run_mujoco_feasibility
            else None
        ),
        "generated_feedback_kinematic_gates_pass": generated_feedback_gate_pass,
        "dynamic_feasibility": "not_evaluated",
    }
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    wandb_url = _log_wandb(args, report, report_path, artifact_paths)
    if wandb_url is not None:
        report["wandb_url"] = wandb_url
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(json.dumps({"verdict": report["verdict"], "comparisons": report["comparisons"]}, indent=2))
    print(f"JSON: {report_path}")
    if wandb_url is not None:
        print(f"W&B: {wandb_url}")
    if args.require_all_kinematic_gates and not report["verdict"]["all_kinematic_gates_pass"]:
        raise AssertionError("At least one rollout comparison variant failed the kinematic gate")


if __name__ == "__main__":
    main(tyro.cli(Args))
