"""Benchmark Stage-10 generator and tracker PyTorch inference latency.

This is deliberately a module-only benchmark: inputs already reside on the
GPU, and no Isaac Sim stepping, terrain ray casting, observation assembly, or
robot command-buffer update is included.  The sequential measurement runs one
2-step generator query followed by one tracker query on the same CUDA stream;
it is not an exported end-to-end closed-loop artifact.  In particular, the
existing fixed-motion tracker ONNX exporter is neither used nor described here
as a closed-loop diffusion export.

Example::

    python -m holosoma.motion_gen.scripts.benchmark_inference_latency \
      --generator-checkpoint logs/motion_gen/terrain_robust_fk_4090/checkpoints/final.pt \
      --tracker-checkpoint logs/WholeBodyTracking/<run>/model_*.pt \
      --output logs/motion_gen/inference_latency/stage10.json
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import tyro
from torch import nn

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.agents.ppo.ppo import EmpiricalNormalization
from holosoma.config_types.algo import LayerConfig, ModuleConfig
from holosoma.motion_gen.dataset import load_wbt_motion
from holosoma.motion_gen.features import quat_yaw
from holosoma.motion_gen.sampling import MotionGenerator, MotionGeneratorInput

_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Args:
    generator_checkpoint: str = "logs/motion_gen/terrain_robust_fk_4090/checkpoints/final.pt"
    tracker_checkpoint: str = (
        "logs/WholeBodyTracking/20260711_122501-g1_29dof_wbt_gen_manager-locomotion/model_12000.pt"
    )
    motion_clip: str = "data/motion_gen/processed_paperscale/lafan1_walk4_subject1.npz"
    motion_start: int = 0
    output: str = "logs/motion_gen/inference_latency/stage10_pytorch_cuda.json"
    device: str = "cuda:0"
    batch_size: int = 1
    ddim_steps: int = 2
    warmup: int = 50
    measure: int = 500
    use_ema: bool = True
    deterministic_sampling: bool = False
    sampling_seed: int = 0
    replan_interval_policy_steps: int = 25


@dataclass(frozen=True)
class LoadedTracker:
    policy: nn.Module
    observation_dim: int
    action_dim: int
    empirical_normalization: bool
    checkpoint_iteration: int
    actor_config: dict[str, Any]


class TrackerInferencePolicy(nn.Module):
    """The same normalization + deterministic actor path used by PPO eval."""

    def __init__(self, actor: nn.Module, normalizer: nn.Module, empirical_normalization: bool):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer
        self.empirical_normalization = empirical_normalization

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        if self.empirical_normalization:
            actor_obs = self.normalizer(actor_obs, update=False)
        return self.actor.act_inference({"actor_obs": actor_obs})


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading the checkpoint twice."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must lie in [0, 1], got {quantile}")
    if not sorted_values:
        raise ValueError("cannot summarize an empty latency sample")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summarize_latencies_ms(samples: list[float]) -> dict[str, float | int]:
    """Summarize samples using linear-interpolated percentiles."""
    if not samples:
        raise ValueError("cannot summarize an empty latency sample")
    if any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("latency samples must be finite and non-negative")
    ordered = sorted(float(value) for value in samples)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": _percentile(ordered, 0.5),
        "p95": _percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
        "population_std": statistics.pstdev(ordered),
    }


def _dataclass_kwargs(cls, values: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    return {key: value for key, value in values.items() if key in allowed}


def _tracker_actor_config(checkpoint: Mapping[str, Any], action_dim: int) -> tuple[ModuleConfig, dict[str, Any]]:
    try:
        raw_actor = dict(checkpoint["experiment_config"]["algo"]["config"]["module_dict"]["actor"])
    except (KeyError, TypeError) as exc:
        raise ValueError("tracker checkpoint does not contain its PPO actor configuration") from exc
    if raw_actor.get("type") != "MLP":
        raise ValueError(
            f"standalone latency loader currently supports the WBT MLP actor, got {raw_actor.get('type')!r}"
        )
    input_spec = list(raw_actor.get("input_dim", []))
    if input_spec != ["actor_obs"]:
        raise ValueError(f"expected the WBT actor input ['actor_obs'], got {input_spec}")

    raw_layer = dict(raw_actor.get("layer_config", {}))
    layer_config = LayerConfig(**_dataclass_kwargs(LayerConfig, raw_layer))
    output_spec = [action_dim if item == "robot_action_dim" else item for item in raw_actor["output_dim"]]
    module_config = ModuleConfig(
        **{
            **_dataclass_kwargs(ModuleConfig, raw_actor),
            "input_dim": input_spec,
            "output_dim": output_spec,
            "layer_config": layer_config,
        }
    )
    serializable = {
        "type": module_config.type,
        "input_dim": list(module_config.input_dim),
        "output_dim": list(module_config.output_dim),
        "layer_config": asdict(module_config.layer_config),
    }
    return module_config, serializable


def load_tracker_policy(checkpoint_path: str | Path, device: torch.device | str) -> LoadedTracker:
    """Reconstruct the inference-only PPO actor without creating an Isaac env."""
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_state = checkpoint.get("actor_model_state_dict")
    if not isinstance(actor_state, Mapping):
        raise ValueError("tracker checkpoint is missing actor_model_state_dict")
    input_weight = actor_state.get("actor_module.module.0.weight")
    action_std = actor_state.get("std")
    if not isinstance(input_weight, torch.Tensor) or input_weight.ndim != 2:
        raise ValueError("tracker checkpoint has no supported actor input layer")
    if not isinstance(action_std, torch.Tensor) or action_std.ndim != 1:
        raise ValueError("tracker checkpoint has no supported PPO action std")
    observation_dim = int(input_weight.shape[1])
    action_dim = int(action_std.shape[0])
    module_config, serializable_config = _tracker_actor_config(checkpoint, action_dim)

    try:
        algo_config = checkpoint["experiment_config"]["algo"]["config"]
    except (KeyError, TypeError) as exc:
        raise ValueError("tracker checkpoint does not contain its PPO config") from exc
    empirical_normalization = bool(algo_config.get("empirical_normalization", False))
    actor = setup_ppo_actor_module(
        obs_dim_dict={"actor_obs": observation_dim},
        module_config=module_config,
        num_actions=action_dim,
        init_noise_std=float(algo_config.get("init_noise_std", 1.0)),
        device=device,
        history_length={"actor_obs": 1},
    )
    actor.load_state_dict(actor_state, strict=True)

    if empirical_normalization:
        normalizer_state = checkpoint.get("actor_obs_normalizer_state_dict")
        if not isinstance(normalizer_state, Mapping):
            raise ValueError("empirical tracker checkpoint is missing actor observation normalizer")
        normalizer: nn.Module = EmpiricalNormalization(shape=observation_dim, device=device)
        normalizer.load_state_dict(normalizer_state, strict=True)
    else:
        normalizer = nn.Identity().to(device)

    policy = TrackerInferencePolicy(actor, normalizer, empirical_normalization)
    policy.eval().requires_grad_(False)
    iteration = int(checkpoint.get("iteration", checkpoint.get("iter", -1)))
    return LoadedTracker(
        policy=policy,
        observation_dim=observation_dim,
        action_dim=action_dim,
        empirical_normalization=empirical_normalization,
        checkpoint_iteration=iteration,
        actor_config=serializable_config,
    )


def _prepare_generator_input(
    generator: MotionGenerator,
    motion_clip_path: str | Path,
    motion_start: int,
    batch_size: int,
) -> tuple[MotionGeneratorInput, dict[str, Any]]:
    clip = load_wbt_motion(
        motion_clip_path,
        generator.layout,
        expected_fps=generator.cfg.data.fps,
    )
    past_frames = generator.cfg.data.past_frames
    if motion_start < 0 or motion_start + past_frames > clip.num_frames:
        raise ValueError(
            f"motion_start {motion_start} cannot provide {past_frames} past frames from a {clip.num_frames}-frame clip"
        )
    past = clip.features[motion_start : motion_start + past_frames].unsqueeze(0)
    past = past.repeat(batch_size, 1, 1).to(generator.device)

    anchor_quat = past[:, -1, generator.layout.root_quat_slice]
    anchor_yaw = quat_yaw(anchor_quat)
    heading = torch.stack([torch.cos(anchor_yaw), torch.sin(anchor_yaw)], dim=-1)

    terrain_source = "zero_flat_scan"
    if clip.terrain_scan is not None:
        terrain_frame = clip.terrain_scan[motion_start + past_frames - 1]
        if terrain_frame.numel() != generator.cfg.data.terrain_dim:
            generator_terrain_dim = generator.cfg.data.terrain_dim
            raise ValueError(
                f"clip terrain dimension {terrain_frame.numel()} != generator dimension {generator_terrain_dim}"
            )
        terrain = terrain_frame.unsqueeze(0).repeat(batch_size, 1).to(generator.device)
        terrain_source = "motion_clip_anchor_scan"
    else:
        terrain = torch.zeros(batch_size, generator.cfg.data.terrain_dim, device=generator.device)

    return (
        MotionGeneratorInput(past_motion=past, target_heading=heading, terrain_height=terrain),
        {
            "motion_clip": str(Path(motion_clip_path).resolve()),
            "motion_start": motion_start,
            "past_frames": past_frames,
            "feature_dim": generator.layout.dim,
            "terrain_dim": generator.cfg.data.terrain_dim,
            "terrain_source": terrain_source,
            "heading_source": "anchor_root_yaw",
        },
    )


def _tracker_input(tracker: LoadedTracker, batch_size: int, device: torch.device) -> tuple[torch.Tensor, str]:
    if tracker.empirical_normalization:
        mean = tracker.policy.normalizer.state_dict()["_mean"]
        return mean.expand(batch_size, -1).clone().to(device), "checkpoint_normalizer_mean"
    return torch.zeros(batch_size, tracker.observation_dim, device=device), "zeros"


def benchmark_cuda_callable(
    function: Callable[[], Any],
    *,
    warmup: int,
    measure: int,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    """Time a callable with CUDA events and a synchronized host clock."""
    if device.type != "cuda":
        raise ValueError(f"CUDA benchmark requires a CUDA device, got {device}")
    if warmup < 0 or measure <= 0:
        raise ValueError(f"warmup must be >= 0 and measure must be > 0, got {warmup}/{measure}")

    with torch.inference_mode():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize(device)

        cuda_samples: list[float] = []
        host_samples: list[float] = []
        for _ in range(measure):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter_ns()
            start.record()
            function()
            end.record()
            end.synchronize()
            host_samples.append((time.perf_counter_ns() - host_start) / 1.0e6)
            cuda_samples.append(float(start.elapsed_time(end)))

    return {
        "cuda_event_ms": summarize_latencies_ms(cuda_samples),
        "host_synchronized_ms": summarize_latencies_ms(host_samples),
    }


def _validate_args(args: Args) -> torch.device:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.ddim_steps != 2:
        raise ValueError("Stage-10 deployment benchmark requires exactly two DDIM steps")
    if args.replan_interval_policy_steps <= 0:
        raise ValueError("replan_interval_policy_steps must be positive")
    if args.sampling_seed < 0:
        raise ValueError("sampling_seed must be non-negative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"an available CUDA device is required, got {args.device!r}")
    return device


def run(args: Args) -> dict[str, Any]:
    """Load frozen modules, execute benchmarks, write JSON, and return it."""
    device = _validate_args(args)
    generator_path = Path(args.generator_checkpoint)
    tracker_path = Path(args.tracker_checkpoint)
    for label, path in (("generator", generator_path), ("tracker", tracker_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")

    generator = MotionGenerator.from_checkpoint(str(generator_path), device=str(device), use_ema=args.use_ema)
    trainable_generator_parameters = sum(
        parameter.numel() for parameter in generator.model.parameters() if parameter.requires_grad
    )
    if trainable_generator_parameters != 0:
        raise RuntimeError(f"generator is not frozen: {trainable_generator_parameters} trainable parameters")
    tracker = load_tracker_policy(tracker_path, device)
    generator_input, generator_input_metadata = _prepare_generator_input(
        generator,
        args.motion_clip,
        args.motion_start,
        args.batch_size,
    )
    tracker_observation, tracker_input_source = _tracker_input(tracker, args.batch_size, device)

    def generator_query():
        return generator.generate(
            generator_input,
            num_steps=args.ddim_steps,
            deterministic=args.deterministic_sampling,
            seed=args.sampling_seed,
        )

    def tracker_query():
        return tracker.policy(tracker_observation)

    def sequential_query():
        generated = generator_query()
        actions = tracker_query()
        return generated, actions

    timings = {
        "generator_2step": benchmark_cuda_callable(
            generator_query,
            warmup=args.warmup,
            measure=args.measure,
            device=device,
        ),
        "tracker_policy": benchmark_cuda_callable(
            tracker_query,
            warmup=args.warmup,
            measure=args.measure,
            device=device,
        ),
        "sequential_generator_plus_one_tracker": benchmark_cuda_callable(
            sequential_query,
            warmup=args.warmup,
            measure=args.measure,
            device=device,
        ),
    }
    generator_mean = timings["generator_2step"]["cuda_event_ms"]["mean"]
    tracker_mean = timings["tracker_policy"]["cuda_event_ms"]["mean"]
    amortized_mean = tracker_mean + generator_mean / args.replan_interval_policy_steps

    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    report: dict[str, Any] = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "runtime": "pytorch_cuda",
            "precision": str(next(generator.model.parameters()).dtype).replace("torch.", ""),
            "batch_size": args.batch_size,
            "warmup_iterations_per_workload": args.warmup,
            "measured_iterations_per_workload": args.measure,
            "ddim_steps": args.ddim_steps,
            "deterministic_sampling": args.deterministic_sampling,
            "sampling_seed": args.sampling_seed if args.deterministic_sampling else None,
            "replan_interval_policy_steps": args.replan_interval_policy_steps,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "hardware": {
            "device": str(device),
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "generator": {
            "checkpoint": str(generator_path.resolve()),
            "checkpoint_sha256": file_sha256(generator_path),
            "checkpoint_size_bytes": generator_path.stat().st_size,
            "checkpoint_step": generator.checkpoint_step,
            "weights": "ema" if args.use_ema else "raw_model",
            "trainable_parameters": trainable_generator_parameters,
            "model_parameters": sum(parameter.numel() for parameter in generator.model.parameters()),
            "future_frames": generator.cfg.data.future_frames,
            "model_config": asdict(generator.cfg.model),
            "input": generator_input_metadata,
        },
        "tracker": {
            "checkpoint": str(tracker_path.resolve()),
            "checkpoint_sha256": file_sha256(tracker_path),
            "checkpoint_size_bytes": tracker_path.stat().st_size,
            "checkpoint_iteration": tracker.checkpoint_iteration,
            "trainable_parameters": sum(
                parameter.numel() for parameter in tracker.policy.parameters() if parameter.requires_grad
            ),
            "observation_dim": tracker.observation_dim,
            "action_dim": tracker.action_dim,
            "empirical_normalization": tracker.empirical_normalization,
            "input_source": tracker_input_source,
            "actor_config": tracker.actor_config,
        },
        "timings": timings,
        "derived": {
            "estimated_amortized_cuda_mean_ms_per_policy_step": amortized_mean,
            "formula": "tracker_policy.mean + generator_2step.mean / replan_interval_policy_steps",
        },
        "scope": {
            "inputs_preallocated_on_gpu": True,
            "includes_generator_canonicalization_and_output_unpack": True,
            "includes_tracker_empirical_normalization": tracker.empirical_normalization,
            "includes_simulator_step": False,
            "includes_terrain_height_scan": False,
            "includes_observation_assembly": False,
            "includes_reference_window_buffer_updates": False,
            "sequential_measurement_is_functional_closed_loop": False,
            "onnx_or_tensorrt": False,
        },
        "argv": sys.argv,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = tyro.cli(Args)
    report = run(args)
    summary = {
        name: values["cuda_event_ms"]
        for name, values in report["timings"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
