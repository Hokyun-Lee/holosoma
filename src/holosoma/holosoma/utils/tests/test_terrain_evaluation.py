from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from holosoma.agents.callbacks.terrain_metrics import TerrainMetricsCallback
from holosoma.config_types.command import GeneratedMotionConfig
from holosoma.config_types.eval_callback import TerrainMetricsConfig
from holosoma.utils.terrain_evaluation import TerrainEvaluationAccumulator, write_terrain_evaluation_outputs


def _start_two(accumulator: TerrainEvaluationAccumulator) -> None:
    accumulator.start_episodes(
        [0, 1],
        target_headings=np.array([[1.0, 0.0], [0.0, 2.0]]),
        terrain_types=["box", "stair"],
        terrain_levels=[3, 3],
    )


def test_accumulator_signed_progress_outcomes_and_raw_denominators() -> None:
    accumulator = TerrainEvaluationAccumulator(num_envs=2, target_episodes=2, success_distance_m=1.5)
    _start_two(accumulator)
    accumulator.observe(
        forward_progress_m=np.array([1.6, -2.0]),
        falls=np.array([False, True]),
        terminated=np.array([False, True]),
        timeouts=np.array([True, False]),
        metrics={
            "motion/heading_error_rad": np.array([0.1, 1.2]),
            "terrain/undesired_contact_any": np.array([1.0, 0.0]),
        },
    )

    assert accumulator.complete
    first, second = accumulator.records
    assert first.success and first.survival and first.undesired_contact
    assert not second.success and second.fall and second.bad_tracking
    overall = accumulator.summary()["overall"]
    assert overall["rates"]["episode/success_rate"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }
    heading = overall["step_metrics"]["motion/heading_error_rad"]
    assert heading["denominator"] == 2
    assert np.isclose(heading["numerator"], 1.3)


def test_lateral_and_backward_progress_never_cross() -> None:
    accumulator = TerrainEvaluationAccumulator(num_envs=2, target_episodes=2, success_distance_m=1.5)
    _start_two(accumulator)
    accumulator.observe(
        # These are already target-heading signed projections: lateral is 0,
        # backward is negative and is clamped out of the episode maximum.
        forward_progress_m=np.array([0.0, -2.0]),
        falls=np.zeros(2, dtype=np.bool_),
        terminated=np.zeros(2, dtype=np.bool_),
        timeouts=np.ones(2, dtype=np.bool_),
        metrics={},
    )
    assert not any(record.success for record in accumulator.records)
    assert [record.max_forward_progress_m for record in accumulator.records] == [0.0, 0.0]


def test_async_fast_failure_type_cannot_exceed_balanced_quota() -> None:
    accumulator = TerrainEvaluationAccumulator(
        num_envs=2,
        target_episodes=4,
        success_distance_m=1.5,
        expected_terrain_types=("box", "stair"),
    )
    _start_two(accumulator)

    # Box fails twice while stair is still in its first long episode.
    for _ in range(2):
        accumulator.observe(
            forward_progress_m=np.zeros(2),
            falls=np.zeros(2, dtype=np.bool_),
            terminated=np.array([True, False]),
            timeouts=np.zeros(2, dtype=np.bool_),
            metrics={},
        )
        accumulator.start_episodes(
            [0],
            target_headings=np.array([[1.0, 0.0]]),
            terrain_types=["box"],
            terrain_levels=[3],
        )

    assert accumulator.completed_per_terrain_type == {"box": 2, "stair": 0}
    # A third box episode is not started or counted after its quota is full.
    assert not accumulator.active_mask[0]

    for _ in range(2):
        accumulator.observe(
            forward_progress_m=np.array([0.0, 2.0]),
            falls=np.zeros(2, dtype=np.bool_),
            terminated=np.zeros(2, dtype=np.bool_),
            timeouts=np.array([False, True]),
            metrics={},
        )
        accumulator.start_episodes(
            [1],
            target_headings=np.array([[0.0, 1.0]]),
            terrain_types=["stair"],
            terrain_levels=[3],
        )

    assert accumulator.complete
    assert accumulator.completed_per_terrain_type == {"box": 2, "stair": 2}
    summary = accumulator.summary()
    assert summary["requested_per_terrain_type"] == {"box": 2, "stair": 2}
    assert summary["completed_per_terrain_type"] == {"box": 2, "stair": 2}


def test_simultaneous_same_type_completions_are_clipped_to_quota() -> None:
    accumulator = TerrainEvaluationAccumulator(
        num_envs=4,
        target_episodes=2,
        success_distance_m=1.0,
        expected_terrain_types=("box", "stair"),
    )
    accumulator.start_episodes(
        [0, 1, 2, 3],
        target_headings=np.tile(np.array([[1.0, 0.0]]), (4, 1)),
        terrain_types=["box", "box", "stair", "stair"],
        terrain_levels=[0, 0, 0, 0],
    )
    accumulator.observe(
        forward_progress_m=np.ones(4),
        falls=np.zeros(4, dtype=np.bool_),
        terminated=np.zeros(4, dtype=np.bool_),
        timeouts=np.ones(4, dtype=np.bool_),
        metrics={},
    )
    assert accumulator.complete
    assert len(accumulator.records) == 2
    assert accumulator.completed_per_terrain_type == {"box": 1, "stair": 1}


def test_outputs_include_raw_json_and_csv(tmp_path: Path) -> None:
    accumulator = TerrainEvaluationAccumulator(num_envs=1, target_episodes=1, success_distance_m=1.0)
    accumulator.start_episodes(
        [0],
        target_headings=np.array([[1.0, 0.0]]),
        terrain_types=["hurdle"],
        terrain_levels=[2],
    )
    accumulator.observe(
        forward_progress_m=np.array([1.2]),
        falls=np.array([False]),
        terminated=np.array([False]),
        timeouts=np.array([True]),
        metrics={"terrain/robot_body_origin_penetration_mean_m": np.array([0.03])},
    )
    paths = write_terrain_evaluation_outputs(
        accumulator=accumulator,
        output_prefix=tmp_path / "result",
        metadata={"variant": "D", "checkpoint_sha256": "abc", "config": {"seed": 7}},
    )
    payload = json.loads(paths["json"].read_text())
    assert "not collision-shape" in payload["metric_definition"]["body_origin_penetration_proxy"]
    assert payload["summary"]["overall"]["rates"]["episode/success_rate"]["numerator"] == 1
    with paths["summary_csv"].open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    quota = next(row for row in rows if row["group"] == "overall" and row["metric"] == "episode/completion_quota")
    assert quota["numerator"] == "1"
    assert quota["denominator"] == "1"
    success = next(row for row in rows if row["group"] == "overall" and row["metric"] == "episode/success_rate")
    assert success["numerator"] == "1"
    assert success["denominator"] == "1"
    assert paths["episodes_csv"].is_file()


class _FakeTerrainState:
    def __init__(self) -> None:
        self.terrain = SimpleNamespace(curriculum_enabled=True)
        self.terrain_type_ids = torch.tensor([0])
        self.terrain_type_names = ("box",)
        self.terrain_levels = torch.tensor([2])
        self.num_curriculum_levels = 4
        self.set_calls: list[int] = []

    def set_curriculum_origins(self, env_ids: torch.Tensor, levels) -> None:
        levels_tensor = torch.as_tensor(levels, dtype=torch.long).reshape(-1)
        if levels_tensor.numel() == 1:
            levels_tensor = levels_tensor.expand(env_ids.numel())
        self.terrain_levels[env_ids] = levels_tensor
        self.set_calls.append(int(levels_tensor[0]))

    def query_terrain_heights(self, xy: torch.Tensor) -> torch.Tensor:
        return torch.zeros(xy.shape[0])


class _FakeCurriculum:
    def __init__(self) -> None:
        self.enabled = False
        self.initial_level = 0
        self.min_level = 0
        self.max_level = 3
        self.episode_start_root_xy = torch.zeros(1, 2)
        self.episode_target_heading_w = torch.tensor([[1.0, 0.0]])


class _FakeRewardManager:
    active_terms: tuple[str, ...] = ()


def _fake_callback_runtime(tmp_path: Path):
    terrain_state = _FakeTerrainState()
    curriculum = _FakeCurriculum()
    generated_cfg = GeneratedMotionConfig(
        motion_file="seed.npz",
        body_name_ref=["torso"],
        body_names_to_track=["torso"],
        generator_checkpoint="generator.pt",
        deterministic_sampling=False,
        sampling_seed=99,
    )
    command = SimpleNamespace(
        motion_cfg=generated_cfg,
        gen_cfg=generated_cfg,
        target_heading_w=torch.tensor([[1.0, 0.0]]),
        robot_body_pos_w=torch.tensor([[[0.0, 0.0, 0.8]]]),
        body_pos_relative_w=torch.tensor([[[0.0, 0.0, 0.8]]]),
        metrics={
            "motion/error_ref_pos": torch.tensor([0.2]),
            "motion/error_ref_rot": torch.tensor([0.1]),
            "motion/error_ref_lin_vel": torch.tensor([0.3]),
            "motion/error_body_pos": torch.tensor([0.4]),
            "motion/error_body_lin_vel": torch.tensor([0.5]),
            "motion/error_joint_pos": torch.tensor([0.6]),
            "motion/error_joint_vel": torch.tensor([0.7]),
        },
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        simulator=SimpleNamespace(
            robot_root_states=torch.tensor([[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        ),
        base_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        command_manager=SimpleNamespace(get_state=lambda _: command),
        terrain_manager=SimpleNamespace(get_state=lambda _: terrain_state),
        curriculum_manager=SimpleNamespace(get_term=lambda _: curriculum),
        termination_manager=SimpleNamespace(
            terminated=torch.tensor([False]),
            time_outs=torch.tensor([False]),
        ),
        reward_manager=_FakeRewardManager(),
        _update_log_dict=lambda: None,
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"tracker")
    generator = tmp_path / "generator.pt"
    generator.write_bytes(b"generator")
    # Callback hashes the configured generator path relative to cwd. Use the
    # absolute fake path for this isolated test.
    generated_cfg = dataclasses.replace(generated_cfg, generator_checkpoint=str(generator))
    command.motion_cfg = generated_cfg
    command.gen_cfg = generated_cfg
    loop = SimpleNamespace(
        device="cpu",
        log_dir=str(tmp_path),
        _unwrap_env=lambda: env,
        _evaluation_checkpoint_path=str(checkpoint),
        _evaluation_config={"training": {"seed": 7}},
    )
    return env, command, terrain_state, curriculum, loop


def test_fake_env_callback_forces_runtime_controls_and_captures_terminal_pose(tmp_path: Path) -> None:
    env, command, terrain_state, curriculum, loop = _fake_callback_runtime(tmp_path)
    callback = TerrainMetricsCallback(
        TerrainMetricsConfig(
            enabled=True,
            output_prefix=str(tmp_path / "callback"),
            variant="C",
            episode_count=1,
            success_distance_m=1.5,
            fixed_terrain_level=1,
            deterministic_generator=True,
            generator_sampling_seed=7,
        ),
        loop,
    )
    callback.on_pre_evaluate_policy()
    assert terrain_state.terrain_levels.tolist() == [1]
    assert curriculum.enabled and curriculum.min_level == curriculum.max_level == 1
    assert command.gen_cfg.deterministic_sampling and command.gen_cfg.sampling_seed == 7

    actor_state = callback.on_pre_eval_env_step({"stop": False})
    assert not actor_state["stop"]
    # The wrapped hook sees this terminal pose before any reset can overwrite it.
    env.simulator.robot_root_states[0, 0] = 1.6
    env.termination_manager.time_outs[0] = True
    env._update_log_dict()
    actor_state = callback.on_post_eval_env_step(actor_state)
    assert actor_state["stop"]
    assert callback.accumulator.records[0].max_forward_progress_m == pytest.approx(1.6)
    assert callback.accumulator.records[0].success
    callback.on_post_evaluate_policy()

    payload = json.loads((tmp_path / "callback.json").read_text())
    assert payload["metadata"]["deterministic_generator"] is True
    # Runtime controls are restored for callers that reuse one algorithm object.
    assert command.gen_cfg.sampling_seed == 99
    assert terrain_state.terrain_levels.tolist() == [2]
    assert not curriculum.enabled
