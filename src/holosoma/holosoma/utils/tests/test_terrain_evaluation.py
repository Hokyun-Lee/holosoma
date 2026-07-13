from __future__ import annotations

import csv
import dataclasses
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from holosoma.agents.callbacks.terrain_metrics import TerrainMetricsCallback
from holosoma.config_types.command import GeneratedMotionConfig
from holosoma.config_types.eval_callback import TerrainMetricsConfig
from holosoma.motion_gen.terrain import ScanGrid
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
        self.reference_height = 0.0
        self.robot_height = 0.0
        self.local_height_scan_configured = False

    def set_curriculum_origins(self, env_ids: torch.Tensor, levels) -> None:
        levels_tensor = torch.as_tensor(levels, dtype=torch.long).reshape(-1)
        if levels_tensor.numel() == 1:
            levels_tensor = levels_tensor.expand(env_ids.numel())
        self.terrain_levels[env_ids] = levels_tensor
        self.set_calls.append(int(levels_tensor[0]))

    def query_terrain_heights(self, xy: torch.Tensor) -> torch.Tensor:
        reference = torch.full(
            (xy.shape[0],),
            self.reference_height,
            dtype=xy.dtype,
            device=xy.device,
        )
        robot = torch.full_like(reference, self.robot_height)
        return torch.where(xy[:, 0] > 0.5, reference, robot)


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


class _FakeRootStatesProxy:
    """Isaac-Sim-shaped boundary: indexable, but no tensor ``dtype``."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __setitem__(self, index, value):
        self.tensor[index] = value


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
        sampling_counter_reset_count=0,
    )
    command.reset_deterministic_sampling_counters = lambda: setattr(
        command,
        "sampling_counter_reset_count",
        command.sampling_counter_reset_count + 1,
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        simulator=SimpleNamespace(
            robot_root_states=_FakeRootStatesProxy(
                torch.tensor([[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            )
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
        episode_length_buf=torch.tensor([0]),
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
    assert command.gen_cfg.deterministic_per_env_sampling
    assert command.gen_cfg.evaluation_phase_mode == "uniform"
    assert command.gen_cfg.reanchor_motion_xy_on_reset
    assert command.gen_cfg.phase_horizon_steps == 500
    assert command.sampling_counter_reset_count == 1

    actor_state = callback.on_pre_eval_env_step(
        {"stop": False, "step": 0, "actions": torch.zeros(1, 29)}
    )
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
    assert payload["metadata"]["deterministic_per_env_sampling"] is True
    assert payload["metadata"]["evaluation_phase_mode"] == "uniform"
    assert payload["metadata"]["reanchor_motion_xy_on_reset"] is True
    assert payload["metadata"]["phase_horizon_steps"] == 500
    assert payload["metadata"]["first_correction_exemplar"]["found"] is False
    assert payload["metadata"]["first_correction_exemplar"]["path"] is None
    assert not (tmp_path / "callback_first_correction_exemplar.npz").exists()
    # Runtime controls are restored for callers that reuse one algorithm object.
    assert command.gen_cfg.sampling_seed == 99
    assert not command.gen_cfg.deterministic_per_env_sampling
    assert command.gen_cfg.evaluation_phase_mode == "zero"
    assert not command.gen_cfg.reanchor_motion_xy_on_reset
    assert command.gen_cfg.phase_horizon_steps == 0
    assert terrain_state.terrain_levels.tolist() == [2]
    assert not curriculum.enabled


def test_callback_reseeds_after_setup_reset_independent_of_variant_rng_history(tmp_path: Path) -> None:
    samples = []
    for variant, burn_count in (("B", 3), ("D", 19)):
        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)
        torch.rand(burn_count)
        np.random.rand(burn_count)
        for _ in range(burn_count):
            random.random()
        _env, command, _terrain, _curriculum, loop = _fake_callback_runtime(tmp_path)
        callback = TerrainMetricsCallback(
            TerrainMetricsConfig(
                enabled=True,
                variant=variant,
                episode_count=1,
                evaluation_seed=71,
            ),
            loop,
        )
        callback.on_pre_evaluate_policy()
        samples.append((torch.rand(4), np.random.rand(4), [random.random() for _ in range(4)]))
        assert command.sampling_counter_reset_count == 1

    torch.testing.assert_close(samples[0][0], samples[1][0])
    np.testing.assert_array_equal(samples[0][1], samples[1][1])
    assert samples[0][2] == samples[1][2]


def test_first_correction_exemplar_threshold_one_shot_and_serialization(tmp_path: Path) -> None:
    env, command, terrain_state, _curriculum, loop = _fake_callback_runtime(tmp_path)
    command.robot_body_pos_w = torch.tensor([[[0.0, 0.0, 0.0]]])
    command.body_pos_relative_w = torch.tensor([[[1.0, 0.0, 0.0]]])
    grid = ScanGrid(x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.1, spacing=0.1)
    terrain_state.local_height_scan_configured = True
    terrain_state._local_scan_grid = grid
    terrain_state._local_height_scan = torch.tensor([[0.0, 0.1, 0.2, 0.3]])
    terrain_state._local_height_scan_valid = torch.tensor([True])
    terrain_state._local_scan_root_xy = torch.tensor([[0.2, -0.1]])
    terrain_state._local_scan_root_yaw = torch.tensor([0.3])
    terrain_state._local_scan_world_xy = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    terrain_state.local_height_scan = terrain_state._local_height_scan
    terrain_state.local_height_scan_valid = terrain_state._local_height_scan_valid
    terrain_state.local_height_scan_root_xy = terrain_state._local_scan_root_xy
    terrain_state.local_height_scan_root_yaw = terrain_state._local_scan_root_yaw

    callback = TerrainMetricsCallback(
        TerrainMetricsConfig(
            enabled=True,
            output_prefix=str(tmp_path / "correction"),
            variant="D",
            episode_count=1,
            body_origin_penetration_threshold_m=0.02,
            body_origin_correction_min_improvement_m=0.01,
        ),
        loop,
    )
    callback.on_pre_evaluate_policy()

    def step(*, step_index: int, reference_height: float, robot_height: float, action: float) -> None:
        terrain_state.reference_height = reference_height
        terrain_state.robot_height = robot_height
        env.episode_length_buf[0] = step_index + 1
        callback.on_pre_eval_env_step(
            {
                "stop": False,
                "step": step_index,
                "actions": torch.full((1, 29), action),
            }
        )
        env._update_log_dict()

    # Below reference threshold, then below improvement threshold: neither qualifies.
    step(step_index=0, reference_height=0.019, robot_height=0.0, action=0.0)
    assert callback._correction_exemplar is None
    step(step_index=1, reference_height=0.03, robot_height=0.025, action=1.0)
    assert callback._correction_exemplar is None

    # First qualifying exact frame is retained even when a stronger case follows.
    step(step_index=2, reference_height=0.03, robot_height=0.01, action=2.0)
    assert callback._correction_exemplar is not None
    step(step_index=3, reference_height=0.08, robot_height=0.0, action=3.0)
    env.termination_manager.time_outs[0] = True
    step(step_index=4, reference_height=0.09, robot_height=0.0, action=4.0)
    callback.on_post_evaluate_policy()

    exemplar_path = tmp_path / "correction_first_correction_exemplar.npz"
    assert exemplar_path.is_file()
    with np.load(exemplar_path, allow_pickle=False) as exemplar:
        assert exemplar["evaluation_step"].item() == 2
        assert exemplar["episode_step"].item() == 3
        assert exemplar["env_id"].item() == 0
        assert exemplar["terrain_type"].item() == "box"
        assert exemplar["terrain_level"].item() == 0
        assert exemplar["target_heading_w"].shape == (2,)
        assert exemplar["action"].shape == (29,)
        assert np.all(exemplar["action"] == 2.0)
        assert exemplar["root_state_w"].shape == (13,)
        assert exemplar["robot_body_pos_w"].shape == (1, 3)
        assert exemplar["reference_body_pos_w"].shape == (1, 3)
        np.testing.assert_allclose(exemplar["reference_body_terrain_height_w"], [0.03])
        np.testing.assert_allclose(exemplar["robot_body_terrain_height_w"], [0.01])
        np.testing.assert_allclose(exemplar["reference_body_origin_penetration_m"], [0.03])
        np.testing.assert_allclose(exemplar["robot_body_origin_penetration_m"], [0.01])
        assert exemplar["reference_max_body_origin_penetration_m"].item() == pytest.approx(0.03)
        assert exemplar["robot_max_body_origin_penetration_m"].item() == pytest.approx(0.01)
        assert exemplar["reference_minus_robot_max_body_origin_penetration_m"].item() == pytest.approx(0.02)
        assert exemplar["correction_proxy_case"].item() is True
        assert exemplar["local_scan_height_w"].shape == (4,)
        assert exemplar["local_scan_local_xy"].shape == (4, 2)
        assert exemplar["local_scan_world_xy"].shape == (4, 2)
        np.testing.assert_allclose(exemplar["local_scan_root_xy_w"], [0.2, -0.1])
        assert exemplar["local_scan_root_yaw_w"].item() == pytest.approx(0.3)
        assert "not collision-shape" in exemplar["proxy_limitation"].item()
        assert "quaternion_xyzw" in exemplar["root_state_layout"].item()

    payload = json.loads((tmp_path / "correction.json").read_text())
    exemplar_metadata = payload["metadata"]["first_correction_exemplar"]
    assert exemplar_metadata["found"] is True
    assert Path(exemplar_metadata["path"]) == exemplar_path.resolve()
    assert exemplar_metadata["selection_condition"] == {
        "reference_max_body_origin_penetration_m_gte": 0.02,
        "reference_minus_robot_max_body_origin_penetration_m_gte": 0.01,
    }
    assert "not proof" in exemplar_metadata["limitation"]
    step_metrics = payload["summary"]["overall"]["step_metrics"]
    assert step_metrics["terrain/tracker_body_origin_correction_case"]["max"] == 1.0
    assert "terrain/tracker_body_origin_correction_proxy_m" in step_metrics
