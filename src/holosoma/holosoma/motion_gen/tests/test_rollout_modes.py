from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.motion_gen.features import FeatureLayout
from holosoma.motion_gen.scripts.evaluate_rollout_modes import (
    Args,
    build_comparisons,
    generate_variant,
    oracle_cycle_heading,
)


def _layout() -> FeatureLayout:
    return FeatureLayout(joint_names=("joint",), body_names=("body",), bone_pairs=())


def _source(frames: int = 20) -> torch.Tensor:
    layout = _layout()
    features = torch.zeros(frames, layout.dim)
    features[:, 0] = torch.arange(frames, dtype=torch.float32)
    features[:, 3] = 1.0
    features[:, layout.body_pos_slice.start] = features[:, 0]
    return features


class _FakeGenerator:
    def __init__(self) -> None:
        self.layout = _layout()
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(
            data=SimpleNamespace(
                past_frames=2,
                future_frames=4,
                min_heading_disp=0.05,
                terrain_dim=3,
            )
        )
        self.calls: list[dict] = []

    def generate(self, inp, *, num_steps, deterministic, seed, guidance_scale):
        self.calls.append(
            {
                "anchor_x": float(inp.past_motion[0, -1, 0]),
                "heading": None if inp.target_heading is None else inp.target_heading.clone(),
                "seed": seed,
                "num_steps": num_steps,
                "deterministic": deterministic,
                "guidance_scale": guidance_scale,
            }
        )
        anchor = inp.past_motion[:, -1:].clone()
        future = anchor.expand(-1, self.cfg.data.future_frames, -1).clone()
        increments = torch.arange(1, self.cfg.data.future_frames + 1).float() * 0.25
        future[0, :, 0] += increments
        future[0, :, self.layout.body_pos_slice.start] = future[0, :, 0]
        return SimpleNamespace(features=future)


def _terrain_scan(past: torch.Tensor) -> torch.Tensor:
    return past[:, -1, :1].expand(-1, 3).clone()


def test_oracle_cycle_heading_uses_source_displacement_and_stationary_yaw() -> None:
    layout = _layout()
    source = _source()
    heading = oracle_cycle_heading(
        source,
        layout,
        cycle_start=0,
        past_frames=2,
        future_frames=4,
        min_displacement_m=0.05,
    )
    torch.testing.assert_close(heading, torch.tensor([1.0, 0.0]))

    source[:, :3] = 0.0
    yaw = torch.tensor(torch.pi / 2)
    source[1, 3:7] = torch.tensor([torch.cos(yaw / 2), 0.0, 0.0, torch.sin(yaw / 2)])
    fallback = oracle_cycle_heading(
        source,
        layout,
        cycle_start=0,
        past_frames=2,
        future_frames=4,
        min_displacement_m=0.05,
    )
    torch.testing.assert_close(fallback, torch.tensor([0.0, 1.0]), atol=1.0e-6, rtol=0.0)


def test_generated_feedback_and_source_history_use_distinct_cycle_inputs() -> None:
    source = _source()
    generator = _FakeGenerator()
    feedback = generate_variant(
        generator,
        source,
        start=0,
        num_cycles=2,
        replan_stride=2,
        seed=7,
        num_steps=2,
        guidance_scale=None,
        history_mode="generated_feedback",
        heading_mode="keep_current",
        terrain_scan_fn=_terrain_scan,
    )
    assert feedback.features.shape == (6, generator.layout.dim)
    assert [call["seed"] for call in generator.calls] == [7, 8]
    assert [call["anchor_x"] for call in generator.calls] == pytest.approx([1.0, 1.5])
    assert all(call["heading"] is not None for call in generator.calls)
    torch.testing.assert_close(generator.calls[0]["heading"], generator.calls[1]["heading"])
    torch.testing.assert_close(generator.calls[0]["heading"], torch.tensor([[1.0, 0.0]]))
    assert feedback.report["cycles"][1]["history_anchor_root_error_m"] == pytest.approx(1.5)
    assert feedback.report["cycles"][1]["terrain_scan_source_rmse_m"] == pytest.approx(1.5)

    generator.calls.clear()
    teacher_forced = generate_variant(
        generator,
        source,
        start=0,
        num_cycles=2,
        replan_stride=2,
        seed=7,
        num_steps=2,
        guidance_scale=None,
        history_mode="source_history",
        heading_mode="oracle_cycle",
        terrain_scan_fn=_terrain_scan,
    )
    assert [call["anchor_x"] for call in generator.calls] == pytest.approx([1.0, 3.0])
    assert all(call["heading"] is not None for call in generator.calls)
    assert all(cycle["history_anchor_root_error_m"] == 0.0 for cycle in teacher_forced.report["cycles"])
    assert all(cycle["terrain_scan_source_rmse_m"] == 0.0 for cycle in teacher_forced.report["cycles"])
    torch.testing.assert_close(feedback.features[:4], teacher_forced.features[:4])
    assert not torch.equal(feedback.features[4:], teacher_forced.features[4:])


def test_rollout_defaults_match_current_production_replan_contract() -> None:
    args = Args()
    assert args.replan_stride == 25
    assert args.num_cycles == 17
    assert args.output_dir.endswith("climb09_rollout_modes_production_2step")


def test_generate_variant_rejects_source_without_full_oracle_horizon() -> None:
    with pytest.raises(ValueError, match="requires frame"):
        generate_variant(
            _FakeGenerator(),
            _source(frames=7),
            start=0,
            num_cycles=2,
            replan_stride=2,
            seed=0,
            num_steps=2,
            guidance_scale=None,
            history_mode="source_history",
            heading_mode="oracle_cycle",
            terrain_scan_fn=_terrain_scan,
        )


def test_build_comparisons_has_controlled_signed_deltas() -> None:
    def variant(value: float) -> dict:
        return {
            "aggregate_source_alignment": {
                "root_position_error_m_mean": value,
                "root_position_error_m_final": value + 1.0,
                "joint_l2_error_rad_mean": value + 2.0,
                "body_mpjpe_m_mean": value + 3.0,
            },
            "mujoco_feasibility": None,
        }

    variants = {
        "generated_feedback__keep_current": variant(4.0),
        "source_history__keep_current": variant(1.0),
        "generated_feedback__oracle_cycle": variant(2.0),
        "source_history__oracle_cycle": variant(0.5),
    }
    comparison = build_comparisons(variants)
    history_delta = comparison["generated_feedback_minus_source_history"]["keep_current"]["metric_delta"]
    assert history_delta["root_position_error_m_mean"] == pytest.approx(3.0)
    heading_delta = comparison["oracle_cycle_minus_keep_current"]["generated_feedback"]["metric_delta"]
    assert heading_delta["body_mpjpe_m_mean"] == pytest.approx(-2.0)
