from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.motion_gen.features import FeatureLayout
from holosoma.motion_gen.sampling import MotionGenerator, MotionGeneratorOutput
from holosoma.motion_gen.scripts.sample import _resolve_replan_stride, _rollout_stem


def test_sample_rollout_resolves_full_horizon_default_in_artifact_name() -> None:
    resolved = _resolve_replan_stride(None, future_frames=25)
    assert resolved == 25
    assert (
        _rollout_stem(
            "climb",
            start=100,
            num_cycles=17,
            resolved_stride=resolved,
        )
        == "climb_s100_rollout17x25"
    )


def test_sample_rollout_keeps_explicit_shorter_stride() -> None:
    assert _resolve_replan_stride(12, future_frames=25) == 12


@pytest.mark.parametrize("requested", [0, 26])
def test_sample_rollout_rejects_invalid_stride(requested: int) -> None:
    with pytest.raises(ValueError, match="replan_stride must be"):
        _resolve_replan_stride(requested, future_frames=25)


class _YawChangingGenerator:
    receding_horizon = MotionGenerator.receding_horizon

    def __init__(self) -> None:
        self.layout = FeatureLayout(
            joint_names=("joint",),
            body_names=("body",),
            bone_pairs=(),
        )
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(data=SimpleNamespace(future_frames=3))
        self.headings: list[torch.Tensor] = []

    def generate(
        self,
        inp,
        *,
        num_steps,
        deterministic,
        seed,
    ) -> MotionGeneratorOutput:
        del num_steps, deterministic, seed
        self.headings.append(inp.target_heading.clone())
        future = inp.past_motion[:, -1:].expand(-1, 3, -1).clone()
        half_sqrt = 2.0**-0.5
        future[..., self.layout.root_quat_slice] = torch.tensor([half_sqrt, 0.0, 0.0, half_sqrt])
        return MotionGeneratorOutput(
            root_pos=future[..., self.layout.root_pos_slice],
            root_quat=future[..., self.layout.root_quat_slice],
            joint_pos=future[..., self.layout.joint_pos_slice],
            body_pos=future[..., self.layout.body_pos_slice].reshape(1, 3, 1, 3),
            features=future,
        )


def _identity_past(generator: _YawChangingGenerator) -> torch.Tensor:
    past = torch.zeros(2, generator.layout.dim)
    past[:, generator.layout.root_quat_slice.start] = 1.0
    return past


def test_receding_horizon_none_heading_is_fixed_from_initial_anchor() -> None:
    generator = _YawChangingGenerator()
    generator.receding_horizon(
        _identity_past(generator),
        num_cycles=2,
        replan_stride=2,
        target_heading=None,
        num_steps=2,
        deterministic=True,
    )

    assert len(generator.headings) == 2
    torch.testing.assert_close(generator.headings[0], torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(generator.headings[1], generator.headings[0])


def test_receding_horizon_callable_heading_remains_cycle_dependent() -> None:
    generator = _YawChangingGenerator()
    requested = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    generator.receding_horizon(
        _identity_past(generator),
        num_cycles=2,
        replan_stride=2,
        target_heading=lambda cycle: requested[cycle],
        num_steps=2,
        deterministic=True,
    )

    torch.testing.assert_close(generator.headings[0], requested[0].unsqueeze(0))
    torch.testing.assert_close(generator.headings[1], requested[1].unsqueeze(0))
