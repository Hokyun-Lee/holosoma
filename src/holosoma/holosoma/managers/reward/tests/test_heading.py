import math

import pytest
import torch
from holosoma.managers.reward.terms.heading import velocity_heading_error_rad, velocity_heading_reward


def test_velocity_heading_reward_exact_equation_and_directions():
    velocity = torch.tensor(
        [
            [3.0, 4.0],
            [-2.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ]
    )
    direction = torch.tensor([[1.0, 0.0]]).expand_as(velocity)
    epsilon = 1.0e-6

    reward = velocity_heading_reward(velocity, direction, epsilon=epsilon)

    expected = torch.tensor([3.0 / (5.0 + epsilon), -2.0 / (2.0 + epsilon), 0.0, 0.0])
    torch.testing.assert_close(reward, expected)
    assert reward.shape == (4,)
    assert torch.isfinite(reward).all()


def test_velocity_heading_error_and_stopped_convention():
    velocity = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.01, 0.0]])
    direction = torch.tensor([[1.0, 0.0]]).expand_as(velocity)

    error = velocity_heading_error_rad(velocity, direction, speed_threshold=0.05)

    torch.testing.assert_close(error, torch.tensor([0.0, math.pi, math.pi / 2.0, math.pi / 2.0]))


@pytest.mark.parametrize("epsilon", [0.0, -1.0])
def test_velocity_heading_reward_rejects_non_positive_epsilon(epsilon: float):
    value = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="epsilon must be positive"):
        velocity_heading_reward(value, value, epsilon=epsilon)


def test_heading_helpers_reject_shape_mismatch():
    with pytest.raises(ValueError, match=r"same .* shape"):
        velocity_heading_reward(torch.zeros(2, 2), torch.zeros(1, 2), epsilon=1.0e-6)
