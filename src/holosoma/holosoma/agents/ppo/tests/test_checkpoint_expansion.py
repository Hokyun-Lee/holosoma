import pytest
import torch
from holosoma.agents.ppo.ppo import (
    EmpiricalNormalization,
    _load_module_with_input_expansion,
    _load_normalizer_with_input_expansion,
    _load_optimizer_with_input_expansion,
)
from torch import nn


class _Actor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 4):
        super().__init__()
        self.actor_module = nn.Module()
        self.actor_module.module = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.std = nn.Parameter(torch.ones(2))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_module.module(obs)


def _migrate_actor(old: _Actor, new: _Actor) -> None:
    _load_module_with_input_expansion(
        new,
        old.state_dict(),
        allowed_weight_key="actor_module.module.0.weight",
        label="actor",
    )


def test_model_input_suffix_expansion_preserves_outputs():
    torch.manual_seed(7)
    old = _Actor(3)
    new = _Actor(5)
    old_state = {key: value.clone() for key, value in old.state_dict().items()}
    _migrate_actor(old, new)

    weight = new.state_dict()["actor_module.module.0.weight"]
    torch.testing.assert_close(weight[:, :3], old_state["actor_module.module.0.weight"])
    torch.testing.assert_close(weight[:, 3:], torch.zeros_like(weight[:, 3:]))
    for key, value in old_state.items():
        if key != "actor_module.module.0.weight":
            torch.testing.assert_close(new.state_dict()[key], value)

    old_obs = torch.randn(8, 3)
    terrain_suffix = torch.randn(8, 2)
    torch.testing.assert_close(old(old_obs), new(torch.cat([old_obs, terrain_suffix], dim=-1)))


def test_normalizer_and_optimizer_suffix_expansion():
    torch.manual_seed(11)
    old = _Actor(3)
    old_optimizer = torch.optim.AdamW(old.parameters(), lr=7.5e-5)
    old_obs = torch.randn(8, 3)
    loss = old(old_obs).square().mean() + old.std.square().mean()
    loss.backward()
    old_optimizer.step()

    new = _Actor(5)
    _migrate_actor(old, new)
    new_optimizer = torch.optim.AdamW(new.parameters(), lr=1e-3)
    expanded_moments = _load_optimizer_with_input_expansion(
        new_optimizer,
        old_optimizer.state_dict(),
        label="actor",
    )
    assert expanded_moments == 2
    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(7.5e-5)

    old_weight = dict(old.named_parameters())["actor_module.module.0.weight"]
    new_weight = dict(new.named_parameters())["actor_module.module.0.weight"]
    old_moment = old_optimizer.state[old_weight]["exp_avg"]
    new_moment = new_optimizer.state[new_weight]["exp_avg"]
    torch.testing.assert_close(new_moment[:, :3], old_moment)
    torch.testing.assert_close(new_moment[:, 3:], torch.zeros_like(new_moment[:, 3:]))

    new_optimizer.zero_grad()
    new_loss = new(torch.randn(8, 5)).square().mean() + new.std.square().mean()
    new_loss.backward()
    new_optimizer.step()

    old_normalizer = EmpiricalNormalization(shape=3, device="cpu")
    old_normalizer._mean.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
    old_normalizer._var.copy_(torch.tensor([[4.0, 5.0, 6.0]]))
    old_normalizer._std.copy_(old_normalizer._var.sqrt())
    old_normalizer.count.fill_(123)
    new_normalizer = EmpiricalNormalization(shape=5, device="cpu")
    _load_normalizer_with_input_expansion(
        new_normalizer,
        old_normalizer.state_dict(),
        label="actor",
    )
    torch.testing.assert_close(new_normalizer._mean[:, :3], old_normalizer._mean)
    torch.testing.assert_close(new_normalizer._mean[:, 3:], torch.zeros(1, 2))
    torch.testing.assert_close(new_normalizer._var[:, 3:], torch.ones(1, 2))
    torch.testing.assert_close(new_normalizer._std[:, 3:], torch.ones(1, 2))
    assert new_normalizer.count == 123


def test_input_expansion_rejects_hidden_width_change():
    old = _Actor(3, hidden_dim=4)
    incompatible = _Actor(5, hidden_dim=6)
    with pytest.raises(RuntimeError, match="not a suffix expansion"):
        _migrate_actor(old, incompatible)
