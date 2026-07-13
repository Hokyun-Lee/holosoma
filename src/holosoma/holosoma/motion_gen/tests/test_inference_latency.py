from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from holosoma.motion_gen.scripts.benchmark_inference_latency import (
    file_sha256,
    load_tracker_policy,
    summarize_latencies_ms,
)


def test_summarize_latencies_uses_interpolated_percentiles():
    summary = summarize_latencies_ms([4.0, 1.0, 3.0, 2.0])

    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["median"] == pytest.approx(2.5)
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["population_std"] == pytest.approx(1.11803398875)


@pytest.mark.parametrize("samples", [[], [-1.0], [float("nan")], [float("inf")]])
def test_summarize_latencies_rejects_invalid_samples(samples):
    with pytest.raises(ValueError, match=r"empty|finite"):
        summarize_latencies_ms(samples)


def test_file_sha256_streams_file(tmp_path: Path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"holosoma-latency")

    assert file_sha256(path, chunk_size=3) == "a4212e72918156e5113e1b8074e0df4ecb8bc679fadd19000b69cf06d44864ec"


def test_tracker_loader_reconstructs_normalized_ppo_actor(tmp_path: Path):
    checkpoint_path = tmp_path / "tracker.pt"
    first_weight = torch.tensor(
        [
            [1.0, -2.0, 0.5],
            [0.25, 0.5, -1.0],
            [-0.75, 1.5, 0.25],
            [0.1, 0.2, 0.3],
        ]
    )
    first_bias = torch.tensor([0.1, -0.2, 0.3, -0.4])
    final_weight = torch.tensor([[0.5, -0.25, 1.0, 0.0], [-1.0, 0.5, 0.25, 2.0]])
    final_bias = torch.tensor([0.2, -0.1])
    mean = torch.tensor([[1.0, 2.0, 3.0]])
    std = torch.tensor([[2.0, 4.0, 5.0]])
    checkpoint = {
        "actor_model_state_dict": {
            "std": torch.ones(2),
            "actor_module.module.0.weight": first_weight,
            "actor_module.module.0.bias": first_bias,
            "actor_module.module.2.weight": final_weight,
            "actor_module.module.2.bias": final_bias,
        },
        "actor_obs_normalizer_state_dict": {
            "_mean": mean,
            "_var": std.square(),
            "_std": std,
            "count": torch.tensor(123),
        },
        "experiment_config": {
            "algo": {
                "config": {
                    "init_noise_std": 1.0,
                    "empirical_normalization": True,
                    "module_dict": {
                        "actor": {
                            "type": "MLP",
                            "input_dim": ["actor_obs"],
                            "output_dim": [2],
                            "layer_config": {
                                "hidden_dims": [4],
                                "activation": "ELU",
                                "dropout_prob": 0.0,
                            },
                            "min_noise_std": None,
                            "min_mean_noise_std": None,
                        }
                    },
                }
            }
        },
        "iteration": 17,
    }
    torch.save(checkpoint, checkpoint_path)

    loaded = load_tracker_policy(checkpoint_path, "cpu")
    observation = torch.tensor([[3.0, -2.0, 8.0]])
    normalized = (observation - mean) / (std + 1.0e-2)
    expected = F.linear(F.elu(F.linear(normalized, first_weight, first_bias)), final_weight, final_bias)

    assert loaded.observation_dim == 3
    assert loaded.action_dim == 2
    assert loaded.checkpoint_iteration == 17
    assert loaded.empirical_normalization
    assert not any(parameter.requires_grad for parameter in loaded.policy.parameters())
    torch.testing.assert_close(loaded.policy(observation), expected)
