"""Feature normalization for the motion generator.

Z-score normalization with per-dimension mean/std computed on the *train*
split only (canonical-frame features). Statistics are stored both as a
standalone ``normalization_stats.npz`` and inside training checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_STD_FLOOR = 1e-4  # dims with (near-)zero variance are left unscaled


class FeatureNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        assert mean.shape == std.shape and mean.ndim == 1
        self.mean = mean.float()
        self.std = torch.where(std < _STD_FLOOR, torch.ones_like(std), std).float()

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean.numpy(), std=self.std.numpy())

    @staticmethod
    def load(path: str | Path) -> "FeatureNormalizer":
        data = np.load(path)
        return FeatureNormalizer(torch.from_numpy(data["mean"]), torch.from_numpy(data["std"]))

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @staticmethod
    def from_state_dict(state: dict) -> "FeatureNormalizer":
        return FeatureNormalizer(state["mean"], state["std"])


def compute_normalizer(
    dataset,
    max_windows: int = 2000,
    seed: int = 0,
) -> FeatureNormalizer:
    """Streaming mean/std over canonical features of sampled train windows.

    Uses both past and future frames of each window. ``max_windows`` bounds
    the cost on large datasets (uniform subsample without replacement).
    """
    n = len(dataset)
    if n == 0:
        raise ValueError("Cannot compute normalization statistics on an empty dataset.")
    rng = np.random.default_rng(seed)
    indices = np.arange(n) if n <= max_windows else rng.choice(n, size=max_windows, replace=False)

    count = 0
    mean = None
    m2 = None
    for i in indices:
        item = dataset[int(i)]
        frames = torch.cat([item["past"], item["x"]], dim=0)  # (T, D)
        for f in frames:
            count += 1
            if mean is None:
                mean = f.double().clone()
                m2 = torch.zeros_like(mean)
                continue
            delta = f.double() - mean
            mean += delta / count
            m2 += delta * (f.double() - mean)
    assert mean is not None and m2 is not None and count > 1
    std = (m2 / (count - 1)).sqrt()
    return FeatureNormalizer(mean.float(), std.float())
