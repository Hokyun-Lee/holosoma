"""Print keys, shapes, dtypes and basic statistics of a motion ``.npz``.

Usage:
    python -m holosoma.motion_gen.scripts.inspect_motion_npz --path <file.npz> [--stats]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tyro


@dataclass
class Args:
    path: str
    stats: bool = False
    """Also print min/max/mean for numeric arrays."""


def inspect(path: str, stats: bool = False) -> None:
    data = np.load(path, allow_pickle=True)
    print(f"=== {path}")
    for key in data.files:
        v = data[key]
        line = f"  {key}: shape={v.shape}, dtype={v.dtype}"
        if v.ndim == 0 or v.size == 1:
            line += f", value={v.reshape(-1)[:1]}"
        elif stats and np.issubdtype(v.dtype, np.number):
            line += f", min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}"
            if not np.isfinite(np.asarray(v, dtype=np.float64)).all():
                line += "  [WARNING: contains NaN/Inf]"
        print(line)
    if "fps" in data.files and "joint_pos" in data.files:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        frames = data["joint_pos"].shape[0]
        print(f"  -> {frames} frames @ {fps:g} fps = {frames / fps:.2f} s")


def main(args: Args) -> None:
    inspect(args.path, args.stats)


if __name__ == "__main__":
    main(tyro.cli(Args))
