"""Train the diffusion motion generator.

Usage (from the repo root, hssim env):
    python -m holosoma.motion_gen.scripts.train smoke
    python -m holosoma.motion_gen.scripts.train debug
    python -m holosoma.motion_gen.scripts.train baseline_4090
    python -m holosoma.motion_gen.scripts.train baseline_4090 --batch-size 128
    python -m holosoma.motion_gen.scripts.train baseline_4090 \\
        --resume logs/motion_gen/baseline_4090/checkpoints/latest.pt
"""

from __future__ import annotations

import tyro

from holosoma.motion_gen.configs import PRESETS
from holosoma.motion_gen.training import Trainer


def main() -> None:
    cfg = tyro.extras.overridable_config_cli(
        {name: (f"{name} preset", fn()) for name, fn in PRESETS.items()}
    )
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
