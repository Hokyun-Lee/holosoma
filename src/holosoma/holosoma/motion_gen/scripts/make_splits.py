"""Write the motion-level train/val split file.

Usage (from the repo root):
    python -m holosoma.motion_gen.scripts.make_splits [--data-root data/motion_gen]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from holosoma.motion_gen.data_manifest import DEFAULT_SPLITS


@dataclass
class Args:
    data_root: str = "data/motion_gen"
    train: list[str] = field(default_factory=lambda: list(DEFAULT_SPLITS["train"]))
    val: list[str] = field(default_factory=lambda: list(DEFAULT_SPLITS["val"]))


def main(args: Args) -> None:
    overlap = set(args.train) & set(args.val)
    if overlap:
        raise ValueError(f"train/val overlap (leakage): {sorted(overlap)}")
    processed = Path(args.data_root) / "processed"
    missing = [s for s in args.train + args.val if not (processed / f"{s}.npz").exists()]
    if missing:
        print(f"[warn] processed files missing (run prepare + conversion first): {missing}")
    out = Path(args.data_root) / "splits" / "splits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"train": args.train, "val": args.val}, indent=2))
    print(f"[ok] {out}: {len(args.train)} train, {len(args.val)} val motions")


if __name__ == "__main__":
    main(tyro.cli(Args))
