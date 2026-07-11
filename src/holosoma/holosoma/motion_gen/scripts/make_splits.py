"""Write the motion-level train/val split file.

Usage (from the repo root):
    python -m holosoma.motion_gen.scripts.make_splits [--profile paperscale]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from holosoma.motion_gen.data_manifest import get_default_splits


@dataclass
class Args:
    data_root: str = "data/motion_gen"
    profile: str = "small"
    """Clip selection: 'small' or 'paperscale'."""
    train: list[str] = field(default_factory=list)
    """Override train stems (default: profile's split)."""
    val: list[str] = field(default_factory=list)
    """Override val stems (default: profile's split)."""


def main(args: Args) -> None:
    defaults = get_default_splits(args.profile)
    train = args.train or defaults["train"]
    val = args.val or defaults["val"]
    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(f"train/val overlap (leakage): {sorted(overlap)}")
    suffix = "" if args.profile == "small" else f"_{args.profile}"
    processed = Path(args.data_root) / f"processed{suffix}"
    missing = [s for s in train + val if not (processed / f"{s}.npz").exists()]
    if missing:
        print(f"[warn] {len(missing)} processed files missing (run prepare + conversion first), e.g. {missing[:5]}")
    out = Path(args.data_root) / "splits" / f"splits{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"train": train, "val": val}, indent=2))
    print(f"[ok] {out}: {len(train)} train, {len(val)} val motions")


if __name__ == "__main__":
    main(tyro.cli(Args))
