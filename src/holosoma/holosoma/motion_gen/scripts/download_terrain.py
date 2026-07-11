"""Download OmniRetarget climb terrain models (multi-box URDFs + obj meshes).

Fetches, for every climb id used by the paperscale profile, the 5 z-scale
URDF variants and the referenced ``box_models/*.obj`` files (a few KB each).
Files land in ``data/motion_gen/raw/omniretarget_terrain/climb_XX/``.

Usage (from the repo root):
    python -m holosoma.motion_gen.scripts.download_terrain
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import tyro

from holosoma.motion_gen.data_manifest import OMNI_CLIMB_IDS, OMNI_SCENE_IDS, OMNIRETARGET_BASE_URL

API_BASE = "https://huggingface.co/api/datasets/omniretarget/OmniRetarget_Dataset/tree/main"

TERRAIN_DIRS = [f"climb_{cid:02d}" for cid in OMNI_CLIMB_IDS] + [f"scene_{sid}" for sid in OMNI_SCENE_IDS]


@dataclass
class Args:
    data_root: str = "data/motion_gen"
    force: bool = False


def fetch_json(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "holosoma-motion-gen/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def download(url: str, dest: Path, force: bool) -> bool:
    if dest.exists() and not force:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "holosoma-motion-gen/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return True


def main(args: Args) -> None:
    out_root = Path(args.data_root) / "raw" / "omniretarget_terrain"
    total_new = 0
    for name in TERRAIN_DIRS:
        listing = fetch_json(f"{API_BASE}/models/terrain/{name}")
        files = [e["path"] for e in listing if e["path"].endswith(".urdf")]
        try:
            box_listing = fetch_json(f"{API_BASE}/models/terrain/{name}/box_models")
            files += [e["path"] for e in box_listing if e["path"].endswith(".obj")]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"{name}: failed to list box_models: {e}") from e
        new = 0
        for path in files:
            rel = path.removeprefix("models/terrain/")
            if download(f"{OMNIRETARGET_BASE_URL}/{path}", out_root / rel, args.force):
                new += 1
        total_new += new
        print(f"[ok] {name}: {len(files)} files ({new} new)")
    print(f"[done] terrain models in {out_root} ({total_new} new files)")


if __name__ == "__main__":
    main(tyro.cli(Args))
