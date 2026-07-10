"""Download the selected training motions (manifest-driven, selective only).

- LAFAN1 G1 CSVs: downloaded one by one from HuggingFace (never the full set).
- OmniRetarget: downloads the two needed zips (15/38 MB), verifies sha256,
  extracts only the manifest members. The full dataset is never mirrored.
- Existing files are skipped; failures raise with a clear message.
- Writes data/motion_gen/metadata/downloads.json (sha256 of every file) and
  SOURCES.md (origin + license notes).

Usage (from the repo root):
    python -m holosoma.motion_gen.scripts.download_data [--data-root data/motion_gen]
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import tyro

from holosoma.motion_gen.data_manifest import MANIFEST, OMNIRETARGET_BASE_URL, OMNIRETARGET_ZIPS


@dataclass
class Args:
    data_root: str = "data/motion_gen"
    force: bool = False


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"[skip] {dest.name} already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "holosoma-motion-gen/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url}: {e}. Check network access to huggingface.co.") from e
    tmp.rename(dest)
    print(f"[ok] {dest} ({dest.stat().st_size / 1e6:.2f} MB)")


def main(args: Args) -> None:
    root = Path(args.data_root)
    raw_lafan = root / "raw" / "lafan1_g1"
    raw_omni = root / "raw" / "omniretarget"
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    hashes: dict[str, str] = {}

    # --- LAFAN1 CSVs -------------------------------------------------------
    for clip in MANIFEST:
        if clip.source != "lafan1":
            continue
        dest = raw_lafan / clip.origin.rsplit("/", 1)[-1]
        download(clip.origin, dest, args.force)
        hashes[str(dest.relative_to(root))] = sha256_of(dest)

    # --- OmniRetarget zips + selective extraction --------------------------
    needed_members: dict[str, list[str]] = {}
    for clip in MANIFEST:
        if clip.source != "omniretarget":
            continue
        zip_name, member = clip.origin.split(":", 1)
        needed_members.setdefault(zip_name, []).append(member)

    for zip_name, members in needed_members.items():
        zip_path = raw_omni / zip_name
        download(f"{OMNIRETARGET_BASE_URL}/{zip_name}", zip_path, args.force)
        digest = sha256_of(zip_path)
        hashes[str(zip_path.relative_to(root))] = digest
        expected = OMNIRETARGET_ZIPS.get(zip_name)
        if expected and digest != expected:
            raise RuntimeError(
                f"{zip_path}: sha256 mismatch (got {digest}, expected {expected}). "
                "The upstream dataset may have changed; re-verify the manifest."
            )
        with zipfile.ZipFile(zip_path) as zf:
            for member in members:
                out = raw_omni / Path(member).name
                if out.exists() and not args.force:
                    print(f"[skip] {out.name} already extracted")
                    continue
                with zf.open(member) as src, open(out, "wb") as dst:
                    dst.write(src.read())
                print(f"[ok] extracted {member} -> {out}")
                hashes[str(out.relative_to(root))] = sha256_of(out)

    (meta_dir / "downloads.json").write_text(json.dumps(hashes, indent=2))

    sources = ["# Motion data sources and licenses\n"]
    for clip in MANIFEST:
        sources.append(
            f"- **{clip.stem}** — {clip.description}\n"
            f"  - source: {clip.source}, origin: `{clip.origin}`\n"
            f"  - license: {clip.license}\n"
        )
    sources.append(
        "\nLAFAN1 data (Ubisoft) is CC BY-NC-ND 4.0: non-commercial use, no "
        "redistribution of derivatives. OmniRetarget dataset is MIT. Do not "
        "commit downloaded data to the repository.\n"
    )
    (meta_dir / "SOURCES.md").write_text("\n".join(sources))
    print(f"[done] hashes -> {meta_dir / 'downloads.json'}, sources -> {meta_dir / 'SOURCES.md'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
