# -*- coding: utf-8 -*-
"""Stage a GitHub Release from models/jevision: hash every file, split what is too big, write manifest.json.

    python release/make_release.py                           # models/jevision -> dist/release/   (fp16 graphs)
    python release/make_release.py --with-fp32               # also the fp32 graphs (about 5.4 GB more)
    python release/make_release.py --models path/to/models   # stage from another checkout's models/

GitHub allows at most 2 GiB per release asset (and 1000 assets per release), so a file larger than ``--max-part-gib`` is cut into
byte parts (``<name>.part000``, ...) that download_model.py joins again and checks against the whole file's sha256. Asset names are
flat, so ``jevision/onnx/decoder_text_fp16.onnx`` is published as ``jevision__onnx__decoder_text_fp16.onnx``; manifest.json keeps the
real path. Nothing is uploaded here. Upload dist/release/* yourself, for example:

    gh release create v0.0.0 dist/release/* --title "v0.0.0" --notes "ONNX files for JEVision-ONNX-CPU"

What goes in (paths relative to models/, as download_model.py puts them back):
    jevision/onnx/*_fp16.onnx + *_fp16.onnx_data  embed_tokens (0.51 GB), vision_encoder (0.20 GB), decoder_text (1.00 GB),
                                                  decoder_visual (1.00 GB)  -- 2.7 GB for the four graphs
    jevision/onnx/*.onnx + *.onnx_data (fp32)     only with --with-fp32: 1.02 + 0.40 + 1.99 + 1.99 GB
    jevision/head_text.npz, head_visual.npz       the pointer heads (2.1 MB each)
    jevision/tokenizer.json, tokenizer_config.json, config.json, preprocessor_config.json
    jevision/LICENSE-JEVISION, LICENSE-QWEN       the upstream licence texts (licenses/ of this repository), unmodified
The largest asset is the fp32 decoder_*.onnx_data (1,992,441,856 bytes = 1.86 GiB), so with the default --max-part-gib 1.9 nothing
has to be split; the option is there for graphs you build yourself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHUNK = 1 << 20
GRAPHS = ("embed_tokens", "vision_encoder", "decoder_text", "decoder_visual")
SIDE_FILES = ["head_text.npz", "head_visual.npz", "tokenizer.json", "tokenizer_config.json", "config.json", "preprocessor_config.json"]
LICENSE_FILES = ["LICENSE-JEVISION", "LICENSE-QWEN"]          # licenses/ of this repository


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # no second copy of a multi-GB file on the same drive
    except OSError:
        shutil.copy2(src, dst)


def split(src: pathlib.Path, out_dir: pathlib.Path, stem: str, part_bytes: int) -> list:
    parts = []
    with open(src, "rb") as handle:
        index = 0
        while True:
            name = "%s.part%03d" % (stem, index)
            digest, written = hashlib.sha256(), 0
            with open(out_dir / name, "wb") as out:
                while written < part_bytes:
                    block = handle.read(min(CHUNK, part_bytes - written))
                    if not block:
                        break
                    out.write(block)
                    digest.update(block)
                    written += len(block)
            if written == 0:
                (out_dir / name).unlink()
                break
            parts.append({"asset": name, "size": written, "sha256": digest.hexdigest()})
            index += 1
            if written < part_bytes:
                break
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=str(ROOT / "models"))
    parser.add_argument("--out", default=str(ROOT / "dist" / "release"))
    parser.add_argument("--name", default="jevision-onnx")
    parser.add_argument("--version", default="v0.0.0")
    parser.add_argument("--with-fp32", action="store_true", help="also stage the fp32 graphs")
    parser.add_argument("--max-part-gib", type=float, default=1.9,
                        help="assets above this are split (GitHub's limit is 2 GiB)")
    args = parser.parse_args()

    models, out = pathlib.Path(args.models), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    home = models / "jevision"
    onnx = [p for p in sorted((home / "onnx").glob("*")) if p.is_file() and p.name.startswith(GRAPHS)
            and (args.with_fp32 or "_fp16" in p.name)]
    chosen = [(p, p.relative_to(models).as_posix()) for p in onnx]
    chosen += [(home / n, "jevision/" + n) for n in SIDE_FILES if (home / n).exists()]
    chosen += [(ROOT / "licenses" / n, "jevision/" + n) for n in LICENSE_FILES]
    if not onnx:
        raise SystemExit("no ONNX graphs under %s (run download_model.py or export/build_onnx.py first)" % (home / "onnx"))
    missing = [n for n in SIDE_FILES if not (home / n).exists()]
    if missing:
        raise SystemExit("missing under %s: %s" % (home, ", ".join(missing)))
    limit = int(args.max_part_gib * (1 << 30))

    entries = []
    for path, rel in chosen:
        asset = rel.replace("/", "__")
        size = path.stat().st_size
        entry = {"path": rel, "asset": asset, "size": size, "sha256": sha256_of(path), "parts": None}
        if size > limit:
            entry["parts"] = split(path, out, asset, limit)
            print("split  %-44s %9.1f MB -> %d parts" % (rel, size / 1e6, len(entry["parts"])))
        else:
            link_or_copy(path, out / asset)
            print("asset  %-44s %9.1f MB" % (rel, size / 1e6))
        entries.append(entry)

    manifest = {"format": 1, "name": args.name, "version": args.version, "files": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "SHA256SUMS").write_text("".join("%s  %s\n" % (e["sha256"], e["path"]) for e in entries), encoding="utf-8")
    too_big = [a.name for a in out.iterdir() if a.stat().st_size >= (2 << 30)]
    if too_big:
        raise SystemExit("asset(s) of 2 GiB or more: %s" % ", ".join(too_big))
    print("wrote %s: %d files, %d assets" % (out, len(entries), len(list(out.iterdir()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
