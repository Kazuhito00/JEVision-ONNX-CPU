# -*- coding: utf-8 -*-
"""Download the ONNX model for demo_inference_text.py / demo_inference_image.py from this repository's GitHub release.

    python download_model.py                                    # latest release of Kazuhito00/JEVision-ONNX-CPU, SHA-256 checked
    python download_model.py --tag v0.0.0                       # a given release
    python download_model.py --exclude "*vision_encoder*"       # text-only use: skip the vision encoder
    python download_model.py --variant fp16                     # only the fp16 graphs (every variant in the release is fetched by default)
    python download_model.py --base-url http://127.0.0.1:8000   # a mirror, or a local release/ output
    python download_model.py --list
    python download_model.py --repo OWNER/NAME                  # a fork's release

The default repository is ``Kazuhito00/JEVision-ONNX-CPU``. Give another one with ``--repo OWNER/NAME`` or through the
environment: ``JEVISION_ONNX_REPO=OWNER/NAME`` (the argument wins over the environment, the environment over the default).

JEVision has no published ONNX (the checkpoint is a LoRA adapter pair plus pointer heads, not weights). The release holds the
graphs built by export/build_onnx.py; to build them yourself use the steps in the README ("ONNX conversion").

The release carries ``manifest.json`` (written by release/make_release.py): every file's path, size and sha256 and, for files that
were split to stay under GitHub's 2 GiB asset limit, the parts. This script fetches what the manifest lists, joins the parts and
checks every sha256 (a mismatch deletes the file and stops). A file already in place with the right sha256 is skipped and an
interrupted download continues where it stopped (HTTP Range), so the same command can simply be run again.
Files land under models/ (``models/jevision/...``). Plain HTTPS from the standard library: no git, no git-lfs, no huggingface_hub.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_REPO = "Kazuhito00/JEVision-ONNX-CPU"
USER_AGENT = "jevision-onnx-cpu/0.1 (urllib)"
CHUNK = 1 << 20  # 1 MiB


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return "%.1f %s" % (size, unit)
        size /= 1024
    return "%.1f GB" % size


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def base_url(args) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")
    repo = args.repo or os.environ.get("JEVISION_ONNX_REPO") or DEFAULT_REPO
    if args.tag == "latest":
        return "https://github.com/%s/releases/latest/download" % repo
    return "https://github.com/%s/releases/download/%s" % (repo, args.tag)


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch(url: str, dest: pathlib.Path, size: int, label: str = "", tries: int = 8) -> pathlib.Path:
    """Download ``url`` to ``dest`` (``size`` bytes), resuming the bytes already there; retries on dropped connections."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for attempt in range(1, tries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have == size:
            return dest
        if have > size:
            dest.unlink()
            have = 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if have and response.status != 206:
                    have = 0  # the server ignored the Range header: start over
                with open(dest, "ab" if have else "wb") as handle:
                    done, last = have, 0.0
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        done += len(block)
                        now = time.time()
                        if now - last > 0.5:
                            last = now
                            rate = (done - have) / max(now - started, 1e-6)
                            sys.stdout.write("\r  %-44s %5.1f%%  %s/s   "
                                             % (label or dest.name, 100 * done / size, _human(rate)))
                            sys.stdout.flush()
            if dest.stat().st_size == size:
                return dest
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            wait = min(60, 2 ** attempt)
            print("\n  %s: %s (try %d/%d, again in %d s)" % (dest.name, exc, attempt, tries, wait), flush=True)
            time.sleep(wait)
    raise SystemExit("could not download %s completely" % url)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", help="GitHub repository that holds the release (OWNER/NAME; default: $JEVISION_ONNX_REPO, else %s)"
                        % DEFAULT_REPO)
    parser.add_argument("--tag", default="latest", help="release tag (default: latest)")
    parser.add_argument("--base-url", help="fetch the assets from this URL instead of a GitHub release (mirror, local server)")
    parser.add_argument("--dest", default=str(ROOT / "models"), help="where the files go (default: models/)")
    parser.add_argument("--include", action="append", help="only paths matching this glob (repeatable)")
    parser.add_argument("--exclude", action="append", help="skip paths matching this glob (repeatable)")
    parser.add_argument("--variant", choices=("fp16", "fp32"), action="append",
                        help="fetch only the graphs of this precision (repeatable); the tokenizer, heads and configs always come "
                             "along. Default: every variant in the release")
    parser.add_argument("--list", action="store_true", help="show the files of the release and exit")
    args = parser.parse_args()

    base = base_url(args)
    try:
        manifest = json.loads(fetch_bytes(base + "/manifest.json"))
    except urllib.error.URLError as exc:
        raise SystemExit("could not read %s/manifest.json: %s" % (base, exc))
    if manifest.get("format") != 1:
        raise SystemExit("unknown manifest format: %r" % manifest.get("format"))

    def wanted(path: str) -> bool:
        name = path.rsplit("/", 1)[-1]
        graph = re.match(r"(?:decoder_text|decoder_visual|embed_tokens|vision_encoder)(_fp16)?\.onnx", name)
        if args.variant and graph and ("fp16" if graph.group(1) else "fp32") not in args.variant:
            return False
        if args.include and not any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(name, g) for g in args.include):
            return False
        return not (args.exclude and any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(name, g) for g in args.exclude))

    files = [f for f in manifest["files"] if wanted(f["path"])]
    total = sum(f["size"] for f in files)
    print("%s %s: %d files, %s" % (manifest.get("name"), manifest.get("version"), len(files), _human(total)))
    if args.list:
        for f in files:
            print("  %10s  %s%s" % (_human(f["size"]), f["path"], "  (%d parts)" % len(f["parts"]) if f.get("parts") else ""))
        return 0

    dest = pathlib.Path(args.dest)
    for f in files:
        out = dest / f["path"]
        if out.exists() and out.stat().st_size == f["size"] and _sha256(out) == f["sha256"]:
            print("  have  %-44s %s" % (f["path"], _human(f["size"])))
            continue
        tmp = out.with_name(out.name + ".part")
        if not f.get("parts"):
            fetch("%s/%s" % (base, f["asset"]), tmp, f["size"], label=f["path"])
        else:  # join the parts into one file, one part at a time
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as joined:
                for part in f["parts"]:
                    piece = tmp.with_name(part["asset"] + ".dl")
                    fetch("%s/%s" % (base, part["asset"]), piece, part["size"], label=part["asset"])
                    if _sha256(piece) != part["sha256"]:
                        piece.unlink()
                        raise SystemExit("%s: SHA-256 mismatch, file deleted" % part["asset"])
                    with open(piece, "rb") as src:
                        for block in iter(lambda: src.read(CHUNK), b""):
                            joined.write(block)
                    piece.unlink()
        if _sha256(tmp) != f["sha256"]:
            tmp.unlink()
            raise SystemExit("%s: SHA-256 mismatch, file deleted" % f["path"])
        os.replace(tmp, out)
        sys.stdout.write("\r  got   %-44s %s\n" % (f["path"], _human(f["size"])))
        sys.stdout.flush()

    print("\nready: %s" % dest)
    print("run:    uv run demo_inference_text.py")
    print("verify: uv run verify/run_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
