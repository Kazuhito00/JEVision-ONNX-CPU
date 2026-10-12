# -*- coding: utf-8 -*-
"""Paths shared by the verification gates (not a gate itself)."""

from __future__ import annotations

import hashlib
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
IMAGES = ARTIFACTS / "images"
#: Optional: a folder in which the golden cases' pictures that are not tracked here (Wikimedia Commons photographs and
#: the like) can be found by their golden path (``eval_common/real_images/...``). Cases whose picture is missing are skipped.
IMAGE_ROOT_ENV = "JEVISION_IMAGE_ROOT"


def model_dir() -> pathlib.Path:
    given = os.environ.get("JEVISION_MODEL_DIR")
    return pathlib.Path(given) if given else ROOT / "models" / "jevision"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_image(ref: str, digest: str = ""):
    """The file a golden image reference points at, or None when it is not available here.

    The synthetic pictures are tracked in artifacts/images/ (found by file name); any other picture is looked up under
    $JEVISION_IMAGE_ROOT. A file whose sha256 differs from the golden's is never used."""
    candidates = [IMAGES / pathlib.PurePosixPath(ref).name]
    root = os.environ.get(IMAGE_ROOT_ENV)
    if root:
        candidates.append(pathlib.Path(root) / ref)
    for p in candidates:
        if p.is_file() and (not digest or sha256(p) == digest):
            return p
    return None
