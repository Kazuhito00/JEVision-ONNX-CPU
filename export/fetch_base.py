# -*- coding: utf-8 -*-
"""Fetch the pinned Qwen3.5-0.8B-Base checkpoint and the JEVision bundle (HTTPS only).

    uv run export/fetch_base.py              # models/hf-base (1.7 GB) + models/bundle (the adapters and heads, ~90 MB)
    uv run export/fetch_base.py --runtime    # also models/bundle/runtime: JEVision's own PyTorch runtime (kev), the oracle
                                             # that export/make_golden.py runs. Not needed to build the ONNX graphs.

Both are Apache-2.0 (see README, "License"). Nothing is imported from the runtime files at inference time.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export.download import fetch_files  # noqa: E402

BASE_REPO, BASE_REV = "Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
JEV_REPO, JEV_REV = "divyanshx11/JEVision", "e1cfde19bfae6548a46812c64a9a6f1a67459c95"
BASE_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
              "video_preprocessor_config.json", "vocab.json", "merges.txt", "LICENSE",
              "model.safetensors-00001-of-00001.safetensors"]
JEV_FILES = ["LICENSE", "jevvision_manifest.json", "report.json", "pointer_head.pt", "adapter/adapter_config.json",
             "adapter/adapter_model.safetensors", "text/jevvision-text/adapter_config.json",
             "text/jevvision-text/adapter_model.safetensors", "text/jevvision-text/head.pt"]
RUNTIME_FILES = ["runtime/kev/__init__.py", "runtime/kev/api.py", "runtime/kev/checkpoint.py", "runtime/kev/device.py",
                 "runtime/kev/jevvision.py", "runtime/kev/media.py", "runtime/kev/model.py", "runtime/kev/serve.py",
                 "runtime/kev/visual_adapter.py", "runtime/kev/visual_model.py", "runtime/kev/visual_prompt.py",
                 "jevvision.py"]

if __name__ == "__main__":
    fetch_files(BASE_REPO, BASE_FILES, ROOT / "models" / "hf-base", revision=BASE_REV)
    files = JEV_FILES + (RUNTIME_FILES if "--runtime" in sys.argv else [])
    fetch_files(JEV_REPO, files, ROOT / "models" / "bundle", revision=JEV_REV)
