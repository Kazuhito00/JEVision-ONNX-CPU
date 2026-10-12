# -*- coding: utf-8 -*-
"""Fetch the graphs (and conversion notes) of onnx-community/OneJev-0.8B-ONNX, the containers build_onnx.py fills.

JEVision shares OneJev's architecture (Qwen3.5-0.8B), so these graphs are only used as containers: every
weight is replaced by JEVision's (see build_onnx.py).

    uv run export/fetch_src_onnx.py                 # the fp32 graphs (~1.9 GB)
    uv run export/fetch_src_onnx.py --fp16          # + the fp16 graphs (needed for `build --precision fp16`)
    uv run export/fetch_src_onnx.py --verify-pair   # + OneJev-0.8B's safetensors (models/onejev-hf; only for `build_onnx.py verify`)
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export.download import fetch_files  # noqa: E402

REPO, REV = "onnx-community/OneJev-0.8B-ONNX", "afc03812cedf078ce660ec410fcf8597b4c4ec5d"
ONEJEV_REPO, ONEJEV_REV = "OmniJev/OneJev-0.8B", "c3939d8bf4cad34549a2b13bbb6aee9bcb6afee8"
FP32 = ["onnx/decoder_model_merged.onnx", "onnx/decoder_model_merged.onnx_data", "onnx/embed_tokens.onnx",
        "onnx/embed_tokens.onnx_data", "onnx/vision_encoder.onnx", "onnx/vision_encoder.onnx_data"]
FP16 = [p.replace(".onnx", "_fp16.onnx") for p in FP32]
NOTES = ["conversion/map_weights.py", "conversion/build_onnx.py", "conversion/build_fp16.py", "config.json",
         "preprocessor_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json"]

if __name__ == "__main__":
    files = NOTES + FP32 + (FP16 if "--fp16" in sys.argv else [])
    fetch_files(REPO, files, ROOT / "models" / "src-onnx", revision=REV)
    if "--verify-pair" in sys.argv:
        fetch_files(ONEJEV_REPO, ["model.safetensors"], ROOT / "models" / "onejev-hf", revision=ONEJEV_REV)
