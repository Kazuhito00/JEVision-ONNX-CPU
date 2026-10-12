# -*- coding: utf-8 -*-
"""The --device / --gpu / --require-gpu options shared by the demos and verify/bench.py."""

from __future__ import annotations

import argparse

from .providers import DEVICES


def add_device_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--device", choices=DEVICES, default="cpu",
                    help="cpu (default) | cuda (CUDA first, CPU as fallback with a warning) | auto (CUDA if this onnxruntime has it)")
    ap.add_argument("--gpu", action="store_true", help="same as --device cuda")
    ap.add_argument("--require-gpu", action="store_true",
                    help="error out instead of falling back to the CPU when CUDA cannot be used (implies --device cuda)")


def device_kwargs(args: argparse.Namespace) -> dict:
    """Keyword arguments for ``Decider``."""
    device = args.device
    if args.gpu or args.require_gpu:
        if device == "cpu":
            device = "cuda"
    return {"device": device, "require_gpu": bool(args.require_gpu)}


def providers_line(decider) -> str:
    """One line saying what really ran where, e.g. ``CUDAExecutionProvider (embed_tokens_fp16.onnx, decoder_text_fp16.onnx)``."""
    used = decider.providers_used
    if not used:
        return decider.provider
    groups = {}
    for graph, provider in used.items():
        groups.setdefault(provider, []).append(graph)
    return "; ".join("%s (%s)" % (p, ", ".join(g)) for p, g in groups.items())


def load_decider(args: argparse.Namespace, model_dir=None, **kwargs):
    """``Decider`` from the --device options; a GPU that cannot be used with --require-gpu ends the program with a message
    (exit status 2) instead of a traceback."""
    from . import Decider, GpuUnavailable, default_model_dir
    try:
        return Decider(model_dir or default_model_dir(), **kwargs, **device_kwargs(args))
    except GpuUnavailable as exc:
        raise SystemExit("error: %s" % exc)
