# -*- coding: utf-8 -*-
"""Which onnxruntime execution providers to ask for, and whether what was granted is what was asked.

Two small pure functions (no onnxruntime import, so they are unit-tested without a GPU) and the constants they share.

onnxruntime can create a session that silently runs on the CPU although CUDA was requested (a missing cuDNN / CUDA DLL, a graph
that does not fit in VRAM). ``check_granted`` turns that into a warning, or into an error with ``require_gpu``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
DEVICES = ("cpu", "cuda", "auto")

#: install hint shown with every "no GPU" message
GPU_HINT = ("install onnxruntime-gpu instead of onnxruntime (uv sync --extra gpu) and put the CUDA 13 and cuDNN 9 bin "
            "folders on PATH; see README, 'Running on a GPU'")


class GpuUnavailable(RuntimeError):
    """--require-gpu was given and the GPU is not usable."""


def select_providers(device: str = "cpu", available: Optional[Iterable[str]] = None, require_gpu: bool = False) -> List[str]:
    """The provider list for ``device``: ``cpu`` -> [CPU]; ``cuda`` -> [CUDA, CPU]; ``auto`` -> CUDA first when this onnxruntime
    build has it, else [CPU]. ``available`` is ``onnxruntime.get_available_providers()``.

    ``cuda`` without a CUDA provider returns [CPU] (the caller warns); with ``require_gpu`` it raises ``GpuUnavailable``, before any
    model is loaded. ``require_gpu`` with ``cpu`` is a contradiction (ValueError).
    """
    if device not in DEVICES:
        raise ValueError("device must be one of %s, got %r" % (", ".join(DEVICES), device))
    has_cuda = CUDA in set(available or ())
    if device == "cpu":
        if require_gpu:
            raise ValueError("--require-gpu contradicts --device cpu")
        return [CPU]
    if has_cuda:
        return [CUDA, CPU]
    if require_gpu:
        raise GpuUnavailable("CUDAExecutionProvider is not available in this onnxruntime build (%s)" % GPU_HINT)
    return [CPU]


def wants_gpu(requested: Sequence[str]) -> bool:
    return bool(requested) and requested[0] != CPU


def check_granted(requested: Sequence[str], used: Sequence[str], require_gpu: bool = False, what: str = "") -> Optional[str]:
    """Compare the providers asked for with the ones a session really got (``InferenceSession.get_providers()``).

    Returns None when they agree (or the CPU was asked for), else a warning text; raises ``GpuUnavailable`` instead when
    ``require_gpu`` is set. ``what`` names the graph in the message.
    """
    if not wants_gpu(requested):
        if require_gpu:
            raise GpuUnavailable("--require-gpu was given but only the CPU was requested")
        return None
    if used and used[0] == requested[0]:
        return None
    got = used[0] if used else "nothing"
    message = "%s was requested%s but the session runs on %s (%s)" % (
        requested[0], " for " + what if what else "", got, "CUDA / cuDNN DLLs not found on PATH, or the graph does not fit in VRAM")
    if require_gpu:
        raise GpuUnavailable(message)
    return "warning: " + message
