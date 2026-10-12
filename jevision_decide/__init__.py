# -*- coding: utf-8 -*-
"""JEVision typed decisions on onnxruntime + numpy (+ Pillow for images)."""

import os as _os
import pathlib as _pathlib

from .protocol import Question, choice, noul, score
from .providers import GpuUnavailable, select_providers
from .runtime import VARIANTS, Decider, gpu_providers

__all__ = ["Decider", "Question", "VARIANTS", "choice", "GpuUnavailable", "default_model_dir", "gpu_providers", "select_providers", "noul", "score"]
__version__ = "0.1.0"

MODEL_DIR_ENV = "JEVISION_MODEL_DIR"


def default_model_dir() -> _pathlib.Path:
    """Where the ONNX files are: ``$JEVISION_MODEL_DIR`` if set, else ``models/jevision`` of this checkout."""
    given = _os.environ.get(MODEL_DIR_ENV)
    return _pathlib.Path(given) if given else _pathlib.Path(__file__).resolve().parents[1] / "models" / "jevision"
