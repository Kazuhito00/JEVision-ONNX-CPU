# -*- coding: utf-8 -*-
"""Image preprocessing for the Qwen3.5-VL vision encoder (JEVision flavour).

Produces exactly what the exported ``vision_encoder.onnx`` expects:

    pixel_values     float32 [num_patches, channel * temporal * patch * patch]
    image_grid_thw   int64   [num_images, 3]   (t, grid_h, grid_w)

Ported from ``transformers.models.qwen2_vl.image_processing_qwen2_vl``
(``smart_resize`` / ``patchify``), with Pillow doing the decode and resample and
numpy doing the rest. Pillow is the only dependency here beyond numpy; the
inference path itself needs just onnxruntime and numpy.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
import pathlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

__all__ = ["ImageConfig", "PreparedImages", "prepare_images", "smart_resize"]

ImageSource = Union[str, pathlib.Path, bytes, "Image.Image"]

# Defaults come from the checkpoint's preprocessor_config.json, which states the
# window in *pixels*: shortest_edge = min_pixels, longest_edge = max_pixels.
# Dividing by factor^2 turns them into post-merge visual token counts.
DEFAULT_MIN_TOKENS = 64        # 65536 px / 32^2
DEFAULT_MAX_TOKENS = 16384     # 16777216 px / 32^2


@dataclass(frozen=True)
class ImageConfig:
    """Vision geometry, read from the checkpoint's preprocessor_config.json."""

    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    # How many post-merge visual tokens one image may cost. The defaults match
    # the checkpoint, so results match inference.py; lowering token_budget caps
    # the prompt but changes the answer.
    min_tokens: int = DEFAULT_MIN_TOKENS
    token_budget: int = DEFAULT_MAX_TOKENS
    max_edge: int = 0              # 0 = no extra cap, as the checkpoint has none
    max_pixels_in: int = 25_000_000  # the reference's MAX_IMAGE_PIXELS
    max_bytes: int = 10 * 1024 * 1024  # the reference's MAX_IMAGE_BYTES

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    @classmethod
    def from_pretrained(cls, model_dir, **overrides) -> "ImageConfig":
        import json

        path = pathlib.Path(model_dir) / "preprocessor_config.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        size = cfg.get("size") or {}
        factor = cfg.get("patch_size", 16) * cfg.get("merge_size", 2)
        defaults = {
            "patch_size": cfg.get("patch_size", 16),
            "temporal_patch_size": cfg.get("temporal_patch_size", 2),
            "merge_size": cfg.get("merge_size", 2),
            "image_mean": tuple(cfg.get("image_mean", (0.5, 0.5, 0.5))),
            "image_std": tuple(cfg.get("image_std", (0.5, 0.5, 0.5))),
        }
        if "shortest_edge" in size:
            defaults["min_tokens"] = int(size["shortest_edge"]) // (factor * factor)
        if "longest_edge" in size:
            defaults["token_budget"] = int(size["longest_edge"]) // (factor * factor)
        defaults.update(overrides)
        return cls(**defaults)


@dataclass
class PreparedImages:
    pixel_values: np.ndarray  # float32 [num_patches, C*T*P*P]
    image_grid_thw: np.ndarray  # int64 [num_images, 3]

    @property
    def token_counts(self) -> List[int]:
        """Post-merge visual tokens each image expands to in the text stream."""
        merge = 4  # merge_size ** 2; the encoder folds 2x2 patches into one token
        return [int(t * h * w) // merge for t, h, w in self.image_grid_thw]


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    """transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got %s"
            % (max(height, width) / min(height, width))
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _open(source: ImageSource, config: ImageConfig) -> "Image.Image":
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, (str, pathlib.Path)):
        text = str(source)
        if text.startswith("data:image/"):
            header, sep, encoded = text.partition(",")
            if not sep or ";base64" not in header:
                raise ValueError("only base64 data URLs are supported")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except binascii.Error as exc:
                raise ValueError("image data URL is not valid base64") from exc
        else:
            raw = pathlib.Path(text).read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        raise TypeError("unsupported image source: %r" % type(source).__name__)

    if len(raw) > config.max_bytes:
        raise ValueError("compressed image exceeds the %d byte limit" % config.max_bytes)
    try:
        picture = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as exc:
        raise ValueError("could not decode the image") from exc
    if picture.width * picture.height > config.max_pixels_in:
        raise ValueError("decoded image exceeds the pixel limit")
    return picture


def _to_rgb_array(source: ImageSource, config: ImageConfig) -> np.ndarray:
    """Decode (no EXIF rotation: the reference decode_image does not rotate either), cap the long
    edge if configured, resize to the encoder grid."""
    picture = _open(source, config)
    with picture:
        upright = picture.convert("RGB")

    if config.max_edge:
        scale = min(1.0, config.max_edge / max(upright.size))
        if scale < 1.0:
            capped = (max(1, round(upright.width * scale)), max(1, round(upright.height * scale)))
            upright = upright.resize(capped, Image.Resampling.BICUBIC, reducing_gap=2.0)

    factor = config.factor
    height, width = smart_resize(
        upright.height,
        upright.width,
        factor,
        config.min_tokens * factor * factor,
        config.token_budget * factor * factor,
    )
    # Plain Pillow BICUBIC, no reducing_gap: torchvision resizes a uint8 tensor with an integer
    # implementation of the same Pillow algorithm, and that is what the reference processor
    # (Qwen2VLImageProcessorFast) runs. Measured against it: identical except +-1 grey level in ~0.3 %
    # of the values (verify/check_image.py).
    if (width, height) != upright.size:
        upright = upright.resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(upright, dtype=np.uint8)  # [H, W, 3]


def _patchify(pixels: np.ndarray, config: ImageConfig) -> Tuple[np.ndarray, int, int]:
    """Qwen2VLImageProcessor.patchify, for one image, in numpy.

    ``pixels`` is [C, H, W] and already rescaled/normalized.
    """
    channel, height, width = pixels.shape
    patch, merge, temporal = config.patch_size, config.merge_size, config.temporal_patch_size
    grid_h, grid_w = height // patch, width // patch

    patches = pixels.reshape(
        channel, grid_h // merge, merge, patch, grid_w // merge, merge, patch
    )
    # -> [grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
    # The same spatial patch is repeated across the temporal axis for a still image.
    patches = np.repeat(patches[..., None, :, :], temporal, axis=-3)
    flat = patches.reshape(grid_h * grid_w, channel * temporal * patch * patch)
    return np.ascontiguousarray(flat, dtype=np.float32), grid_h, grid_w


def prepare_images(
    sources: Sequence[ImageSource], config: ImageConfig | None = None
) -> PreparedImages:
    """Decode and patchify images into the vision encoder's inputs."""
    config = config or ImageConfig()
    mean = np.asarray(config.image_mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(config.image_std, dtype=np.float32).reshape(3, 1, 1)

    all_patches, grids = [], []
    for source in sources:
        rgb = _to_rgb_array(source, config)
        pixels = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        pixels = (pixels - mean) / std
        flat, grid_h, grid_w = _patchify(pixels, config)
        all_patches.append(flat)
        grids.append([1, grid_h, grid_w])

    if not all_patches:
        dim = 3 * config.temporal_patch_size * config.patch_size * config.patch_size
        return PreparedImages(
            np.zeros((0, dim), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)
        )
    return PreparedImages(
        np.concatenate(all_patches, axis=0),
        np.asarray(grids, dtype=np.int64),
    )
