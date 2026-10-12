# -*- coding: utf-8 -*-
"""3-D M-RoPE position ids for Qwen3.5 (text + still images), without transformers.

Ports ``Qwen3_5Model.get_rope_index`` / ``get_vision_position_ids``. Text advances all three axes
together; an image lays its visual tokens out on a (t, h, w) grid starting at the current position and
then advances the cursor by ``max(grid_h, grid_w) // merge_size``.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

__all__ = ["position_ids"]


def _runs(types: np.ndarray) -> Iterable[Tuple[int, int, int]]:
    if types.size == 0:
        return
    boundaries = np.flatnonzero(np.diff(types)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [types.size]))
    for start, end in zip(starts, ends):
        yield int(types[start]), int(start), int(end)


def _vision_positions(start: int, grid: Sequence[int], merge: int) -> np.ndarray:
    grid_t, grid_h, grid_w = int(grid[0]), int(grid[1]) // merge, int(grid[2]) // merge
    t = np.arange(grid_t, dtype=np.int64)
    h = np.arange(grid_h, dtype=np.int64) + start
    w = np.arange(grid_w, dtype=np.int64) + start
    T, H, W = np.meshgrid(t, h, w, indexing="ij")
    out = np.stack([T, H, W], axis=0).reshape(3, -1)
    out[0] += start
    return out


def position_ids(input_ids: Sequence[int], image_grid_thw, image_token_id: int, merge_size: int = 2) -> np.ndarray:
    """int64 [3, 1, len(input_ids)]."""
    ids = np.asarray(input_ids, dtype=np.int64)
    types = (ids == image_token_id).astype(np.int64)
    grids = [] if image_grid_thw is None or len(image_grid_thw) == 0 else list(image_grid_thw)
    grid_iter = iter(grids)
    chunks: List[np.ndarray] = []
    cursor = 0
    for kind, start, end in _runs(types):
        if kind == 0:
            length = end - start
            chunks.append(np.broadcast_to(np.arange(length, dtype=np.int64), (3, length)) + cursor)
            cursor += length
        else:
            try:
                grid = next(grid_iter)
            except StopIteration:
                raise ValueError("more image token runs than image grids") from None
            chunks.append(_vision_positions(cursor, grid, merge_size))
            cursor += max(int(grid[1]), int(grid[2])) // merge_size
    if sum(1 for _ in grid_iter):
        raise ValueError("image grids were never placed in the prompt")
    if not chunks:
        return np.zeros((3, 1, 0), dtype=np.int64)
    positions = np.concatenate(chunks, axis=1)
    if positions.shape[1] != len(ids):
        raise ValueError("position_ids length %d does not match %d input ids" % (positions.shape[1], len(ids)))
    return positions[:, None, :].astype(np.int64)
