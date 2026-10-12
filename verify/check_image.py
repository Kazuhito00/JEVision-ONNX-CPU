# -*- coding: utf-8 -*-
"""Gate: image preprocessing vs the reference processor's own pixel_values / image_grid_thw.

The goldens (export/make_golden.py visual) keep what ``AutoProcessor`` (Qwen2VLImageProcessorFast) produced
for each image case in artifacts/golden_visual_pixels.npz (the synthetic pictures only). Our decode + smart_resize + antialiased bicubic +
patchify must give the same grid exactly and the same float32 pixels (a difference is one 8-bit grey level
= 2/255 after normalisation). The reference resizes with torchvision's integer uint8 kernel and we use
Pillow's: they agree except for +-1 (rarely +-2) grey levels in well under 1 % of the values, so the gate
allows two grey levels; anything larger is a bug.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jevision_decide.image import ImageConfig, prepare_images  # noqa: E402

sys.path.insert(0, str(ROOT / "verify"))
from _common import model_dir, resolve_image  # noqa: E402


def main() -> int:
    golden = json.loads((ROOT / "artifacts/golden_visual.json").read_text(encoding="utf-8"))
    px = np.load(ROOT / "artifacts" / "golden_visual_pixels.npz")   # only the synthetic pictures (the others are not tracked)
    cfg = ImageConfig.from_pretrained(model_dir())
    failures, worst, checked = 0, 0.0, 0
    for case in golden["cases"]:
        if case["id"] + "__pixel_values" not in px.files:
            continue
        files = [resolve_image(i["ref"], i["sha256"]) for i in case["images"]]
        if None in files:
            continue
        files = [str(f) for f in files]
        checked += 1
        got = prepare_images(files, cfg)
        ref = px[case["id"] + "__pixel_values"]
        grid = px[case["id"] + "__grid"]
        same_grid = got.image_grid_thw.tolist() == grid.tolist()
        same_shape = got.pixel_values.shape == ref.shape
        if same_grid and same_shape:
            d = np.abs(got.pixel_values - ref)
            dmax, n_diff = float(d.max()), int((d > 1e-6).sum())
        else:
            dmax, n_diff = float("inf"), -1
        ok = same_grid and same_shape and dmax <= 2 * 2 / 255 + 1e-6
        failures += not ok
        worst = max(worst, dmax if dmax != float("inf") else 0)
        print("%s %-16s grid %s %-14s max|d|=%.2e  elements differing: %d of %d" %
              ("ok " if ok else "XX ", case["id"], "same" if same_grid else "DIFFERENT", got.image_grid_thw.tolist(),
               dmax, n_diff, ref.size))
    print("\nworst |d| %.2e (one grey level = %.2e)   failures: %d / %d" % (worst, 2 / 255, failures, checked))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
