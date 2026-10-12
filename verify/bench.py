# -*- coding: utf-8 -*-
"""Median latency of a 2-question text request and a 1-image request (1 warm-up + N runs).

    python verify/bench.py [--device cpu|cuda|auto] [--require-gpu] [--runs 7]

Timing is only meaningful on a quiet machine; the script says which provider really ran.
"""
import argparse
import pathlib
import statistics
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify"))
from _common import model_dir  # noqa: E402
from jevision_decide import choice, noul  # noqa: E402
from jevision_decide.cli import add_device_args, load_decider, providers_line  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    add_device_args(ap)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--variant", default="fp16")
    a = ap.parse_args()
    from PIL import Image
    img = pathlib.Path(tempfile.gettempdir()) / "jev_bench.png"
    Image.new("RGB", (224, 224), (200, 30, 30)).save(img)
    d = load_decider(a, model_dir(), variant=a.variant, intra_op_threads=a.threads)
    text = ("My card was charged twice for the same order and support has not replied in 3 days.", {
        "angry": noul("The customer is angry"),
        "team": choice("Which team?", {"billing": "charges and refunds", "tech": "bugs and outages"})}, [])
    image = ("Look at the attached image.", {"q": noul("The picture is mostly red")}, [str(img)])
    for label, (state, qs, images) in (("text, 2 questions", text), ("1 image, 1 question", image)):
        d.decide(state, qs, images)
        ts = []
        for _ in range(a.runs):
            t = time.perf_counter()
            d.decide(state, qs, images)
            ts.append((time.perf_counter() - t) * 1000)
        print("%-20s median %.0f ms (min %.0f, max %.0f)  provider %s" % (label, statistics.median(ts), min(ts), max(ts), providers_line(d)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
