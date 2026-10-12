# -*- coding: utf-8 -*-
"""Gate (informational on 0.8B): do choice answers flip when the options are reordered?

JEVision reads each option at its own <|box_end|> in a causal row, so the answer is NOT permutation
invariant by construction (the reference's option_isolation, which would make it so, is unavailable on
hybrid backbones). This measures how often the argmax changes under reversal, on the text choice cases
of the goldens, so a regression is visible; it fails only if more than --max-flip of the cases flip.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify"))
from _common import model_dir  # noqa: E402
from jevision_decide import Decider, choice  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fp16")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-flip", type=float, default=0.35)
    args = ap.parse_args()
    d = Decider(model_dir(), variant=args.variant, intra_op_threads=args.threads, load_vision=False)
    golden = json.loads((ROOT / "artifacts/golden_text.json").read_text(encoding="utf-8"))
    n = flips = 0
    for case in golden["cases"]:
        for qid, spec in case["questions"].items():
            if spec["type"] != "choice" or len(spec["criteria"]) < 2:
                continue
            items = list(spec["criteria"].items())
            fwd = d.decide(case["state"], {"q": choice(spec["instructions"], dict(items))})["q"]
            rev = d.decide(case["state"], {"q": choice(spec["instructions"], dict(reversed(items)))})["q"]
            n += 1
            same = fwd["choice"] == rev["choice"]
            flips += not same
            print("%s %-18s %-8s fwd=%-18s rev=%-18s" % ("ok " if same else "-- ", case["id"], qid, fwd["choice"], rev["choice"]))
    print("\nflips under reversal: %d / %d" % (flips, n))
    return 1 if n and flips / n > args.max_flip else 0


if __name__ == "__main__":
    raise SystemExit(main())
