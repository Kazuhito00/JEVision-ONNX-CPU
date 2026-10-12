# -*- coding: utf-8 -*-
"""Text decisions with JEVision, on onnxruntime + numpy.

    python demo_inference_text.py
    python demo_inference_text.py --state "..." --choice "team=billing,technical"
    python demo_inference_text.py --variant fp32

No generation and no vocabulary head: the question's options are read out by a pointer head, so the
answer is always one of the options supplied.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from jevision_decide import choice, default_model_dir, noul, score  # noqa: E402
from jevision_decide.cli import add_device_args, load_decider, providers_line  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent

EXAMPLE_STATE = {
    "ticket": "My card was charged twice for the same order and support has not replied in 3 days.",
    "customer": {"tenure_months": 14, "vip": False},
}
EXAMPLE_QUESTIONS = {
    "angry": noul("The customer is angry"),
    "team": choice("Which team should handle this?",
                   {"billing": "charges and refunds", "tech": "bugs and outages", "sales": "pricing and contracts"}),
    "urgency": score("How urgent is this?", ["low", "medium", "high", "critical"]),
}


def parse_questions(args):
    questions = {}
    for spec in args.choice or []:
        name, _, rest = spec.partition("=")
        options = [v.strip() for v in rest.split(",") if v.strip()]
        questions[name.strip()] = choice(args.instructions or name.strip(), {o: "" for o in options})
    for spec in args.score or []:
        name, _, rest = spec.partition("=")
        questions[name.strip()] = score(args.instructions or name.strip(), [v.strip() for v in rest.split(",") if v.strip()])
    for spec in args.noul or []:
        name, _, rest = spec.partition("=")
        questions[name.strip()] = noul(rest.strip() or name.strip())
    return questions


def show(name: str, a: dict) -> None:
    if a["type"] == "noul":
        print("%-10s noul   p(yes)=%.3f" % (name, a["raw_probabilities"][1]))
        return
    spread = "  ".join("%s=%.3f" % kv for kv in sorted(a["probabilities"].items(), key=lambda kv: -kv[1]))
    if a["type"] == "choice":
        print("%-10s choice %-20s confidence %.2f\n           %s" % (name, a["choice"], a["confidence"], spread))
    else:
        best = max(a["probabilities"], key=a["probabilities"].get)
        print("%-10s score  %-20s expected %.2f, confidence %.2f\n           %s"
              % (name, a["legend"][best], a["score"], a["confidence"], spread))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", help="text or JSON; omit to run the built-in example")
    ap.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    ap.add_argument("--score", action="append", metavar="NAME=low,...,high")
    ap.add_argument("--noul", action="append", metavar="NAME=statement")
    ap.add_argument("--instructions", default="")
    ap.add_argument("--model", default=None, help="model folder (default: $JEVISION_MODEL_DIR, else models/jevision)")
    ap.add_argument("--variant", default="fp16")
    ap.add_argument("--threads", type=int, default=6)
    add_device_args(ap)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.state is None:
        state, questions = EXAMPLE_STATE, EXAMPLE_QUESTIONS
    else:
        try:
            state = json.loads(args.state)
        except json.JSONDecodeError:
            state = args.state
        questions = parse_questions(args)
        if not questions:
            ap.error("give at least one --choice, --score or --noul")

    t = time.perf_counter()
    decider = load_decider(args, args.model or default_model_dir(), variant=args.variant, intra_op_threads=args.threads, load_vision=False)
    load = time.perf_counter() - t
    t = time.perf_counter()
    answers = decider.decide(state, questions)
    elapsed = time.perf_counter() - t
    if args.json:
        print(json.dumps(answers, ensure_ascii=False, indent=2))
        return 0
    print("state:\n  %s\n" % json.dumps(state, ensure_ascii=False)[:160])
    for name, a in answers.items():
        show(name, a)
    print("\n%d questions, the state is prefilled once (%d input tokens)" % (len(answers), decider.last_usage["input_tokens"]))
    print("load %.1f s, inference %.2f s (%s)" % (load, elapsed, args.variant))
    print("execution providers: " + providers_line(decider))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
