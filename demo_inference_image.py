# -*- coding: utf-8 -*-
"""Image decisions with JEVision, on onnxruntime + numpy + Pillow.

    python demo_inference_image.py                       # draws a red circle and asks about it
    python demo_inference_image.py --image photo.jpg --noul "a person is visible" --choice "kind=indoor,outdoor"

1..4 images per request (PNG / JPEG / WebP). A request with images runs the visual decoder; the
images are passed beside the state, they are not referenced from it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from jevision_decide import choice, default_model_dir, noul  # noqa: E402
from jevision_decide.cli import add_device_args, load_decider, providers_line  # noqa: E402
from demo_inference_text import parse_questions, show  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent


def example_image() -> str:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (448, 448), (255, 255, 255))
    ImageDraw.Draw(im).ellipse([96, 96, 352, 352], fill=(220, 30, 30))
    path = pathlib.Path(tempfile.gettempdir()) / "jevision_demo_circle.png"
    im.save(path)
    return str(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", action="append", help="image path (repeat for up to 4)")
    ap.add_argument("--state", default="Look at the attached image.")
    ap.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    ap.add_argument("--score", action="append", metavar="NAME=low,...,high")
    ap.add_argument("--noul", action="append", metavar="NAME=statement")
    ap.add_argument("--instructions", default="")
    ap.add_argument("--model", default=None, help="model folder (default: $JEVISION_MODEL_DIR, else models/jevision)")
    ap.add_argument("--variant", default="fp16")
    ap.add_argument("--threads", type=int, default=6)
    add_device_args(ap)
    args = ap.parse_args(argv)

    images = args.image or [example_image()]
    questions = parse_questions(args) or {
        "shape": choice("What shape is drawn?", {"circle": "a circle", "square": "a square", "triangle": "a triangle"}),
        "colour": choice("What colour is the shape?", {"red": "mostly red", "green": "mostly green", "blue": "mostly blue"}),
        "drawn": noul("The picture contains a drawn shape"),
    }
    t = time.perf_counter()
    decider = load_decider(args, args.model or default_model_dir(), variant=args.variant, intra_op_threads=args.threads, load_vision=True)
    load = time.perf_counter() - t
    t = time.perf_counter()
    answers = decider.decide(args.state, questions, images=images)
    elapsed = time.perf_counter() - t
    print("state: %s   images: %s\n" % (json.dumps(args.state, ensure_ascii=False)[:100], ", ".join(images)))
    for name, a in answers.items():
        show(name, a)
    u = decider.last_usage
    print("\n%d questions, one vision pass, images + state prefilled once (%d input tokens)" % (len(answers), u["input_tokens"]))
    print("load %.1f s, inference %.2f s (%s)" % (load, elapsed, args.variant))
    print("execution providers: " + providers_line(decider))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
