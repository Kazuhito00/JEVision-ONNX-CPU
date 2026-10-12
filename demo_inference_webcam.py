# -*- coding: utf-8 -*-
"""Webcam demo: is there a person in the picture?

Grabs frames with OpenCV, asks JEVision a yes/no question about the newest frame and draws the answer on the live image.

    uv run --group webcam demo_inference_webcam.py                      # camera 0
    uv run --group webcam demo_inference_webcam.py --camera 1 --mirror
    uv run --group webcam demo_inference_webcam.py --question "Is the person wearing glasses?"
    uv run --group webcam demo_inference_webcam.py --camera photo.jpg   # a still image or a video file instead of a camera
    uv run --no-sync demo_inference_webcam.py --device cuda --require-gpu   # GPU (onnxruntime-gpu installed)

One decision takes about a second on the CPU (the first one also loads the decoder), so it runs in its own thread: the window keeps showing the live camera image and
the banner changes whenever a decision for the newest frame is ready ("analyzing..." until the first one). Press q or Esc to quit.

OpenCV is needed for this demo only (the ``webcam`` dependency group); the runtime itself still needs only onnxruntime, numpy and pillow.
The text on the window is ASCII because OpenCV's putText cannot draw Japanese; the question may be in any language.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from jevision_decide import Decider, default_model_dir, noul  # noqa: E402
from jevision_decide.cli import add_device_args, load_decider, providers_line  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GREEN, RED, GREY = (60, 180, 60), (50, 50, 210), (110, 110, 110)


class Worker(threading.Thread):
    """Decide on the newest frame, over and over, without ever blocking the display."""

    def __init__(self, decider: Decider, question: str, size: int):
        super().__init__(daemon=True)
        self.decider, self.question, self.size = decider, question, size
        self.lock = threading.Lock()
        self.frame = None            # newest BGR frame, set by the display loop
        self.result = None           # (probability of "yes", seconds, count)
        self.error = None
        self.count = 0
        self.stop = threading.Event()

    def submit(self, frame) -> None:
        with self.lock:
            self.frame = frame

    def latest(self):
        with self.lock:
            return self.result

    def run(self) -> None:
        while not self.stop.is_set():
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.02)
                continue
            picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            picture.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)   # fewer image tokens
            started = time.perf_counter()
            try:
                out = self.decider.decide("Look at the attached image.", {"q": noul(self.question)}, images=[picture])
                probability = out["q"]["raw_probabilities"][1]
            except Exception as exc:      # keep the window alive and show the problem
                self.error = str(exc).splitlines()[0][:100]
                time.sleep(1.0)
                continue
            self.count += 1
            with self.lock:
                self.result = (probability, time.perf_counter() - started, self.count)


def open_source(source: str):
    """(read_function, is_still): a camera index, a video file, or a still image that is returned again and again."""
    path = pathlib.Path(source)
    if path.suffix.lower() in IMAGE_SUFFIXES and path.exists():
        still = cv2.imread(str(path))
        if still is None:
            raise SystemExit("cannot read the image %s" % source)
        return (lambda: (True, still.copy())), True, None
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise SystemExit("cannot open the camera / video %r (try --camera 1, or give an image or video file)" % source)
    return capture.read, False, capture


def draw(frame, result, question: str, provider: str, threshold: float, error):
    height, width = frame.shape[:2]
    bar = max(44, height // 9)
    if error:
        colour, text = GREY, "error: %s" % error
    elif result is None:
        colour, text = GREY, "analyzing..."
    else:
        probability, seconds, count = result
        yes = probability >= threshold
        colour = GREEN if yes else RED
        text = "%s  yes %.0f%%  (%.1f s, #%d)" % ("YES" if yes else "NO", 100 * probability, seconds, count)
    cv2.rectangle(frame, (0, 0), (width, bar), colour, -1)
    scale = max(0.5, bar / 60.0)
    cv2.putText(frame, text, (10, int(bar * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    if result is not None and not error:                       # a probability bar along the bottom of the banner
        cv2.rectangle(frame, (0, bar), (int(width * result[0]), bar + 6), (255, 255, 255), -1)
    footer = "%s | %s | q / Esc: quit" % (provider.replace("ExecutionProvider", ""), question)
    cv2.putText(frame, footer.encode("ascii", "replace").decode(), (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="0", help="camera index, or an image / video file (default: 0)")
    parser.add_argument("--question", default="Is there a person in the picture?", help="a yes/no question about the image")
    parser.add_argument("--threshold", type=float, default=0.5, help="probability of yes above which the banner says YES")
    parser.add_argument("--size", type=int, default=224, help="long edge of the picture given to the model (default: 224)")
    parser.add_argument("--mirror", action="store_true", help="flip the live image horizontally")
    parser.add_argument("--width", type=int, default=0, help="ask the camera for this width")
    parser.add_argument("--height", type=int, default=0, help="ask the camera for this height")
    parser.add_argument("--model", default=None, help="model folder (default: $JEVISION_MODEL_DIR, else models/jevision)")
    parser.add_argument("--variant", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--threads", type=int, default=6)
    add_device_args(parser)
    parser.add_argument("--no-window", action="store_true", help="no window: print each decision (for a quick check)")
    parser.add_argument("--max-decisions", type=int, default=0, help="stop after this many decisions (0: run until quit)")
    args = parser.parse_args(argv)

    print("loading the model ...", flush=True)
    started = time.perf_counter()
    decider = load_decider(args, args.model or default_model_dir(), variant=args.variant, intra_op_threads=args.threads)
    print("loaded in %.1f s" % (time.perf_counter() - started), flush=True)

    read, is_still, capture = open_source(args.camera)
    if capture is not None and args.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if capture is not None and args.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    worker = Worker(decider, args.question, args.size)
    worker.start()
    shown = 0
    try:
        while True:
            ok, frame = read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            worker.submit(frame)
            result = worker.latest()
            if args.no_window:
                time.sleep(0.05)
                if result is not None and result[2] > shown:
                    shown = result[2]
                    print("decision #%d: yes %.3f -> %s  (%.2f s)"
                          % (shown, result[0], "YES" if result[0] >= args.threshold else "NO", result[1]), flush=True)
            else:
                cv2.imshow("JEVision webcam", draw(frame, result, args.question, decider.provider,
                                                          args.threshold, worker.error))
                key = cv2.waitKey(1 if not is_still else 30) & 0xFF
                if key in (27, ord("q")):
                    break
                if cv2.getWindowProperty("JEVision webcam", cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.max_decisions and result is not None and result[2] >= args.max_decisions:
                break
    finally:
        worker.stop.set()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
    print("execution providers: " + providers_line(decider))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
