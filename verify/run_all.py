# -*- coding: utf-8 -*-
"""Run every gate this repository has, in order, and summarise.

Gates that need only the runtime dependencies run anywhere; gates that need an oracle (transformers, the
goldens' hidden-vector files) are skipped with a reason. Model folder: $JEVISION_MODEL_DIR (default models/jevision).
For the full set (the tokenizer oracle gate needs the verify group and models/hf-base):

    uv run --group verify verify/run_all.py [--variant fp16|fp32] [--fuzz 5000]
"""
import argparse
import importlib.util
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: (script, argv builder, files it needs, modules it needs, purpose)
GATES = [
    ("check_prompt.py", lambda a: [], [], [], "token ids + readout offsets vs the reference (no model)"),
    ("check_image.py", lambda a: [], ["artifacts/golden_visual_pixels.npz"], ["PIL"], "image preprocessing vs the processor's pixel_values"),
    ("check_tokenizer_fuzz.py", lambda a: [str(a.fuzz)], [], ["transformers", "tokenizers"], "tokenizers vs transformers AutoTokenizer"),
    ("verify.py", lambda a: ["--variant", a.variant, "--route", "all", "--threads", str(a.threads)],
     ["artifacts/golden_text.npz"], ["PIL"], "hidden vectors, z, probabilities, decisions vs PyTorch fp32"),
    ("verify.py", lambda a: ["--variant", a.variant, "--route", "text", "--threads", str(a.threads), "--chunk", "32"],
     ["artifacts/golden_text.npz"], [], "same on the text route with 32-token prefill chunks"),
    ("check_bias.py", lambda a: ["--variant", a.variant, "--threads", str(a.threads)], [], [], "answers when the options are reversed"),
]


def have(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--variant", default="fp16")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--fuzz", type=int, default=5000)
    a = ap.parse_args()
    print(ROOT.name, "\n")
    failed = passed = skipped = 0
    for script, argv, files, needs, purpose in GATES:
        absent = [m for m in needs if not have(m)] + [f for f in files if not (ROOT / f).exists()]
        if absent:
            print("  %-24s skip  needs %s -- %s" % (script[:-3], ", ".join(absent), purpose))
            skipped += 1
            continue
        t0 = time.perf_counter()
        r = subprocess.run([a.python, str(ROOT / "verify" / script), *argv(a)], cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        lines = [l.strip() for l in (r.stdout + r.stderr).splitlines() if l.strip() and not l.startswith(("\x1b", "[0;"))]
        summary = next((l for l in reversed(lines) if any(k in l for k in ("failures:", "decisions", "mismatches:", "flips"))), lines[-1] if lines else "")
        ok = r.returncode == 0
        failed += not ok
        passed += ok
        print("  %-24s %-4s %-60s %5.1fs" % (script[:-3], "ok" if ok else "XX", summary[:60], time.perf_counter() - t0))
        if not ok:
            for l in lines[-6:]:
                print("      " + l[:110])
    print("\n%d passed, %d failed, %d skipped" % (passed, failed, skipped))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
