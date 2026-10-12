# -*- coding: utf-8 -*-
"""Gate: the per-request limits, and that large requests run in bounded memory (peak working set is printed,
nothing is timed).

    python verify/check_limits.py [--big]      # --big also runs a 16k-token text and a 2560x1440 image
"""
import argparse
import ctypes
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify"))
from _common import model_dir  # noqa: E402
from jevision_decide import Decider, choice, noul, score  # noqa: E402


def peak_mb() -> float:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [(n, ctypes.c_size_t) for n in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(pmc)
    k32, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 2**20


def expect_error(label, fn):
    try:
        fn()
    except (ValueError, RuntimeError) as e:
        print("ok  %-38s -> error: %s" % (label, str(e)[:70]))
        return 0
    print("XX  %-38s -> no error" % label)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    from PIL import Image
    tmp = pathlib.Path(tempfile.gettempdir())
    small = tmp / "jev_small.png"
    Image.new("RGB", (224, 224), (200, 30, 30)).save(small)
    d = Decider(model_dir(), variant="fp16", intra_op_threads=a.threads)
    bad = 0
    q = {"q": noul("x")}
    bad += expect_error("5 images", lambda: d.decide("s", q, [str(small)] * 5))
    bad += expect_error("256 options", lambda: d.decide("s", {"q": choice("x", {"k%d" % i: "t" for i in range(256)})}))
    bad += expect_error("score with 1 level", lambda: d.decide("s", {"q": score("x", ["only"])}))
    bad += expect_error("image option empty", lambda: d.decide("s", {"q": choice("x", {" ": ""})}, [str(small)]))
    bad += expect_error("no question", lambda: d.decide("s", {}))
    d2 = Decider(model_dir(), variant="fp16", intra_op_threads=a.threads, load_vision=False, max_tokens=500)
    bad += expect_error("text above max_tokens (500)", lambda: d2.decide("word " * 600, q))
    bad += expect_error("images with load_vision=False", lambda: d2.decide("s", q, [str(small)]))
    ok = d.decide("s", {"q": choice("x", {"k%d" % i: "t" for i in range(255)})})
    print("ok  255 options run (argmax %s)  peak RSS %.0f MB" % (ok["q"]["choice"], peak_mb()))
    img4 = d.decide("s", q, [str(small)] * 4)
    print("ok  4 images run: p(yes)=%.3f  tokens=%d  peak RSS %.0f MB" % (img4["q"]["raw_probabilities"][1], d.last_usage["input_tokens"], peak_mb()))
    if a.big:
        r = d.decide("Record number one about invoices and parcels. " * 1800, q)
        print("ok  %d-token text: p(yes)=%.3f  peak RSS %.0f MB" % (d.last_usage["input_tokens"], r["q"]["raw_probabilities"][1], peak_mb()))
        big = tmp / "jev_big.png"
        im = Image.effect_noise((2560, 1440), 60).convert("RGB")
        im.save(big)
        r = d.decide("s", q, [str(big)])
        print("ok  2560x1440 image (%d tokens): p(yes)=%.3f  peak RSS %.0f MB" % (d.last_usage["input_tokens"], r["q"]["raw_probabilities"][1], peak_mb()))
    print("\nfailures: %d" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
