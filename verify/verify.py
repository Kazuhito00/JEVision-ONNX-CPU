# -*- coding: utf-8 -*-
"""Gate: the ONNX runtime vs the reference runtime (PyTorch fp32 goldens, artifacts/golden_*.json).

Three stages, reported separately so a failure says where it started:

  1. token ids        EXACT match of the ids the reference ran (a mismatch here is a prompt/tokenizer
                      bug, never numerical noise, and is judged before any model output)
  2. hidden vectors   h_decide and h_options at the readout positions (max |d| and cosine)
  3. z logits, probs  max |dz| (raw and shift-free), max |dp|, and decision agreement

Decisions are noul p(yes) >= 0.5, choice argmax, score argmax-level; a different label counts as a real
disagreement only if the reference was not already within the numeric tolerance of a tie.

    python verify/verify.py --variant fp32 --route all
    python verify/verify.py --variant fp16 --route text

Needs the goldens (artifacts/golden_*.json and .npz, tracked). Image cases whose picture is not tracked here
(artifacts/images/) are looked up under $JEVISION_IMAGE_ROOT, sha256 checked first, and skipped when missing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify"))
from _common import model_dir, resolve_image  # noqa: E402
from jevision_decide import Decider  # noqa: E402,F401
from jevision_decide.cli import add_device_args, load_decider, providers_line  # noqa: E402
from jevision_decide.protocol import render  # noqa: E402

#: variant -> (max |dz|, max |dp|, max |dh| relative to the reference's max |h|)
TOLERANCE = {("fp32", "text"): (2e-3, 2e-3, 2e-3), ("fp32", "visual"): (0.1, 5e-3, 2e-3),
             ("fp16", "text"): (0.25, 0.02, 0.01), ("fp16", "visual"): (0.25, 0.02, 0.01)}


def pick(answer):
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "noul":
        return "yes" if answer["noul"] >= 0.5 else "no"
    p = answer["probabilities"]
    return max(p, key=p.get)


def margin(probs):
    """Top-1 minus top-2 probability of the reference (noul: |p-0.5|*2 is the same quantity for 2 options)."""
    r = sorted(probs, reverse=True)
    return r[0] - r[1] if len(r) > 1 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="fp16")
    ap.add_argument("--route", default="all", choices=["text", "visual", "all"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--model", default=None, help="model folder (default: $JEVISION_MODEL_DIR, else models/jevision)")
    ap.add_argument("--chunk", type=int, default=1024, help="prefill chunk (small values test chunked prefill)")
    ap.add_argument("--only", default="")
    add_device_args(ap)
    args = ap.parse_args()

    routes = ["text", "visual"] if args.route == "all" else [args.route]
    decider = load_decider(args, pathlib.Path(args.model) if args.model else model_dir(), variant=args.variant, intra_op_threads=args.threads, load_vision="visual" in routes, prefill_chunk=args.chunk)
    totals = dict(q=0, ids_ok=0, agree=0, ties=0, worst_dz=0.0, worst_dzc=0.0, worst_dp=0.0, worst_dh=0.0, worst_cos=1.0)
    failures = skipped = 0
    for route in routes:
        golden = json.loads((ROOT / "artifacts" / ("golden_%s.json" % route)).read_text(encoding="utf-8"))
        hid = np.load(ROOT / "artifacts" / ("golden_%s.npz" % route))
        z_tol, p_tol, h_tol = TOLERANCE[(args.variant, route)]
        print("== %s route, %s, %d cases (reference %s)" % (route, args.variant, len(golden["cases"]), golden["dtype"]))
        for case in golden["cases"]:
            if args.only and case["id"] not in args.only.split(","):
                continue
            images = [resolve_image(im["ref"], im["sha256"]) for im in case["images"]]
            if None in images:
                skipped += 1
                print("-- %-18s skipped (picture not tracked here; set JEVISION_IMAGE_ROOT)" % case["id"])
                continue
            images = [str(p) for p in images]
            problems = []
            if render(case["state"]) != case["record_state"]:
                problems.append("state rendering differs from the reference's")
            results = decider.score_request(case["state"], case["questions"], images)
            lines = []
            for r, b in zip(results, case["branches"]):
                ref_ids = b["ids"]
                if route == "text":
                    ref_decide, ref_opts = b["decide"], b["opts"]
                    ref_head = ref_ids
                    tail_ok = True
                else:                                   # reference ids also hold <|im_end|>\n after the decide token
                    ref_decide, ref_opts = b["qpos"] + b["decide"], [b["qpos"] + o for o in b["opts"]]
                    ref_head = ref_ids[:ref_decide + 1]
                    tail_ok = ref_ids[ref_decide + 1:] == decider.prompter.visual_tail()
                ids_ok = (r["ids"] == ref_head and r["decide"] == ref_decide and r["opts"] == ref_opts and tail_ok)
                totals["q"] += 1
                totals["ids_ok"] += ids_ok
                if not ids_ok:
                    k = next((i for i, (x, y) in enumerate(zip(r["ids"], ref_head)) if x != y), min(len(r["ids"]), len(ref_head)))
                    problems.append("%s: TOKEN IDS differ at %d (len ours=%d ref=%d)" % (b["qid"], k, len(r["ids"]), len(ref_head)))
                    lines.append("      %-9s XX token ids differ" % b["qid"][:9])
                    continue
                hd, ho = hid["%s/%s/hd" % (case["id"], b["qid"])], hid["%s/%s/ho" % (case["id"], b["qid"])]
                scale = float(max(np.abs(hd).max(), np.abs(ho).max()))
                dh = float(max(np.abs(r["h_decide"] - hd).max(), np.abs(r["h_opts"] - ho).max())) / scale
                a, c = r["h_decide"].astype(np.float64), hd.astype(np.float64)
                cos = float((a @ c) / (np.linalg.norm(a) * np.linalg.norm(c)))
                z_ref = np.asarray(b["z"])
                dz = float(np.abs(r["z"] - z_ref).max())
                dzc = float(np.abs((r["z"] - r["z"].mean()) - (z_ref - z_ref.mean())).max())
                p_ref = np.asarray(b["probs"])
                dp = float(np.abs(r["probs"] - p_ref).max())
                ours = r["rq"]
                from jevision_decide.protocol import make_answer
                ans_o, ans_r = make_answer(ours, r["probs"]), case["answers"][b["qid"]]
                same = pick(ans_o) == pick(ans_r)
                near_tie = (not same) and margin(list(p_ref)) <= 2 * p_tol
                ok = (same or near_tie) and dz < z_tol and dp < p_tol and dh < h_tol
                for k_, v in (("worst_dz", dz), ("worst_dzc", dzc), ("worst_dp", dp), ("worst_dh", dh)):
                    totals[k_] = max(totals[k_], v)
                totals["worst_cos"] = min(totals["worst_cos"], cos)
                totals["agree"] += same or near_tie
                totals["ties"] += near_tie
                if not ok:
                    problems.append("%s: outside tolerance" % b["qid"])
                lines.append("      %-9s %s ours=%-12s ref=%-12s |dh|/max=%.1e cos=%.6f |dz|=%.4f (shift-free %.4f) |dp|=%.5f%s"
                             % (b["qid"][:9], "ok " if ok else "XX ", pick(ans_o), pick(ans_r), dh, cos, dz, dzc, dp,
                                "  (near-tie)" if near_tie else ""))
            failures += bool(problems)
            print("%s %-18s %s" % ("ok " if not problems else "XX ", case["id"],
                                   "(%d image%s)" % (len(images), "s" if len(images) != 1 else "") if images else ""))
            for line in lines:
                print(line)
            for p in problems:
                print("      !! " + p)
    print("\nvariant=%s  token-id exact %d/%d   decisions %d/%d agree (%d near-tie)" %
          (args.variant, totals["ids_ok"], totals["q"], totals["agree"], totals["q"], totals["ties"]))
    print("worst |dh|/max %.2e   worst cosine(h_decide) %.6f   worst |dz| %.4f (shift-free %.4f)   worst |dp| %.5f" %
          (totals["worst_dh"], totals["worst_cos"], totals["worst_dz"], totals["worst_dzc"], totals["worst_dp"]))
    print("execution providers actually used: %s" % providers_line(decider))
    print("image cases skipped (picture not available): %d" % skipped)
    print("failures: %d" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
