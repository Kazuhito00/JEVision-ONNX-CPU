# -*- coding: utf-8 -*-
"""Golden outputs from JEVision's OWN runtime (runtime/kev, PyTorch fp32, CPU).

    uv run --group export export/make_golden.py text      # artifacts/golden_text.json
    uv run --group export export/make_golden.py visual    # artifacts/golden_visual.json

One process per route (the two routes load different adapters). Nothing here is reimplemented: the
reference ``kev.checkpoint.Checkpoint`` / ``kev.visual_model.VisualDecisionModel`` / ``kev.api`` do
the work, and only observers are attached:

  * ``encode`` / ``_prepared`` are wrapped to record the token ids and the readout offsets,
  * a forward hook on ``model.head`` records h_decide, h_options and the z logits.

Per question the JSON holds the token ids (the exact-match gate), the offsets, z, the unrounded
probabilities and the typed answer. Hidden vectors go to ``artifacts/golden_<route>.npz`` and, for images,
the processor's pixel_values to ``_work/golden_visual_pixels.npz`` (untracked scratch; artifacts/golden_visual_pixels.npz
keeps the synthetic pictures only). Pictures other than the synthetic ones come from $JEVISION_IMAGE_ROOT..
The JSON only names images by path and sha256.

Needs the pinned base under models/hf-base and the reference runtime under models/bundle/runtime (export/fetch_base.py
--runtime); the HF cache that
``from_pretrained`` wants is assembled offline from it (hard links).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = ROOT / "_work"
BASE_REV = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
os.environ["HF_HOME"] = str(WORK / "hf_home")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, str(ROOT / "models" / "bundle" / "runtime"))
sys.path.insert(0, str(ROOT / "verify"))


def build_hf_cache():
    snap = WORK / "hf_home" / "hub" / "models--Qwen--Qwen3.5-0.8B-Base" / "snapshots" / BASE_REV
    snap.mkdir(parents=True, exist_ok=True)
    refs = snap.parent.parent / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(BASE_REV)
    for f in (ROOT / "models" / "hf-base").iterdir():
        dst = snap / f.name
        if not dst.exists():
            try:
                os.link(f, dst)
            except OSError:
                shutil.copyfile(f, dst)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_image(ref: str) -> pathlib.Path:
    """@tiny and the synthetic pictures are tracked in artifacts/images/; any other picture is looked up under
    $JEVISION_IMAGE_ROOT (the golden's path, e.g. eval_common/real_images/final/bananas.jpg)."""
    from _common import resolve_image as find
    if ref == "@tiny":
        p = ROOT / "artifacts" / "images" / "tiny.png"
        if not p.exists():
            from PIL import Image, ImageDraw
            im = Image.new("RGB", (40, 30), (230, 20, 20))
            ImageDraw.Draw(im).rectangle([10, 8, 30, 22], fill=(250, 120, 120))
            im.save(p)
        return p
    p = find(ref)
    if p is None:
        raise SystemExit("picture %r not found: set JEVISION_IMAGE_ROOT to the folder that holds it" % ref)
    return p


def data_url(path: pathlib.Path) -> dict:
    """A System One image input. The media type is taken from the content, not the file name
    (one eval image is an RGBA PNG called .jpg)."""
    from PIL import Image
    raw = path.read_bytes()
    fmt = Image.open(io.BytesIO(raw)).format
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[fmt]
    return {"data_url": "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode("ascii"))}


def tolist(t):
    return [float(x) for x in t.detach().double().flatten().tolist()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("route", choices=["text", "visual"])
    ap.add_argument("--text-adapter", default="jevvision-text")
    ap.add_argument("--only", default="", help="comma separated case ids (debug)")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    build_hf_cache()
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    import transformers
    from kev.api import SystemOneRequest, to_answers, to_record
    from cases import all_cases

    bundle = ROOT / "models" / "bundle"
    cases = all_cases(args.route)
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]

    captured = {"head": [], "enc": [], "prep": []}

    def hook(_module, inputs, output):
        captured["head"].append((inputs[0].detach().float().clone(), inputs[1].detach().float().clone(),
                                 output.detach().float().clone()))

    started = time.time()
    if args.route == "text":
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev import model as kev_model
        run = bundle / "text" / args.text_adapter
        ck = Checkpoint(str(run))
        tok, model = ck.load("cpu", LoadOptions(dtype=torch.float32, backend="torch"))
        model.head.register_forward_hook(hook)
        encode = model.encode

        def spy_encode(tok_, rec, **kw):
            enc = encode(tok_, rec, **kw)
            captured["enc"].append(enc)
            return enc

        model.encode = spy_encode
        from kev.serve import Server
        server = Server(ck, tok, model, "cpu")
        info = {"adapter": args.text_adapter, "head_temperature": float(model.head.temperature)}
        pixel_store = None
    else:
        from kev.visual_model import VisualDecisionModel
        visual = VisualDecisionModel.from_run(bundle, "cpu", max_input_tokens=80000)
        visual.head.register_forward_hook(hook)
        prepared_fn = visual._prepared

        def spy_prepared(rec, images):
            out = prepared_fn(rec, images)
            captured["prep"].append(out)
            return out

        visual._prepared = spy_prepared
        info = {"adapter": "adapter (visual sidecar)", "head_temperature": float(getattr(visual.head, "temperature", 1.0))}
        pixel_store = {}
    print("loaded in %.0f s  (transformers %s, torch %s)" % (time.time() - started, transformers.__version__, torch.__version__),
          flush=True)

    hidden_store = {}
    out_cases = []
    for case in cases:
        t0 = time.time()
        for k in captured:
            captured[k].clear()
        images_in = [resolve_image(r) for r in case.get("images", [])]
        req = SystemOneRequest(state=case["state"], questions=case["questions"],
                               images=[data_url(p) for p in images_in] or None)
        rec, meta = to_record(req)
        record = {"id": case["id"], "state": case["state"], "questions": case["questions"],
                  "images": [{"ref": r, "sha256": sha256(p), "bytes": p.stat().st_size}
                             for r, p in zip(case.get("images", []), images_in)],
                  "record_state": rec["state"], "branches": []}
        if args.route == "text":
            probs, m = server.probs(rec)
            enc = captured["enc"][-1]
            S, Sp, rows = kev_model.rows_of(enc)
            for q, r in enumerate(rows):
                record["branches"].append({
                    "qid": meta[q]["id"], "ids": S + r["ids"], "pos": Sp + r["pos"],
                    "decide": len(S) + r["decide"], "opts": [len(S) + o for o in r["opts"]],
                    "options": rec["questions"][q]["options"], "instr": rec["questions"][q]["instr"]})
            record["usage"] = m
        else:
            from kev.media import decode_image
            pil = [decode_image(i) for i in req.images]
            probs = [p.tolist() for p in visual.probs(rec, pil)]
            prepared, tokens = captured["prep"][-1]
            for q, (batch, qpos, options, decision) in enumerate(prepared):
                record["branches"].append({
                    "qid": meta[q]["id"], "ids": batch["input_ids"][0].tolist(), "qpos": qpos,
                    "decide": decision, "opts": options,       # offsets relative to qpos
                    "options": rec["questions"][q]["options"], "instr": rec["questions"][q]["instr"]})
            first = prepared[0][0]
            record["image_grid_thw"] = first["image_grid_thw"].tolist()
            record["pixel_values_shape"] = list(first["pixel_values"].shape)
            pixel_store[case["id"] + "__pixel_values"] = first["pixel_values"].numpy().astype(np.float32)
            pixel_store[case["id"] + "__grid"] = first["image_grid_thw"].numpy().astype(np.int64)
            record["usage"] = dict(visual.last_usage)
        # head observations are in question order (the hybrid path calls the head once per question, in order)
        if len(captured["head"]) != len(rec["questions"]):
            raise SystemExit("%s: expected %d head calls, saw %d" % (case["id"], len(rec["questions"]), len(captured["head"])))
        for q, (hd, ho, z) in enumerate(captured["head"]):
            b = record["branches"][q]
            b["z"] = tolist(z)
            b["probs"] = [float(x) for x in probs[q]]
            hidden_store["%s/%s/hd" % (case["id"], b["qid"])] = hd.numpy()
            hidden_store["%s/%s/ho" % (case["id"], b["qid"])] = ho.numpy()
        record["answers"] = to_answers(probs, meta)
        out_cases.append(record)
        print("  %-18s %5.1f s  %s" % (case["id"], time.time() - t0, {
            k: (v.get("choice") if v["type"] == "choice" else v["noul"] if v["type"] == "noul" else v["score"])
            for k, v in record["answers"].items()}), flush=True)

    out = ROOT / "artifacts" / ("golden_%s.json" % args.route)
    out.parent.mkdir(exist_ok=True)
    doc = {"model": "JEVision", "route": args.route, "dtype": "float32", "device": "cpu",
           "reference": "runtime/kev of divyanshx11/JEVision (revision e1cfde19), torch %s, transformers %s"
                        % (torch.__version__, transformers.__version__), **info, "cases": out_cases}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    WORK.mkdir(exist_ok=True)
    np.savez_compressed(ROOT / "artifacts" / ("golden_%s.npz" % args.route), **hidden_store)
    if pixel_store is not None:
        np.savez(WORK / "golden_visual_pixels.npz", **pixel_store)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
