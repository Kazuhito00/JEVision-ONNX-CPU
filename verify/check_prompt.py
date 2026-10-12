# -*- coding: utf-8 -*-
"""Gate: token ids (and readout offsets) vs the reference runtime, with NO model involved.

artifacts/golden_{text,visual}.json hold, per question, the exact ids the reference ran. This rebuilds them
from the request alone (protocol.render_question + prompt.Prompter, tokenizers + numpy only; the image token
counts come from image.prepare_images when Pillow is there) and demands exact equality, so a prompt or
tokenizer bug is caught here, before any numerical comparison could blur it.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify"))
from _common import model_dir, resolve_image  # noqa: E402
from jevision_decide.prompt import Prompter  # noqa: E402
from jevision_decide.protocol import question_from_spec, render, render_question  # noqa: E402


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def main() -> int:
    model = model_dir()
    P = Prompter(model)
    failures = total = skipped = 0
    for route in ("text", "visual"):
        golden = json.loads((ROOT / "artifacts" / ("golden_%s.json" % route)).read_text(encoding="utf-8"))
        cfg = None
        for case in golden["cases"]:
            problems = []
            if render(case["state"]) != case["record_state"]:
                problems.append("state rendering differs")
            counts = []
            if route == "visual":
                from jevision_decide.image import ImageConfig, prepare_images
                cfg = cfg or ImageConfig.from_pretrained(model)
                files = [resolve_image(i["ref"], i["sha256"]) for i in case["images"]]
                if None in files:
                    skipped += 1
                    print("-- %-7s %-18s skipped (picture not tracked here; set JEVISION_IMAGE_ROOT)" % (route, case["id"]))
                    continue
                counts = prepare_images([str(f) for f in files], cfg).token_counts
            for b in case["branches"]:
                total += 1
                rq = render_question(question_from_spec(case["questions"][b["qid"]]))
                if rq.options != b["options"] or rq.instr != b["instr"]:
                    problems.append("%s: option/instruction texts differ" % b["qid"])
                ids, ends, decide = P.branch(rq)
                prefix = P.visual_prefix(case["record_state"], counts) if route == "visual" else P.text_prefix(case["record_state"])
                ours = prefix + ids
                if route == "text":
                    ref, ref_decide, ref_opts = b["ids"], b["decide"], b["opts"]
                else:
                    ref_decide = b["qpos"] + b["decide"]
                    ref_opts = [b["qpos"] + o for o in b["opts"]]
                    ref = b["ids"][:ref_decide + 1]
                    if b["ids"][ref_decide + 1:] != P.visual_tail():
                        problems.append("%s: tail after <|fim_suffix|> differs" % b["qid"])
                if ours != ref:
                    problems.append("%s: ids differ at %d (len ours=%d ref=%d)" % (b["qid"], first_diff(ours, ref), len(ours), len(ref)))
                if len(prefix) + decide != ref_decide or [len(prefix) + e for e in ends] != ref_opts:
                    problems.append("%s: readout offsets differ" % b["qid"])
            failures += bool(problems)
            print("%s %-7s %-18s %d questions%s" % ("ok " if not problems else "XX ", route, case["id"], len(case["branches"]),
                                                    "  (%d image%s)" % (len(counts), "s" if len(counts) != 1 else "") if counts else ""))
            for p in problems:
                print("      " + p)
    print("\nfailures: %d cases, %d questions checked, %d image cases skipped" % (failures, total, skipped))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
