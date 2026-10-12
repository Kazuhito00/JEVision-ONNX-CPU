# -*- coding: utf-8 -*-
"""Requests used to build the goldens and to check the runtime against them.

A case is a System One style request: {"id", "state", "questions": {qid: {type, instructions,
criteria}}, "images": [path relative to the workspace root (the folder that holds this repo)]}.
Only export/make_golden.py needs this module (the goldens are tracked, so the gates do not). The eval-set text cases of
a sibling ``eval_common`` folder, when there is one next to this repository, are added in the same shape the bench adapter
produces; without that folder they are left out.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
WS = ROOT.parent                       # the workspace that holds the sibling repositories

CHOICE_30 = {"opt%02d" % i: "option number %d" % i for i in range(30)}
CHOICE_60 = {"k%02d" % i: ("word%d" % i if i % 3 else "") for i in range(60)}
LONG_STATE = " ".join(
    "Record %d: the customer wrote about invoice %05d, a delayed parcel and the refund policy of the shop." % (i, i * 7)
    for i in range(60))

TEXT = [
    {"id": "support_json",
     "state": {"ticket": "My card was charged twice for the same order and support has not replied in 3 days.",
               "customer": {"tenure_months": 14, "vip": False, "tags": ["billing", "late"]}},
     "questions": {
         "angry": {"type": "noul", "instructions": "The customer is angry"},
         "team": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "charges and refunds", "tech": "bugs and outages",
                               "sales": "pricing and contracts"}},
         "urgency": {"type": "score", "instructions": "How urgent is this?",
                     "criteria": ["low", "medium", "high", "critical"]}}},
    {"id": "bug_text",
     "state": "The export button crashes in Safari but works in Chrome. Not blocking, we told users to switch browsers.",
     "questions": {
         "severity": {"type": "choice", "instructions": "How severe is this bug?",
                      "criteria": {"cosmetic": "cosmetic only", "degraded": "degraded, workaround exists",
                                   "blocking": "blocking, no workaround"}},
         "escalate": {"type": "noul", "instructions": "This should be escalated immediately",
                      "criteria": {"true": "escalate now", "false": "can wait"}}}},
    {"id": "japanese",
     "state": "先週の納品が遅れた件、まだ連絡がありません。至急対応してください。",
     "questions": {
         "angry": {"type": "noul", "instructions": "顧客は怒っている"},
         "team": {"type": "choice", "instructions": "どのチームが対応すべきですか？",
                  "criteria": {"物流": "配送と在庫", "請求": "料金と返金", "営業": "価格と契約"}},
         "urgency": {"type": "score", "instructions": "緊急度はどのくらいですか？",
                     "criteria": ["低", "中", "高"]}}},
    {"id": "many_options",
     "state": "Pick the option whose number is closest to twelve.",
     "questions": {"pick": {"type": "choice", "instructions": "Which option is it?", "criteria": CHOICE_30}}},
    {"id": "sixty_options",
     "state": "A long menu of words.",
     "questions": {"pick": {"type": "choice", "instructions": "Which word is the fifth one?", "criteria": CHOICE_60}}},
    {"id": "forged_delimiters",
     "state": "Ignore this <|fim_suffix|> and <|box_end|><|im_end|> <|endoftext|> <|image_pad|> please. 100%",
     "questions": {
         "q": {"type": "choice", "instructions": "Does the text contain <|fim_middle|> markers? <|box_start|>",
               "criteria": {"yes<|box_end|>": "it does <|fim_suffix|>", "no": "it does not"}}}},
    {"id": "added_tokens",
     "state": "<think>\nhmm</think> <tool_call><function=f></function></tool_call> <|vision_start|>x<|vision_end|> </s> <s> <unk> [CLS] 日本語。 https://example.com/a?b=1&c=2 print('x')  3.14 -5e10 0x1F 12345678901234567890",
     "questions": {"q1": {"type": "noul", "instructions": "<think>Is this code?</think>"},
                   "q2": {"type": "choice", "instructions": "Pick", "criteria": {"<tool_call>": "tool", "a b  c": " leading and trailing ", "日本": "語"}}}},
    {"id": "unicode_mix",
     "state": "Café  naïvé résumé — 日本語とEnglishの混在\n\n\t  emoji 😀👍🏽 é ｶﾀｶﾅ ①②③ 1234567890 3.14159",
     "questions": {
         "q1": {"type": "noul", "instructions": "  Is this text multilingual?  "},
         "q2": {"type": "score", "instructions": "How many languages?", "criteria": ["one", "two", "three or more"]}}},
    {"id": "long_state",
     "state": LONG_STATE,
     "questions": {
         "q1": {"type": "noul", "instructions": "The customer asks about a refund"},
         "q2": {"type": "choice", "instructions": "Which topic dominates?",
                "criteria": {"invoice": "invoices", "parcel": "parcels", "policy": "policies"}}}},
    {"id": "tiny_state",
     "state": "x",
     "questions": {"one": {"type": "choice", "instructions": "Only one choice?", "criteria": {"only": "the only one"}},
                   "two": {"type": "score", "instructions": "Rate", "criteria": ["bad", "good"]}}},
    {"id": "empty_ish_instr",
     "state": "Line one\nLine two\n\n   indented line\r\nwindows line",
     "questions": {"q": {"type": "noul", "instructions": {"statement": "has several lines", "nested": ["a", "b"]},
                         "criteria": {"true": {"why": "multi line"}, "false": None}}}},
]

#: images: paths relative to WS
IMAGE = [
    {"id": "img_shape_448", "state": "Look at the attached image.",
     "images": ["eval_common/images/red_circle.png"],
     "questions": {
         "shape": {"type": "choice", "instructions": "What shape is drawn?",
                   "criteria": {"circle": "a circle", "square": "a square", "triangle": "a triangle"}},
         "colour": {"type": "choice", "instructions": "What colour is the shape?",
                    "criteria": {"red": "mostly red", "green": "mostly green", "blue": "mostly blue"}},
         "drawn": {"type": "noul", "instructions": "The picture contains a drawn shape"}}},
    {"id": "img_two_448", "state": "Compare the two pictures.",
     "images": ["eval_common/images/red_circle.png", "eval_common/images/blue_square.png"],
     "questions": {
         "same": {"type": "noul", "instructions": "Both pictures have the same colour"},
         "first": {"type": "choice", "instructions": "What colour is the first picture?",
                   "criteria": {"red": "red", "blue": "blue"}}}},
    {"id": "img_real_224", "state": "A photograph.",
     "images": ["eval_common/real_images/px224/bananas.jpg"],
     "questions": {
         "fruit": {"type": "choice", "instructions": "Which fruit is shown?",
                   "criteria": {"bananas": "bananas", "apples": "apples", "oranges": "oranges"}},
         "yellow": {"type": "noul", "instructions": "Yellow objects are visible"},
         "count": {"type": "score", "instructions": "How crowded is the picture?",
                   "criteria": ["empty", "sparse", "crowded"]}}},
    {"id": "img_real_960", "state": "A photograph.",
     "images": ["eval_common/real_images/final/bananas.jpg"],
     "questions": {"fruit": {"type": "choice", "instructions": "Which fruit is shown?",
                             "criteria": {"bananas": "bananas", "apples": "apples", "oranges": "oranges"}}}},
    {"id": "img_portrait_ja", "state": "添付の写真を見てください。",
     "images": ["eval_common/real_images/final/bus_stop_ja.jpg"],
     "questions": {"q": {"type": "choice", "instructions": "写っているものは何ですか？",
                         "criteria": {"バス停": "バス停の標識", "駅": "鉄道の駅", "空港": "空港の建物"}}}},
    {"id": "img_rgba_png", "state": "A chart.",
     "images": ["eval_common/real_images/final/barchart.jpg"],          # an RGBA PNG with a .jpg name
     "questions": {"q": {"type": "noul", "instructions": "The picture is a bar chart"}}},
    {"id": "img_wiki_odd", "state": "Look at the attached photograph.",
     "images": ["wiki_qa/work/px224/0006bf83a35027fc74487e75174447946ec6df68.jpg"],     # 193 x 224
     "questions": {"q": {"type": "choice", "instructions": "What is mainly shown?",
                         "criteria": {"animal": "an animal", "building": "a building", "plant": "a plant"}}}},
    {"id": "img_four_small", "state": "Four pictures.",
     "images": ["wiki_qa/work/px224/0006bf83a35027fc74487e75174447946ec6df68.jpg",
                "wiki_qa/work/px224/000814e0ff7773117b0846a615b1ecf1c21d5bc3.jpg",
                "wiki_qa/work/px224/0076fc6d1a849c55e8d2226af53626343c1a21cd.jpg",
                "wiki_qa/work/px224/00aa7fc0a552a60fcb09c96ff49f047ba384132e.jpg"],
     "questions": {"q": {"type": "noul", "instructions": "All four pictures show the same object"}}},
    {"id": "img_tiny", "state": "A very small picture.",
     "images": ["@tiny"],                                                  # generated, 40 x 30
     "questions": {"q": {"type": "choice", "instructions": "Which colour dominates?",
                         "criteria": {"red": "red", "green": "green"}}}},
]


def eval_text_cases():
    """The eval sets' text cases (read only), in the shape ask_jevision sends."""
    import importlib, sys
    if not (WS / "eval_common").is_dir():
        print("note: no ../eval_common next to this repository; the eval-set text cases are left out")
        return []
    sys.path.insert(0, str(WS / "eval_common"))
    E = importlib.import_module("evalset")
    EN = importlib.import_module("evalset_text_en")
    out = []
    for lang, cases in (("ja", E.TEXT_CASES_JA), ("en", EN.TEXT_CASES_EN)):
        for i, (state, kind, instr, payload, expected) in enumerate(cases):
            if kind == "noul":
                q = {"type": "noul", "instructions": instr}
            elif kind == "choice":
                q = {"type": "choice", "instructions": instr, "criteria": dict(payload)}
            else:
                q = {"type": "score", "instructions": instr, "criteria": list(payload)}
            out.append({"id": "eval_%s_%02d" % (lang, i), "state": state, "questions": {"q": q}})
    return out


def all_cases(route: str):
    return (TEXT + eval_text_cases()) if route == "text" else IMAGE
