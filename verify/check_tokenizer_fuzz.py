# -*- coding: utf-8 -*-
"""Gate: the pure-Python runtime tokenizer vs TWO oracles, exact id equality only:
  * the Rust ``tokenizers`` library on the same (patched) tokenizer.json,
  * transformers' AutoTokenizer of the pinned Base checkpoint (what the reference runs).
Neither oracle is ever imported by jevision_decide. Corpus: random mixes of scripts, digits, punctuation,
10+ Unicode whitespace kinds, emoji (ZWJ, flags, skin tones), combining marks, ligatures, every added token,
special tokens inside user text (raw and escaped), very long strings, and real eval-style JEVision strings.

    uv run --group export verify/check_tokenizer_fuzz.py [N=100000]
"""
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify"))
from _common import model_dir  # noqa: E402
from jevision_decide.protocol import SPECIAL, escape  # noqa: E402
from jevision_decide.tokenizer import Tokenizer  # noqa: E402

WS_KINDS = [" ", "  ", "   ", "\t", "\n", "\r", "\r\n", "\n\n", "\x0b", "\x0c", "\x1c", "\x1f", "\x85", " ", " ", " ", " ",
            " ", " ", " ", " ", " ", " ", "　"]
PIECES = [
    "a", "Z", "the", "The", "ing", "'s", "'LL", "don't", "0", "7", "123", "3.14", "-", "--", "_", "/", "\\", "(", ")", "[", "]", "{", "}", "<", ">", "&",
    "=", "?", "!", "...", "。", "、", "「", "」", "日本語", "漢字", "ひらがな", "カタカナ", "ｶﾀｶﾅ", "ﾞ", "한국어", "русский", "العربية", "हिन्दी", "ไทย", "é",
    "é", "ñ", "ñ", "ß", "İ", "ǅ", "①", "½", "²", "ﬁ", "ﬂ", "ﬃ", "Ĳ", "㌔", "ｱ", "Å", "Å", "😀", "👍🏽", "🇯🇵", "🇺🇸", "👨‍👩‍👧‍👦",
    "🏳️‍🌈", "❤️", "​", "‍", "﻿", "\u0000", "\x7f", "‮", "́", "゙", "ั", "ि",
    "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "<|audio_pad|>", "<tts_pad>", "<¦fim_middle¦>", "<|", "|>", "<|x|",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|image_pad|>", "<|vision_start|>", "<|video_pad|>",
] + list(SPECIAL) + WS_KINDS
REAL = [
    "My card was charged twice for the same order and support has not replied in 3 days.", "ticket: Hi\ncustomer:\n  tenure_months: 14\n  vip: False",
    "先週の納品が遅れた件、まだ連絡がありません。至急対応してください。", "顧客は怒っている", "billing: charges and refunds", "技術サポート: 不具合・障害・エラー",
    "level 2", "Which team should handle this?", "どのチームが対応すべきですか？", "Pro プランの料金が 2 回引き落とされています。対応をお願いします。",
]


def build_corpus(n, rng):
    texts = []
    spec = list(SPECIAL)
    for r in REAL:       # the exact request shapes the runtime sends
        texts += [r, escape(r), "<|fim_prefix|>" + escape(r), "".join(["<|fim_middle|>", r, "<|box_start|>", r, "<|box_end|>", "<|fim_suffix|>"])]
    for _ in range(n - len(texts)):
        mode = rng.random()
        if mode < 0.55:
            k = rng.choice([1, 2, 3, 5, 8, 13, 30])
            t = "".join(rng.choice(PIECES) if rng.random() < 0.85 else chr(rng.choice([rng.randrange(32, 127), rng.randrange(0xA0, 0x3000), rng.randrange(0x3000, 0xD800), rng.randrange(0xE000, 0x11000), rng.randrange(0x1F000, 0x1FB00)])) for _ in range(k))
        elif mode < 0.75:
            t = " ".join(rng.choice(REAL) for _ in range(rng.randrange(1, 6)))
            t = t.replace(" ", rng.choice(WS_KINDS), rng.randrange(0, 4))
        elif mode < 0.9:
            body = rng.choice(REAL) + rng.choice(PIECES) + rng.choice(REAL)
            t = rng.choice(spec) + body + rng.choice(spec) + rng.choice(REAL) + rng.choice(spec)
            if rng.random() < 0.5:
                t = escape(t)
        elif mode < 0.97:
            t = "".join(rng.choice(PIECES) for _ in range(rng.randrange(200, 1500)))
        else:
            t = "".join(rng.choice(PIECES) for _ in range(rng.randrange(1500, 6000)))
        texts.append(t)
    return texts


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    from tokenizers import Tokenizer as Rust            # oracle 1
    from transformers import AutoTokenizer              # oracle 2
    tj = str(model_dir() / "tokenizer.json")
    rust = Rust.from_file(tj)
    hf = AutoTokenizer.from_pretrained(str(ROOT / "models" / "hf-base"))
    ours = Tokenizer(tj)
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 20261010)
    texts = build_corpus(n, rng)
    bad = 0
    for t in texts:
        a = ours.encode(t)
        b = rust.encode(t, add_special_tokens=False).ids
        c = hf(t, add_special_tokens=False).input_ids
        if not (a == b == c):
            bad += 1
            if bad <= 5:
                k = next((i for i, (x, y) in enumerate(zip(a, c)) if x != y), min(len(a), len(c)))
                print("MISMATCH rust=%s hf=%s first diff at %d: %r" % (a == b, a == c, k, t[:120]))
    print("cases: %d   mismatches: %d   (oracles: tokenizers + transformers)" % (len(texts), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
