# -*- coding: utf-8 -*-
"""Translate a Rust `regex` pre-tokenizer pattern into one Python's `re` accepts.

A tokenizer.json states its pre-tokenizer pattern in the Rust regex dialect,
which uses Unicode property escapes (``\\p{L}``, ``\\p{N}``, ``\\p{M}``) that
Python's ``re`` does not support and the ``regex`` package would -- but that is
a dependency this runtime does not want. Each class is expanded to explicit
codepoint ranges from the generated ``_uniprops`` table instead.

``\\s`` is expanded too, and deliberately: Rust matches whitespace by the
Unicode White_Space property, while Python's ``re.UNICODE`` ``\\s`` also
matches U+001C..U+001F. Leaving it alone would silently split differently on
those four characters.

Translating the pattern rather than hardcoding one keeps the same code working
across checkpoints whose patterns differ -- Qwen3 uses ``\\p{L}+`` where
Qwen3.5 uses ``[\\p{L}\\p{M}]+``.
"""

from __future__ import annotations

import re
from typing import List

from ._uniprops import char_body

__all__ = ["translate", "UnsupportedPattern"]

#: Rust property name -> key in the generated range table.
_CLASSES = {"L": "L", "N": "N", "M": "M"}


class UnsupportedPattern(ValueError):
    """The pattern uses a construct this translator does not implement."""


def translate(pattern: str) -> str:
    """Return an equivalent pattern for Python's ``re``."""
    out: List[str] = []
    in_class = False
    i = 0
    while i < len(pattern):
        char = pattern[i]

        if char == "\\" and i + 1 < len(pattern):
            nxt = pattern[i + 1]

            if nxt == "p" or nxt == "P":
                if pattern[i + 2 : i + 3] != "{":
                    raise UnsupportedPattern("expected \\p{...} at %d" % i)
                end = pattern.index("}", i + 3)
                name = pattern[i + 3 : end]
                if name not in _CLASSES:
                    raise UnsupportedPattern("unsupported Unicode class \\p{%s}" % name)
                body = char_body(_CLASSES[name])
                if nxt == "P":  # negated property
                    if in_class:
                        raise UnsupportedPattern("\\P{...} inside a character class")
                    out.append("[^" + body + "]")
                else:
                    out.append(body if in_class else "[" + body + "]")
                i = end + 1
                continue

            if nxt == "s":
                body = char_body("S")
                out.append(body if in_class else "[" + body + "]")
                i += 2
                continue

            if nxt == "S":
                if in_class:
                    raise UnsupportedPattern("\\S inside a character class")
                out.append("[^" + char_body("S") + "]")
                i += 2
                continue

            out.append(pattern[i : i + 2])  # any other escape passes through
            i += 2
            continue

        if char == "[" and not in_class:
            in_class = True
        elif char == "]" and in_class:
            in_class = False
        out.append(char)
        i += 1

    translated = "".join(out)
    try:
        re.compile(translated)
    except re.error as exc:  # pragma: no cover - a bug in this translator
        raise UnsupportedPattern("translated pattern does not compile: %s" % exc) from exc
    return translated
