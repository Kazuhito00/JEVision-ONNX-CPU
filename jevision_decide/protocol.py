# -*- coding: utf-8 -*-
"""The JEVision request protocol, without transformers or pydantic.

Ported from the reference runtime of ``divyanshx11/JEVision`` (``kev/api.py``, ``kev/model.py``,
``kev/visual_prompt.py``). The checkpoint was trained on these exact strings.

A question becomes one pointer-readout problem, not a prompt asking for a letter::

    <|fim_prefix|> state <|fim_middle|> instructions
        <|box_start|> option 1 <|box_end|> <|box_start|> option 2 <|box_end|> ... <|fim_suffix|>

The hidden state at ``<|fim_suffix|>`` is the query and the hidden state at each ``<|box_end|>`` is a
key; the answer is the softmax over the options of q . k (see runtime.py).

    noul   -> options ["no", "yes"]  (or "no: <criteria.false>", "yes: <criteria.true>"); answer p(yes)
    choice -> options "name" or "name: description"
    score  -> options = the level descriptions, in order; answer = sum(i * p_i)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

__all__ = ["Question", "choice", "score", "noul", "question_from_spec", "render", "escape",
           "RenderedQuestion", "render_question", "make_answer", "SPECIAL", "MAX_OPTIONS", "MAX_IMAGES"]

#: kev.model.SPECIAL: state, question, option start, option end, decide
SPECIAL = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
MAX_OPTIONS = 255
MAX_IMAGES = 4

_CONTROL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def escape(text: str) -> str:
    """User text must never produce a control token: ``<|name|>`` is rewritten to ``<¦name¦>``."""
    return _CONTROL.sub(r"<¦\1¦>", text)


@dataclass
class Question:
    type: str
    instructions: Any = None
    #: choice: {name: description}; score: [description, ...]; noul: {"true": ..., "false": ...} or None
    criteria: Any = None


def choice(instructions: Any, criteria: Mapping[str, Any]) -> Question:
    """Pick one of the named options (the description may be empty / None)."""
    return Question("choice", instructions, dict(criteria))


def score(instructions: Any, levels: Sequence[Any]) -> Question:
    """Rate on an ordered scale; the first level is the lowest."""
    return Question("score", instructions, list(levels))


def noul(instructions: Any = None, true: Any = None, false: Any = None) -> Question:
    """A yes/no gate. ``true`` / ``false`` describe the two answers (optional)."""
    criteria = None
    if true is not None or false is not None:
        criteria = {"true": true, "false": false}
    return Question("noul", instructions, criteria)


def question_from_spec(spec: Mapping[str, Any]) -> Question:
    """A System One style dict: {type, instructions, criteria}."""
    kind = spec.get("type")
    if kind not in ("choice", "score", "noul"):
        raise ValueError("question type must be choice, score or noul, got %r" % (kind,))
    criteria = spec.get("criteria")
    if kind == "choice" and (not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= MAX_OPTIONS):
        raise ValueError("a choice question needs a criteria object with 1..%d options" % MAX_OPTIONS)
    if kind == "score" and (not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= MAX_OPTIONS):
        raise ValueError("a score question needs a criteria list with 2..%d levels" % MAX_OPTIONS)
    return Question(kind, spec.get("instructions"), criteria)


def render(v: Any, indent: int = 0) -> str:
    """kev.api.render: flatten str | object | array into text. Field names are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join("%s- %s" % (pad, render(x, indent + 1).lstrip()) for x in v)
    return "\n".join(
        "%s%s:\n%s" % (pad, k, render(x, indent + 1)) if isinstance(x, (dict, list))
        else "%s%s: %s" % (pad, k, render(x))
        for k, x in v.items())


def option_text(name: str, desc: Any) -> str:
    return name if desc is None or desc == "" else "%s: %s" % (name, render(desc))


@dataclass
class RenderedQuestion:
    kind: str
    keys: List[str]            # the keys the probabilities are reported under
    instr: str                 # rendered instructions
    options: List[str]         # option texts, in order
    legend: Dict[str, str]     # score only

    @property
    def n_options(self) -> int:
        return len(self.options)


def render_question(q: Question) -> RenderedQuestion:
    """kev.api.to_record, one question."""
    if q.type == "noul":
        c = q.criteria or {}
        opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        keys = ["false", "true"]
        legend: Dict[str, str] = {}
    elif q.type == "choice":
        keys = list(q.criteria)
        opts = [option_text(k, v) for k, v in q.criteria.items()]
        legend = {}
    elif q.type == "score":
        opts = [render(x) for x in q.criteria]
        keys = [str(i) for i in range(len(opts))]
        legend = dict(zip(keys, opts))
    else:
        raise TypeError("unknown question type %r" % (q.type,))
    return RenderedQuestion(q.type, keys, render(q.instructions), opts, legend)


def check_options(kind_name: str, rq: RenderedQuestion, visual: bool) -> None:
    """The limits of the reference serving path, as errors (the reference answers 422)."""
    if not 1 <= rq.n_options <= MAX_OPTIONS:
        raise ValueError("%s: needs 1..%d options, got %d" % (kind_name, MAX_OPTIONS, rq.n_options))
    if rq.kind == "score" and rq.n_options < 2:
        raise ValueError("%s: a score question needs at least 2 levels" % kind_name)
    if visual and any(not o.strip() for o in rq.options):
        raise ValueError("%s: visual options must be non-empty strings" % kind_name)


# ------------------------------------------------------------------------ answers (kev.api.to_answers)

def _r2(x: float) -> float:
    return round(float(x), 2)


def _choice_confidence(p: Sequence[float]) -> float:
    k = len(p)
    return 1.0 if k == 1 else (max(p) - 1 / k) / (1 - 1 / k)


def _score_confidence(p: Sequence[float]) -> float:
    """TypeSafe's 'distance from the modal level' approximation: 1 - E|level - mode| / (L - 1)."""
    n = len(p)
    mode = max(range(n), key=lambda i: p[i])
    return 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / (n - 1)


def make_answer(rq: RenderedQuestion, p: Sequence[float]) -> Dict[str, Any]:
    """Probabilities -> the typed answer the reference returns (rounded to 2 decimals), plus
    ``raw_probabilities`` (full precision, in option order) for callers that need an exact argmax."""
    p = [float(x) for x in p]
    if rq.kind == "noul":
        out: Dict[str, Any] = {"type": "noul", "noul": _r2(p[1])}
    elif rq.kind == "choice":
        out = {"type": "choice", "choice": rq.keys[max(range(len(p)), key=lambda i: p[i])],
               "confidence": _r2(_choice_confidence(p)),
               "probabilities": {k: _r2(v) for k, v in zip(rq.keys, p)}}
    else:
        s = sum(i * pi for i, pi in enumerate(p))
        out = {"type": "score", "score": _r2(s), "legend": dict(rq.legend),
               "probabilities": {str(i): _r2(v) for i, v in enumerate(p)},
               "confidence": _r2(_score_confidence(p))}
    out["raw_probabilities"] = p
    return out
