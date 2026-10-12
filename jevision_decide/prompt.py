# -*- coding: utf-8 -*-
"""Token sequences, with no model involved (so the token-id gate runs without onnxruntime sessions).

Text route   : <|fim_prefix|> state
               per question: <|fim_middle|> instr (<|box_start|> option <|box_end|>)... <|fim_suffix|>
Visual route : <|im_start|>user\n  (<|vision_start|> <|image_pad|> x N <|vision_end|>) per image
               <|fim_prefix|> state, then the same per-question branch, then <|im_end|>\n (never read)
"""

from __future__ import annotations

import json
import pathlib
from typing import List, Sequence, Tuple, Union

from .protocol import SPECIAL, RenderedQuestion, escape
from .tokenizer import Tokenizer

__all__ = ["Prompter"]


class Prompter:
    def __init__(self, model_dir: Union[str, pathlib.Path]):
        model_dir = pathlib.Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.image_token_id: int = config["image_token_id"]
        self.tokenizer = Tokenizer(str(model_dir / "tokenizer.json"))
        t = self.tokenizer
        self.ids = {name: t.token_id(name) for name in SPECIAL + ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>")}
        self.ids["<|image_pad|>"] = self.image_token_id
        if t.token_id("<|image_pad|>") != self.image_token_id:
            raise ValueError("config image_token_id does not match the tokenizer's <|image_pad|>")

    def encode_user(self, text: str) -> List[int]:
        """kev.model.user_tokens: caller text can never produce a control token."""
        return self.tokenizer.encode(escape(text))

    def branch(self, rq: RenderedQuestion) -> Tuple[List[int], List[int], int]:
        """-> (ids, offsets of each <|box_end|>, offset of <|fim_suffix|>), all within the branch."""
        ids = self.ids
        out = [ids["<|fim_middle|>"]] + self.encode_user(rq.instr)
        ends = []
        for o in rq.options:
            out += [ids["<|box_start|>"]] + self.encode_user(o) + [ids["<|box_end|>"]]
            ends.append(len(out) - 1)
        out.append(ids["<|fim_suffix|>"])
        return out, ends, len(out) - 1

    def text_prefix(self, state_text: str) -> List[int]:
        return [self.ids["<|fim_prefix|>"]] + self.encode_user(state_text)

    def visual_prefix(self, state_text: str, token_counts: Sequence[int]) -> List[int]:
        out = [self.ids["<|im_start|>"]] + self.tokenizer.encode("user\n")
        for n in token_counts:
            out += [self.ids["<|vision_start|>"]] + [self.image_token_id] * int(n) + [self.ids["<|vision_end|>"]]
        return out + self.text_prefix(state_text)

    def visual_tail(self) -> List[int]:
        return [self.ids["<|im_end|>"]] + self.tokenizer.encode("\n")
