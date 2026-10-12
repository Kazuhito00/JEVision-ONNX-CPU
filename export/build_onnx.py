# -*- coding: utf-8 -*-
"""Build the JEVision ONNX graphs by swapping weights into the published Qwen3.5-0.8B graphs.

JEVision = Qwen3.5-0.8B-Base (pinned revision) + a merged LoRA + a pointer head. The graphs of
``onnx-community/OneJev-0.8B-ONNX`` (Qwen3.5-0.8B architecture, same shapes) are used as containers and
EVERY decoder / embedding / merger tensor is replaced, so nothing of the OneJev fine-tune survives.

How an ONNX tensor derives from a safetensors tensor (a transpose, -exp(A_log), norm + 1, ...) is not
documented and a wrong guess does not fail -- it yields a model that runs and answers wrongly. So the
rules are first checked on a pair where both sides are known (``verify``: OneJev-0.8B safetensors vs its
own ONNX, fp16 and fp32, decoder + embedding + vision tower), and only then applied to Base + LoRA.

    uv run --group export export/build_onnx.py verify
    uv run --group export export/build_onnx.py build --precision fp16
    uv run --group export export/build_onnx.py build --precision fp32 --parts text

Changes to the graph itself (nothing else is touched):
  * decoder: the pointer head reads the post-final-norm hidden states (HF ``last_hidden_state``), so the
    tensor that feeds ``/lm_head`` is exported as ``hidden_states`` [batch, seq, 1024] and the lm_head
    (Slice + MatMul + 248320 x 1024 weight) and its ``logits`` / ``num_logits_to_keep`` are removed.
  * the weights are rewritten into a fresh ``.onnx_data`` file (lm_head is gone, nothing else moves).

Layout written under models/jevision/onnx:
    embed_tokens[_fp16]      shared by both routes (Base embeddings; no LoRA targets the embedding)
    vision_encoder[_fp16]    shared (Base merger + pos_embed; the other 146 vision tensors are byte equal)
    decoder_text[_fp16]      Base + text LoRA (text/jevvision-text)
    decoder_visual[_fp16]    Base + visual-sidecar LoRA (adapter/)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import pathlib
import re
import shutil
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "models" / "hf-base"                      # export/fetch_base.py
BUNDLE = ROOT / "models" / "bundle"                         # export/fetch_base.py (the JEVision adapters and heads)
SRC32 = ROOT / "models" / "src-onnx" / "onnx"               # export/fetch_src_onnx.py
SRC16 = SRC32                                               # export/fetch_src_onnx.py --fp16 (the *_fp16 graphs sit beside the fp32 ones)
ONEJEV_HF = ROOT / "models" / "onejev-hf"                   # export/fetch_src_onnx.py --verify-pair (only for the `verify` command)
OUT = ROOT / "models" / "jevision"

NP_OF = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16, TensorProto.INT64: np.int64}
SUFFIX = {"fp32": "", "fp16": "_fp16"}
LM = "model.language_model."
VIS = "model.visual."
ADAPTERS = {
    "text": (BUNDLE / "text" / "jevvision-text", "base_model.model.layers."),
    "visual": (BUNDLE / "adapter", "base_model.model.model.language_model.layers."),
}


class SafeTensors:
    """Minimal safetensors reader (bf16 included) over one or more files."""

    def __init__(self, paths):
        self.index = {}
        self.maps = {}
        for p in paths:
            p = pathlib.Path(p)
            f = open(p, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            n = int.from_bytes(mm[:8], "little")
            header = json.loads(mm[8:8 + n])
            header.pop("__metadata__", None)
            self.maps[p] = (f, mm, 8 + n)
            for k, v in header.items():
                self.index[k] = (p, v["dtype"], v["shape"], v["data_offsets"])

    def __contains__(self, name):
        return name in self.index

    def keys(self):
        return self.index.keys()

    def get(self, name) -> np.ndarray:
        p, dtype, shape, (a, b) = self.index[name]
        _, mm, base = self.maps[p]
        raw = np.frombuffer(mm, dtype=np.uint8, count=b - a, offset=base + a)
        if dtype == "BF16":
            out = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F16":
            out = raw.view(np.float16).astype(np.float32)
        elif dtype == "F32":
            out = raw.view(np.float32)
        else:
            raise ValueError(f"{name}: unsupported dtype {dtype}")
        return out.reshape(shape)


class Merged:
    """Base checkpoint with a LoRA folded in, in fp32: W' = W + (alpha / r) * B @ A.

    Same arithmetic as PEFT's ``merge_and_unload`` on an fp32 model (what ``kev.checkpoint`` does).
    ``get`` returns fp32 for every tensor; tensors without an adapter pass through unchanged.
    """

    def __init__(self, base: SafeTensors, adapter_dir: pathlib.Path | None, prefix: str | None):
        self.base = base
        self.deltas: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.scale = 1.0
        self.used: set[str] = set()
        if adapter_dir is None:
            return
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        if cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("lora_bias") or cfg.get("trainable_token_indices"):
            raise SystemExit("adapter uses an unsupported LoRA variant: %s" % cfg)
        self.scale = float(cfg["lora_alpha"]) / float(cfg["r"])
        ad = SafeTensors([adapter_dir / "adapter_model.safetensors"])
        pat = re.compile(r"^" + re.escape(prefix) + r"(\d+)\.(.+)\.lora_([AB])\.weight$")
        parts: dict[str, dict[str, str]] = {}
        for key in ad.keys():
            m = pat.match(key)
            if not m:
                raise SystemExit(f"adapter key outside the expected prefix {prefix!r}: {key}")
            target = f"{LM}layers.{m.group(1)}.{m.group(2)}.weight"
            parts.setdefault(target, {})[m.group(3)] = key
        for target, ab in parts.items():
            if set(ab) != {"A", "B"}:
                raise SystemExit(f"{target}: incomplete LoRA pair {sorted(ab)}")
            if target not in base:
                raise SystemExit(f"{target}: not in the base checkpoint")
            self.deltas[target] = (ad.get(ab["A"]), ad.get(ab["B"]))
        self.adapter_tensors = len(ad.keys())

    def __contains__(self, name):
        return name in self.base or (name == "lm_head.weight" and f"{LM}embed_tokens.weight" in self.base)

    def get(self, name) -> np.ndarray:
        if name == "lm_head.weight" and name not in self.base:
            name = f"{LM}embed_tokens.weight"           # tied
        w = self.base.get(name)
        if name in self.deltas:
            a, b = self.deltas[name]
            self.used.add(name)
            w = w + np.float32(self.scale) * (b @ a)
        return w


# ---- name rules: ONNX initializer -> (safetensors name, transform) ----------------------------------
LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

#: decoder: ONNX suffix -> (safetensors suffix under language_model.layers.N., transform)
DECODER_RULES = {
    "input_layernorm.weight": ("input_layernorm.weight", "norm"),
    "post_attention_layernorm.weight": ("post_attention_layernorm.weight", "norm"),
    "gdn.conv1d.weight": ("linear_attn.conv1d.weight", "squeeze1"),
    "gdn.conv1d.weight_3d": ("linear_attn.conv1d.weight", "id"),
    "gdn.A_neg_exp": ("linear_attn.A_log", "negexp"),
    "gdn.dt_bias": ("linear_attn.dt_bias", "id"),
    "gdn.norm.weight": ("linear_attn.norm.weight", "keep"),
    "gdn.in_proj_qkv.MatMul.weight": ("linear_attn.in_proj_qkv.weight", "T"),
    "gdn.in_proj_z.MatMul.weight": ("linear_attn.in_proj_z.weight", "T"),
    "gdn.in_proj_b.MatMul.weight": ("linear_attn.in_proj_b.weight", "T"),
    "gdn.in_proj_a.MatMul.weight": ("linear_attn.in_proj_a.weight", "T"),
    "gdn.out_proj.MatMul.weight": ("linear_attn.out_proj.weight", "T"),
    "attn.q_proj.MatMul.weight": ("self_attn.q_proj.weight", "T"),
    "attn.k_proj.MatMul.weight": ("self_attn.k_proj.weight", "T"),
    "attn.v_proj.MatMul.weight": ("self_attn.v_proj.weight", "T"),
    "attn.o_proj.MatMul.weight": ("self_attn.o_proj.weight", "T"),
    "attn.q_norm.layernorm.weight": ("self_attn.q_norm.weight", "norm"),
    "attn.k_norm.layernorm.weight": ("self_attn.k_norm.weight", "norm"),
    "mlp.gate_proj.MatMul.weight": ("mlp.gate_proj.weight", "T"),
    "mlp.up_proj.MatMul.weight": ("mlp.up_proj.weight", "T"),
    "mlp.down_proj.MatMul.weight": ("mlp.down_proj.weight", "T"),
}

#: vision encoder: ONNX suffix under model.layers.N. -> (safetensors suffix under visual.blocks.N., transform)
VISION_RULES = {
    "norm1_layernorm.weight": ("norm1.weight", "id"), "norm1_layernorm.bias": ("norm1.bias", "id"),
    "norm2_layernorm.weight": ("norm2.weight", "id"), "norm2_layernorm.bias": ("norm2.bias", "id"),
    "attn.qkv.MatMul.weight": ("attn.qkv.weight", "T"), "attn.qkv.Add.bias": ("attn.qkv.bias", "id"),
    "attn.proj.MatMul.weight": ("attn.proj.weight", "T"), "attn.proj.Add.bias": ("attn.proj.bias", "id"),
    "mlp.linear_fc1.MatMul.weight": ("mlp.linear_fc1.weight", "T"), "mlp.linear_fc1.Add.bias": ("mlp.linear_fc1.bias", "id"),
    "mlp.linear_fc2.MatMul.weight": ("mlp.linear_fc2.weight", "T"), "mlp.linear_fc2.Add.bias": ("mlp.linear_fc2.bias", "id"),
}
VISION_FIXED = {
    "model.embeddings.patch_embed.weight": (VIS + "patch_embed.proj.weight", "convT"),
    "model.embeddings.patch_embed.Add.bias": (VIS + "patch_embed.proj.bias", "id"),
    "model.pos_embed.weight": (VIS + "pos_embed.weight", "id"),
    "model.merger.linear1.MatMul.weight": (VIS + "merger.linear_fc1.weight", "T"),
    "model.merger.linear1.Add.bias": (VIS + "merger.linear_fc1.bias", "id"),
    "model.merger.linear2.MatMul.weight": (VIS + "merger.linear_fc2.weight", "T"),
    "model.merger.linear2.Add.bias": (VIS + "merger.linear_fc2.bias", "id"),
    "model.layers.merger.merger_norm_layernorm.weight": (VIS + "merger.norm.weight", "id"),
    "model.layers.merger.merger_norm_layernorm.bias": (VIS + "merger.norm.bias", "id"),
}


def rule_for(graph: str, name: str):
    """(safetensors name, transform) or None when the tensor is a constant / not covered."""
    if graph.startswith("decoder"):
        m = LAYER.match(name)
        if m and m.group(2) in DECODER_RULES:
            suffix, tf = DECODER_RULES[m.group(2)]
            return f"{LM}layers.{m.group(1)}.{suffix}", tf
        if name.endswith(".final_norm_layernorm.weight"):
            return f"{LM}norm.weight", "norm"
        return None                      # lm_head.MatMul.weight is removed from the graph (see cut_lm_head)
    if graph.startswith("embed") and name == "model.embed_tokens.weight":
        return f"{LM}embed_tokens.weight", "id"
    if graph.startswith("vision"):
        if name in VISION_FIXED:
            return VISION_FIXED[name]
        m = LAYER.match(name)
        if m and m.group(2) in VISION_RULES:
            suffix, tf = VISION_RULES[m.group(2)]
            return f"{VIS}blocks.{m.group(1)}.{suffix}", tf
    return None


def to_bf16(w: np.ndarray) -> np.ndarray:
    u = w.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16
    return u.astype(np.uint32).view(np.float32)


def apply(tf: str, w: np.ndarray, norm_offset: float) -> np.ndarray:
    if tf in ("id", "keep"):
        return w
    if tf == "T":
        return np.ascontiguousarray(w.T)
    if tf == "convT":                     # Conv3d [out, 3, 2, 16, 16] -> MatMul [3*2*16*16, out]
        return np.ascontiguousarray(w.reshape(w.shape[0], -1).T)
    if tf == "squeeze1":
        return w.reshape(w.shape[0], w.shape[-1])
    if tf == "negexp":
        return -np.exp(w.astype(np.float32))
    if tf == "norm":
        return w + norm_offset
    raise ValueError(tf)


# ---- ONNX helpers -----------------------------------------------------------------------------------

def graph_path(graph: str, precision: str, src: pathlib.Path | None = None) -> pathlib.Path:
    src = src or (SRC32 if precision == "fp32" else SRC16)
    return src / f"{graph}{SUFFIX[precision]}.onnx"


def load_graph(path: pathlib.Path):
    return onnx.load(str(path), load_external_data=False)


def ext_info(t: TensorProto) -> dict:
    return {e.key: e.value for e in t.external_data}


def read_any(t: TensorProto, onnx_dir: pathlib.Path) -> np.ndarray:
    if t.data_location == TensorProto.EXTERNAL:
        info = ext_info(t)
        count = int(np.prod(t.dims)) if t.dims else 1
        data = np.fromfile(onnx_dir / info["location"], dtype=NP_OF[t.data_type], count=count, offset=int(info.get("offset", 0)))
        return data.reshape(tuple(t.dims))
    return numpy_helper.to_array(t)


def covered(model, graph):
    for t in model.graph.initializer:
        r = rule_for(graph, t.name)
        if r:
            yield t, r


def detect_norm_offset(model, graph, onnx_dir, src) -> float:
    for t, (name, tf) in covered(model, graph):
        if tf == "norm":
            onnx_w = read_any(t, onnx_dir).astype(np.float32)
            w = src.get(name).astype(np.float32)
            for off in (0.0, 1.0):
                if np.allclose(onnx_w, w + off, atol=1e-2, rtol=1e-2):
                    return off
            raise SystemExit(f"cannot explain norm tensor {t.name} (max |onnx-w|={np.abs(onnx_w - w).max():.4f})")
    return 0.0


# ---- verify -----------------------------------------------------------------------------------------

def cmd_verify(a) -> int:
    """Rules vs the OneJev pair (safetensors + its published ONNX): every covered tensor must reproduce."""
    one = SafeTensors(sorted(ONEJEV_HF.glob("*.safetensors")))
    bad = 0
    for precision in ("fp16", "fp32"):
        onnx_dir = (SRC16 if precision == "fp16" else SRC32)
        for graph in ("decoder_model_merged", "embed_tokens", "vision_encoder"):
            path = onnx_dir / f"{graph}{SUFFIX[precision]}.onnx"
            if not path.exists():
                print(f"{graph}{SUFFIX[precision]}: missing, skipped")
                continue
            model = load_graph(path)
            off = detect_norm_offset(model, graph, onnx_dir, one) if graph.startswith("decoder") else 0.0
            n, worst = 0, 0.0
            for t, (name, tf) in covered(model, graph):
                want = read_any(t, onnx_dir)
                got = apply(tf, one.get(name), off)
                if tf == "keep" and precision == "fp16":
                    got = to_bf16(got)
                got = got.astype(want.dtype)
                if got.shape != want.shape:
                    print(f"  SHAPE {t.name}: onnx {want.shape} vs rule {got.shape}")
                    bad += 1
                    continue
                d = float(np.abs(got.astype(np.float32) - want.astype(np.float32)).max())
                worst = max(worst, d)
                n += 1
                # "keep": gdn.norm.weight is an fp32 parameter that the published graph stores rounded to
                # bf16 (and OneJev's fine-tune moved it by ~1 bf16 ulp): accept one bf16 ulp, build uses the exact value
                tol = 2.0 ** -7 * float(np.abs(want.astype(np.float32)).max()) if tf == "keep" else                     1e-3 * max(1.0, float(np.abs(want.astype(np.float32)).max()))
                if d > tol:
                    print(f"  MISMATCH {t.name} <- {name} [{tf}]  max|d|={d:.5f}")
                    bad += 1
            rest = [t.name for t in model.graph.initializer
                    if rule_for(graph, t.name) is None and not t.name.startswith("/model/constants/")]
            print(f"{graph}{SUFFIX[precision]}: {n} tensors reproduced, norm offset {off}, worst |d| {worst:.2e}; "
                  f"not covered: {rest}")
    print("RESULT:", "FAIL" if bad else "OK (the rules reproduce the published OneJev ONNX)")
    return 1 if bad else 0


# ---- graph surgery ----------------------------------------------------------------------------------

def producer_cast_type(model, tensor: str):
    for n in model.graph.node:
        if tensor in n.output and n.op_type == "Cast":
            return next(at.i for at in n.attribute if at.name == "to")
    return None


def cut_lm_head(model, hidden_name: str = "hidden_states") -> str:
    """Export the tensor that feeds the lm_head as ``hidden_states`` and drop the lm_head."""
    graph = model.graph
    nodes = {n.name: n for n in graph.node}
    slice_node = nodes["/lm_head/num_logits_to_keep/Slice"]
    tap = slice_node.input[0]
    head_nodes = [n for n in graph.node if n.name.startswith("/lm_head/")]
    removed_out = {o for n in head_nodes for o in n.output}
    if {o.name for o in graph.output if o.name == "logits"} - removed_out:
        raise SystemExit("logits is not produced by the lm_head nodes")
    elem = producer_cast_type(model, tap)
    if elem is None:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        elem = next(v.type.tensor_type.elem_type for v in inferred.graph.value_info if v.name == tap)
    for n in head_nodes:
        graph.node.remove(n)
    for o in [o for o in graph.output if o.name == "logits"]:
        graph.output.remove(o)
    for i in [i for i in graph.input if i.name == "num_logits_to_keep"]:
        graph.input.remove(i)
    for v in [v for v in graph.value_info if v.name in removed_out]:
        graph.value_info.remove(v)
    for t in [t for t in graph.initializer if t.name == "lm_head.MatMul.weight"]:
        graph.initializer.remove(t)
    # the tap keeps its name inside the graph; an Identity gives the output its public name
    graph.node.append(onnx.helper.make_node("Identity", [tap], [hidden_name], name="/hidden_states/Identity"))
    graph.output.insert(0, onnx.helper.make_tensor_value_info(hidden_name, elem, ["batch_size", "sequence_length", 1024]))
    return tap


def write_graph(model, src_dir: pathlib.Path, out_dir: pathlib.Path, stem: str, repl: dict[str, np.ndarray]) -> dict:
    """Save ``model`` as ``stem``.onnx with a fresh ``stem``.onnx_data: replaced tensors from ``repl``,
    everything else copied from ``src_dir`` byte for byte."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data_name = stem + ".onnx_data"
    stats = {"replaced": 0, "copied": 0, "bytes": 0}
    srcs: dict[str, np.memmap] = {}
    with open(out_dir / data_name, "wb") as fh:
        for t in model.graph.initializer:
            if t.data_location != TensorProto.EXTERNAL:
                if t.name in repl:
                    t.raw_data = repl[t.name].astype(NP_OF[t.data_type]).tobytes()
                    stats["replaced"] += 1
                continue
            info = ext_info(t)
            if t.name in repl:
                raw = np.ascontiguousarray(repl[t.name].astype(NP_OF[t.data_type])).tobytes()
                stats["replaced"] += 1
            else:
                if info["location"] not in srcs:
                    srcs[info["location"]] = np.memmap(src_dir / info["location"], dtype=np.uint8, mode="r")
                o, n = int(info.get("offset", 0)), int(info["length"])
                raw = bytes(srcs[info["location"]][o:o + n])
                stats["copied"] += 1
            pad = (-fh.tell()) % 64
            fh.write(b"\0" * pad)
            offset = fh.tell()
            fh.write(raw)
            del t.external_data[:]
            for k, v in (("location", data_name), ("offset", str(offset)), ("length", str(len(raw)))):
                e = t.external_data.add()
                e.key, e.value = k, v
            stats["bytes"] += len(raw)
    onnx.save(model, str(out_dir / (stem + ".onnx")))
    return stats


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ---- build ------------------------------------------------------------------------------------------

def build_graph(kind: str, precision: str, source: Merged, out_dir: pathlib.Path, stem: str) -> dict:
    graph = {"embed": "embed_tokens", "vision": "vision_encoder", "text": "decoder_model_merged",
             "visual": "decoder_model_merged"}[kind]
    src_dir = SRC32 if precision == "fp32" else SRC16
    model = load_graph(graph_path(graph, precision))
    off = detect_norm_offset(model, graph, src_dir, source) if graph.startswith("decoder") else 0.0
    repl = {}
    for t, (name, tf) in covered(model, graph):
        new = apply(tf, source.get(name), off)
        if tuple(new.shape) != tuple(t.dims):
            raise SystemExit(f"{t.name}: tensor shape {new.shape} != graph {tuple(t.dims)}")
        repl[t.name] = new
    tap = cut_lm_head(model) if graph.startswith("decoder") else None
    stats = write_graph(model, src_dir, out_dir, stem, repl)
    stats.update({"tap": tap, "norm_offset": off})
    return stats


def head_npz(route: str, out: pathlib.Path):
    import torch
    if route == "text":
        d = torch.load(BUNDLE / "text" / "jevvision-text" / "head.pt", map_location="cpu")
        state, temp = d["head"], float(d.get("temperature", 1.0))
    else:
        state, temp = torch.load(BUNDLE / "pointer_head.pt", map_location="cpu", weights_only=True), 1.0
    arrays = {k: v.float().numpy() for k, v in state.items()}
    arrays["temperature"] = np.float32(temp)
    arrays["scale"] = np.float32(1.0 / np.sqrt(arrays["q.weight"].shape[0]))
    np.savez(out, **arrays)


def patch_tokenizer(base_dir: pathlib.Path, out: pathlib.Path) -> list:
    """The Base repo's tokenizer.json lacks 11 added tokens (<think>, </think>, <tool_response>, ...) that
    its tokenizer_config.json declares; transformers adds them at load time, the ``tokenizers`` library
    reading tokenizer.json alone would not (and would split ``<think>`` into pieces). Same ids, same flags."""
    tj = json.loads((base_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tc = json.loads((base_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    have = {t["id"] for t in tj["added_tokens"]}
    added = []
    for i, t in sorted(tc["added_tokens_decoder"].items(), key=lambda kv: int(kv[0])):
        if int(i) not in have:
            tj["added_tokens"].append({"id": int(i), "content": t["content"], "single_word": t["single_word"],
                                       "lstrip": t["lstrip"], "rstrip": t["rstrip"], "normalized": t["normalized"],
                                       "special": t["special"]})
            added.append(t["content"])
    # transformers 5.17 (the reference's pinned version) builds its Qwen tokenizer with a split regex that has
    # ``\p{L}+`` where tokenizer.json / tokenizer_config.json say ``[\p{L}\p{M}]+`` (combining marks, e.g.
    # Devanagari, split differently). The reference runs the former, so that is what is matched here.
    split = tj["pre_tokenizer"]["pretokenizers"][0]["pattern"]
    split["Regex"] = split["Regex"].replace(r"[\p{L}\p{M}]+", r"\p{L}+").replace(r"[^\s\p{L}\p{M}\p{N}]", r"[^\s\p{L}\p{N}]")
    out.write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    return added


def cmd_build(a) -> int:
    base = SafeTensors([BASE_DIR / "model.safetensors-00001-of-00001.safetensors"])
    precision = a.precision
    out_dir = pathlib.Path(a.out) / "onnx"
    manifest_path = pathlib.Path(a.out) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    parts = a.parts.split(",")
    for part in parts:
        t0 = time.time()
        if part in ("text", "visual"):
            adir, prefix = ADAPTERS[part]
            source = Merged(base, adir, prefix)
            stem = f"decoder_{part}{SUFFIX[precision]}"
            kind = part
        else:
            source = Merged(base, None, None)
            stem = {"embed": "embed_tokens", "vision": "vision_encoder"}[part] + SUFFIX[precision]
            kind = part
        stats = build_graph(kind, precision, source, out_dir, stem)
        if part in ("text", "visual"):
            if len(source.used) != len(source.deltas):
                raise SystemExit(f"{part}: {len(source.deltas) - len(source.used)} LoRA targets were not used by the graph")
            stats["lora_modules_merged"] = len(source.used)
            stats["lora_scale"] = source.scale
        print(f"{stem}: {stats}  ({time.time() - t0:.0f} s)", flush=True)
        for f in (stem + ".onnx", stem + ".onnx_data"):
            manifest["files"][f] = {"bytes": (out_dir / f).stat().st_size, "sha256": sha256(out_dir / f)}
    for part in ("text", "visual"):
        head_npz(part, pathlib.Path(a.out) / f"head_{part}.npz")
    for f in ("config.json", "tokenizer_config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        shutil.copyfile(BASE_DIR / f, pathlib.Path(a.out) / f)
    manifest["tokenizer_added_tokens_patched"] = patch_tokenizer(BASE_DIR, pathlib.Path(a.out) / "tokenizer.json")
    manifest.update({"base": "Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
                     "jevision": "divyanshx11/JEVision@e1cfde19bfae6548a46812c64a9a6f1a67459c95",
                     "graph_containers": "onnx-community/OneJev-0.8B-ONNX@afc03812cedf078ce660ec410fcf8597b4c4ec5d"})
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("written to", a.out)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["verify", "build"])
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--parts", default="embed,vision,text,visual", help="comma separated: embed, vision, text, visual")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    sys.exit({"verify": cmd_verify, "build": cmd_build}[a.cmd](a))


if __name__ == "__main__":
    main()
