# -*- coding: utf-8 -*-
"""JEVision typed decisions on onnxruntime + numpy (+ Pillow for images).

No generation and no vocabulary head. Each question is read out with a pointer head::

    h        = decoder(tokens)                       # post-final-norm hidden states [seq, 1024]
    q        = Wq h[<|fim_suffix|>] + bq             # one query
    k_j      = Wk h[<|box_end|> of option j] + bk    # one key per option
    z_j      = (k_j . q) / sqrt(256) / temperature
    p        = softmax(z)

Routing (as in the reference): a request without images runs the TEXT decoder (Base + the
``jevvision-text`` LoRA); a request with images runs the VISUAL decoder (Base + the visual-sidecar
LoRA) and the vision encoder. Both decoders are separate graph files, loaded on first use.

Execution: the request's common prefix (images + state) is prefilled once and each question runs its
own branch on top of a copy of that state -- the reference's "row form" -- so a request with several
questions pays for the state once. A graph state (DeltaNet conv/recurrent + KV) is immutable here:
ORT never writes into its inputs, so the prefix state is simply fed again.

Graphs (models/jevision/onnx):
    embed_tokens          input_ids -> inputs_embeds
    vision_encoder        pixel_values, image_grid_thw -> image_features
    decoder_{text,visual} inputs_embeds, attention_mask, position_ids, past_* -> hidden_states, present_*
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort

from .mrope import position_ids
from .protocol import (MAX_IMAGES, Question, RenderedQuestion, check_options, escape, make_answer,
                       question_from_spec, render, render_question)
from .prompt import Prompter
from .providers import CPU, GpuUnavailable, check_granted, select_providers, wants_gpu

__all__ = ["Decider", "VARIANTS", "gpu_providers"]

VARIANTS = ("fp32", "fp16")
_SUFFIX = {"fp32": "", "fp16": "_fp16"}
_NUMPY_DTYPE = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int64)": np.int64}

#: the reference serving limits (kev.model.SERVE_MAX_CONTEXT, kev.visual_model's first-chunk bound)
DEFAULT_MAX_TOKENS = 80_000
DEFAULT_MAX_IMAGE_REGION = 8192


def gpu_providers() -> List[str]:
    """[CUDA, CPU] when this onnxruntime build has CUDA, else [CPU] (kept for callers that pass ``providers=`` themselves)."""
    return select_providers("auto", ort.get_available_providers())


def _session(path: pathlib.Path, threads: Optional[int], providers: Optional[Sequence[str]] = None,
             require_gpu: bool = False, log: Optional[list] = None) -> ort.InferenceSession:
    """One InferenceSession. What it really got is checked right after creation (onnxruntime may hand out a CPU-only session without
    raising): a warning on stderr, or ``GpuUnavailable`` with ``require_gpu``. ``log`` collects (file name, first provider used)."""
    if not path.exists():
        raise FileNotFoundError("%s not found (run download_model.py or export/build_onnx.py)" % path)
    options = ort.SessionOptions()
    options.log_severity_level = 3                  # errors only: the graphs carry unused-initializer notes that are not actionable
    if threads:
        options.intra_op_num_threads = threads
    wanted = list(providers) if providers else [CPU]
    try:
        sess = ort.InferenceSession(str(path), sess_options=options, providers=wanted)
    except Exception as exc:                        # a GPU that is too small / missing DLLs
        if not wants_gpu(wanted):
            raise
        message = "%s could not load %s (%s)" % (wanted[0], path.name, str(exc).splitlines()[0][:160])
        if require_gpu:
            raise GpuUnavailable(message) from exc
        print("warning: %s; using the CPU" % message, file=sys.stderr)
        sess = ort.InferenceSession(str(path), sess_options=options, providers=[CPU])
    else:
        note = check_granted(wanted, sess.get_providers(), require_gpu, path.name)
        if note:
            print(note, file=sys.stderr)
    if log is not None:
        log.append((path.name, sess.get_providers()[0]))
    return sess


def _softmax(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    e = np.exp(v - v.max())
    return e / e.sum()


class _Head:
    """The pointer head, fp32 numpy: two Linear(1024 -> 256)."""

    def __init__(self, path: pathlib.Path):
        d = np.load(path)
        self.wq, self.bq = d["q.weight"], d["q.bias"]
        self.wk, self.bk = d["k.weight"], d["k.bias"]
        self.scale = float(d["scale"])
        self.temperature = float(d["temperature"])

    def logits(self, h_decide: np.ndarray, h_opts: np.ndarray) -> np.ndarray:
        q = self.wq @ h_decide + self.bq
        k = h_opts @ self.wk.T + self.bk
        z = (k @ q) * self.scale
        return z if self.temperature == 1.0 else z / self.temperature


class _Route:
    """One decoder graph + its head."""

    def __init__(self, model_dir: pathlib.Path, name: str, variant: str, threads: Optional[int], providers=None,
                 require_gpu: bool = False, log: Optional[list] = None):
        self.name = name
        self.sess = _session(model_dir / "onnx" / ("decoder_%s%s.onnx" % (name, _SUFFIX[variant])), threads, providers,
                             require_gpu, log)
        self.head = _Head(model_dir / ("head_%s.npz" % name))
        ins = self.sess.get_inputs()
        self.embed_dtype = _NUMPY_DTYPE[next(i for i in ins if i.name == "inputs_embeds").type]
        self.empty: Dict[str, np.ndarray] = {}
        for i in ins:
            if i.name.startswith("past_"):
                shape = [1 if a == 0 else (d if isinstance(d, int) else 0) for a, d in enumerate(i.shape)]
                self.empty[i.name] = np.zeros(shape, dtype=_NUMPY_DTYPE[i.type])
        outs = [o.name for o in self.sess.get_outputs()]
        if outs[0] != "hidden_states":
            raise RuntimeError("decoder output 0 is %r, expected hidden_states" % outs[0])
        self.state_out = outs[1:]
        self.out_to_in = {o: self._past_name(o) for o in self.state_out}
        missing = set(self.out_to_in.values()) - set(self.empty)
        if missing:
            raise RuntimeError("no past input for %s" % sorted(missing)[:3])

    @staticmethod
    def _past_name(present: str) -> str:
        if present.startswith("present_conv."):
            return "past_conv." + present.split(".", 1)[1]
        if present.startswith("present_recurrent."):
            return "past_recurrent." + present.split(".", 1)[1]
        if present.startswith("present."):                    # present.N.key / present.N.value
            _, n, kv = present.split(".")
            return "past_key_values.%s.%s" % (n, kv)
        raise RuntimeError("unexpected decoder output %r" % present)

    def run(self, embeds: np.ndarray, positions: np.ndarray, state: Optional[Dict[str, np.ndarray]],
            past_len: int, keep_state: bool) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        n = embeds.shape[1]
        feeds: Dict[str, Any] = dict(state) if state is not None else dict(self.empty)
        feeds["inputs_embeds"] = embeds.astype(self.embed_dtype, copy=False)
        feeds["attention_mask"] = np.ones((1, past_len + n), dtype=np.int64)
        feeds["position_ids"] = positions
        names = ["hidden_states"] + (self.state_out if keep_state else [])
        out = self.sess.run(names, feeds)
        new_state = {self.out_to_in[o]: a for o, a in zip(self.state_out, out[1:])} if keep_state else None
        return out[0][0].astype(np.float32), new_state


class Decider:
    """Loads one variant of the graphs and answers typed questions."""

    def __init__(
        self,
        model_dir: Union[str, pathlib.Path],
        variant: str = "fp16",
        intra_op_threads: Optional[int] = None,
        load_vision: bool = True,
        image_config=None,
        media_root: Optional[Union[str, pathlib.Path]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_image_region: int = DEFAULT_MAX_IMAGE_REGION,
        prefill_chunk: int = 1024,
        providers: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        require_gpu: bool = False,
    ):
        """Where the graphs run (every graph, same providers). Default: the CPU.

        ``device``: "cpu" | "cuda" | "auto" (see providers.select_providers); or ``providers``: an explicit list such as
        ``["CUDAExecutionProvider", "CPUExecutionProvider"]``. ``require_gpu``: raise ``GpuUnavailable`` (before anything is loaded
        when CUDA is not in this build, else as soon as a session lands on the CPU) instead of warning and using the CPU.
        What was really used: ``decider.provider`` / ``decider.providers_used``."""
        if variant not in VARIANTS:
            raise ValueError("unknown variant %r; pick one of %s" % (variant, list(VARIANTS)))
        if device is not None:
            if providers is not None:
                raise ValueError("give device or providers, not both")
            providers = select_providers(device, ort.get_available_providers(), require_gpu)
            if device == "cuda" and not wants_gpu(providers):
                print("warning: CUDAExecutionProvider is not available in this onnxruntime build; running on the CPU", file=sys.stderr)
        elif require_gpu and not wants_gpu(providers or [CPU]):
            raise GpuUnavailable("require_gpu needs device='cuda' / 'auto' or a GPU provider list")
        self.require_gpu = require_gpu
        self._log: list = []
        model_dir = pathlib.Path(model_dir)
        self.model_dir, self.variant, self.threads, self.providers = model_dir, variant, intra_op_threads, providers
        self.media_root = pathlib.Path(media_root) if media_root else None
        self.max_tokens, self.max_image_region, self.prefill_chunk = max_tokens, max_image_region, prefill_chunk
        suffix = _SUFFIX[variant]

        self.embed = _session(model_dir / "onnx" / ("embed_tokens%s.onnx" % suffix), intra_op_threads, providers, require_gpu, self._log)
        self.vision = (_session(model_dir / "onnx" / ("vision_encoder%s.onnx" % suffix), intra_op_threads, providers,
                                require_gpu, self._log)
                       if load_vision else None)
        self._routes: Dict[str, _Route] = {}

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.image_token_id: int = config["image_token_id"]
        self.merge_size: int = config["vision_config"]["spatial_merge_size"]
        self.hidden_size: int = config["text_config"]["hidden_size"]
        self.prompter = Prompter(model_dir)
        self._image_config = image_config
        self.last_usage: Dict[str, Any] = {}

    # ------------------------------------------------------------- plumbing

    @property
    def image_config(self):
        """Lazy: importing image.py pulls in Pillow, which text-only use avoids."""
        if self._image_config is None:
            from .image import ImageConfig
            self._image_config = ImageConfig.from_pretrained(self.model_dir)
        return self._image_config

    @property
    def provider(self) -> str:
        """The execution provider the graphs actually run on (the decoder if one is loaded, else the embedding graph)."""
        sess = next(iter(self._routes.values())).sess if self._routes else self.embed
        return sess.get_providers()[0]

    @property
    def providers_used(self) -> Dict[str, str]:
        """{graph file: first execution provider it really got}, for every graph loaded so far (the decoders load on first use)."""
        return dict(self._log)

    def route(self, name: str) -> _Route:
        if name not in self._routes:
            self._routes[name] = _Route(self.model_dir, name, self.variant, self.threads, self.providers,
                                          self.require_gpu, self._log)
        return self._routes[name]

    def _embed(self, ids: Sequence[int]) -> np.ndarray:
        return self.embed.run(["inputs_embeds"], {"input_ids": np.asarray([list(ids)], dtype=np.int64)})[0]

    def _image_features(self, prepared) -> np.ndarray:
        if self.vision is None:
            raise RuntimeError("this Decider was built with load_vision=False; images are unavailable")
        return self.vision.run(["image_features"], {"pixel_values": prepared.pixel_values.astype(np.float32),
                                                    "image_grid_thw": prepared.image_grid_thw.astype(np.int64)})[0]

    def _resolve_images(self, images: Sequence[Any]) -> List[Any]:
        out = []
        for source in images:
            if isinstance(source, (str, pathlib.Path)):
                text = str(source)
                if not text.startswith("data:image/") and not pathlib.Path(text).is_absolute() and self.media_root:
                    text = str(self.media_root / text)
                out.append(text)
            else:
                out.append(source)
        return out

    # ------------------------------------------------------------- forward

    def _prefill(self, route: _Route, embeds: np.ndarray, positions: np.ndarray):
        """Run the prefix in chunks, carrying the state. Returns (state, length)."""
        n = embeds.shape[1]
        state, done = None, 0
        while done < n:
            end = min(done + self.prefill_chunk, n)
            _, state = route.run(embeds[:, done:end], positions[:, :, done:end], state, done, keep_state=True)
            done = end
        return state, n

    def score_request(self, state: Any, questions: Mapping[str, Union[Question, Mapping[str, Any]]],
                      images: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        """Per question: ids, offsets, hidden vectors at the readout positions, z and probabilities.
        decide() is this plus the typed answers; the verify gates read this."""
        state_text = state if isinstance(state, str) else render(state)
        visual = len(images) > 0
        if visual and not 1 <= len(images) <= MAX_IMAGES:
            raise ValueError("a request takes 1..%d images, got %d" % (MAX_IMAGES, len(images)))
        rqs: List[Tuple[str, RenderedQuestion]] = []
        for qid, spec in questions.items():
            q = spec if isinstance(spec, Question) else question_from_spec(spec)
            rq = render_question(q)
            check_options(repr(qid), rq, visual)
            rqs.append((qid, rq))
        if not rqs:
            raise ValueError("a request needs at least one question")

        route = self.route("visual" if visual else "text")
        prepared = features = None
        if visual:
            from .image import prepare_images
            prepared = prepare_images(self._resolve_images(images), self.image_config)
            counts = prepared.token_counts
            prefix = self.prompter.visual_prefix(state_text, counts)
            region_end = max(i for i, t in enumerate(prefix) if t == self.image_token_id) + 1
            if region_end > self.max_image_region:
                raise ValueError("image token region is %d tokens; the limit is %d" % (region_end, self.max_image_region))
            features = self._image_features(prepared)
        else:
            prefix = self.prompter.text_prefix(state_text)

        branches = [self.prompter.branch(rq) for _, rq in rqs]
        tail = 2 if visual else 0                       # <|im_end|>\n after the decide token in the reference prompt
        total = len(prefix) + sum(len(b[0]) + tail for b in branches)
        if total > self.max_tokens:
            raise ValueError("request uses %d tokens; the limit is %d" % (total, self.max_tokens))

        embeds = self._embed(prefix)
        if visual:
            slots = np.flatnonzero(np.asarray(prefix) == self.image_token_id)
            if slots.size != features.shape[0]:
                raise RuntimeError("vision encoder produced %d features for %d image tokens" % (features.shape[0], slots.size))
            embeds[0, slots, :] = features.astype(embeds.dtype)
            pos_prefix = position_ids(prefix, prepared.image_grid_thw, self.image_token_id, self.merge_size)
        else:
            pos_prefix = np.broadcast_to(np.arange(len(prefix), dtype=np.int64), (3, 1, len(prefix))).copy()
        state_in, n_prefix = self._prefill(route, embeds, pos_prefix)

        results = []
        for (qid, rq), (ids, ends, decide) in zip(rqs, branches):
            if visual:
                full = position_ids(prefix + ids, prepared.image_grid_thw, self.image_token_id, self.merge_size)
                pos = full[:, :, n_prefix:]
            else:
                pos = np.broadcast_to(np.arange(n_prefix, n_prefix + len(ids), dtype=np.int64), (3, 1, len(ids))).copy()
            hidden, _ = route.run(self._embed(ids), pos, state_in, n_prefix, keep_state=False)
            h_decide, h_opts = hidden[decide], hidden[ends]
            z = route.head.logits(h_decide, h_opts)
            results.append({"qid": qid, "rq": rq, "ids": prefix + ids, "decide": n_prefix + decide,
                            "opts": [n_prefix + e for e in ends], "h_decide": h_decide, "h_opts": h_opts,
                            "z": z, "probs": _softmax(z)})
        self.last_usage = {"input_tokens": total, "state_tokens": n_prefix, "images": len(images),
                           "route": route.name, "questions": len(rqs)}
        return results

    # --------------------------------------------------------------- decide

    def decide(self, state: Any, questions: Mapping[str, Union[Question, Mapping[str, Any]]],
               images: Sequence[Any] = ()) -> Dict[str, Dict[str, Any]]:
        """Answer every question about ``state`` (text, or any JSON-able object).

        Images (1..4, paths / bytes / PIL images / data URLs) go beside the state; unlike OneJev the
        state does not reference them. Without images the text decoder runs, with images the visual one.
        """
        return {r["qid"]: make_answer(r["rq"], r["probs"]) for r in self.score_request(state, questions, images)}

    def predict(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """A System One style request: {"state", "images": [path | {"data_url"}], "questions": {id: {...}}}."""
        images = [i["data_url"] if isinstance(i, Mapping) else i for i in (request.get("images") or [])]
        answers = self.decide(request.get("state"), request["questions"], images)
        return {"model": "jevision", "answers": answers, "backend": "onnxruntime"}
