[[Japanese](README.md)/[English](README_EN.md)]

# JEVision-ONNX-CPU

A minimal-dependency inference implementation of [`divyanshx11/JEVision`](https://huggingface.co/divyanshx11/JEVision) (a typed decision model for text and images: Qwen3.5-0.8B-Base + LoRA + pointer heads) that runs on onnxruntime and numpy only. GPU (CUDA) inference is supported too.<br>
It does not use the official runtime (which needs PyTorch, transformers and peft).

# Features
- Minimal dependencies: only onnxruntime and numpy are needed at run time (plus pillow when images are used)
- Two scripts are enough to run it: `download_model.py` (fetch the model) and `demo_inference_text.py` (inference). Images: `demo_inference_image.py`; camera: `demo_inference_webcam.py`
- Decisions match the reference implementation (PyTorch fp32): with the fp16 graphs, token ids 58/58 and decisions 58/58; the largest probability difference is 0.0046
- No generation and no vocabulary head. The hidden states of special tokens in the prompt are read out by pointer heads, so the answer is always one of the options you supplied
- Image support (1 to 4 images per request). A request without images runs the text decoder, a request with images runs the visual decoder
- The preprocessing (Qwen byte-level BPE, 3-D M-RoPE, image patching) is ported to pure Python, with cross-checks against the reference implementation included
- Runs on CPU and on GPU (CUDA) (`--device cuda`). To avoid silently falling back to the CPU, the providers that were really used are shown, and `--require-gpu` turns "no usable GPU" into an error

# Purpose of This Repository
It exists to find out:
- how far JEVision's decisions can be run with only 2 to 3 run-time dependencies
- whether a pure-Python port of the reference preprocessing builds the same inputs
- what speed and memory look like on CPU (and GPU)

The grounds for the port are in [artifacts/report.md](artifacts/report.md).

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
pillow       10.0 or later   (only when images are used)
```
That is all that running needs. Fetching the model uses the standard library only.<br>
The environment is set up with [uv](https://docs.astral.sh/uv/) (no pip / conda).<br>
Optional parts:
- extra `gpu`: `onnxruntime-gpu` (CUDA build). **It cannot coexist with `onnxruntime` (the CPU build)**; see "Running on a GPU"
- group `test`: `pytest` (`tests/`, no model needed)
- group `verify`: `tokenizers` and `transformers` for the tokenizer cross-check (`check_tokenizer_fuzz`)
- group `webcam`: OpenCV for `demo_inference_webcam.py`
- group `export`: torch, transformers and so on, for exporting the ONNX yourself (not used at run time)

# Installation

```bash
git clone https://github.com/Kazuhito00/JEVision-ONNX-CPU
cd JEVision-ONNX-CPU

# install the run-time dependencies only (onnxruntime, numpy, pillow)
uv sync
```

# Download Model
JEVision publishes neither an ONNX nor merged weights (only two LoRA adapters and the heads). The ONNX files built from Base weights with the LoRA merged in are therefore placed in [Releases](https://github.com/Kazuhito00/JEVision-ONNX-CPU/releases) (v0.0.0 is the intended tag); they are not in the repository.<br>
The script below fetches them into `models/`. It uses the standard library's HTTPS only: no git, no git-lfs.
```bash
uv run download_model.py
```
- **With no arguments it fetches every file of the Release (the four fp16 graphs, the heads, the tokenizer, the configs and the licences: 16 files, about 2.72 GB)**
- It fetches from the latest Release of `Kazuhito00/JEVision-ONNX-CPU` by default (another repository with `--repo OWNER/NAME` or the environment variable `JEVISION_ONNX_REPO`; precedence: argument > environment > default)
- Every file is checked against the SHA-256 in the Release's `manifest.json` (a mismatch deletes the file and stops)
- Files split to stay under GitHub's limit (below 2 GiB per file) are joined after download and checked (the largest file of this Release is 996,327,424 bytes, so nothing is split)
- Files already in place are skipped, and an interrupted download resumes with the same command

```bash
uv run download_model.py --list                          # list the files
uv run download_model.py --exclude "*vision_encoder*"    # text only (skip the vision encoder; about 0.2 GB less)
uv run download_model.py --tag v0.0.0                    # a given tag (default: latest)
```

A model kept elsewhere is pointed to with the environment variable `JEVISION_MODEL_DIR` (the folder that corresponds to `models/jevision`) or with `--model` of each demo.

### Files of the Release (output of `release/make_release.py`)
`uv run release/make_release.py` writes the assets, `manifest.json` and `SHA256SUMS` to `dist/release/` (**it does not upload anything**; upload them yourself, for example with `gh release create v0.0.0 dist/release/*`). Asset names are flat: the `/` of the path becomes `__`.

| Asset | Size (bytes) |
|---|---:|
| `jevision__onnx__embed_tokens_fp16.onnx` / `.onnx_data` | 637 / 508,559,360 |
| `jevision__onnx__vision_encoder_fp16.onnx` / `.onnx_data` | 187,476 / 204,798,464 |
| `jevision__onnx__decoder_text_fp16.onnx` / `.onnx_data` | 597,430 / 996,327,424 |
| `jevision__onnx__decoder_visual_fp16.onnx` / `.onnx_data` | 597,967 / 996,327,424 |
| `jevision__head_text.npz` / `jevision__head_visual.npz` | 2,100,710 each |
| `jevision__tokenizer.json` | 9,833,522 |
| `jevision__tokenizer_config.json` / `jevision__config.json` / `jevision__preprocessor_config.json` | 16,712 / 2,907 / 390 |
| `jevision__LICENSE-JEVISION` / `jevision__LICENSE-QWEN` | 11,545 / 11,343 |
| `manifest.json` / `SHA256SUMS` | read by the download script (path, size, SHA-256) |

The total is about 2.72 GB (2.53 GiB). Every file is below 2 GiB, so no splitting is needed (the default of `--max-part-gib` is 1.9).<br>
The fp32 graphs (about 5.4 GB more; the largest file, `decoder_*.onnx_data`, is 1,992,441,856 bytes = 1.86 GiB, also without splitting) are not in the Release by default (add them with `make_release.py --with-fp32`).

To build the ONNX files yourself (torch is required), see "ONNX conversion".

# Usage

### Examples (Python)
```bash
uv run demo_inference_text.py
uv run demo_inference_image.py
```
The text demo decides three questions about a support ticket at once: is the customer angry (noul), which team (choice), how urgent (score). Your own input:
```bash
uv run demo_inference_text.py --state "Server is down since 5am." --choice "team=billing,tech,sales" --noul "angry=The customer is angry" --score "urgency=low,medium,high"
```
The image demo draws a red circle by default and asks about its shape and colour. Your own pictures: `--image` (up to 4, repeatable).
```bash
uv run demo_inference_image.py --image shot.png --choice "status=paid,pending,failed" --noul "paid=The invoice is paid"
```
A 224 px thumbnail is enlarged to 256 px by the processor's minimum pixel count (65536) and costs 64 tokens per image.

```python
from jevision_decide import Decider, choice, noul, score

d = Decider("models/jevision", variant="fp16", intra_op_threads=6)   # variant: "fp16" (default) | "fp32"
d.decide({"ticket": "My card was charged twice."},
         {"angry": noul("The customer is angry"),
          "team": choice("Which team?", {"billing": "charges and refunds", "tech": "bugs"}),
          "urgency": score("How urgent?", ["low", "medium", "high"])})

d.decide("Look at the attached image.", {"q": noul("It shows a cat")}, images=["cat.jpg"])   # 1..4 images
```
Unlike OneJev, images are **not** referenced from the state with `<image:N>` (JEVision's own format). `images` accepts file paths, `data:image/...;base64,` URLs, bytes and Pillow images.<br>
Answers have the same shape as the reference implementation's (probabilities rounded to 2 decimals) plus `raw_probabilities` (unrounded).

**How it works:** it does not read the last character of the prompt but reads out pointers. Nothing is generated.
```
<|fim_prefix|> state <|fim_middle|> instr <|box_start|> opt1 <|box_end|> <|box_start|> opt2 <|box_end|> ... <|fim_suffix|>
```
The final-norm hidden state at `<|fim_suffix|>` is the query and the one at each `<|box_end|>` is a key; `z_j = (Wk h_j + bk)·(Wq h_q + bq) / sqrt(256)` goes through a softmax. The part shared by a request (images + state) is prefilled once and each question runs its own branch on top of it.

### Is there a person in the picture? (camera)
Grabs frames with OpenCV, decides "is there a person in the picture?" for the newest frame and draws the result on the live image.
```bash
uv run --group webcam demo_inference_webcam.py                        # camera 0
uv run --group webcam demo_inference_webcam.py --camera 1 --mirror
uv run --group webcam demo_inference_webcam.py --question "Is the person wearing glasses?"
uv run --group webcam demo_inference_webcam.py --camera artifacts/images/red_circle.png --no-window --max-decisions 1   # an image or video file instead of a camera
```
- Decisions run in their own thread so the live image never stalls. Each time a decision is ready the banner at the top (green = yes, red = no) and the probability of "yes" are updated
- Change the question with `--question` (a yes/no question). The picture given to the model is reduced to a 224 px long edge (`--size`)
- Press `q` or `Esc` to quit. OpenCV is a dependency of this demo only (a later `uv sync` removes it again)

### Question types
| Type | Answer |
|---|---|
| `choice(instructions, {key: description})` | one of the options (`choice`, `confidence`, `probabilities`). Options are passed as `key` or `key: description` |
| `score(instructions, [level description, ...])` | expected value on an ordinal scale (`score`, `legend`); the first level is the lowest |
| `noul(instructions)` | whether a statement holds (`noul` = probability of yes); the options are fixed to `no` / `yes` |

# Running on a GPU
`--device cuda` runs inference on CUDA (all graphs: text and visual decoders, embed, vision). The default is the CPU.

**What you need:**
- an NVIDIA GPU and the `bin` folders of CUDA 13 and cuDNN 9 on PATH (`onnxruntime-gpu` does not ship the CUDA DLLs and looks for them on PATH; `CUDA_PATH` may point to an older CUDA, so check PATH)
- `onnxruntime-gpu` **instead of** `onnxruntime`. **Never install both** (they write into the same `onnxruntime/` folder)

```bash
# take the CPU build out and put the GPU build in (pyproject.toml forbids the combination that has both)
uv sync --extra gpu --no-group cpu
# from then on run with uv run --no-sync (without --no-sync the default CPU build is put back)
uv run --no-sync demo_inference_text.py --device cuda
uv run --no-sync demo_inference_text.py --require-gpu        # stop with an error if the GPU cannot be used
uv run --no-sync verify/bench.py --device cuda
```
Back to the CPU build: `uv sync`. Example Windows PATH entries (adapt the versions; reopen the terminal after changing PATH):
```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64
C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64
```
```python
from jevision_decide import Decider, noul

d = Decider("models/jevision", device="cuda", require_gpu=True)   # device: "cpu" (default) | "cuda" | "auto"
d.decide("hello", {"q": noul("This is a greeting")})
print(d.providers_used)   # {graph file: provider it really got}; a decoder is loaded by the first decision
```
- `--device cpu` (default) is CPU only; `cuda` is CUDAExecutionProvider then CPUExecutionProvider; `auto` is `cuda` when this ONNX Runtime has CUDA and `cpu` otherwise. `--gpu` is an alias of `--device cuda`
- **It never falls back to the CPU silently.** Right after each session is created the provider it really got is checked. When CUDA was requested but the session runs on the CPU a warning is printed, and the last line of the demos (`execution providers: ...`) and the output of `verify/bench.py` list the real provider per graph
- `--require-gpu` (`require_gpu=True` in Python) makes that an error instead of a warning when CUDA cannot be used (no CUDA in this ONNX Runtime, DLLs not found, does not fit on the GPU); the exit status is not 0. It is also a way to check the environment
- Probabilities on the GPU do not match the CPU exactly (fp16 rounding, order of operations). In the measurement below the answer differed between CPU and GPU in 4 of 5364 cases, and correctness changed in 0
- What was checked: Windows 11, NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM), `onnxruntime-gpu` 1.31, CUDA 13.4, cuDNN 9.20, fp16 graphs. The fp32 graphs, DirectML and GPU runs on Linux are **not verified**

**VRAM (fp16):** in the GPU measurement below the peak VRAM was 3192 MiB for text only and 2452 MiB with images (minus about 106 MiB at the start; only the decoder that is used, text or visual, is loaded). It ran on a 4 GB GPU. fp32 is not verified.

# Performance
These are numbers from existing measurements (sources given). The conditions differ, so they are in separate tables. Numbers without grounds are not included.

### CPU vs GPU speed (GPU bench)
RTX 3050 Ti Laptop 4 GB, CUDA 13.4, cuDNN 9.20, onnxruntime-gpu 1.31, fp16, 224 px, CPU with 8 threads.

| Input | CPU per question | GPU (CUDA) per question | Ratio | Peak VRAM |
|---|---:|---:|---:|---:|
| text | 607 ms | 60 ms | 10.1x | 3192 MiB |
| image | 767 ms | 69 ms | 11.1x | 2452 MiB |

Source: `eval_common/gpu_bench/report.md` of the working workspace (not part of this repository; 2026-10-10; the same 3 text and 3 image questions; median per question of the whole `decide` call including preprocessing and tokenizing, not including load time). The GLiNER calibration of the CPU was not run.<br>
In the same measurement, the other models: OneJev 0.8B 12.6 to 16x, OpenJev 0.8B 3.2 to 3.9x. Intern-Decision, Strands, QEV and OneJev 4B were not faster on the GPU, and clef-flash ran out of VRAM and fell back to the CPU.

### Wikipedia vision bench (v1 trial run, 5364 cases = question x language)
Source: `results/runs/v1/summary_v1.md` of `wikipedia-vision-decision-bench` (CPU with 8 threads, blocks with a valid calibration; GPU is the same RTX 3050 Ti).

| | all | en | ja |
|---|---:|---:|---:|
| accuracy | 76.8% (4120/5364) | 80.7% | 73.2% |

| Subset | cases | CPU per question (median) | GPU per question (median) | Ratio |
|---|---:|---:|---:|---:|
| all | 5364 | 785 ms | 71 ms | 11.1x |
| with picture | 5017 | 787 ms | 71 ms | 11.1x |
| **text only** | 347 | 622 ms | **402 ms** | **1.5x** |

- **For the text-only questions the GPU ratio was only 1.5x** (which does not agree with the 10.1x for text in the GPU bench above). The cause was not investigated
- In 4 of the 5364 cases CPU and GPU gave different answers; correctness was the same

### Accuracy on the Wikipedia image set (まとめ（Wikipedia画像セット）.html)
Accuracy of JEVision 0.8B (fp16, CPU 8 threads, pictures at 224 px, options in their original order, asked once) on the same questions in English and Japanese. The values were recounted from the per-question results embedded in that file (time: median CPU time per question).

| Dataset | questions | en | ja | median CPU per question (en / ja) |
|---|---:|---:|---:|---:|
| 1. photos used in Wikipedia articles | 797 | 92.1% (734) | 82.9% (661) | 727 / 748 ms |
| 2. Commons expansion | 471 | 92.8% (437) | 89.4% (421) | 743 / 754 ms |
| 3. synthetic business images | 382 | 87.7% (335) | 86.6% (331) | 721 / 722 ms |
| 4. reasoning (synthetic) | 225 | 42.7% (96) | 40.9% (92) | 728 / 740 ms |

On the same questions OneJev 0.8B got 1. 91.3% / 83.9%, 2. 95.8% / 92.1%, 3. 91.6% / 89.0%, 4. 45.8% / 47.6% (en / ja). On set 4 (questions that need thinking after looking: arithmetic, applying rules and so on) both JEVision and OneJev are below half.

### For reference: JEVision's own reports
JEVision's model card lists self-reported numbers for the visual sidecar (Beans validation 51/60, generated tasks 147/150 and so on). **They are self-reported by the provider and have not been reproduced with this implementation.**

### Memory and length
- Peak RSS was about 10 GB for a 14k-token text and about 17 GB for a 3.6k-token image (2560x1440), because the vision encoder's attention grows with the square of the pixel count (`artifacts/report.md`)
- The 80000-token request limit and the 8192-token image region limit were not exercised on a real run

# ONNX conversion
JEVision ships only LoRA adapters (text and image) and the pointer heads. So the LoRA is merged into the Base weights in fp32 (`W += (alpha/r)·B@A`), and the published Qwen3.5-0.8B ONNX graphs ([onnx-community/OneJev-0.8B-ONNX](https://huggingface.co/onnx-community/OneJev-0.8B-ONNX)) are used as containers whose **every weight is replaced** (no torch export). The replacement rules are first shown to reproduce a pair where both sides are known, OneJev's safetensors and its published ONNX (`build_onnx.py verify`).
```bash
uv sync --group export
uv run --group export export/fetch_base.py                         # Base weights + the JEVision bundle (HTTPS only; about 1.8 GB)
uv run --group export export/fetch_src_onnx.py --fp16 --verify-pair   # the graph containers (OneJev-0.8B-ONNX, fp32 and fp16) and OneJev's safetensors for the check
uv run --group export export/build_onnx.py verify                  # check the conversion rules
uv run --group export export/build_onnx.py build --precision fp16
uv run --group export export/build_onnx.py build --precision fp32 --parts text   # if needed
```
The output is in `models/jevision/`. The decoder outputs the post-final-norm hidden states as `hidden_states` (the lm_head is removed). fp16 is the default because its decisions agree with fp32 and a decoder is 1.0 GB (fp32: 2.0 GB).

# Verification
```bash
uv run --no-sync verify/run_all.py                       # the gates that pass with the run-time dependencies only (the model must be in models/jevision)
uv run --group verify verify/run_all.py --fuzz 100000    # also run the tokenizer fuzz (needs models/hf-base from export/fetch_base.py)
uv sync --group test && uv run --no-sync pytest          # unit tests (no model)
```

| Gate | What it checks | Needs |
|---|---|---|
| check_prompt | token ids and readout offsets match exactly | run-time deps + `models/jevision/tokenizer.json` |
| check_image | preprocessing equals the processor's `pixel_values` (within 2 grey levels) | + pillow |
| check_tokenizer_fuzz | cross-check against Rust `tokenizers` and transformers' `AutoTokenizer` | group `verify` + `models/hf-base` |
| verify | hidden states, logits, probabilities and decisions match PyTorch fp32 (`--variant fp16`/`fp32`, `--route`) | + model |
| check_bias | answers when the options are reversed (1 of 18 flips) | + model |
| check_limits | the limits (255 options, 4 images, a 14411-token text, a 2560x1440 image) (RSS reaches 10 to 17 GB; not part of `run_all.py`) | + model |

`artifacts/golden_*.json` and `.npz` are golden data made by really running the reference implementation (JEVision's bundled `runtime/kev`, PyTorch fp32, CPU): 34 text cases with 44 questions and 9 image cases with 14 questions. Both run the same weights, so a difference is a bug in the port. The photographs among the image cases are not in the repository, so by default only the cases with synthetic pictures are checked (point `JEVISION_IMAGE_ROOT` at a folder of pictures to check the ones found there).

Results (`artifacts/report.md`):
| Item | Result |
|---|---|
| token ids | 58/58 equal |
| decisions (fp16) | 58/58 equal, probability difference at most 0.0046, hidden-state cosine similarity at least 0.999998 |
| decisions (fp32) | 58/58 equal, probability difference at most 7e-4 |
| tokenizer fuzz | 200000/200000 equal |

# Limitations
- Images: 1 to 4 per request, PNG/JPEG/WebP, 25 MP and 10 MiB. Options: 1 to 255 (score: at least 2 levels), no empty option strings when images are present. The image-token region is limited to 8192 tokens and the whole request to 80000 tokens (the sum over all question branches). Beyond a limit it raises `ValueError` (the same limits as the reference serving path)
- The reference does not apply EXIF rotation, so neither does this
- The order of a question's options affects the result (1 of 18 flipped when reversed)
- The kernels the reference uses on a GPU may differ from these at the level of fp32 noise
- Video input is not handled. The numbers on the model card are self-reported by the provider (see above)

# Project Structure

```text
README.md                # README (Japanese)
README_EN.md             # README (English)
LICENSE                  # Apache-2.0
licenses/                # licence texts of the original models (LICENSE-JEVISION, LICENSE-QWEN; also shipped in the Release)
pyproject.toml           # dependencies (run time: onnxruntime, numpy, pillow. extra: gpu; groups: cpu / test / verify / webcam / export)
download_model.py        # fetch the model (GitHub Releases, SHA-256 check, joins split files)
demo_inference_text.py   # text inference demo
demo_inference_image.py  # image inference demo
demo_inference_webcam.py # "is there a person in the picture?" from a camera (OpenCV)
jevision_decide/         # the inference code (light dependencies only; never imports export/)
  runtime.py             #   ORT sessions, chunked prefill, branch execution, pointer heads
  providers.py           #   resolving --device and checking the providers really granted (pure functions)
  cli.py                 #   --device / --gpu / --require-gpu (shared by the demos and verify)
  prompt.py              #   builds the token sequence (no model)
  tokenizer.py           #   pure-Python Qwen byte-level BPE (_regex_translate.py / _uniprops.py are generated)
  protocol.py            #   question rendering, escaping, answer building (a port of kev/api.py)
  image.py               #   Pillow + smart_resize + patching
  mrope.py               #   3-D M-RoPE
export/                  # heavy dependencies (torch / transformers / onnx)
  fetch_base.py          #   fetch the Base weights + the JEVision bundle
  fetch_src_onnx.py      #   fetch the graph containers (OneJev-0.8B-ONNX)
  build_onnx.py          #   build the ONNX by weight replacement (verify / build)
  gen_uniprops.py        #   regenerate the Unicode tables from tokenizers
  make_golden.py         #   make artifacts/golden_*.json / .npz
release/
  make_release.py        # stage the Release files from models/ (split, SHA-256, manifest.json)
tests/                   # unit tests (no model: provider selection, protocol, download scripts, ...)
verify/                  # verification gates (run all with run_all.py) and bench.py
artifacts/               # golden data (the oracle), synthetic pictures, report.md
models/                  # ONNX, tokenizer, configs (gitignored)
```

# Reference
- [divyanshx11/JEVision](https://huggingface.co/divyanshx11/JEVision) (the original model; revision `e1cfde19`; Apache-2.0)
- [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) (the base model; revision `dc7cdfe2`; Alibaba Cloud; Apache-2.0)
- [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) (the starting point of JEVision's text LoRA; Apache-2.0)
- [onnx-community/OneJev-0.8B-ONNX](https://huggingface.co/onnx-community/OneJev-0.8B-ONNX) (the ONNX graph containers; revision `afc03812`; Apache-2.0; every weight is replaced)

# License
JEVision-ONNX-CPU is under [Apache-2.0 license](LICENSE).<br>
The model weights follow the licence of the original model ([divyanshx11/JEVision](https://huggingface.co/divyanshx11/JEVision), Apache-2.0; the base Qwen3.5-0.8B-Base is Apache-2.0 too). The Release ships both licence texts unmodified (`LICENSE-JEVISION`, `LICENSE-QWEN`, the same as `licenses/` of this repository). Neither repository has a NOTICE file (as of 2026-10-11).

JEVision's model card puts the adapters and the runtime under Apache-2.0 and says to follow "the base model's terms and the terms of any datasets you use". **The text adapter was trained starting from [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) (Apache-2.0); kev-0.8b's model card lists datasets whose Hugging Face dataset cards show CC BY-SA, other or unknown licences (BoolQ, DBpedia, Yelp, IMDB, AG News and others).** The visual sidecar is said to be trained on Beans (MIT) and programmatically generated images. What the terms of that data mean for redistributing trained weights is not assessed in this repository; please check for your use.

## About this being a converted work
The ONNX files in the Release are a conversion, made with `export/build_onnx.py` of this repository, of the checkpoint of [divyanshx11/JEVision](https://huggingface.co/divyanshx11/JEVision) (Apache-2.0), for which no ONNX is officially published. Concretely: the LoRA is merged into the Base weights in fp32, every weight of the published Qwen3.5-0.8B ONNX graphs (OneJev-0.8B-ONNX) is replaced, the graph is changed to output the post-final-norm hidden states (the lm_head is removed), and the result is converted to fp16. The tokenizer is the Base `tokenizer.json` plus the 11 special tokens it lacks (such as `<think>`), with the splitting regex matched to transformers 5.17. Agreement of the decisions with the reference implementation (PyTorch fp32) was checked (fp16: 58/58, probability difference at most 0.0046). The provider of the original model does not guarantee the original model or the behaviour of this conversion.
