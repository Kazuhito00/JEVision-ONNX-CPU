# JEVision ONNX port: notes and numbers

Model: `divyanshx11/JEVision@e1cfde19` = Qwen3.5-0.8B-Base@dc7cdfe2 + merged LoRA + pointer head (two LoRAs: text
`jevvision-text` [== `jevbench-m3`, same sha256], visual sidecar `adapter/`; heads 1024->256 q/k, temperature 1.0 both).

## Method
1. Goldens from JEVision's own runtime (runtime/kev, torch 2.14 CPU fp32, transformers 5.17.0, peft 0.21): 34 text
   cases (44 questions; EN/JA, 60 options, forged delimiters, added tokens, long state, eval-set cases) and 9 image
   cases (14 questions; 1/2/4 images, 40x30 .. 960x1280, RGBA PNG, JPEG). Token ids, offsets, hidden vectors, z, p.
2. Weight swap into the OneJev-0.8B-ONNX graphs. Rules verified first on the OneJev pair (safetensors vs its own
   ONNX, fp16+fp32): 319 decoder + 1 embed + 153 vision tensors reproduce (gdn.norm.weight within 1 bf16 ulp: the
   published graph stores it bf16-rounded; the build uses the exact fp32 value). Every decoder/embed tensor is replaced
   with Base+LoRA (186 LoRA modules, scale 2.0, merged in fp32); lm_head and its Slice/MatMul are removed and the
   post-final-norm tensor `/model/layers.24/final_norm_layernorm/output_0` is exported as `hidden_states`.
3. Vision: Base visual.* differs from OneJev's in 7 tensors (merger fc1/fc2/norm, pos_embed); the other 146 are
   byte-identical. All 153 are written from Base.
4. Tokenizer: Base tokenizer.json lacks 11 added tokens (`<think>` ...) and transformers 5.17 splits with `\p{L}+`
   (not `[\p{L}\p{M}]+`); the runtime tokenizer.json is patched to match (30000 fuzz strings, 0 mismatches).
   Update: the runtime tokenizer is now pure Python (OneJev BPE + regex translator); `tokenizers` is an oracle only.
   `_uniprops.py` is regenerated from the oracle's own Unicode tables (Python 3.12's unicodedata missed newly assigned
   letters: 2/100000 mismatches before). Fuzz: 2 x 100000 cases (eval-style prompts EN/JA, CJK, ZWJ/flag emoji, combining
   marks, ligatures, 24 whitespace kinds, specials inside user text, 6k-piece strings) vs tokenizers AND transformers: 0
   mismatches. Gates re-run with tokenizers/transformers/torch blocked in sys.modules: token ids 58/58, ask smoke unchanged.
   Tokenization per typical request (3 questions + states, warm cache): ~0.2 ms pure vs ~0.3 ms Rust; load 0.43 s vs 0.36 s.

## Results (re-run with verify/run_all.py; the original logs are not kept in this repository)
| gate | result |
|---|---|
| check_prompt (no model) | 58/58 questions: token ids + readout offsets exact |
| check_tokenizer_fuzz | 200000/200000 equal to both tokenizers and AutoTokenizer |
| check_image | grids identical; pixels equal except <= 2 grey levels in <1 % of values (torchvision integer resize vs Pillow) |
| verify fp32 (all) | token ids 58/58, decisions 58/58, max rel dh 7.5e-4 (images; text 1.7e-6), max dz 0.046, max dp 7e-4 |
| verify fp16 (all) | token ids 58/58, decisions 58/58, max rel dh 1.4e-3, cosine >= 0.999998, max dz 0.110, max dp 0.0046 |
| verify fp16, 32-token prefill chunks (text) | 44/44, max dp 0.007 |
| check_bias | 1/18 choice answers flip when options are reversed |
| check_limits | limits raise; 255 options, 4 images, 14411-token text, 2560x1440 image run |

Choice: **fp16** decoder (parity holds, 1.0 GB per decoder vs 2.0 GB). fp32 also built and verified.

## Files (models/jevision, gitignored)
fp16: decoder_text 996 MB, decoder_visual 996 MB, embed_tokens 509 MB, vision_encoder 205 MB (~2.7 GB total).
fp32: 1992 MB x2, 1017 MB, 402 MB. sha256 in manifest.json.

## Smoke runs through an ask adapter of the working workspace (not part of this repository)
Smoke (fp16, 6 threads): wiki image 20 cases 19/20 (choice fwd+rev both correct 17/18); wiki text(ja) 10/10;
text en 11/12 (the reference itself gets the same 11/12); generated image(ja) 14/15. No wiki case has > 4 images.

## Limits / open
- Peak RSS: 14k-token text ~10 GB, 3.6k-token image ~17 GB (vision attention is O(n^2)); 80000 tokens and the
  8192-token image region were not exercised.
- No timing was measured for this report (machine shared). Speed numbers are in the README (bench results, GPU bench).
- Reference runs the torch fallback for DeltaNet (no fla); GPU kernels may differ at fp32 noise level.
