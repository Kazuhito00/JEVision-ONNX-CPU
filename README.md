[[Japanese](README.md)/[English](README_EN.md)]

# JEVision-ONNX-CPU

[`divyanshx11/JEVision`](https://huggingface.co/divyanshx11/JEVision)（Qwen3.5-0.8B-Base + LoRA + ポインタヘッドの、テキストと画像に対応した型付き判定モデル）を、onnxruntime と numpy だけで推論する最小依存の推論実装です。 ※GPU（CUDA）推論も可<br>
公式のランタイム（PyTorch・transformers・peft が必要）は使いません。

# Features
以下の特徴があります。
- 最小依存：実行時に必要なパッケージは onnxruntime と numpy の2つだけ（画像を使う場合は pillow を足した3つ）
- 動かすのに必要なスクリプトは `download_model.py`（モデル取得）と `demo_inference_text.py`（推論）の2つ。画像は `demo_inference_image.py`、カメラは `demo_inference_webcam.py`
- 判定は参照実装（PyTorch fp32）と一致（fp16 のグラフで、トークン列 58/58、判定 58/58。確率の差は最大 0.0046）
- 生成も語彙ヘッドも無い。プロンプト中の特殊トークンの隠れ状態を読み出すポインタ方式なので、答えは必ず与えた選択肢のどれか
- 画像対応（1回に1〜4枚）。画像なしのリクエストはテキスト用デコーダ、画像ありのリクエストは視覚用デコーダが処理する
- 前処理（Qwen byte-level BPE、3次元 M-RoPE、画像のパッチ化）を純Pythonで移植し、参照実装との突き合わせ検証を同梱
- CPU のほか、GPU（CUDA）でも動かせる（`--device cuda`。CPU へ黙って切り替わらないよう、実際に使われたプロバイダを表示し、`--require-gpu` で GPU が使えなければエラーにできる）

# Purpose of This Repository
以下の検証を目的としています。
- JEVision の判定を、実行時依存2〜3つだけでどこまで動かせるか
- 参照実装の前処理を純Pythonに移植して、同じ入力が作れるか
- CPU（と GPU）で速度・メモリがどうなるか

移植の根拠は [artifacts/report.md](artifacts/report.md) を参照してください。

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
pillow       10.0 or later   （画像を使う場合のみ）
```
実行に必要なのはこれだけです。モデルの取得も標準ライブラリのみで動きます。<br>
環境構築は[uv](https://docs.astral.sh/uv/)を使います（pip / conda は使いません）。<br>
オプションがあります。
- extra `gpu`：`onnxruntime-gpu`（CUDA 版）。**`onnxruntime`（CPU 版）とは同居できません**。入れ方は「GPUで実行する」を参照
- group `test`：`pytest`（`tests/`。モデル不要）
- group `verify`：トークナイザの突き合わせ検証（`check_tokenizer_fuzz`）用の `tokenizers`・`transformers`
- group `webcam`：`demo_inference_webcam.py` 用の OpenCV
- group `export`：ONNX を自分で書き出すときの torch・transformers など（実行時には使いません）

# Installation

```bash
git clone https://github.com/Kazuhito00/JEVision-ONNX-CPU
cd JEVision-ONNX-CPU

# 実行時の依存（onnxruntime・numpy・pillow）のみインストール
uv sync
```

# Download Model
JEVision は ONNX も、重みそのものも公開していません（LoRA アダプタ2組とヘッドだけを配布）。そこで、Base 重みに LoRA を統合して書き出した ONNX を[Releases](https://github.com/Kazuhito00/JEVision-ONNX-CPU/releases)（v0.0.0 を想定）に置きます（リポジトリには含めていません）。<br>
以下のスクリプトで `models/` に取得します。標準ライブラリのHTTPSのみで動き、gitもgit-lfsも不要です。
```bash
uv run download_model.py
```
- **引数なしでは、Release の全ファイル（fp16 の4グラフ、ヘッド、トークナイザ、設定、ライセンス。16 ファイル、約 2.72 GB）を取得します**
- 取得元は、既定で `Kazuhito00/JEVision-ONNX-CPU` の最新の Release です（`--repo OWNER/NAME` か、環境変数 `JEVISION_ONNX_REPO` で別のリポジトリにできます。優先順位は 引数 > 環境変数 > 既定）
- 取得したファイルは、Release の `manifest.json` に書かれた SHA-256 と照合します（不一致ならファイルを削除して停止）
- GitHub の制限（1ファイル 2 GiB 未満）で分割されたファイルは、取得後に結合して照合します（この Release の最大ファイルは 996,327,424 バイトなので、分割は起きません）
- 取得済みなら何もせず、中断した場合は同じコマンドで続きから再開します

```bash
uv run download_model.py --list                          # ファイル一覧
uv run download_model.py --exclude "*vision_encoder*"    # テキスト専用（視覚エンコーダを取らない。約 0.2 GB 少ない）
uv run download_model.py --tag v0.0.0                    # タグを指定（既定は latest）
```

別の場所に置いたモデルは、環境変数 `JEVISION_MODEL_DIR`（`models/jevision` に当たるフォルダ）か、各デモの `--model` で指します。

### Release に置くファイル（`release/make_release.py` の出力）
`uv run release/make_release.py` が `dist/release/` にアセットと `manifest.json`・`SHA256SUMS` を作ります（**アップロードはしません**。`gh release create v0.0.0 dist/release/*` などで自分で上げます）。アセット名はフラットで、パスの `/` が `__` になります。

| アセット名 | サイズ（バイト） |
|---|---:|
| `jevision__onnx__embed_tokens_fp16.onnx` / `.onnx_data` | 637 / 508,559,360 |
| `jevision__onnx__vision_encoder_fp16.onnx` / `.onnx_data` | 187,476 / 204,798,464 |
| `jevision__onnx__decoder_text_fp16.onnx` / `.onnx_data` | 597,430 / 996,327,424 |
| `jevision__onnx__decoder_visual_fp16.onnx` / `.onnx_data` | 597,967 / 996,327,424 |
| `jevision__head_text.npz` / `jevision__head_visual.npz` | 2,100,710 ずつ |
| `jevision__tokenizer.json` | 9,833,522 |
| `jevision__tokenizer_config.json` / `jevision__config.json` / `jevision__preprocessor_config.json` | 16,712 / 2,907 / 390 |
| `jevision__LICENSE-JEVISION` / `jevision__LICENSE-QWEN` | 11,545 / 11,343 |
| `manifest.json` / `SHA256SUMS` | 取得スクリプトが読む（パス・サイズ・SHA-256） |

合計は約 2.72 GB（2.53 GiB）。すべて 1 ファイル 2 GiB 未満なので分割は不要です（`--max-part-gib` の既定は 1.9）。<br>
fp32 のグラフ（約 5.4 GB 増。最大は `decoder_*.onnx_data` の 1,992,441,856 バイト = 1.86 GiB で、これも分割不要）は、既定では Release に入れません（`make_release.py --with-fp32` で追加できます）。

自分で書き出す場合は、「ONNX への変換」を参照してください（torch が要ります）。

# Usage

### 実行例(Python)
```bash
uv run demo_inference_text.py
uv run demo_inference_image.py
```
テキストのデモは、サポートの問い合わせ文について、怒っているか（noul）・どのチームか（choice）・緊急度（score）の3問を一度に判定します。自分の入力は次のように指定します。
```bash
uv run demo_inference_text.py --state "Server is down since 5am." --choice "team=billing,tech,sales" --noul "angry=The customer is angry" --score "urgency=low,medium,high"
```
画像のデモは、既定で赤い円を描いた画像を作って、形・色などを判定します。自分の画像は `--image`（最大4枚、繰り返し指定）で指定します。
```bash
uv run demo_inference_image.py --image shot.png --choice "status=paid,pending,failed" --noul "paid=The invoice is paid"
```
224 px のサムネイルは、processor の最小画素数（65536）で 256 px に拡大され、1枚 64 トークンになります。

```python
from jevision_decide import Decider, choice, noul, score

d = Decider("models/jevision", variant="fp16", intra_op_threads=6)   # variant: "fp16"（既定）| "fp32"
d.decide({"ticket": "My card was charged twice."},
         {"angry": noul("The customer is angry"),
          "team": choice("Which team?", {"billing": "charges and refunds", "tech": "bugs"}),
          "urgency": score("How urgent?", ["low", "medium", "high"])})

d.decide("Look at the attached image.", {"q": noul("It shows a cat")}, images=["cat.jpg"])   # 1..4 枚
```
画像は、OneJev と違い、state から `<image:N>` で参照**しません**（JEVision の形式）。`images` には、ファイルのパス、`data:image/...;base64,` の URL、バイト列、Pillow の画像を渡せます。<br>
回答は参照実装と同じ形（確率は小数 2 桁）で、`raw_probabilities`（丸めなし）を追加で返します。

**仕組み：** プロンプトの最後の文字を読むのではなく、ポインタ読み出しをします。生成は走りません。
```
<|fim_prefix|> state <|fim_middle|> instr <|box_start|> opt1 <|box_end|> <|box_start|> opt2 <|box_end|> ... <|fim_suffix|>
```
`<|fim_suffix|>` の最終正規化後の隠れ状態が query、各 `<|box_end|>` が key で、`z_j = (Wk h_j + bk)·(Wq h_q + bq) / sqrt(256)` を softmax します。1リクエストの共通部分（画像 + state）は1回だけプレフィルし、質問ごとにその上で枝を走らせます。

### カメラで「人が映っているか」を判定する
OpenCV でカメラの映像を取得し、最新のフレームについて「人が映っているか」を判定して、結果を映像に重ねて表示します。
```bash
uv run --group webcam demo_inference_webcam.py                        # カメラ 0
uv run --group webcam demo_inference_webcam.py --camera 1 --mirror
uv run --group webcam demo_inference_webcam.py --question "Is the person wearing glasses?"
uv run --group webcam demo_inference_webcam.py --camera artifacts/images/red_circle.png --no-window --max-decisions 1   # カメラの代わりに画像・動画ファイル
```
- 判定は別スレッドで動かし、映像の表示は止めません。判定が出るたびに、上部の帯（緑＝はい／赤＝いいえ）と「はい」の確率が更新されます
- 質問は `--question`（はい・いいえの質問）で変えられます。モデルに渡す画像は長辺 224 px に縮小します（`--size`）
- `q` または `Esc` で終了します。OpenCV は、このデモだけの依存です（あとで `uv sync` を実行すると、OpenCV は外れます）

### 質問の種類
| 種類 | 答え |
|---|---|
| `choice(instructions, {key: 説明})` | 選択肢から1つ（`choice`、`confidence`、`probabilities`）。選択肢は `key` または `key: 説明` として渡される |
| `score(instructions, [レベル説明, ...])` | 順序尺度の期待値（`score`、`legend`）。先頭が最低 |
| `noul(instructions)` | 文が成り立つか（`noul` = はいの確率）。選択肢は `no` / `yes` 固定 |

# GPUで実行する
`--device cuda` を付けると、CUDA で推論します（全グラフ: テキスト用・視覚用デコーダ、embed、vision）。既定は CPU です。

**必要なもの：**
- NVIDIA の GPU と、CUDA 13 と cuDNN 9 の `bin` フォルダ（PATH に通す。`onnxruntime-gpu` は CUDA の DLL を同梱せず、PATH から探します。`CUDA_PATH` が古い CUDA を指していることがあるので、PATH を確認してください）
- `onnxruntime` の代わりに `onnxruntime-gpu`。**両方は同居させないでください**（同じ `onnxruntime/` フォルダにファイルを書きます）

```bash
# CPU 版を外して GPU 版を入れる（pyproject.toml で両方が入る組み合わせを禁止しています）
uv sync --extra gpu --no-group cpu
# 以降は uv run --no-sync で実行（--no-sync を付けないと、既定の CPU 版に戻されます）
uv run --no-sync demo_inference_text.py --device cuda
uv run --no-sync demo_inference_text.py --require-gpu        # GPU が使えなければエラーで止まる
uv run --no-sync verify/bench.py --device cuda
```
CPU 版に戻すときは `uv sync` です。Windows の PATH の例です（バージョンは環境に合わせてください。PATH を変えたあとはターミナルを開き直します）。
```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64
C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64
```
```python
from jevision_decide import Decider, noul

d = Decider("models/jevision", device="cuda", require_gpu=True)   # device: "cpu"（既定）| "cuda" | "auto"
d.decide("hello", {"q": noul("This is a greeting")})
print(d.providers_used)   # {グラフのファイル名: 実際に使われたプロバイダ}。デコーダは最初の判定で読み込まれる
```
- `--device cpu`（既定）は CPU のみ、`cuda` は CUDAExecutionProvider → CPUExecutionProvider の順、`auto` は CUDA が使える ONNX Runtime なら `cuda`、なければ `cpu` です。`--gpu` は `--device cuda` の別名です
- **CPU へ黙って切り替わることはありません。** 各セッションを作った直後に、実際に使われたプロバイダを確認します。CUDA を求めたのに CPU になったときは警告を出し、デモの最終行（`execution providers: ...`）と `verify/bench.py` の出力に、グラフごとの実際のプロバイダを表示します
- `--require-gpu`（Python では `require_gpu=True`）は、CUDA が使えない場合（この ONNX Runtime に CUDA が無い、DLL が見つからない、GPU に載らない）に、警告ではなくエラーで止めます（終了コードは 0 ではありません）。環境の確認にも使えます
- GPU での確率は、CPU と完全には一致しません（fp16 の丸め・演算順序の違い）。下の測定では、CPU と GPU で答えが違ったのは 5364 件中 4 件で、正誤が変わったものは 0 件でした
- 動作を確認したのは、Windows 11、NVIDIA GeForce RTX 3050 Ti Laptop GPU（VRAM 4 GB）、`onnxruntime-gpu` 1.31、CUDA 13.4、cuDNN 9.20、fp16 のグラフです。fp32 のグラフ、DirectML、Linux での GPU 実行は**未検証**です

**VRAM の目安（fp16）：** 下の GPU 測定で、ピーク VRAM はテキストのみで 3192 MiB、画像で 2452 MiB でした（開始時の約 106 MiB を引いた値。テキスト用・視覚用デコーダのうち、使う方だけを載せます）。4 GB の GPU で動いた大きさです。fp32 は未検証です。

# 性能
以下は、既存の測定の数字です（出典を併記）。測定の条件が異なるので、表を分けています。根拠のない数字は載せていません。

### CPU と GPU の速さ（GPU ベンチ）
RTX 3050 Ti Laptop 4 GB、CUDA 13.4、cuDNN 9.20、onnxruntime-gpu 1.31、fp16、224 px、CPU は 8 スレッド。

| 入力 | CPU 1問 | GPU（CUDA）1問 | 倍率 | ピーク VRAM |
|---|---:|---:|---:|---:|
| テキスト | 607 ms | 60 ms | 10.1 倍 | 3192 MiB |
| 画像 | 767 ms | 69 ms | 11.1 倍 | 2452 MiB |

出典：作業用ワークスペースの `eval_common/gpu_bench/report.md`（このリポジトリには含みません。2026-10-10。同一の問題をテキスト3問・画像3問。1問あたりの中央値で、前処理・トークナイズを含む `decide` 呼び出し全体。ロード時間は含みません）。GLiNER による CPU の較正は未実施です。<br>
同じ測定の他のモデルは、OneJev 0.8B が 12.6〜16 倍、OpenJev 0.8B が 3.2〜3.9 倍でした。Intern-Decision・Strands・QEV・OneJev 4B は GPU で速くならず、clef-flash は VRAM 不足で CPU に切り替わりました。

### Wikipedia 画像ベンチ（v1 の試行測定。5364 件 = 問 × 言語）
出典：`wikipedia-vision-decision-bench` の `results/runs/v1/summary_v1.md`（CPU は 8 スレッド、基準測定で較正済みのブロック。GPU は同じ RTX 3050 Ti）。

| | 全体 | 英語 | 日本語 |
|---|---:|---:|---:|
| 正解率 | 76.8%（4120/5364） | 80.7% | 73.2% |

| 部分 | 件数 | CPU 1問（中央値） | GPU 1問（中央値） | 倍率 |
|---|---:|---:|---:|---:|
| 全体 | 5364 | 785 ms | 71 ms | 11.1 倍 |
| 画像あり | 5017 | 787 ms | 71 ms | 11.1 倍 |
| **テキストのみ** | 347 | 622 ms | **402 ms** | **1.5 倍** |

- **テキストのみの設問では、GPU の倍率が 1.5 倍に留まりました**（上の GPU ベンチのテキスト 10.1 倍とは食い違います）。原因は未調査です
- 5364 件のうち、CPU と GPU で答えが違ったのは 4 件で、正誤は同じでした

### Wikipedia 画像セット（まとめ（Wikipedia画像セット）.html）の正解率
同じ質問を英語・日本語で聞いたときの、JEVision 0.8B（fp16、CPU 8 スレッド、画像は 224 px、選択肢は元の順で1回）の正解率です。値は、同ファイルに埋め込まれた設問ごとの結果から数え直しました（時間は、その設問の CPU での 1 問あたりの中央値）。

| データセット | 設問数 | 英語 | 日本語 | CPU 1問の中央値（英 / 日） |
|---|---:|---:|---:|---:|
| ① Wikipedia 記事の写真 | 797 | 92.1%（734） | 82.9%（661） | 727 / 748 ms |
| ② Commons 拡充 | 471 | 92.8%（437） | 89.4%（421） | 743 / 754 ms |
| ③ 合成の実務画像 | 382 | 87.7%（335） | 86.6%（331） | 721 / 722 ms |
| ④ 推論・知性（合成） | 225 | 42.7%（96） | 40.9%（92） | 728 / 740 ms |

同じ設問で、OneJev 0.8B は ① 91.3% / 83.9%、② 95.8% / 92.1%、③ 91.6% / 89.0%、④ 45.8% / 47.6%（英 / 日）でした。④（計算・規則の適用など、見たあとに考える設問）は、JEVision も OneJev も半分を下回っています。

### 参考：JEVision 自身の報告
JEVision のモデルカードには、視覚サイドカーの自己申告値（Beans 検証 51/60、生成タスク 147/150 など）が載っています。**これらは提供元の自己申告で、この実装では再現していません。**

### メモリと系列長
- ピーク RSS は、14k トークンのテキストで約 10 GB、3.6k トークンの画像（2560x1440）で約 17 GB でした（視覚エンコーダの attention が画素数の2乗で増えるため。`artifacts/report.md`）
- 80000 トークンのリクエストと、8192 トークンの画像領域の上限は、実機では確認していません

# ONNX への変換
JEVision は、LoRA アダプタ（テキスト用・画像用）とポインタヘッドだけを配っています。そこで、Base 重みに LoRA を fp32 で統合し（`W += (alpha/r)·B@A`）、公開されている Qwen3.5-0.8B 系の ONNX グラフ（[onnx-community/OneJev-0.8B-ONNX](https://huggingface.co/onnx-community/OneJev-0.8B-ONNX)）を「器」として、**全重みを差し替えて**作ります（torch での export はしません）。差し替えの規則は、先に OneJev の safetensors と公開 ONNX の組で再現できることを確認します（`build_onnx.py verify`）。
```bash
uv sync --group export
uv run --group export export/fetch_base.py                         # Base 重み + JEVision バンドル（HTTPS のみ。約 1.8 GB）
uv run --group export export/fetch_src_onnx.py --fp16 --verify-pair   # グラフの器（OneJev-0.8B-ONNX の fp32 と fp16）と、検証用の OneJev の safetensors
uv run --group export export/build_onnx.py verify                  # 変換規則の検証
uv run --group export export/build_onnx.py build --precision fp16
uv run --group export export/build_onnx.py build --precision fp32 --parts text   # 必要なら
```
出力は `models/jevision/` です。デコーダは、最終正規化後の隠れ状態を `hidden_states` として出力します（lm_head は取り除きます）。fp16 を既定にしたのは、判定が fp32 と一致し、デコーダが 1.0 GB（fp32 は 2.0 GB）で済むためです。

# Verification
```bash
uv run --no-sync verify/run_all.py                       # 実行時依存だけで通るゲート（モデルが models/jevision にあること）
uv run --group verify verify/run_all.py --fuzz 100000    # トークナイザのファズも回す（export/fetch_base.py で取った models/hf-base が要る）
uv sync --group test && uv run --no-sync pytest          # 単体テスト（モデル不要）
```

| ゲート | 内容 | 必要なもの |
|---|---|---|
| check_prompt | プロンプトのトークン列・読み出し位置の完全一致 | 実行時依存 + `models/jevision/tokenizer.json` |
| check_image | 前処理が processor の `pixel_values` と一致（差は 2 階調以内） | + pillow |
| check_tokenizer_fuzz | Rust `tokenizers` と transformers の `AutoTokenizer` との突合 | group `verify` + `models/hf-base` |
| verify | 隠れ状態・logit・確率・判定が PyTorch fp32 と一致（`--variant fp16`/`fp32`、`--route`） | + モデル |
| check_bias | 選択肢を逆順にしたときの答え（1/18 件で反転） | + モデル |
| check_limits | 上限（255 選択肢、4 枚、14411 トークンのテキスト、2560x1440 の画像）の確認（RSS が 10〜17 GB になる。`run_all.py` には含まれない） | + モデル |

`artifacts/golden_*.json` と `.npz` は、参照実装（JEVision 同梱の `runtime/kev`、PyTorch fp32、CPU）を実際に動かして作った正解データです（テキスト 34 ケース 44 問、画像 9 ケース 14 問）。同じ重みを両者が走らせるので、差が出たら移植のバグと断定できます。画像ケースのうち、実写の画像はリポジトリに含めていないので、既定では合成画像のケースだけを検証します（`JEVISION_IMAGE_ROOT` で画像のフォルダを指すと、ある分だけ検証します）。

結果（`artifacts/report.md`）：
| 項目 | 結果 |
|---|---|
| トークン列 | 58/58 一致 |
| 判定（fp16） | 58/58 一致、確率の差は最大 0.0046、隠れ状態のコサイン類似度 0.999998 以上 |
| 判定（fp32） | 58/58 一致、確率の差は最大 7e-4 |
| トークナイザのファズ | 200000/200000 一致 |

# Limitations
- 画像：1リクエスト 1〜4 枚、PNG/JPEG/WebP、25 MP・10 MiB。選択肢は 1〜255（score は 2 レベル以上）、画像があるときは空文字の選択肢は不可。画像トークン領域は 8192 トークンまで、リクエスト全体は 80000 トークンまで（全質問の枝の合計）。超えると `ValueError`（参照実装の配信パスと同じ上限）
- 参照実装は EXIF の回転をしないので、ここでもしません
- 質問（枝）の選択肢の順序は結果に影響します（18 件中 1 件が逆順で反転）
- 参照実装が GPU で使うカーネルとは、fp32 のノイズ程度の違いがあり得ます
- 動画の入力は扱いません。モデルカードの評価値は、提供元の自己申告です（上述）

# Project Structure

```text
README.md                # README（日本語）
README_EN.md             # README（英語）
LICENSE                  # Apache-2.0
licenses/                # 元のモデルのライセンス文（LICENSE-JEVISION、LICENSE-QWEN。Release にも同梱）
pyproject.toml           # 依存定義（実行時は onnxruntime・numpy・pillow。extra: gpu、group: cpu / test / verify / webcam / export）
download_model.py        # モデル取得（GitHub Releases、SHA-256照合、分割ファイルの結合）
demo_inference_text.py   # テキストの推論デモ
demo_inference_image.py  # 画像の推論デモ
demo_inference_webcam.py # カメラで「人が映っているか」を判定（OpenCV）
jevision_decide/         # 推論実装（軽い依存のみ。export/ を import しない）
  runtime.py             #   ORT セッション、チャンク付きプレフィル、枝の実行、ポインタヘッド
  providers.py           #   --device の解決と、実際に使われたプロバイダの検査（純粋な関数）
  cli.py                 #   --device / --gpu / --require-gpu（デモと verify 共通）
  prompt.py              #   トークン列の組み立て（モデル不要）
  tokenizer.py           #   Qwen byte-level BPE の純Python実装（_regex_translate.py / _uniprops.py は生成物）
  protocol.py            #   質問の描画・エスケープ・回答の構築（kev/api.py の移植）
  image.py               #   Pillow + smart_resize + パッチ化
  mrope.py               #   3次元 M-RoPE
export/                  # 重い依存（torch / transformers / onnx）
  fetch_base.py          #   Base 重み + JEVision バンドルの取得
  fetch_src_onnx.py      #   グラフの器（OneJev-0.8B-ONNX）の取得
  build_onnx.py          #   重みの差し替えで ONNX を作る（verify / build）
  gen_uniprops.py        #   Unicode 表を tokenizers から再生成
  make_golden.py         #   artifacts/golden_*.json / .npz を作る
release/
  make_release.py        # models/ から Release 用のファイルを作る（分割・SHA-256・manifest.json）
tests/                   # 単体テスト（モデル不要。プロバイダ選択、プロトコル、取得スクリプトなど）
verify/                  # 検証ゲート（run_all.pyで一括）と bench.py
artifacts/               # golden（検証の正解データ）、合成画像、report.md
models/                  # ONNX・トークナイザ・設定（gitignore）
```

# Reference
- [divyanshx11/JEVision](https://huggingface.co/divyanshx11/JEVision)（元のモデル。revision `e1cfde19`。Apache-2.0）
- [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)（ベースモデル。revision `dc7cdfe2`。Alibaba Cloud。Apache-2.0）
- [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b)（JEVision のテキスト用 LoRA の出発点。Apache-2.0）
- [onnx-community/OneJev-0.8B-ONNX](https://huggingface.co/onnx-community/OneJev-0.8B-ONNX)（ONNX グラフの器。revision `afc03812`。Apache-2.0。重みはすべて差し替え）

# License
JEVision-ONNX-CPU is under [Apache-2.0 license](LICENSE).<br>
モデルの重みは、元のモデル（[divyanshx11/JEVision](https://huggingface.co/divyanshx11/JEVision)、Apache-2.0。ベースの Qwen3.5-0.8B-Base も Apache-2.0）のライセンスに従います。Release には、両方のライセンス文（`LICENSE-JEVISION`、`LICENSE-QWEN`。このリポジトリの `licenses/` と同じもの）を元のまま同梱しています。どちらのリポジトリにも NOTICE ファイルは見当たりませんでした（2026-10-11 時点）。

JEVision のモデルカードは、アダプタとランタイムを Apache-2.0 とし、「ベースモデルの条件と、使うデータセットの条件に従うこと」と書いています。**テキスト用アダプタは、[jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b)（Apache-2.0）から学習を始めたもので、kev-0.8b のモデルカードには、Hugging Face 上のデータセットのカードで CC BY-SA、other、unknown と表示されているデータセット（BoolQ、DBpedia、Yelp、IMDB、AG News など）が挙げられています。** 視覚用サイドカーは、Beans（MIT）と、プログラムで生成した画像で学習したと書かれています。これらのデータの条件が、学習済み重みの再配布に何を及ぼすかは、このリポジトリでは判断していません。利用する側で確認してください。

## 変換物であることについて
Release に含まれる ONNX ファイルは、公式には ONNX が公開されていない [divyanshx11/JEVision](https://huggingface.co/divyanshx11/JEVision)（Apache-2.0）のチェックポイントを、このリポジトリの `export/build_onnx.py` で ONNX 形式に変換したものです。具体的には、Base 重みに LoRA を fp32 で統合し、公開済みの Qwen3.5-0.8B 系 ONNX グラフ（OneJev-0.8B-ONNX）に全重みを差し替え、最終正規化後の隠れ状態を出力するようにグラフを変更し（lm_head は除去）、fp16 に変換しています。トークナイザは、Base の `tokenizer.json` に無い 11 個の特殊トークン（`<think>` など）を足し、分割の正規表現を transformers 5.17 に合わせたものです。参照実装（PyTorch fp32）との判定の一致を確認しています（fp16 で 58/58、確率の差は最大 0.0046）。元のモデルの提供元およびこの変換物の動作を、提供元は保証していません。
