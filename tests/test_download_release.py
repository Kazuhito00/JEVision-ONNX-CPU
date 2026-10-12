import argparse
import hashlib
import importlib
import json

import download_model as dm  # noqa: E402  (repository root is on sys.path through conftest)


def args(**kw):
    base = dict(base_url=None, repo=None, tag="latest")
    base.update(kw)
    return argparse.Namespace(**base)


def test_default_repo_constant():
    assert dm.DEFAULT_REPO == "Kazuhito00/JEVision-ONNX-CPU"


def test_repo_precedence_argument_over_environment_over_default(monkeypatch):
    monkeypatch.delenv("JEVISION_ONNX_REPO", raising=False)
    assert dm.base_url(args()) == "https://github.com/Kazuhito00/JEVision-ONNX-CPU/releases/latest/download"
    monkeypatch.setenv("JEVISION_ONNX_REPO", "env/Repo")
    assert dm.base_url(args()) == "https://github.com/env/Repo/releases/latest/download"
    assert dm.base_url(args(repo="arg/Repo", tag="v0.0.0")) == "https://github.com/arg/Repo/releases/download/v0.0.0"
    assert dm.base_url(args(base_url="http://127.0.0.1:8000/")) == "http://127.0.0.1:8000"


def test_make_release_splits_big_files_and_hashes(tmp_path):
    mr = importlib.import_module("make_release")
    src = tmp_path / "big.bin"
    data = bytes(range(256)) * 100
    src.write_bytes(data)
    parts = mr.split(src, tmp_path, "big.bin", 10000)
    assert [p["size"] for p in parts] == [10000, 10000, 5600]
    joined = b"".join((tmp_path / p["asset"]).read_bytes() for p in parts)
    assert joined == data and hashlib.sha256(joined).hexdigest() == mr.sha256_of(src)


def test_make_release_stages_fp16_only_by_default(tmp_path, monkeypatch):
    mr = importlib.import_module("make_release")
    home = tmp_path / "models" / "jevision"
    (home / "onnx").mkdir(parents=True)
    for name in ("embed_tokens_fp16.onnx", "embed_tokens_fp16.onnx_data", "decoder_text.onnx", "decoder_text.onnx_data"):
        (home / "onnx" / name).write_bytes(b"x" * 10)
    for name in mr.SIDE_FILES:
        (home / name).write_bytes(b"{}")
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["make_release.py", "--models", str(tmp_path / "models"), "--out", str(out)])
    assert mr.main() == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    paths = [f["path"] for f in manifest["files"]]
    assert "jevision/onnx/embed_tokens_fp16.onnx_data" in paths and "jevision/onnx/decoder_text.onnx" not in paths
    assert "jevision/LICENSE-JEVISION" in paths and "jevision/LICENSE-QWEN" in paths
    assert (out / "jevision__onnx__embed_tokens_fp16.onnx").exists()
