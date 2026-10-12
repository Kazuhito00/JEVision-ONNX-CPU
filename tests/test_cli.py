import argparse

import pytest

from jevision_decide.cli import add_device_args, device_kwargs, providers_line


def parse(*argv):
    ap = argparse.ArgumentParser()
    add_device_args(ap)
    return ap.parse_args(argv)


def test_default_is_cpu():
    assert device_kwargs(parse()) == {"device": "cpu", "require_gpu": False}


def test_gpu_alias_means_cuda():
    assert device_kwargs(parse("--gpu")) == {"device": "cuda", "require_gpu": False}


def test_require_gpu_implies_cuda():
    assert device_kwargs(parse("--require-gpu")) == {"device": "cuda", "require_gpu": True}


def test_auto_and_bad_device():
    assert device_kwargs(parse("--device", "auto"))["device"] == "auto"
    with pytest.raises(SystemExit):
        parse("--device", "tpu")


def test_providers_line_groups_graphs():
    class D:
        provider = "CPUExecutionProvider"
        providers_used = {"a.onnx": "CUDAExecutionProvider", "b.onnx": "CUDAExecutionProvider", "c.onnx": "CPUExecutionProvider"}
    line = providers_line(D())
    assert "CUDAExecutionProvider (a.onnx, b.onnx)" in line and "CPUExecutionProvider (c.onnx)" in line
