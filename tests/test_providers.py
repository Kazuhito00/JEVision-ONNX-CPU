"""Execution-provider selection and the "did we really get the GPU" check. Pure logic: no GPU, no model, no onnxruntime build needed."""
import pytest

from jevision_decide.providers import CPU, CUDA, GpuUnavailable, check_granted, select_providers, wants_gpu

BOTH = [CUDA, CPU]


def test_cpu_is_cpu_only_even_if_cuda_is_available():
    assert select_providers("cpu", BOTH) == [CPU]


def test_cuda_orders_cuda_before_cpu():
    assert select_providers("cuda", [CPU, CUDA]) == [CUDA, CPU]


def test_cuda_without_cuda_build_falls_back_to_cpu_list():
    assert select_providers("cuda", [CPU]) == [CPU]


def test_require_gpu_without_cuda_raises_before_any_load():
    with pytest.raises(GpuUnavailable):
        select_providers("cuda", [CPU], require_gpu=True)
    with pytest.raises(GpuUnavailable):
        select_providers("auto", [CPU], require_gpu=True)


def test_require_gpu_with_cpu_device_is_a_contradiction():
    with pytest.raises(ValueError):
        select_providers("cpu", BOTH, require_gpu=True)


def test_auto():
    assert select_providers("auto", BOTH) == [CUDA, CPU]
    assert select_providers("auto", [CPU]) == [CPU]


def test_unknown_device():
    with pytest.raises(ValueError):
        select_providers("tpu", BOTH)


def test_wants_gpu():
    assert wants_gpu([CUDA, CPU]) and not wants_gpu([CPU]) and not wants_gpu([])


def test_check_granted_agree_is_silent():
    assert check_granted([CUDA, CPU], [CUDA, CPU]) is None
    assert check_granted([CPU], [CPU]) is None


def test_check_granted_silent_cpu_fallback_warns():
    note = check_granted([CUDA, CPU], [CPU], what="decoder_text_fp16.onnx")
    assert note.startswith("warning:") and "decoder_text_fp16.onnx" in note and CPU in note


def test_check_granted_silent_cpu_fallback_raises_with_require_gpu():
    with pytest.raises(GpuUnavailable):
        check_granted([CUDA, CPU], [CPU], require_gpu=True)


def test_check_granted_empty_used_counts_as_not_granted():
    assert check_granted([CUDA, CPU], []) is not None
