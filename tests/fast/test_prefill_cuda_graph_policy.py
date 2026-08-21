"""Orbit defaults sglang's prefill CUDA-graph backend to "disabled" and refuses
other backends under OFT (Phase-0 finding, 2026-08-21: the breakable backend
refuses memory-saver/colocate and its replay does not apply OFT adapters)."""

from argparse import Namespace

import pytest

from orbit.backends.sglang_utils.arguments import apply_prefill_cuda_graph_policy


def test_unset_backend_defaults_to_disabled():
    args = Namespace(sglang_cuda_graph_backend_prefill=None, peft_method="oft")
    apply_prefill_cuda_graph_policy(args)
    assert args.sglang_cuda_graph_backend_prefill == "disabled"


def test_missing_attribute_defaults_to_disabled():
    args = Namespace(peft_method="none")
    apply_prefill_cuda_graph_policy(args)
    assert args.sglang_cuda_graph_backend_prefill == "disabled"


def test_explicit_disabled_is_kept():
    args = Namespace(sglang_cuda_graph_backend_prefill="disabled", peft_method="oft")
    apply_prefill_cuda_graph_policy(args)
    assert args.sglang_cuda_graph_backend_prefill == "disabled"


def test_explicit_backend_rejected_under_oft():
    args = Namespace(sglang_cuda_graph_backend_prefill="breakable", peft_method="oft")
    with pytest.raises(ValueError, match="not supported with --peft-method oft"):
        apply_prefill_cuda_graph_policy(args)


@pytest.mark.parametrize("peft", ["lora", "none"])
def test_explicit_backend_allowed_without_oft(peft):
    args = Namespace(sglang_cuda_graph_backend_prefill="breakable", peft_method=peft)
    apply_prefill_cuda_graph_policy(args)
    assert args.sglang_cuda_graph_backend_prefill == "breakable"
