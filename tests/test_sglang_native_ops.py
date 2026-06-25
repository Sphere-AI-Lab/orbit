import os
from types import SimpleNamespace

from orbit.backends.sglang_utils.native_ops import force_native_forward_after_init
from orbit.backends.sglang_utils.sglang_engine import (
    _configure_peft_runtime_kwargs,
    _prepare_child_peft_cache_env,
)


def test_force_native_forward_after_init_uses_native_forward():
    class Op:
        def __init__(self):
            self._forward_method = self.forward_cuda

        def forward_cuda(self):
            return "cuda"

        def forward_native(self):
            return "native"

    force_native_forward_after_init(Op)

    assert Op()._forward_method() == "native"


def test_force_native_forward_after_init_is_idempotent():
    class Op:
        init_count = 0

        def __init__(self):
            type(self).init_count += 1
            self._forward_method = self.forward_cuda

        def forward_cuda(self):
            return "cuda"

        def forward_native(self):
            return "native"

    force_native_forward_after_init(Op)
    force_native_forward_after_init(Op)
    instance = Op()

    assert instance._forward_method() == "native"
    assert Op.init_count == 1


def test_prepare_child_peft_cache_env_disables_cpp_radix_for_oft(monkeypatch):
    monkeypatch.setenv("SGLANG_EXPERIMENTAL_CPP_RADIX_TREE", "1")

    _prepare_child_peft_cache_env(SimpleNamespace(enable_oft=True, enable_lora=None))

    assert os.environ["SGLANG_EXPERIMENTAL_CPP_RADIX_TREE"] == "0"


def test_prepare_child_peft_cache_env_disables_cpp_radix_for_lora(monkeypatch):
    monkeypatch.setenv("SGLANG_EXPERIMENTAL_CPP_RADIX_TREE", "true")

    _prepare_child_peft_cache_env(SimpleNamespace(enable_oft=None, enable_lora=True))

    assert os.environ["SGLANG_EXPERIMENTAL_CPP_RADIX_TREE"] == "0"


def test_prepare_child_peft_cache_env_leaves_non_peft_server_unchanged(monkeypatch):
    monkeypatch.setenv("SGLANG_EXPERIMENTAL_CPP_RADIX_TREE", "1")

    _prepare_child_peft_cache_env(SimpleNamespace(enable_oft=None, enable_lora=None))

    assert os.environ["SGLANG_EXPERIMENTAL_CPP_RADIX_TREE"] == "1"


def test_configure_peft_runtime_kwargs_disables_radix_for_oft():
    kwargs = {"disable_radix_cache": False}

    _configure_peft_runtime_kwargs(kwargs, "oft")

    assert kwargs["disable_radix_cache"] is True


def test_configure_peft_runtime_kwargs_disables_radix_for_lora():
    kwargs = {}

    _configure_peft_runtime_kwargs(kwargs, "lora")

    assert kwargs["disable_radix_cache"] is True


def test_configure_peft_runtime_kwargs_disables_cuda_graph_for_oft():
    kwargs = {"disable_cuda_graph": False}

    _configure_peft_runtime_kwargs(kwargs, "oft")

    assert kwargs["disable_cuda_graph"] is True


def test_configure_peft_runtime_kwargs_leaves_cuda_graph_enabled_for_lora():
    kwargs = {}

    _configure_peft_runtime_kwargs(kwargs, "lora")

    assert "disable_cuda_graph" not in kwargs


def test_configure_peft_runtime_kwargs_leaves_non_peft_unchanged():
    kwargs = {}

    _configure_peft_runtime_kwargs(kwargs, None)

    assert "disable_radix_cache" not in kwargs
    assert "disable_cuda_graph" not in kwargs
