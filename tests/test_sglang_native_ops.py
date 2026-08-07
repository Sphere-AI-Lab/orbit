import os
from types import SimpleNamespace

from orbit.backends.sglang_utils.native_ops import force_native_forward_after_init
from orbit.backends.sglang_utils.sglang_engine import (
    _compute_server_args,
    _configure_peft_cache_kwargs,
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


def test_configure_peft_cache_kwargs_disables_radix_for_oft():
    kwargs = {"disable_radix_cache": False}

    _configure_peft_cache_kwargs(kwargs, "oft")

    assert kwargs["disable_radix_cache"] is True


def test_configure_peft_cache_kwargs_disables_radix_for_lora():
    kwargs = {}

    _configure_peft_cache_kwargs(kwargs, "lora")

    assert kwargs["disable_radix_cache"] is True


def test_configure_peft_cache_kwargs_leaves_non_peft_unchanged():
    kwargs = {}

    _configure_peft_cache_kwargs(kwargs, None)

    assert "disable_radix_cache" not in kwargs


def _oft_server_args(
    *,
    adapter_double_buffer: bool,
    oft_adapter_path: str | None,
    opd_teacher_url: str | None = None,
):
    return SimpleNamespace(
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        hf_checkpoint="/base",
        seed=1,
        offload_rollout=False,
        sglang_dp_size=1,
        sglang_attn_cp_size=1,
        sglang_moe_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
        bf16=True,
        opd_type="sglang",
        opd_teacher="adapter:/teacher",
        opd_teacher_load=None,
        opd_teacher_url=opd_teacher_url,
        opd_teacher_urls=None,
        opd_serve_teacher=False,
        opd_teacher_pool=None,
        peft_method="oft",
        offload_rollout_adapter=False,
        target_modules=["linear_qkv"],
        oft_block_size=8,
        oft_type="canonical_oft",
        adapter_double_buffer=adapter_double_buffer,
        sglang_oft_backend="triton",
        oft_adapter_path=oft_adapter_path,
    )


def _compute_oft_server_args(args, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    kwargs, _ = _compute_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:29500",
        nccl_port=29501,
        host="127.0.0.1",
        port=30000,
        base_gpu_id=0,
    )
    return kwargs


def test_compute_server_args_merges_frozen_oft_teacher_into_peft_paths(monkeypatch):
    kwargs = _compute_oft_server_args(
        _oft_server_args(adapter_double_buffer=False, oft_adapter_path="/student"),
        monkeypatch,
    )

    assert kwargs["peft_paths"] == {
        "orbit_oft": "/student",
        "orbit_teacher": "/teacher",
    }
    assert "oft_paths" not in kwargs
    assert "lora_paths" not in kwargs
    assert kwargs["max_ofts_per_batch"] == 3
    assert kwargs["peft_double_buffer"] is False


def test_compute_server_args_reserves_frozen_teacher_with_double_buffer(monkeypatch):
    kwargs = _compute_oft_server_args(
        _oft_server_args(adapter_double_buffer=True, oft_adapter_path="/student"),
        monkeypatch,
    )

    assert kwargs["peft_paths"] == {
        "orbit_oft": "/student",
        "orbit_teacher": "/teacher",
    }
    assert kwargs["max_ofts_per_batch"] == 4
    assert kwargs["peft_double_buffer"] is True


def test_compute_server_args_keeps_teacher_when_student_has_no_adapter_path(monkeypatch):
    kwargs = _compute_oft_server_args(
        _oft_server_args(adapter_double_buffer=False, oft_adapter_path=None),
        monkeypatch,
    )

    assert kwargs["peft_paths"] == {"orbit_teacher": "/teacher"}


def test_compute_server_args_does_not_preload_teacher_for_external_opd(monkeypatch):
    kwargs = _compute_oft_server_args(
        _oft_server_args(
            adapter_double_buffer=False,
            oft_adapter_path="/student",
            opd_teacher_url="http://teacher/generate",
        ),
        monkeypatch,
    )

    assert kwargs["peft_paths"] == {"orbit_oft": "/student"}
    assert kwargs["max_ofts_per_batch"] == 2
