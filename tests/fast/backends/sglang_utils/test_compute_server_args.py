from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from orbit.backends.sglang_utils import sglang_engine
from orbit.backends.sglang_utils.sglang_engine import _compute_server_args


def make_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        hf_checkpoint="/fake/model",
        seed=0,
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=1,
        offload_rollout=False,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        sglang_mem_fraction_static=0.7,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        fp16=False,
        lora_adapter_path=None,
        multi_lora_n_adapters=1,
        target_modules=["linear_qkv"],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def compute(args, **kwargs) -> dict:
    server_args, _ = _compute_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:1234",
        nccl_port=5000,
        host="127.0.0.1",
        port=30000,
        base_gpu_id=0,
        **kwargs,
    )
    return server_args


class TestSglangOverridePrecedence:
    """An override must win over every args-derived default, including the conditional ones."""

    def test_override_wins_over_conditional_args_defaults(self):
        args = make_args(fp16=True, use_rollout_routing_replay=True, use_rollout_indexer_replay=True)

        server_args = compute(args, sglang_overrides={"dtype": "bfloat16"})

        assert server_args["dtype"] == "bfloat16"

    def test_override_wins_over_lora_defaults(self):
        args = make_args(lora_rank=8)

        server_args = compute(args, sglang_overrides={"enable_lora": False})

        assert server_args["enable_lora"] is False

    @pytest.mark.parametrize("value", [0.5, 0.95])
    def test_override_wins_over_base_sglang_args(self, value):
        args = make_args(sglang_mem_fraction_static=0.7)

        server_args = compute(args, sglang_overrides={"mem_fraction_static": value})

        assert server_args["mem_fraction_static"] == value

    def test_no_overrides_keeps_args_derived_values(self):
        args = make_args(fp16=True, lora_rank=8)

        server_args = compute(args)

        assert server_args["dtype"] == "float16"
        assert server_args["enable_lora"] is True
        assert server_args["mem_fraction_static"] == 0.7


@pytest.mark.parametrize("double_buffer,capacity", [(False, 2), (True, 3)])
def test_oft_options_reach_submission_sglang(double_buffer, capacity):
    args = make_args(
        peft_method="oft",
        bf16=True,
        oft_block_size=128,
        oft_type="canonical_oft",
        adapter_double_buffer=double_buffer,
        sglang_oft_impl="staged",
    )

    server_args = compute(args)

    assert server_args["enable_oft"] is True
    assert server_args["oft_double_buffer"] is double_buffer
    assert server_args["oft_target_modules"] == ["q_proj", "k_proj", "v_proj"]
    assert server_args["oft_dtype"] == "bf16"
    assert server_args["max_ofts_per_batch"] == capacity
    assert server_args["oft_impl"] == "staged"
    assert server_args["disable_radix_cache"] is True


def test_oft_preserves_unified_sglang_api(monkeypatch):
    unified_args = dataclasses.make_dataclass(
        "UnifiedServerArgs",
        ["peft_method", "peft_target_modules", "peft_double_buffer", "peft_paths"],
        bases=(sglang_engine.ServerArgs,),
        kw_only=True,
    )
    monkeypatch.setattr(sglang_engine, "ServerArgs", unified_args)
    args = make_args(
        peft_method="oft",
        oft_block_size=128,
        oft_type="canonical_oft",
        adapter_double_buffer=True,
        oft_adapter_path="/fake/adapter",
    )

    server_args = compute(args)

    assert server_args["peft_method"] == "oft"
    assert server_args["peft_target_modules"] == ["q_proj", "k_proj", "v_proj"]
    assert server_args["peft_double_buffer"] is True
    assert server_args["peft_paths"] == {"orbit_oft": "/fake/adapter"}
    assert "enable_oft" not in server_args


def test_oft_submission_override_wins():
    args = make_args(
        peft_method="oft", oft_block_size=128, oft_type="canonical_oft", adapter_double_buffer=True
    )

    server_args = compute(args, sglang_overrides={"oft_double_buffer": False})

    assert server_args["oft_double_buffer"] is False
