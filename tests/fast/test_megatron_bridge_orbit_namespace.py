"""Contract between Orbit and its private Megatron-Bridge namespace."""

from __future__ import annotations

import importlib

import pytest


CONSUMED_BRIDGE_SYMBOLS = {
    "megatron.bridge.orbit.low_precision.common": (
        "TensorSpillManager",
        "build_single_rank_meta_provider",
        "patch_meta_init_for_te_modules",
    ),
    "megatron.bridge.orbit.low_precision.fp8": (
        "apply_modelopt_fp8_to_meta_model",
        "build_fp8_direct_model_state_dict",
    ),
    "megatron.bridge.orbit.low_precision.int4": (
        "build_int4_direct_model_state_dict",
        "dequantize_int4",
        "quantize_to_int4",
        "register_int4_buffers_after_load_dense",
        "transform_sharded_state_dict_for_int4_dense",
    ),
    "megatron.bridge.orbit.low_precision.nvfp4": (
        "apply_modelopt_nvfp4_to_meta_model",
        "build_nvfp4_direct_model_state_dict",
        "collect_nvfp4_target_module_names",
        "is_nvfp4_source",
        "register_nvfp4_buffers_after_load_dense",
        "transform_sharded_state_dict_for_nvfp4_dense",
    ),
    "megatron.bridge.orbit.model_bridges.deepseek_v3_int4_bridge": ("DeepSeekV3INT4Bridge",),
    "megatron.bridge.orbit.model_bridges.deepseek_v4_bridge": ("DSV4OFT", "DeepSeekV4Bridge"),
    "megatron.bridge.orbit.model_bridges.llama_int4_bridge": ("LlamaINT4Bridge",),
    "megatron.bridge.orbit.model_bridges.qwen3_int4_bridge": ("Qwen3INT4Bridge", "Qwen3MoEINT4Bridge"),
    "megatron.bridge.orbit.oft.canonical_oft": ("CanonicalOFT",),
    "megatron.bridge.orbit.oft.oft": ("OFT",),
    "megatron.bridge.orbit.oft.oft_layers": ("OFTVocabParallelEmbedding",),
    "megatron.bridge.orbit.oft.param_names": (
        "CANONICAL_OFT_SLICE_NAMES",
        "is_peft_adapter_param_name",
    ),
    "megatron.bridge.orbit.quant.fp8_utils": (
        "merge_gated_mlp_scale_inv",
        "merge_qkv_scale_inv",
        "register_fp8_scale_inv_buffers_after_load",
        "transform_sharded_state_dict_for_fp8",
    ),
    "megatron.bridge.orbit.quant.int4_utils": (
        "register_int4_buffers_after_load",
        "transform_sharded_state_dict_for_int4",
    ),
    "megatron.bridge.orbit.quant.nvfp4_utils": (
        "register_nvfp4_buffers_after_load",
        "transform_sharded_state_dict_for_nvfp4",
    ),
}


@pytest.mark.parametrize(("module_name", "symbol_names"), CONSUMED_BRIDGE_SYMBOLS.items())
def test_restructured_bridge_exports_orbit_contract(module_name: str, symbol_names: tuple[str, ...]) -> None:
    module = importlib.import_module(module_name)

    missing = [name for name in symbol_names if not hasattr(module, name)]
    assert not missing, f"{module_name} is missing expected symbols: {', '.join(missing)}"
