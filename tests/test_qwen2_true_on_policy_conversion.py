"""Ported from miles ``tests/fast/backends/megatron_utils/test_qwen2_true_on_policy_conversion.py``,
then generalized to cover every megatron_to_hf converter.

--true-on-policy forces --transformer-impl local (orbit/true_on_policy/config.py),
under which Megatron emits layernorm params as bare "input_layernorm.weight" /
"pre_mlp_layernorm.weight" instead of the TE-fused
"self_attention.linear_qkv.layer_norm_weight" / "mlp.linear_fc1.layer_norm_weight"
names. Every convert_*_to_hf function must accept both spellings.
"""

from argparse import Namespace

import pytest
import torch

from miles.backends.megatron_utils.megatron_to_hf.deepseekv3 import convert_deepseekv3_to_hf
from miles.backends.megatron_utils.megatron_to_hf.glm4 import convert_glm4_to_hf
from miles.backends.megatron_utils.megatron_to_hf.glm4moe import convert_glm4moe_to_hf
from miles.backends.megatron_utils.megatron_to_hf.llama import convert_llama_to_hf
from miles.backends.megatron_utils.megatron_to_hf.mimo import convert_mimo_to_hf
from miles.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf
from miles.backends.megatron_utils.megatron_to_hf.qwen3_5 import convert_qwen3_5_to_hf
from miles.backends.megatron_utils.megatron_to_hf.qwen3_next import convert_qwen3_next_to_hf
from miles.backends.megatron_utils.megatron_to_hf.qwen3moe import convert_qwen3moe_to_hf

ARGS = Namespace(
    hidden_size=4,
    kv_channels=2,
    num_attention_heads=2,
    num_query_groups=1,
)

# (converter, HF layer-prefix that converter emits for layer 0)
CONVERTERS = [
    (convert_qwen2_to_hf, "model.layers.0"),
    (convert_llama_to_hf, "model.layers.0"),
    (convert_glm4_to_hf, "model.layers.0"),
    (convert_glm4moe_to_hf, "model.layers.0"),
    (convert_qwen3_5_to_hf, "model.language_model.layers.0"),
    (convert_qwen3_next_to_hf, "model.layers.0"),
    (convert_qwen3moe_to_hf, "model.layers.0"),
    (convert_mimo_to_hf, "model.layers.0"),
    (convert_deepseekv3_to_hf, "model.layers.0"),
]
CONVERTER_IDS = ["qwen2", "llama", "glm4", "glm4moe", "qwen3_5", "qwen3_next", "qwen3moe", "mimo", "deepseekv3"]


@pytest.mark.parametrize(("convert_fn", "hf_prefix"), CONVERTERS, ids=CONVERTER_IDS)
def test_converter_accepts_explicit_true_on_policy_layernorm_names(convert_fn, hf_prefix):
    param = torch.ones(4)

    assert convert_fn(
        ARGS,
        "module.module.decoder.layers.0.input_layernorm.weight",
        param,
    ) == [(f"{hf_prefix}.input_layernorm.weight", param)]
    assert convert_fn(
        ARGS,
        "module.module.decoder.layers.0.pre_mlp_layernorm.weight",
        param,
    ) == [(f"{hf_prefix}.post_attention_layernorm.weight", param)]
