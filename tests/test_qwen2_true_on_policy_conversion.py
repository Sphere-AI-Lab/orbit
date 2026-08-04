"""Ported from miles ``tests/fast/backends/megatron_utils/test_qwen2_true_on_policy_conversion.py``.

--true-on-policy forces --transformer-impl local (orbit/true_on_policy/config.py),
under which Megatron emits layernorm params as bare "input_layernorm.weight" /
"pre_mlp_layernorm.weight" instead of the TE-fused
"self_attention.linear_qkv.layer_norm_weight" / "mlp.linear_fc1.layer_norm_weight"
names. convert_qwen2_to_hf must accept both.
"""

from argparse import Namespace

import torch


def test_qwen2_converter_accepts_explicit_true_on_policy_layernorm_names():
    from orbit.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf

    args = Namespace(
        hidden_size=4,
        kv_channels=2,
        num_attention_heads=2,
        num_query_groups=1,
    )
    param = torch.ones(4)

    assert convert_qwen2_to_hf(
        args,
        "module.module.decoder.layers.0.input_layernorm.weight",
        param,
    ) == [("model.layers.0.input_layernorm.weight", param)]
    assert convert_qwen2_to_hf(
        args,
        "module.module.decoder.layers.0.pre_mlp_layernorm.weight",
        param,
    ) == [("model.layers.0.post_attention_layernorm.weight", param)]
