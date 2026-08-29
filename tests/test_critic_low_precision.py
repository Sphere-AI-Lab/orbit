import argparse

import pytest

from orbit.peft.megatron.low_precision_bootstrap import validate_low_precision_bootstrap_args


def test_adapter_mode_rejects_low_precision_bridge_checkpoint_early():
    args = argparse.Namespace(megatron_to_hf_mode="bridge", critic_mode="adapter")
    hf_config = {"quantization_config": {"quant_method": "int4"}}

    with pytest.raises(ValueError, match="critic-mode adapter.*low-precision/quantized"):
        validate_low_precision_bootstrap_args(args, hf_config=hf_config)
