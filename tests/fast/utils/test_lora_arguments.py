"""`--lora-a-init-method` survives PEFT arg normalization.

Scoped deliberately: this file carries only the LoRA-A-init assertions the
lora-without-regret port owns. The old repo's `test_lora_arguments.py` also
covered target-module parsing and exclusion, but that behaviour is exercised
here by `test_peft_arguments.py`'s sibling suite and by the production helper's
own callers -- copying it wholesale would fork two copies of assertions about
code this port never touched.

The init method is what the campaign varies: Bridge's default `xavier` is
`xavier_normal_`, while `kaiming` is `kaiming_uniform_(a=sqrt(5))` -- HF PEFT's
spelling and the LoRA-without-regret paper's. They differ by ~2.4x in std, which
moves the optimal learning rate, so a value that silently fails to survive
normalization would shift every LR sweep in the study.
"""

from argparse import Namespace
from copy import deepcopy

import pytest

from miles.utils.arguments import _normalize_peft_args


def _make_args(**overrides) -> Namespace:
    args = {
        "peft_method": "lora",
        "target_modules": None,
        "exclude_modules": None,
        "peft_adapter_path": None,
        "lora_rank": 0,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_type": "lora",
        "lora_adapter_path": None,
        "lora_sync_from_tensor": False,
        "lora_a_init_method": "xavier",
        "oft_type": "canonical_oft",
        "oft_block_size": 0,
        "oft_coft": False,
        "oft_eps": 1e-5,
        "oft_block_share": False,
        "oft_adapter_path": None,
        "adapter_double_buffer": False,
        "colocate": True,
        "megatron_to_hf_mode": "bridge",
    }
    args.update(overrides)
    return Namespace(**args)


def _apply_peft_validation(args: Namespace) -> Namespace:
    return _normalize_peft_args(deepcopy(args))


class TestLoraAInitMethod:
    def test_default_is_xavier(self):
        args = _make_args(lora_rank=16, target_modules="q_proj", lora_a_init_method="xavier")
        result = _apply_peft_validation(args)
        assert result.lora_a_init_method == "xavier"

    def test_kaiming_survives_normalization(self):
        args = _make_args(lora_rank=16, target_modules="q_proj", lora_a_init_method="kaiming")
        result = _apply_peft_validation(args)
        assert result.lora_a_init_method == "kaiming"

    def test_non_default_init_rejected_when_peft_method_is_oft(self):
        """`lora_a_init_method` must be listed in `_PEFT_LORA_DEFAULTS`.

        That membership is what makes a LoRA-only flag an error under
        `--peft-method oft`. If the entry is ever dropped, a `--lora-a-init-method`
        passed alongside OFT is silently accepted and silently ignored.
        """
        args = _make_args(
            peft_method="oft",
            lora_rank=0,
            oft_block_size=64,
            target_modules="q_proj",
            lora_a_init_method="kaiming",
        )
        with pytest.raises(AssertionError, match="LoRA flags require --peft-method lora"):
            _apply_peft_validation(args)

    def test_default_init_accepted_when_peft_method_is_oft(self):
        args = _make_args(
            peft_method="oft",
            lora_rank=0,
            oft_block_size=64,
            target_modules="q_proj",
            lora_a_init_method="xavier",
        )
        result = _apply_peft_validation(args)
        assert result.peft_method == "oft"
