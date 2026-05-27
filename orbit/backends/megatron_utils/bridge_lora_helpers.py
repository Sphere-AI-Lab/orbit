"""Compatibility wrappers for LoRA-specific bridge imports."""

from .bridge_peft_helpers import _ensure_model_list, _setup_peft_model_via_bridge


def _setup_lora_model_via_bridge(args):
    return _setup_peft_model_via_bridge(args)
