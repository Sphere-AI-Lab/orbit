# ORBIT-SEAM: whole-file rewrite - base's Bridge/LoRA model-setup implementation (config dataclass,
# value-model pre-wrap hook, model-list helper, and the full _setup_lora_model_via_bridge body) moved
# to orbit.megatron.bridge_peft_helpers to also serve OFT; this module keeps only a delegation shim.
"""Compatibility wrappers for LoRA-specific bridge imports."""

from orbit.megatron.bridge_peft_helpers import _ensure_model_list, _setup_peft_model_via_bridge


# ORBIT-SEAM: delegates to the shared PEFT (LoRA + OFT) bridge model builder in
# orbit.megatron.bridge_peft_helpers instead of the base's LoRA-only implementation removed above
def _setup_lora_model_via_bridge(args):
    return _setup_peft_model_via_bridge(args)
