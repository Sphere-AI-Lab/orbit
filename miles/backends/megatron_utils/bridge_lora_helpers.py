# ORBIT-SEAM: whole-file rewrite - base's Bridge/LoRA model-setup implementation (config dataclass,
# value-model pre-wrap hook, model-list helper, and the full _setup_lora_model_via_bridge body) moved
# to miles.orbit.megatron.bridge_peft_helpers to also serve OFT; this module keeps only a delegation shim.
# PORT DEBT (phase-4 merge): upstream evolved the extracted implementation - see the flag in the
# merge report for the list (provider recompute/attention-backend forwarding, multi-LoRA MoE
# validation, muon/multi-LoRA ddp_config, offload_train grad-buffer patch, load_hf_config).
"""Compatibility wrappers for LoRA-specific bridge imports."""

from miles.orbit.megatron.bridge_peft_helpers import _ensure_model_list, _setup_peft_model_via_bridge


# ORBIT-SEAM: delegates to the shared PEFT (LoRA + OFT) bridge model builder in
# miles.orbit.megatron.bridge_peft_helpers instead of the base's LoRA-only implementation removed above
def _setup_lora_model_via_bridge(args):
    return _setup_peft_model_via_bridge(args)
