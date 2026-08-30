"""``_compute_server_args`` MoE-parity helpers.

Home for the SGLang <-> Megatron MoE numerical-parity helpers lifted out of
miles/backends/sglang_utils/sglang_engine.py (Phase 3 isolation, slice 3c).
``_compute_server_args`` (a pre-fork miles function, heavily modified in
place and left in the miles file per the P1/P2 pattern split) calls
``_configure_megatron_moe_parity_kwargs`` and ``_training_adapter_dtype_arg``
directly.
"""

import logging

logger = logging.getLogger(__name__)


def _target_modules_request_moe_lora(target_modules) -> bool:
    if target_modules is None:
        return False
    if isinstance(target_modules, str):
        values = [part.strip().lower() for part in target_modules.split(",")]
    else:
        values = [str(part).strip().lower() for part in target_modules]
    return bool(
        {value for value in values if value}
        & {
            "all",
            "all-linear",
            "all_linear",
            "gate_proj",
            "up_proj",
            "down_proj",
            "linear_fc1",
            "linear_fc2",
            "linear_fc1_gate",
            "linear_fc1_up",
        }
    )


def _training_adapter_dtype_arg(args) -> str:
    if getattr(args, "fp16", False):
        return "float16"
    if getattr(args, "bf16", False):
        return "bfloat16"
    return "float32"


def _args_indicate_moe_model(args) -> bool:
    # num_experts is the authoritative MoE signal. moe_layer_freq is meaningful
    # only when MoE is active, and Megatron's parser defaults it to 1 even for
    # dense models, so it cannot be used as a fallback indicator.
    num_experts = getattr(args, "num_experts", None)
    if num_experts is None:
        return False
    try:
        return int(num_experts) > 0
    except (TypeError, ValueError):
        return False


def _configure_megatron_moe_parity_kwargs(kwargs: dict, args, sglang_overrides: dict | None) -> None:
    if not _args_indicate_moe_model(args):
        return

    explicit_overrides = set(sglang_overrides or {})

    if (
        not getattr(args, "moe_apply_probs_on_input", False)
        and "moe_megatron_weighted_swiglu" not in explicit_overrides
    ):
        if not kwargs.get("moe_megatron_weighted_swiglu", False):
            logger.info(
                "Megatron MoE rollout: enabling SGLang moe_megatron_weighted_swiglu "
                "to match Megatron Core weighted_bias_swiglu_impl."
            )
        kwargs["moe_megatron_weighted_swiglu"] = True

    if (
        str(getattr(args, "moe_router_dtype", "")).lower() == "fp32"
        and "moe_router_force_fp32" not in explicit_overrides
    ):
        if not kwargs.get("moe_router_force_fp32", False):
            logger.info(
                "Megatron MoE rollout: enabling SGLang moe_router_force_fp32 "
                "because Megatron moe_router_dtype=fp32."
            )
        kwargs["moe_router_force_fp32"] = True
