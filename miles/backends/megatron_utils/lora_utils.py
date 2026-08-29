"""LoRA utilities for Megatron backend using Megatron-Bridge PEFT integration."""

import logging
# ORBIT-SEAM: removed base's `import os`: os.sync() after the HF PEFT adapter write now lives in
# orbit.megatron.peft_utils.save_peft_adapter_checkpoint
from argparse import Namespace
from collections.abc import Sequence
# ORBIT-SEAM: removed base's `from pathlib import Path`: adapter save/load path handling moved into
# orbit.megatron.peft_utils
from typing import Any

import torch
# ORBIT-SEAM: removed base's `torch.distributed` / `megatron.core.mpu` imports: rank and TP/PP-rank
# bookkeeping for adapter save/load moved into orbit.megatron.peft_utils
# ORBIT-SEAM: base's `get_parallel_state` import replaced by this block - PEFT-method detection,
# target-module conversion, and adapter checkpoint save/load now delegate to orbit.megatron.peft_utils
from orbit.megatron.peft_utils import (
    PeftCheckpointPreflight,
    convert_target_modules_to_megatron,
    get_peft_method,
    load_peft_adapter_checkpoint,
    parse_exclude_modules,
    resolve_target_modules_hf,
    save_peft_adapter_checkpoint,
)

logger = logging.getLogger(__name__)

LORA_ADAPTER_NAME = "miles_lora"

# ORBIT-SEAM: removed base's ~60-line HF<->Megatron module-name mapping tables
# (_STANDARD_LORA_HF_TO_MEGATRON, _CANONICAL_LORA_HF_TO_MEGATRON, _MEGATRON_TO_HF_MODULES,
# _HF_MODULE_NAMES) and their section-header comments: name conversion now lives in
# orbit.megatron.peft_utils (convert_target_modules_to_megatron / resolve_target_modules_hf)


def is_lora_enabled(args: Namespace) -> bool:
    # ORBIT-SEAM: base's lora_rank/lora_adapter_path check plus the whole is_lora_model() function
    # (removed) are replaced by orbit's single PEFT-method source of truth (LoRA vs OFT vs none)
    return get_peft_method(args) == "lora"


def is_lora_weight_name(name: str) -> bool:
    # ORBIT-SEAM: docstring dropped (logic unchanged): part of the file's delegation-focused rewrite
    return ".lora_A." in name or ".lora_B." in name


# ORBIT-SEAM: base's _is_adapter_param_name, _get_lora_class_name, convert_target_modules_to_megatron,
# convert_target_modules_to_hf, and parse_exclude_modules helpers (97 lines) removed - name-conversion
# logic moved to orbit.megatron.peft_utils (imported above); _lora_variant below is new orbit logic
# replacing the old lora_type-only class-name switch with an MLA-aware variant selector
def _lora_variant(args: Namespace) -> str:
    if getattr(args, "multi_latent_attention", False):
        return "mla"
    lora_type_name = getattr(args, "lora_type", "lora").lower()
    return "canonical" if lora_type_name == "canonical_lora" else "standard"


def create_lora_instance(args: Namespace):
    # ORBIT-SEAM: docstring dropped (logic below rewritten to delegate); still creates the
    # LoRA/CanonicalLoRA instance, now via _lora_variant
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.lora import LoRA

    # ORBIT-SEAM: base's lora_type_name lookup + if/else lora_cls selection replaced by _lora_variant,
    # which additionally special-cases multi_latent_attention (MLA) to force the "mla" variant
    variant = _lora_variant(args)
    # CanonicalLoRA splits merged Q/K/V; MLA already exposes split projections,
    # so plain LoRA is the right wrapper there.
    lora_cls = CanonicalLoRA if variant == "canonical" else LoRA

    # ORBIT-SEAM: convert_target_modules_to_megatron / parse_exclude_modules now take variant=
    # (from _lora_variant) instead of lora_type=lora_cls - matches the orbit.megatron.peft_utils signatures
    target_modules = convert_target_modules_to_megatron(args.target_modules, variant=variant)
    exclude_modules = parse_exclude_modules(args, variant=variant)

    lora = lora_cls(
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        # ORBIT-SEAM: args attribute renamed lora_A_init_method -> lora_a_init_method (lowercase 'a'):
        # matches orbit's argument-registration casing convention
        lora_A_init_method=getattr(args, "lora_a_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )

    logger.info(
        f"Created {lora_cls.__name__}: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        # ORBIT-SEAM: log line gains a_init=<lora_a_init_method> alongside the existing fields -
        # reflects the renamed arg above
        f"dropout={args.lora_dropout}, a_init={getattr(args, 'lora_a_init_method', 'xavier')}, "
        f"target_modules={target_modules}, exclude_modules={exclude_modules}"
    )
    return lora


# ORBIT-SEAM: relocated here from the end of the base file (base's checkpoint save/load section
# header comment stood at this spot) so save_lora_checkpoint's build_config lambda below can see it;
# target-module HF listing now delegates to orbit.megatron.peft_utils.resolve_target_modules_hf
# instead of computing target_modules_hf inline via the removed convert_target_modules_to_hf
def build_lora_sync_config(args: Namespace) -> dict[str, Any]:
    """Build LoRA config dict for syncing weights to SGLang engines."""
    return {
        "peft_type": "LORA",
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": resolve_target_modules_hf(args),
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def save_lora_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
    # ORBIT-SEAM: active_student_version param added - threaded through to
    # orbit.megatron.peft_utils.save_peft_adapter_checkpoint for OPD self-teacher version tagging
    active_student_version: str | None = None,
) -> str:
    # ORBIT-SEAM: base's ~90-line LoRA save (Megatron-native per-rank .pt write + HF PEFT bridge
    # export + adapter_config.json + optimizer/scheduler training-state write) excised and replaced
    # by a single delegation to orbit.megatron.peft_utils.save_peft_adapter_checkpoint(method="lora"),
    # which implements the same two-format save plus PEFT method dispatch and the self-teacher sidecar
    return save_peft_adapter_checkpoint(
        model,
        args,
        save_dir,
        method="lora",
        build_config=lambda: build_lora_sync_config(args),
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        iteration=iteration,
        active_student_version=active_student_version,
    )


def load_lora_adapter(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    # ORBIT-SEAM: 3 params added (expected_iteration, expected_active_student_version,
    # checkpoint_preflight) - support orbit's checkpoint-preflight validation and self-teacher-version
    # consistency check, threaded through to orbit.megatron.peft_utils.load_peft_adapter_checkpoint
    expected_iteration: int | None = None,
    expected_active_student_version: str | None = None,
    checkpoint_preflight: PeftCheckpointPreflight | None = None,
) -> tuple[bool, int | None]:
    # ORBIT-SEAM: base's ~90-line LoRA load (native .pt tensor restore + HF-PEFT-not-supported
    # warning + optimizer/scheduler training-state restore) plus the standalone _load_training_state()
    # helper (removed entirely) excised and replaced by a delegation to
    # orbit.megatron.peft_utils.load_peft_adapter_checkpoint(label="LoRA"), which folds in
    # checkpoint-preflight validation and self-teacher-version checks
    return load_peft_adapter_checkpoint(
        model,
        adapter_path,
        label="LoRA",
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        expected_iteration=expected_iteration,
        expected_active_student_version=expected_active_student_version,
        checkpoint_preflight=checkpoint_preflight,
    )


# ORBIT-SEAM: base's build_lora_sync_config (inline target_modules_hf computation via
# convert_target_modules_to_hf, then this same dict literal) stood at the end of the file here -
# it was relocated above create_lora_instance and now delegates to
# orbit.megatron.peft_utils.resolve_target_modules_hf; see the ORBIT-SEAM stamp there
