"""LoRA utilities for Megatron backend using Megatron-Bridge PEFT integration."""

import logging
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import torch

from .peft_utils import (
    convert_target_modules_to_hf,
    convert_target_modules_to_megatron,
    get_peft_method,
    is_adapter_param_name,
    load_peft_adapter_checkpoint,
    parse_exclude_modules,
    resolve_target_modules_hf,
    save_peft_adapter_checkpoint,
)

logger = logging.getLogger(__name__)

LORA_ADAPTER_NAME = "orbit_lora"


def is_lora_enabled(args: Namespace) -> bool:
    return get_peft_method(args) == "lora"


def is_lora_weight_name(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def _lora_variant(args: Namespace) -> str:
    if getattr(args, "multi_latent_attention", False):
        return "mla"
    lora_type_name = getattr(args, "lora_type", "lora").lower()
    return "canonical" if lora_type_name == "canonical_lora" else "standard"


def create_lora_instance(args: Namespace):
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.lora import LoRA

    variant = _lora_variant(args)
    # CanonicalLoRA splits merged Q/K/V; MLA already exposes split projections,
    # so plain LoRA is the right wrapper there.
    lora_cls = CanonicalLoRA if variant == "canonical" else LoRA

    target_modules = convert_target_modules_to_megatron(args.target_modules, variant=variant)
    exclude_modules = parse_exclude_modules(args, variant=variant)

    lora = lora_cls(
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )

    logger.info(
        f"Created {lora_cls.__name__}: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, target_modules={target_modules}, "
        f"exclude_modules={exclude_modules}"
    )
    return lora


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
) -> str:
    return save_peft_adapter_checkpoint(
        model,
        args,
        save_dir,
        method="lora",
        build_config=lambda: build_lora_sync_config(args),
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        iteration=iteration,
    )


def load_lora_adapter(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
) -> tuple[bool, int | None]:
    return load_peft_adapter_checkpoint(
        model,
        adapter_path,
        label="LoRA",
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
    )
