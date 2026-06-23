"""OFT utilities for Megatron backend using Megatron-Bridge PEFT integration."""

import logging
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import torch

from .peft_utils import (
    convert_target_modules_to_megatron,
    detect_peft_variant,
    parse_exclude_modules,
    resolve_target_modules_hf,
)


OFT_ADAPTER_NAME = "orbit_oft"
logger = logging.getLogger(__name__)
_BRIDGE_OFT_EMBEDDING_WEIGHT_PROXY_PATCHED = False


def _patch_bridge_oft_embedding_weight_proxy() -> None:
    """Expose the wrapped embedding weight on Bridge's OFT embedding wrapper.

    Megatron reads ``embedding.word_embeddings.weight`` when embeddings and
    output weights are tied. Bridge's ``OFTVocabParallelEmbedding`` stores the
    real embedding module in ``to_wrap`` but does not proxy ``weight``, so
    ``--target-modules all`` can fail after wrapping ``word_embeddings``.
    """
    global _BRIDGE_OFT_EMBEDDING_WEIGHT_PROXY_PATCHED
    if _BRIDGE_OFT_EMBEDDING_WEIGHT_PROXY_PATCHED:
        return

    from megatron.bridge.peft.oft_layers import OFTVocabParallelEmbedding

    current = getattr(OFTVocabParallelEmbedding, "weight", None)
    if current is None:
        OFTVocabParallelEmbedding.weight = property(lambda self: self.to_wrap.weight)
    elif not isinstance(current, property):
        logger.warning(
            "OFTVocabParallelEmbedding already exposes non-property weight=%r; "
            "leaving Bridge behavior unchanged.",
            current,
        )
    _BRIDGE_OFT_EMBEDDING_WEIGHT_PROXY_PATCHED = True


def _oft_type(args: Namespace) -> str:
    oft_type = getattr(args, "oft_type", "canonical_oft")
    if oft_type not in {"oft", "canonical_oft"}:
        raise ValueError(f"Unsupported OFT type: {oft_type!r}. Expected 'oft' or 'canonical_oft'.")
    return oft_type


def is_oft_weight_name(name: str) -> bool:
    """Check if a weight name corresponds to an OFT adapter weight.

    HF PEFT OFT names contain ``.oft_r`` (the skew-symmetric generator); the
    substring ``.oft_`` also matches the uppercase ``.oft_R`` form used by
    some loaders.
    """
    return ".oft_" in name


def create_oft_instance(args: Namespace):
    """Return the OFT PEFT instance used by orbit.

    DSV4 keeps the dedicated native MoE wrapper. Other OFT runs default to
    ``CanonicalOFT`` so fused QKV / gate-up layers receive independent split
    rotations, and unmerged leaves fall back inside Megatron-Bridge to plain
    per-module OFT wrappers. ``--oft-type oft`` intentionally selects the
    legacy shared-R ``OFT`` wrapper.
    """
    _patch_bridge_oft_embedding_weight_proxy()

    variant = detect_peft_variant(args)
    adapter_dtype = (
        torch.float16
        if getattr(args, "fp16", False)
        else torch.bfloat16
        if getattr(args, "bf16", False)
        else torch.float32
    )

    if variant == "dsv4":
        from megatron.bridge.models.deepseek.deepseek_v4_bridge import DSV4OFT

        target_modules = convert_target_modules_to_megatron(args.target_modules, variant="dsv4")
        exclude_modules = parse_exclude_modules(args, variant="dsv4")
        return DSV4OFT(
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            block_size=args.oft_block_size,
            coft=args.oft_coft,
            eps=args.oft_eps,
            block_share=args.oft_block_share,
            adapter_dtype=adapter_dtype,
        )

    if _oft_type(args) == "oft":
        from megatron.bridge.peft.oft import OFT

        target_modules = convert_target_modules_to_megatron(args.target_modules, variant=variant)
        exclude_modules = parse_exclude_modules(args, variant=variant)
        logger.warning(
            "OFT: legacy shared-R (--oft-type oft) — fused QKV/gate-up share one rotation. "
            "targets=%s exclude=%s",
            target_modules,
            exclude_modules,
        )
        return OFT(
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            block_size=args.oft_block_size,
            coft=args.oft_coft,
            eps=args.oft_eps,
            block_share=args.oft_block_share,
        )

    from megatron.bridge.peft.canonical_oft import CanonicalOFT

    target_modules = convert_target_modules_to_megatron(args.target_modules, variant="canonical")
    exclude_modules = parse_exclude_modules(args, variant="canonical")
    logger.info(
        "OFT: using CanonicalOFT (peft_variant=%r targets=%s exclude=%s)",
        getattr(args, "peft_variant", "standard"),
        target_modules,
        exclude_modules,
    )
    return CanonicalOFT(
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        block_size=args.oft_block_size,
        coft=args.oft_coft,
        eps=args.oft_eps,
        block_share=args.oft_block_share,
    )


def build_oft_sync_config(args: Namespace) -> dict[str, Any]:
    """Build OFT config dict for syncing weights to SGLang engines."""
    return {
        "peft_type": "OFT",
        "oft_type": getattr(args, "oft_type", "canonical_oft"),
        "oft_block_size": args.oft_block_size,
        "coft": args.oft_coft,
        "eps": args.oft_eps,
        "block_share": args.oft_block_share,
        "target_modules": resolve_target_modules_hf(args),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def save_oft_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
) -> str:
    from . import peft_utils

    return peft_utils.save_peft_adapter_checkpoint(
        model,
        args,
        save_dir,
        method="oft",
        build_config=lambda: build_oft_sync_config(args),
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        iteration=iteration,
    )


def load_oft_adapter(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
) -> tuple[bool, int | None]:
    from . import peft_utils

    return peft_utils.load_peft_adapter_checkpoint(
        model,
        adapter_path,
        label="OFT",
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
    )
