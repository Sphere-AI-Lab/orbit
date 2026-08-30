"""Shared helpers used by both backends.

`validate_adapter_weight_chunk` is moved verbatim from
`update_weight_from_tensor.py:444-468` and adapted to take a `PeftMethodSpec`
instead of a `PeftSyncSpec` (so the transport layer is method-agnostic at the
call site).

`coalesce_lora_hf_weight_chunks` is moved from `update_weight_from_tensor.py`
to live next to the SGLang LoRA call that requires coalesced input.
"""
from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterable

import torch

from miles.orbit.transport.registry import PeftMethodSpec


def peft_adapter_preloaded(args: Namespace, peft_method: str) -> bool:
    """Return True when a LoRA adapter was already loaded into the engine at
    startup (via lora_adapter_path / peft_adapter_path). Used to decide
    whether an unload-before-reload step is needed on the first sync.
    """
    if peft_method != "lora":
        return False
    return bool(getattr(args, "lora_adapter_path", None) or getattr(args, "peft_adapter_path", None))


def validate_adapter_weight_chunk(
    hf_named_tensors: Iterable[tuple[str, torch.Tensor]],
    method_spec: PeftMethodSpec,
) -> list[tuple[str, torch.Tensor]]:
    """Filter to method-specific adapter tensors. Raise on empty or mixed."""
    named_tensors = list(hf_named_tensors)
    is_adapter_name = method_spec.weight_name_predicate
    label = method_spec.label
    sample_names = method_spec.sample_names

    weight_tensors = [(n, t) for n, t in named_tensors if is_adapter_name(n)]
    if not weight_tensors:
        raise RuntimeError(
            f"{label} weight sync failed: no {label} weights ({sample_names}) found in the "
            "HF weight chunk produced by the weight iterator. This usually means the "
            "Megatron-Bridge or SGLang version is incompatible and adapter weights were "
            "not exported. Check that `megatron_to_hf_mode` and bridge version match."
        )
    if len(weight_tensors) != len(named_tensors):
        non_matching = [n for n, _ in named_tensors if not is_adapter_name(n)]
        raise RuntimeError(
            f"{label} weight sync failed: mixed {label} adapter weights with non-{label} tensors "
            f"in the HF weight chunk: {', '.join(non_matching)}"
        )
    return weight_tensors


def coalesce_lora_hf_weight_chunks(
    hf_weight_chunks: Iterable[list[tuple[str, torch.Tensor]]],
) -> tuple[int, list[list[tuple[str, torch.Tensor]]]]:
    """SGLang LoRA adapter loading replaces the full adapter per call.

    The HF iterator may split a large MoE LoRA adapter according to
    update_weight_buffer_size. Sending those chunks one by one drops all but
    the last chunk on the rollout side, so LoRA must be coalesced into one
    logical adapter payload before calling load_lora_adapter_from_tensors.

    Returns (chunk_count, [combined_chunk]) — the combined list always has
    exactly one element (or zero if the input was empty).
    """
    combined: list[tuple[str, torch.Tensor]] = []
    chunk_count = 0
    for chunk in hf_weight_chunks:
        chunk_count += 1
        combined.extend(chunk)

    if chunk_count == 0:
        return 0, []

    return chunk_count, [combined]


def coalesce_oft_hf_weight_chunks(
    hf_weight_chunks: Iterable[list[tuple[str, torch.Tensor]]],
) -> tuple[int, list[list[tuple[str, torch.Tensor]]]]:
    """SGLang's OFT streamed loader needs all CanonicalOFT siblings together.

    For CanonicalOFT, Bridge HF-exports each split slice (``q_proj.oft_R`` /
    ``k_proj.oft_R`` / ``v_proj.oft_R`` and ``gate_proj.oft_R`` /
    ``up_proj.oft_R``) as an independent tensor. SGLang's
    ``load_streamed_oft_adapter`` calls ``normalize_merged_oft_weights`` on
    each ``update_weights_from_tensor`` chunk to pre-stack siblings into the
    runtime's fused ``qkv_proj.oft_R`` / ``gate_up_proj.oft_R`` stacked-R
    buffer, but the stacker requires *all* sibling leaves to be present in
    the same per-chunk dict. The orbit ``update_weight_buffer_size`` chunker
    splits siblings across chunks for a handful of layers per sync, after
    which the dense resolver silently drops the non-primary leaves via
    ``_OFT_STACKED_PARAMS`` ``is_duplicate`` skipping — k/v slices stay at
    identity on the rollout while training has the full canonical R, and
    train/rollout logprobs drift super-linearly with training steps.

    Coalescing into one logical chunk guarantees all siblings of each fused
    target are co-located, mirroring :func:`coalesce_lora_hf_weight_chunks`.
    The rollout-side OFT loader is unchanged.

    Returns ``(chunk_count, [combined_chunk])`` — the combined list always
    has exactly one element (or zero if the input was empty).

    The iteration order of tensors within and across chunks is preserved.
    Downstream ``dedupe_named_tensors_by_storage`` assigns ``unique_index``
    in first-seen order and the OFT name-resolver depends on that
    ordering remaining stable across the sync.
    """
    combined: list[tuple[str, torch.Tensor]] = []
    chunk_count = 0
    for chunk in hf_weight_chunks:
        chunk_count += 1
        combined.extend(chunk)

    if chunk_count == 0:
        return 0, []

    return chunk_count, [combined]
