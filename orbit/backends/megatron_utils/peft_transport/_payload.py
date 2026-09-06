"""OFT-specific payload shaping. Moved verbatim from `_sync_peft_adapter`'s OFT
branch (update_weight_from_tensor.py:488-525), exposed as a payload_shaper hook.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from orbit.backends.megatron_utils.sglang import FlattenedTensorBucket

from .interface import PeftPayload


def _get_tensor_alias_key(tensor: torch.Tensor) -> tuple:
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        storage.data_ptr(),
    )


def _dedupe_named_tensors_by_storage(
    named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, int]]]:
    # Preserve the legacy SGLang OFT alias key and first-seen payload order.
    # Submission receivers accept this schema but no longer export the helper.
    unique_named_tensors: list[tuple[str, torch.Tensor]] = []
    entries: list[tuple[str, int]] = []
    key_to_index: dict[tuple, int] = {}

    for name, tensor in named_tensors:
        alias_key = _get_tensor_alias_key(tensor)
        unique_index = key_to_index.get(alias_key)
        if unique_index is None:
            unique_index = len(unique_named_tensors)
            key_to_index[alias_key] = unique_index
            unique_named_tensors.append((name, tensor))
        entries.append((name, unique_index))

    return unique_named_tensors, entries


def build_oft_flattened_payload(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> PeftPayload:
    """Dedupe by storage, build flat tensor + metadata.

    The flat tensor stays on GPU. The serialization (CUDA IPC for IPC backend,
    NCCL broadcast for NCCL backend) is the backend's responsibility — this
    helper only shapes the payload.
    """
    unique_named_tensors, entries = _dedupe_named_tensors_by_storage(named_tensors)
    flattened_bucket = FlattenedTensorBucket(named_tensors=unique_named_tensors)
    flat_tensor = flattened_bucket.get_flattened_tensor().detach()
    return PeftPayload(
        flat_tensor=flat_tensor,
        metadata=flattened_bucket.get_metadata(),
        extra={"entries": entries},
    )
