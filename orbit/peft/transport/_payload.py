"""OFT-specific payload shaping. Moved verbatim from `_sync_peft_adapter`'s OFT
branch (update_weight_from_tensor.py:488-525), exposed as a payload_shaper hook.
"""
from __future__ import annotations

import torch

from orbit.backends.megatron_utils.sglang import FlattenedTensorBucket

from orbit.peft.transport.interface import PeftPayload


def build_oft_flattened_payload(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> PeftPayload:
    """Dedupe by storage, build flat tensor + metadata.

    The flat tensor stays on GPU. The serialization (CUDA IPC for IPC backend,
    NCCL broadcast for NCCL backend) is the backend's responsibility — this
    helper only shapes the payload.
    """
    from sglang.srt.peft.oft.streamed_weight_loader import (
        dedupe_named_tensors_by_storage,
    )

    unique_named_tensors, entries = dedupe_named_tensors_by_storage(named_tensors)
    flattened_bucket = FlattenedTensorBucket(named_tensors=unique_named_tensors)
    flat_tensor = flattened_bucket.get_flattened_tensor().detach()
    return PeftPayload(
        flat_tensor=flat_tensor,
        metadata=flattened_bucket.get_metadata(),
        extra={"entries": entries},
    )
