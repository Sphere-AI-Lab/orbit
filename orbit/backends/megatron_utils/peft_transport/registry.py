from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from orbit.backends.megatron_utils.lora_utils import is_lora_weight_name
from orbit.backends.megatron_utils.oft_utils import is_oft_weight_name

if TYPE_CHECKING:
    from .interface import PeftPayload


@dataclass(frozen=True)
class PeftMethodSpec:
    """Static per-method metadata used by PeftWeightTransport implementations."""

    name: str  # "lora" | "oft"
    sglang_load_format: str  # "lora_adapter" | "oft_adapter"
    weight_name_predicate: Callable[[str], bool]
    dedupe_by_storage: bool
    payload_shaper: Callable[[list], PeftPayload] | None
    sample_names: str  # for diagnostics — e.g., "lora_A/lora_B"
    label: str  # for diagnostics — "LoRA" | "OFT"


def _oft_payload_shaper(named_tensors: list) -> PeftPayload:
    """Shape an OFT payload, importing the sglang helper on first call.

    The import is deferred because ``_payload`` pulls in
    ``sglang.srt.oft.streamed_weight_loader``, an optional heavy dependency that
    need not be importable when this registry is loaded. Resolving the shaper
    eagerly defeated that: every consumer of ``PEFT_METHODS`` — LoRA included —
    then died at import time whenever the module was absent or had moved.
    """
    from ._payload import build_oft_flattened_payload

    return build_oft_flattened_payload(named_tensors)


PEFT_METHODS: dict[str, PeftMethodSpec] = {
    "lora": PeftMethodSpec(
        name="lora",
        sglang_load_format="lora_adapter",
        weight_name_predicate=is_lora_weight_name,
        # No payload_shaper: LoRA loads through sglang's native per-tensor
        # adapter interface (load_lora_adapter_from_ray_tensors ->
        # load_lora_adapter_from_tensors), the shaper-less branch every backend
        # in this package already implements.
        #
        # The flattened-payload path is OFT-only by construction. sglang ships
        # serialize_flattened_oft_payload / normalize_oft_weight_payload and has
        # no LoRA counterpart, so shaping LoRA sent a "flattened_lora_payload"
        # tag that nothing on the server side accepts.
        #
        # Upstream radixark/miles has no flattened-LoRA machinery either, but it
        # does not use this entrypoint: it syncs adapters as "<lora_name>:<hf_name>"
        # named tensors through the generic update_weights_from_tensor protocol
        # (update_weight/hf_weight_iterator.py). This fork keeps a dedicated
        # adapter transport, so it drives sglang's adapter entrypoint instead —
        # per-tensor in both designs, never flattened.
        dedupe_by_storage=False,
        payload_shaper=None,
        sample_names="lora_A/lora_B",
        label="LoRA",
    ),
    "oft": PeftMethodSpec(
        name="oft",
        sglang_load_format="oft_adapter",
        weight_name_predicate=is_oft_weight_name,
        dedupe_by_storage=True,
        payload_shaper=_oft_payload_shaper,
        sample_names="oft_r/oft_R",
        label="OFT",
    ),
}
