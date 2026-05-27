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
    name: str                                                    # "lora" | "oft"
    sglang_load_format: str                                      # "lora_adapter" | "oft_adapter"
    weight_name_predicate: Callable[[str], bool]
    dedupe_by_storage: bool
    payload_shaper: Callable[[list], "PeftPayload"] | None
    sample_names: str  # for diagnostics — e.g., "lora_A/lora_B"
    label: str         # for diagnostics — "LoRA" | "OFT"


def _build_oft_payload_shaper():
    # Late import — _payload imports sglang.srt.oft.streamed_weight_loader, which
    # is an optional heavy dependency that may not be present at registry-load time.
    from ._payload import build_oft_flattened_payload
    return build_oft_flattened_payload


PEFT_METHODS: dict[str, PeftMethodSpec] = {
    "lora": PeftMethodSpec(
        name="lora",
        sglang_load_format="lora_adapter",
        weight_name_predicate=is_lora_weight_name,
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
        payload_shaper=_build_oft_payload_shaper(),
        sample_names="oft_r/oft_R",
        label="OFT",
    ),
}
