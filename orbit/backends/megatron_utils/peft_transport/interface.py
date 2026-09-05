from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from ray import ObjectRef
from ray.actor import ActorHandle


@dataclass(frozen=True)
class PeftPayload:
    """Output of a method-specific payload_shaper. Carries the shaped flat
    tensor (still on GPU) plus any metadata the engine needs to interpret it."""

    flat_tensor: torch.Tensor
    metadata: dict[str, Any]
    extra: dict[str, Any]  # OFT dedupe entries, etc.


@dataclass(frozen=True)
class PeftSendResult:
    """Result of dispatching adapter weights.

    ``results`` is populated when a transport must wait internally to preserve
    its own synchronization semantics. When it is None, the caller owns waiting
    on ``refs``.
    """

    refs: list[ObjectRef]
    results: list[Any] | None = None


class PeftWeightTransport(ABC):
    """Sends PEFT adapter weights from trainer rank-0 to all rollout engines.

    Lifecycle: build_peft_transport(...) -> connect(...) -> send_adapter(...) [* N]
    -> disconnect()."""

    @abstractmethod
    def connect(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
    ) -> None: ...

    @abstractmethod
    def send_adapter(
        self,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        weight_version: int,
    ) -> PeftSendResult: ...

    @abstractmethod
    def disconnect(self) -> None: ...
