"""MergeStrategy interface + registry. Seam for OFT now, non-OFT (procrustes) later."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

# state dict: parameter name -> tensor
StateDict = dict[str, torch.Tensor]


class MergeStrategy(ABC):
    name: str = ""

    @abstractmethod
    def merge(
        self,
        adapters: list[StateDict],
        weights: list[float] | None = None,
    ) -> StateDict:
        """Merge N adapter state dicts into one."""


_REGISTRY: dict[str, MergeStrategy] = {}


def register(strategy: MergeStrategy) -> MergeStrategy:
    if not strategy.name:
        raise ValueError("strategy.name must be set")
    _REGISTRY[strategy.name] = strategy
    return strategy


def get_strategy(name: str) -> MergeStrategy:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown merge method {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]
