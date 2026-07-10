"""Value-swap of adapter tensors for teacher-forcing forwards.

Swaps VALUES, not modules: optimizer state, grad buffers, and the weight-sync
registry all key on parameter object identity, which must stay stable. The
adapter-param predicate is injected so this module needs no megatron import
(CPU unit tests use a toy predicate).
"""

from collections.abc import Callable, Sequence
from contextlib import contextmanager

import torch


@contextmanager
def swap_adapter_tensors(
    model: Sequence[torch.nn.Module],
    teacher_tensors: dict[str, torch.Tensor],
    is_adapter_name: Callable[[str], bool],
):
    params: dict[str, torch.nn.Parameter] = {}
    for chunk in model:
        for name, param in chunk.named_parameters():
            if is_adapter_name(name):
                params[name] = param

    missing = sorted(set(params) - set(teacher_tensors))
    if missing:
        raise ValueError(f"Teacher tensors missing for adapter params: {missing[:5]} (+{max(0, len(missing) - 5)} more)")
    extra = sorted(set(teacher_tensors) - set(params))
    if extra:
        raise ValueError(f"Teacher tensors reference unknown adapter params: {extra[:5]} (+{max(0, len(extra) - 5)} more)")

    stash = {name: param.data.clone() for name, param in params.items()}
    try:
        with torch.no_grad():
            for name, param in params.items():
                param.data.copy_(teacher_tensors[name].to(device=param.device, dtype=param.dtype))
        yield
    finally:
        with torch.no_grad():
            for name, param in params.items():
                param.data.copy_(stash[name])
