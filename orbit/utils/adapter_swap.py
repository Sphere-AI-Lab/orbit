"""Value-swap of adapter tensors for teacher-forcing forwards.

Swaps VALUES, not modules: optimizer state, grad buffers, and the weight-sync
registry all key on parameter object identity, which must stay stable. The
adapter-param predicate is injected so this module needs no megatron import
(CPU unit tests use a toy predicate).
"""

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager

import torch

from orbit.utils.adapter_tensors import AdapterTensorKey, adapter_named_parameters


@contextmanager
def swap_adapter_tensors(
    model: Sequence[torch.nn.Module],
    teacher_tensors: Mapping[AdapterTensorKey, torch.Tensor],
    is_adapter_name: Callable[[str], bool],
):
    params = adapter_named_parameters(model, is_adapter_name)

    missing = sorted(set(params) - set(teacher_tensors))
    if missing:
        raise ValueError(f"Teacher tensors missing for adapter params: {missing[:5]} (+{max(0, len(missing) - 5)} more)")
    extra = sorted(set(teacher_tensors) - set(params))
    if extra:
        raise ValueError(f"Teacher tensors reference unknown adapter params: {extra[:5]} (+{max(0, len(extra) - 5)} more)")

    prepared: dict[AdapterTensorKey, torch.Tensor] = {}
    for key, param in params.items():
        source = teacher_tensors[key]
        if not isinstance(source, torch.Tensor):
            raise TypeError(f"Teacher adapter tensor {key!r} is not a tensor")
        if source.shape != param.shape:
            raise ValueError(
                f"Teacher adapter tensor {key!r} shape {tuple(source.shape)} "
                f"does not match parameter shape {tuple(param.shape)}"
            )
        prepared[key] = source.detach().to(device=param.device, dtype=param.dtype)

    stash = {key: param.detach().clone() for key, param in params.items()}
    try:
        with torch.no_grad():
            for key, param in params.items():
                param.copy_(prepared[key])
        yield
    finally:
        with torch.no_grad():
            for key, param in params.items():
                param.copy_(stash[key])
