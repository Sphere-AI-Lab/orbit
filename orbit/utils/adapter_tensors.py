"""Canonical identity for adapter tensors across virtual pipeline chunks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence

import torch


type AdapterTensorKey = tuple[int, str]


def _validate_adapter_tensor_key(key: object) -> AdapterTensorKey:
    if (
        type(key) is not tuple
        or len(key) != 2
        or type(key[0]) is not int
        or key[0] < 0
        or type(key[1]) is not str
        or not key[1].strip()
    ):
        raise ValueError("adapter tensor key is invalid")
    return key


def adapter_named_parameters(
    model: Sequence[torch.nn.Module],
    is_adapter_name: Callable[[str], bool],
) -> dict[AdapterTensorKey, torch.nn.Parameter]:
    """Enumerate adapter parameters without collapsing VPP chunk identity."""

    if not isinstance(model, Sequence) or isinstance(model, (str, bytes)) or not model:
        raise TypeError("model must be a nonempty module sequence")
    if not callable(is_adapter_name):
        raise TypeError("is_adapter_name must be callable")

    chunks = tuple(model)
    if len(chunks) != len(model) or any(model[index] is not chunk for index, chunk in enumerate(chunks)):
        raise ValueError("model chunk sequence must have deterministic index order")

    params: dict[AdapterTensorKey, torch.nn.Parameter] = {}
    chunk_ids: set[int] = set()
    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, torch.nn.Module) or id(chunk) in chunk_ids:
            raise ValueError("model chunks must be distinct torch modules")
        chunk_ids.add(id(chunk))
        for local_name, parameter in chunk.named_parameters():
            if type(local_name) is not str or not local_name.strip():
                raise ValueError("adapter parameter local name is invalid")
            if not is_adapter_name(local_name):
                continue
            if not isinstance(parameter, torch.nn.Parameter):
                raise ValueError("selected adapter tensor is not a parameter")
            key = (chunk_index, local_name)
            if key in params:
                raise ValueError(f"duplicate adapter tensor key {key!r}")
            params[key] = parameter

    if not params:
        raise ValueError("model contains no adapter parameters")
    return params


def adapter_tensor_key_digest(keys: Iterable[AdapterTensorKey]) -> str:
    """Hash a canonical sorted JSON encoding of chunk-aware tensor keys."""

    if not isinstance(keys, Iterable) or isinstance(keys, (str, bytes)):
        raise TypeError("adapter tensor keys must be an iterable")
    normalized = [_validate_adapter_tensor_key(key) for key in keys]
    if not normalized:
        raise ValueError("adapter tensor keys must be nonempty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("adapter tensor keys must be unique")
    encoded = json.dumps(
        sorted([chunk_index, local_name] for chunk_index, local_name in normalized),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
