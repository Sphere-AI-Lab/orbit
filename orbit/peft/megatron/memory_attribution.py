from __future__ import annotations

import os
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

import torch

from orbit.backends.megatron_utils.update_weight.common import is_named_adapter_tensor


DEFAULT_BUCKET_NAMES = (
    "base_param",
    "adapter_param",
    "trainable_non_adapter_param",
    "base_buffer",
    "adapter_buffer",
    "base_grad",
    "adapter_grad",
    "trainable_non_adapter_grad",
    "base_optimizer_state",
    "adapter_optimizer_state",
    "trainable_non_adapter_optimizer_state",
    "unknown_optimizer_state",
    "rollout_tensor",
)


@dataclass
class MemoryBucket:
    cpu_bytes: int = 0
    cuda_bytes: int = 0
    other_bytes: int = 0
    meta_logical_bytes: int = 0
    tensor_count: int = 0

    @property
    def total_bytes(self) -> int:
        return self.cpu_bytes + self.cuda_bytes + self.other_bytes + self.meta_logical_bytes

    def add_tensor(self, tensor: torch.Tensor) -> None:
        nbytes = _tensor_nbytes(tensor)
        device_type = tensor.device.type
        if device_type == "cuda":
            self.cuda_bytes += nbytes
        elif device_type == "cpu":
            self.cpu_bytes += nbytes
        elif device_type == "meta":
            self.meta_logical_bytes += nbytes
        else:
            self.other_bytes += nbytes
        self.tensor_count += 1


@dataclass
class TensorMemoryRecord:
    bucket: str
    name: str
    nbytes: int
    device: str
    dtype: str
    shape: tuple[int, ...]


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _as_model_sequence(model: torch.nn.Module | Sequence[torch.nn.Module]) -> Sequence[torch.nn.Module]:
    if isinstance(model, torch.nn.Module):
        return [model]
    return model


def _empty_buckets() -> dict[str, MemoryBucket]:
    return {name: MemoryBucket() for name in DEFAULT_BUCKET_NAMES}


def _parameter_bucket(name: str, param: torch.nn.Parameter) -> str:
    if is_named_adapter_tensor(name):
        return "adapter_param"
    if param.requires_grad:
        return "trainable_non_adapter_param"
    return "base_param"


def _grad_bucket(param_bucket: str) -> str:
    if param_bucket == "adapter_param":
        return "adapter_grad"
    if param_bucket == "trainable_non_adapter_param":
        return "trainable_non_adapter_grad"
    return "base_grad"


def _optimizer_bucket(param_bucket: str | None) -> str:
    if param_bucket == "adapter_param":
        return "adapter_optimizer_state"
    if param_bucket == "trainable_non_adapter_param":
        return "trainable_non_adapter_optimizer_state"
    if param_bucket == "base_param":
        return "base_optimizer_state"
    return "unknown_optimizer_state"


def _buffer_bucket(name: str) -> str:
    if is_named_adapter_tensor(name):
        return "adapter_buffer"
    return "base_buffer"


def _add_tensor_once(
    buckets: dict[str, MemoryBucket],
    bucket_name: str,
    tensor: torch.Tensor,
    seen: set[int],
    *,
    records: list[TensorMemoryRecord] | None = None,
    name: str | None = None,
) -> None:
    tensor_id = id(tensor)
    if tensor_id in seen:
        return
    seen.add(tensor_id)
    buckets.setdefault(bucket_name, MemoryBucket()).add_tensor(tensor)
    if records is not None:
        records.append(
            TensorMemoryRecord(
                bucket=bucket_name,
                name=name or "<unnamed>",
                nbytes=_tensor_nbytes(tensor),
                device=str(tensor.device),
                dtype=str(tensor.dtype),
                shape=tuple(int(dim) for dim in tensor.shape),
            )
        )


def _iter_inner_optimizers(optimizer: Any) -> Iterable[Any]:
    seen: set[int] = set()
    stack = [optimizer]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        children = []
        if hasattr(current, "chained_optimizers"):
            children.extend(getattr(current, "chained_optimizers") or [])
        if hasattr(current, "optimizers"):
            children.extend(getattr(current, "optimizers") or [])
        wrapped = getattr(current, "optimizer", None)
        if wrapped is not None and wrapped is not current:
            children.append(wrapped)

        if children:
            stack.extend(children)
            continue

        yield current


def _iter_nested_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_nested_tensors(nested)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _iter_nested_tensors(nested)


def _iter_nested_tensors_with_path(value: Any, path: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield path or "<tensor>", value
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _iter_nested_tensors_with_path(nested, child_path)
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            yield from _iter_nested_tensors_with_path(nested, child_path)
        return
    if isinstance(value, set):
        for index, nested in enumerate(value):
            child_path = f"{path}{{{index}}}" if path else f"{{{index}}}"
            yield from _iter_nested_tensors_with_path(nested, child_path)


def _collect_memory_buckets_and_records(
    model: torch.nn.Module | Sequence[torch.nn.Module],
    *,
    optimizer: Any | None = None,
    rollout_data: Any | None = None,
    collect_records: bool = False,
) -> tuple[dict[str, MemoryBucket], list[TensorMemoryRecord]]:
    """Collect semantic tensor memory buckets for Megatron actor diagnostics.

    This accounts for tensors reachable from model modules, optimizer state,
    and optional rollout data. It intentionally does not claim to account for
    transient activations, CUDA/NCCL workspaces, allocator fragmentation, or
    non-PyTorch CUDA allocations; those show up as allocator deltas around
    these known buckets.
    """
    buckets = _empty_buckets()
    param_bucket_by_id: dict[int, str] = {}
    param_name_by_id: dict[int, str] = {}
    seen_model_tensors: set[int] = set()
    records: list[TensorMemoryRecord] = [] if collect_records else None

    for model_chunk in _as_model_sequence(model):
        for name, param in model_chunk.named_parameters():
            bucket_name = _parameter_bucket(name, param)
            param_bucket_by_id[id(param)] = bucket_name
            param_name_by_id[id(param)] = name
            _add_tensor_once(buckets, bucket_name, param, seen_model_tensors, records=records, name=name)
            if param.grad is not None:
                _add_tensor_once(
                    buckets,
                    _grad_bucket(bucket_name),
                    param.grad,
                    seen_model_tensors,
                    records=records,
                    name=f"{name}.grad",
                )

        for name, buffer in model_chunk.named_buffers():
            _add_tensor_once(
                buckets,
                _buffer_bucket(name),
                buffer,
                seen_model_tensors,
                records=records,
                name=name,
            )

    if optimizer is not None:
        seen_optimizer_tensors: set[int] = set()
        for inner_optimizer in _iter_inner_optimizers(optimizer):
            state = getattr(inner_optimizer, "state", None)
            if not isinstance(state, Mapping):
                continue
            for param_or_key, param_state in state.items():
                param_bucket = param_bucket_by_id.get(id(param_or_key))
                bucket_name = _optimizer_bucket(param_bucket)
                param_name = param_name_by_id.get(id(param_or_key), repr(param_or_key))
                optimizer_name = type(inner_optimizer).__name__
                for tensor_path, tensor in _iter_nested_tensors_with_path(param_state):
                    _add_tensor_once(
                        buckets,
                        bucket_name,
                        tensor,
                        seen_optimizer_tensors,
                        records=records,
                        name=f"{optimizer_name}.state[{param_name}].{tensor_path}",
                    )

    if rollout_data is not None:
        seen_rollout_tensors: set[int] = set()
        for tensor_path, tensor in _iter_nested_tensors_with_path(rollout_data):
            _add_tensor_once(
                buckets,
                "rollout_tensor",
                tensor,
                seen_rollout_tensors,
                records=records,
                name=tensor_path,
            )

    return buckets, records or []


def collect_memory_buckets(
    model: torch.nn.Module | Sequence[torch.nn.Module],
    *,
    optimizer: Any | None = None,
    rollout_data: Any | None = None,
) -> dict[str, MemoryBucket]:
    buckets, _ = _collect_memory_buckets_and_records(model, optimizer=optimizer, rollout_data=rollout_data)

    return buckets


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.3f}GiB"


def format_memory_bucket_line(label: str, buckets: Mapping[str, MemoryBucket], *, rank: int | None = None) -> str:
    rank_value = "unknown" if rank is None else str(rank)
    parts = [f"[mem-bucket] label={label} rank={rank_value}"]
    for name in sorted(buckets):
        bucket = buckets[name]
        if bucket.total_bytes == 0 and bucket.tensor_count == 0:
            continue
        parts.append(
            f"{name}={_format_bytes(bucket.total_bytes)}"
            f"(cuda={_format_bytes(bucket.cuda_bytes)},cpu={_format_bytes(bucket.cpu_bytes)},"
            f"other={_format_bytes(bucket.other_bytes)},meta={_format_bytes(bucket.meta_logical_bytes)},"
            f"tensors={bucket.tensor_count})"
        )
    return " ".join(parts)


def _bucket_payload(bucket: MemoryBucket) -> dict[str, int]:
    return {
        "cpu_bytes": bucket.cpu_bytes,
        "cuda_bytes": bucket.cuda_bytes,
        "other_bytes": bucket.other_bytes,
        "meta_logical_bytes": bucket.meta_logical_bytes,
        "total_bytes": bucket.total_bytes,
        "tensor_count": bucket.tensor_count,
    }


def _top_tensor_payload(records: Sequence[TensorMemoryRecord], limit: int) -> dict[str, list[dict[str, Any]]]:
    if limit <= 0:
        return {}
    by_bucket: dict[str, list[TensorMemoryRecord]] = {}
    for record in records:
        by_bucket.setdefault(record.bucket, []).append(record)
    payload = {}
    for bucket_name, bucket_records in sorted(by_bucket.items()):
        top_records = sorted(bucket_records, key=lambda record: record.nbytes, reverse=True)[:limit]
        if not top_records:
            continue
        payload[bucket_name] = [
            {
                "name": record.name,
                "bytes": record.nbytes,
                "device": record.device,
                "dtype": record.dtype,
                "shape": list(record.shape),
            }
            for record in top_records
        ]
    return payload


def _top_tensor_limit() -> int:
    return 20


def _cuda_allocator_payload() -> dict[str, int | str] | None:
    try:
        if not torch.cuda.is_available():
            return None
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        return {
            "device": int(device),
            "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "mem_get_info_free_bytes": int(free),
            "mem_get_info_total_bytes": int(total),
        }
    except Exception as exc:  # pragma: no cover - defensive for OOM paths
        return {"error": repr(exc)}


def _safe_memory_summary() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.memory_summary()
    except Exception as exc:  # pragma: no cover - defensive for OOM paths
        return f"torch.cuda.memory_summary() failed: {exc!r}"


def dump_memory_diagnostics(
    output_dir: str | os.PathLike[str],
    label: str,
    model: torch.nn.Module | Sequence[torch.nn.Module],
    *,
    optimizer: Any | None = None,
    rollout_data: Any | None = None,
    rank: int | None = None,
    include_memory_summary: bool = True,
) -> Path:
    resolved_rank = _current_rank() if rank is None else rank
    buckets, tensor_records = _collect_memory_buckets_and_records(
        model,
        optimizer=optimizer,
        rollout_data=rollout_data,
        collect_records=True,
    )
    top_tensor_limit = _top_tensor_limit()
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    dump_path = path / f"mem_diag_rank{resolved_rank}_{label}.json"
    payload = {
        "label": label,
        "rank": resolved_rank,
        "pid": os.getpid(),
        "time": time(),
        "buckets": {name: _bucket_payload(bucket) for name, bucket in sorted(buckets.items())},
        "top_tensors": _top_tensor_payload(tensor_records, top_tensor_limit),
        "cuda_allocator": _cuda_allocator_payload(),
    }
    if include_memory_summary:
        payload["memory_summary"] = _safe_memory_summary()
    dump_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return dump_path


def _current_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0
