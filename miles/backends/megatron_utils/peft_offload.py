from __future__ import annotations

import gc
import logging
import os
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from miles.backends.megatron_utils.update_weight.common import is_named_adapter_tensor

logger = logging.getLogger(__name__)

_SourceKey = tuple[object, ...]


@dataclass
class _FlatEntry:
    name: str
    nelem: int
    shape: torch.Size
    source_key: _SourceKey
    offset: int = 0
    param: torch.nn.Parameter | None = None
    parent: torch.nn.Module | None = None
    buffer_name: str | None = None
    grouped_module: torch.nn.Module | None = None
    grouped_tensor_name: str | None = None

    def get(self) -> torch.Tensor:
        if self.grouped_module is not None:
            assert self.grouped_tensor_name is not None
            getter = self.grouped_module.grouped_fp4_expert_tensors
            return getter()[self.grouped_tensor_name]
        if self.param is not None:
            return self.param.data
        assert self.parent is not None and self.buffer_name is not None
        return self.parent._buffers[self.buffer_name]

    def set(self, tensor: torch.Tensor) -> None:
        if self.grouped_module is not None:
            assert self.grouped_tensor_name is not None
            _set_grouped_fp4_expert_tensor(self.grouped_module, self.grouped_tensor_name, tensor)
            return
        if self.param is not None:
            self.param.data = tensor
            return
        assert self.parent is not None and self.buffer_name is not None
        self.parent._buffers[self.buffer_name] = tensor


@dataclass
class _FlatGroup:
    dtype: torch.dtype
    entries: list[_FlatEntry]
    cpu_flat: torch.Tensor
    total_nelem: int
    device_flat: torch.Tensor | None = None


_FLAT_GROUPS: dict[int, dict[torch.dtype, _FlatGroup]] = {}
_ADAPTER_FLAT_GROUPS: dict[int, dict[torch.dtype, _FlatGroup]] = {}
_PIN_FALLBACK_WARNED: set[tuple[int, torch.dtype]] = set()
_FLAT_ENTRY_ALIGNMENT_BYTES = 256


def _root_module(model_chunk: torch.nn.Module) -> torch.nn.Module:
    return getattr(model_chunk, "module", model_chunk)


def _is_real_tensor(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and tensor.device.type != "meta"


def _iter_model_roots(models: Sequence[torch.nn.Module]) -> Iterable[torch.nn.Module]:
    for model_chunk in models:
        yield _root_module(model_chunk)


def _resolve_load_device(device: torch.device | str | int | None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    if isinstance(device, int):
        return torch.device("cuda", device)

    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())

    return resolved


def _empty_cuda_cache_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def offload_megatron_grad_buffers(model: Sequence[torch.nn.Module]) -> None:
    for model_chunk in model:
        offload_grad_buffers = getattr(model_chunk, "offload_grad_buffers", None)
        if offload_grad_buffers is not None:
            offload_grad_buffers()
    gc.collect()
    _empty_cuda_cache_if_available()


def load_megatron_grad_buffers(model: Sequence[torch.nn.Module]) -> None:
    for model_chunk in model:
        restore_grad_buffers = getattr(model_chunk, "restore_grad_buffers", None)
        if restore_grad_buffers is not None:
            restore_grad_buffers()
    gc.collect()
    _empty_cuda_cache_if_available()


def offload_megatron_optimizer(optimizer) -> None:
    offload_to_cpu = getattr(optimizer, "offload_to_cpu", None)
    if offload_to_cpu is not None:
        offload_to_cpu()
    gc.collect()
    _empty_cuda_cache_if_available()


def load_megatron_optimizer(optimizer) -> None:
    load_from_cpu = getattr(optimizer, "restore_from_cpu", None)
    if load_from_cpu is not None:
        load_from_cpu()
    gc.collect()
    _empty_cuda_cache_if_available()


def _named_buffers_with_duplicates(root: torch.nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    try:
        yield from root.named_buffers(remove_duplicate=False)
        return
    except TypeError:
        pass

    yield from _named_buffers_with_duplicates_fallback(root)


def _named_buffers_with_duplicates_fallback(
    module: torch.nn.Module,
    prefix: str = "",
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, buffer in module._buffers.items():
        if buffer is not None:
            yield prefix + name, buffer

    for name, child in module._modules.items():
        if child is None:
            continue
        yield from _named_buffers_with_duplicates_fallback(child, prefix + name + ".")


def _parent_module_for_buffer(root: torch.nn.Module, buffer_name: str) -> tuple[torch.nn.Module, str]:
    if "." not in buffer_name:
        return root, buffer_name

    parent_name, local_name = buffer_name.rsplit(".", 1)
    return root.get_submodule(parent_name), local_name


def _rebind_buffer(root: torch.nn.Module, buffer_name: str, buffer: torch.Tensor) -> None:
    parent, local_name = _parent_module_for_buffer(root, buffer_name)
    parent._buffers[local_name] = buffer


def _iter_frozen_named_params(root: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Parameter]]:
    for name, param in root.named_parameters():
        if is_named_adapter_tensor(name) or param.requires_grad:
            continue
        if not _is_real_tensor(param):
            continue
        yield name, param


def _iter_base_named_buffers(root: torch.nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    for name, buffer in _named_buffers_with_duplicates(root):
        if is_named_adapter_tensor(name):
            continue
        if not _is_real_tensor(buffer):
            continue
        yield name, buffer


def _iter_adapter_named_params(root: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Parameter]]:
    for name, param in root.named_parameters():
        if not is_named_adapter_tensor(name):
            continue
        if not _is_real_tensor(param):
            continue
        yield name, param


def _move_trainable_adapter_param_buffers(
    model_chunk: torch.nn.Module,
    root: torch.nn.Module,
    method_name: str,
    groups: dict[torch.dtype, _FlatGroup] | None,
) -> set[int]:
    """Move trainable adapter params through Megatron's own param buffers.

    Megatron DDP maps trainable parameters into param buffers before the
    distributed optimizer is created. Rebinding those Parameter.data tensors in
    the flat adapter mirror can disconnect optimizer and grad-norm views. When
    Megatron param buffers are available, keep trainable adapter params out of
    the flat mirror and move their backing storage through Megatron's buffer
    API instead.
    """
    trainable_adapter_ids = {id(param) for _, param in _iter_adapter_named_params(root) if param.requires_grad}
    if not trainable_adapter_ids:
        return set()

    param_buffers = []
    for attr_name in ("buffers", "expert_parallel_buffers"):
        buffers = getattr(model_chunk, attr_name, None)
        if buffers is None or callable(buffers):
            continue
        try:
            iterator = iter(buffers)
        except TypeError:
            continue
        for buffer in iterator:
            if hasattr(buffer, "params"):
                param_buffers.append(buffer)

    if not param_buffers:
        return set()

    def exclude_trainable_adapters_from_flat_groups() -> set[int]:
        if groups is not None:
            for group in groups.values():
                group.entries = [
                    entry
                    for entry in group.entries
                    if not (entry.param is not None and id(entry.param) in trainable_adapter_ids)
                ]
        return trainable_adapter_ids

    names_by_id = {id(param): name for name, param in root.named_parameters()}
    selected_buffers = []
    covered_adapter_ids: set[int] = set()

    for buffer in param_buffers:
        try:
            buffer_params = list(getattr(buffer, "params", ()))
        except TypeError:
            continue
        for param in buffer_params:
            if param.requires_grad:
                continue
            param_name = names_by_id.get(id(param), "<unnamed>")
            raise RuntimeError(
                "Megatron param buffer contains non-trainable parameter "
                f"{param_name}; OFFLOAD_TRAIN_ADAPTER assumes Megatron param buffers "
                "contain trainable parameters only."
            )

        buffer_param_ids = {id(param) for param in buffer_params}
        if not buffer_param_ids & trainable_adapter_ids:
            continue

        for param in buffer_params:
            param_name = names_by_id.get(id(param), "<unnamed>")
            if not is_named_adapter_tensor(param_name):
                raise RuntimeError(
                    "Megatron param buffer selected for adapter offload also contains "
                    f"trainable non-adapter parameter {param_name}; OFFLOAD_TRAIN_ADAPTER "
                    "expects adapter trainable params to be isolated in Megatron param buffers."
                )

        selected_buffers.append(buffer)
        covered_adapter_ids.update(buffer_param_ids & trainable_adapter_ids)

    if covered_adapter_ids != trainable_adapter_ids:
        missing_names = sorted(
            names_by_id.get(param_id, "<unnamed>") for param_id in trainable_adapter_ids - covered_adapter_ids
        )
        raise RuntimeError(
            "Megatron param buffers do not cover all trainable adapter params for "
            f"chunk {id(model_chunk)}; missing={missing_names}."
        )

    for buffer in selected_buffers:
        method = getattr(buffer, method_name, None)
        if callable(method):
            continue
        raise RuntimeError(
            f"Megatron param buffer on chunk {id(model_chunk)} has no {method_name}(); "
            "cannot move trainable adapter params safely."
        )

    for buffer in selected_buffers:
        getattr(buffer, method_name)(move_params=True, move_grads=False)

    return exclude_trainable_adapters_from_flat_groups()


def _iter_adapter_named_buffers(root: torch.nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    for name, buffer in _named_buffers_with_duplicates(root):
        if not is_named_adapter_tensor(name):
            continue
        if not _is_real_tensor(buffer):
            continue
        yield name, buffer


def _is_grouped_fp4_expert_module(module: torch.nn.Module) -> bool:
    has_setter = callable(getattr(module, "set_grouped_fp4_expert_tensor", None)) or callable(
        getattr(module, "set_grouped_fp4_expert_tensors", None)
    )
    return callable(getattr(module, "grouped_fp4_expert_tensors", None)) and has_setter


def _iter_grouped_fp4_expert_modules(root: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Module]]:
    for module_name, module in root.named_modules():
        if _is_grouped_fp4_expert_module(module):
            yield module_name, module


def _set_grouped_fp4_expert_tensor(
    module: torch.nn.Module,
    tensor_name: str,
    tensor: torch.Tensor,
) -> None:
    setter = getattr(module, "set_grouped_fp4_expert_tensor", None)
    if callable(setter):
        setter(tensor_name, tensor)
        return

    tensors = dict(module.grouped_fp4_expert_tensors())
    tensors[tensor_name] = tensor
    module.set_grouped_fp4_expert_tensors(tensors)


def _set_grouped_fp4_expert_tensors(
    module: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
) -> None:
    setter = getattr(module, "set_grouped_fp4_expert_tensors", None)
    if callable(setter):
        setter(tensors)
        return

    for tensor_name, tensor in tensors.items():
        module.set_grouped_fp4_expert_tensor(tensor_name, tensor)


def _is_grouped_fp4_expert_param(name: str, module_prefixes: set[str]) -> bool:
    for prefix in module_prefixes:
        if prefix:
            if not name.startswith(prefix + "."):
                continue
            local_name = name[len(prefix) + 1 :]
        else:
            local_name = name

        parts = local_name.split(".")
        if (
            len(parts) == 4
            and parts[0] == "experts"
            and parts[2] in {"w1", "w2", "w3"}
            and parts[3] in {"weight", "scale"}
        ):
            return True

    return False


def _grouped_fp4_expert_entry(
    module_name: str,
    module: torch.nn.Module,
    tensor_name: str,
    tensor: torch.Tensor,
) -> _FlatEntry:
    entry_name = (
        f"{module_name}.grouped_fp4_expert_tensors.{tensor_name}"
        if module_name
        else f"grouped_fp4_expert_tensors.{tensor_name}"
    )
    return _FlatEntry(
        name=entry_name,
        nelem=tensor.numel(),
        shape=tensor.shape,
        source_key=("grouped_fp4_expert_tensor", id(module), tensor_name),
        grouped_module=module,
        grouped_tensor_name=tensor_name,
    )


def _buffer_entry(root: torch.nn.Module, name: str, buffer: torch.Tensor) -> _FlatEntry:
    parent, local_name = _parent_module_for_buffer(root, name)
    return _FlatEntry(
        name=name,
        nelem=buffer.numel(),
        shape=buffer.shape,
        source_key=("buffer", id(buffer)),
        parent=parent,
        buffer_name=local_name,
    )


def _param_entry(name: str, param: torch.nn.Parameter) -> _FlatEntry:
    return _FlatEntry(
        name=name,
        nelem=param.numel(),
        shape=param.shape,
        source_key=("param", id(param)),
        param=param,
    )


def _allocate_pinned(total_nelem: int, dtype: torch.dtype, chunk_id: int) -> torch.Tensor:
    # Diagnostic knob: MILES_PEFT_OFFLOAD_PIN=0 forces pageable host memory to
    # test whether torch's pinned-memory allocator is doubling our flat-buffer
    # RSS footprint (suspected cause of host OOM on DSV4-Pro EP=8).
    pin = os.environ.get("MILES_PEFT_OFFLOAD_PIN", "1") != "0"
    if not pin:
        return torch.empty(total_nelem, dtype=dtype, device="cpu")
    try:
        return torch.empty(total_nelem, dtype=dtype, device="cpu", pin_memory=True)
    except RuntimeError as exc:
        warning_key = (chunk_id, dtype)
        if warning_key not in _PIN_FALLBACK_WARNED:
            logger.warning(
                "pin_memory failed for chunk %s dtype %s (%s); using pageable host buffer",
                chunk_id,
                dtype,
                exc,
            )
            _PIN_FALLBACK_WARNED.add(warning_key)
        return torch.empty(total_nelem, dtype=dtype, device="cpu")


def _alignment_in_elements(dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    return max(1, (_FLAT_ENTRY_ALIGNMENT_BYTES + element_size - 1) // element_size)


def _align_offset(offset: int, dtype: torch.dtype) -> int:
    alignment = _alignment_in_elements(dtype)
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + alignment - remainder


def _assign_flat_offsets(entries: list[_FlatEntry], dtype: torch.dtype) -> int:
    offset_by_source: dict[_SourceKey, int] = {}
    nelem_by_source: dict[_SourceKey, int] = {}
    total_nelem = 0

    for entry in entries:
        if entry.source_key in offset_by_source:
            if nelem_by_source[entry.source_key] != entry.nelem:
                raise RuntimeError(
                    f"Aliased buffer {entry.name!r} changed shape; cannot preserve alias in PEFT offload."
                )
            entry.offset = offset_by_source[entry.source_key]
            continue

        total_nelem = _align_offset(total_nelem, dtype)
        entry.offset = total_nelem
        offset_by_source[entry.source_key] = total_nelem
        nelem_by_source[entry.source_key] = entry.nelem
        total_nelem += entry.nelem

    return total_nelem


def _plan_groups(chunk_id: int, root: torch.nn.Module) -> dict[torch.dtype, _FlatGroup]:
    by_dtype: dict[torch.dtype, list[_FlatEntry]] = {}
    grouped_module_prefixes: set[str] = set()

    for module_name, module in _iter_grouped_fp4_expert_modules(root):
        grouped_module_prefixes.add(module_name)
        for tensor_name, tensor in module.grouped_fp4_expert_tensors().items():
            if not _is_real_tensor(tensor):
                continue
            by_dtype.setdefault(tensor.dtype, []).append(
                _grouped_fp4_expert_entry(module_name, module, tensor_name, tensor)
            )

    for name, param in _iter_frozen_named_params(root):
        if _is_grouped_fp4_expert_param(name, grouped_module_prefixes):
            continue
        by_dtype.setdefault(param.dtype, []).append(_param_entry(name, param))

    for name, buffer in _iter_base_named_buffers(root):
        by_dtype.setdefault(buffer.dtype, []).append(_buffer_entry(root, name, buffer))

    groups: dict[torch.dtype, _FlatGroup] = {}
    for dtype, entries in by_dtype.items():
        total_nelem = _assign_flat_offsets(entries, dtype)
        groups[dtype] = _FlatGroup(
            dtype=dtype,
            entries=entries,
            cpu_flat=_allocate_pinned(total_nelem, dtype, chunk_id),
            total_nelem=total_nelem,
        )

    return groups


def _plan_adapter_groups(
    chunk_id: int,
    root: torch.nn.Module,
    exclude_param_ids: set[int] | None = None,
) -> dict[torch.dtype, _FlatGroup]:
    by_dtype: dict[torch.dtype, list[_FlatEntry]] = {}
    exclude_param_ids = exclude_param_ids or set()

    for name, param in _iter_adapter_named_params(root):
        if id(param) in exclude_param_ids:
            continue
        by_dtype.setdefault(param.dtype, []).append(_param_entry(name, param))

    for name, buffer in _iter_adapter_named_buffers(root):
        by_dtype.setdefault(buffer.dtype, []).append(_buffer_entry(root, name, buffer))

    groups: dict[torch.dtype, _FlatGroup] = {}
    for dtype, entries in by_dtype.items():
        total_nelem = _assign_flat_offsets(entries, dtype)
        groups[dtype] = _FlatGroup(
            dtype=dtype,
            entries=entries,
            cpu_flat=_allocate_pinned(total_nelem, dtype, chunk_id),
            total_nelem=total_nelem,
        )

    return groups


def _slice_view(flat: torch.Tensor, entry: _FlatEntry) -> torch.Tensor:
    return flat[entry.offset : entry.offset + entry.nelem].view(entry.shape)


def _copy_entry_to_cpu_flat(group: _FlatGroup, entry: _FlatEntry) -> None:
    source = entry.get()
    target = _slice_view(group.cpu_flat, entry)
    target.copy_(source.detach().reshape(-1).view(entry.shape), non_blocking=source.device.type != "cpu")
    entry.set(target)


def _initial_offload_into_flat(group: _FlatGroup) -> None:
    copied_sources: set[_SourceKey] = set()
    for entry in group.entries:
        if entry.source_key not in copied_sources:
            _copy_entry_to_cpu_flat(group, entry)
            copied_sources.add(entry.source_key)
            continue
        entry.set(_slice_view(group.cpu_flat, entry))
    group.device_flat = None


def _bulk_offload_into_flat(group: _FlatGroup) -> None:
    assert group.device_flat is not None
    group.cpu_flat.copy_(group.device_flat, non_blocking=group.cpu_flat.is_pinned())
    group.device_flat = None
    for entry in group.entries:
        entry.set(_slice_view(group.cpu_flat, entry))


def _load_group_from_flat(group: _FlatGroup, device: torch.device) -> None:
    if device.type == "cpu":
        group.device_flat = None
        for entry in group.entries:
            entry.set(_slice_view(group.cpu_flat, entry))
        return

    if group.device_flat is None or group.device_flat.device != device:
        group.device_flat = torch.empty(group.total_nelem, dtype=group.dtype, device=device)
        group.device_flat.copy_(group.cpu_flat, non_blocking=group.cpu_flat.is_pinned())

    for entry in group.entries:
        entry.set(_slice_view(group.device_flat, entry))


def _move_grouped_fp4_expert_modules_to_device(
    root: torch.nn.Module,
    device: torch.device,
) -> tuple[set[str], int, int]:
    grouped_module_prefixes: set[str] = set()
    moved_tensors = 0
    skipped_meta = 0

    for module_name, module in _iter_grouped_fp4_expert_modules(root):
        grouped_module_prefixes.add(module_name)
        grouped_tensors = module.grouped_fp4_expert_tensors()
        moved_grouped_tensors: dict[str, torch.Tensor] = {}
        changed = False

        for tensor_name, tensor in grouped_tensors.items():
            if not _is_real_tensor(tensor):
                moved_grouped_tensors[tensor_name] = tensor
                skipped_meta += 1
                continue
            if tensor.device == device:
                moved_grouped_tensors[tensor_name] = tensor
                continue

            moved_grouped_tensors[tensor_name] = tensor.to(device=device, non_blocking=True)
            moved_tensors += 1
            changed = True

        if changed:
            _set_grouped_fp4_expert_tensors(module, moved_grouped_tensors)

    return grouped_module_prefixes, moved_tensors, skipped_meta


def _move_unplanned_root_to_device(root: torch.nn.Module, device: torch.device) -> tuple[int, int, int]:
    moved_params = 0
    moved_buffers = 0
    skipped_meta = 0
    moved_buffer_cache: dict[int, torch.Tensor] = {}
    grouped_module_prefixes, moved_grouped_tensors, skipped_grouped_meta = _move_grouped_fp4_expert_modules_to_device(
        root, device
    )
    moved_params += moved_grouped_tensors
    skipped_meta += skipped_grouped_meta

    for name, param in root.named_parameters():
        if is_named_adapter_tensor(name) or param.requires_grad:
            continue
        if _is_grouped_fp4_expert_param(name, grouped_module_prefixes):
            continue
        if not _is_real_tensor(param):
            skipped_meta += 1
            continue
        if param.device != device:
            param.data = param.data.to(device=device, non_blocking=True)
            moved_params += 1

    for name, buffer in _named_buffers_with_duplicates(root):
        if is_named_adapter_tensor(name):
            continue
        if not _is_real_tensor(buffer):
            skipped_meta += 1
            continue
        if buffer.device == device:
            continue

        moved_buffer = moved_buffer_cache.get(id(buffer))
        if moved_buffer is None:
            moved_buffer = buffer.to(device=device, non_blocking=True)
            moved_buffer_cache[id(buffer)] = moved_buffer
        _rebind_buffer(root, name, moved_buffer)
        moved_buffers += 1

    return moved_params, moved_buffers, skipped_meta


def _move_unplanned_adapter_root_to_device(
    root: torch.nn.Module,
    device: torch.device,
    *,
    exclude_param_ids: set[int] | None = None,
) -> tuple[int, int, int]:
    moved_params = 0
    moved_buffers = 0
    skipped_meta = 0
    moved_buffer_cache: dict[int, torch.Tensor] = {}
    exclude_param_ids = exclude_param_ids or set()

    for name, param in root.named_parameters():
        if not is_named_adapter_tensor(name):
            continue
        if id(param) in exclude_param_ids:
            continue
        if not _is_real_tensor(param):
            skipped_meta += 1
            continue
        if param.device != device:
            param.data = param.data.to(device=device, non_blocking=True)
            moved_params += 1

    for name, buffer in _named_buffers_with_duplicates(root):
        if not is_named_adapter_tensor(name):
            continue
        if not _is_real_tensor(buffer):
            skipped_meta += 1
            continue
        if buffer.device == device:
            continue

        moved_buffer = moved_buffer_cache.get(id(buffer))
        if moved_buffer is None:
            moved_buffer = buffer.to(device=device, non_blocking=True)
            moved_buffer_cache[id(buffer)] = moved_buffer
        _rebind_buffer(root, name, moved_buffer)
        moved_buffers += 1

    return moved_params, moved_buffers, skipped_meta


@torch.no_grad()
def offload_megatron_frozen_base_to_cpu(models: Sequence[torch.nn.Module]) -> None:
    for model_chunk in models:
        root = _root_module(model_chunk)
        chunk_id = id(model_chunk)
        groups = _FLAT_GROUPS.get(chunk_id)

        if groups is None:
            groups = _plan_groups(chunk_id, root)
            _FLAT_GROUPS[chunk_id] = groups
            for group in groups.values():
                _initial_offload_into_flat(group)
            continue

        for group in groups.values():
            if group.device_flat is None:
                _initial_offload_into_flat(group)
            else:
                _bulk_offload_into_flat(group)

    gc.collect()
    _empty_cuda_cache_if_available()


@torch.no_grad()
def offload_megatron_adapter_to_cpu(models: Sequence[torch.nn.Module]) -> None:
    for model_chunk in models:
        root = _root_module(model_chunk)
        chunk_id = id(model_chunk)
        groups = _ADAPTER_FLAT_GROUPS.get(chunk_id)

        if groups is None:
            megatron_param_ids = _move_trainable_adapter_param_buffers(
                model_chunk,
                root,
                "offload_to_cpu",
                None,
            )
            groups = _plan_adapter_groups(chunk_id, root, exclude_param_ids=megatron_param_ids)
            _ADAPTER_FLAT_GROUPS[chunk_id] = groups
            for group in groups.values():
                _initial_offload_into_flat(group)
            continue

        _move_trainable_adapter_param_buffers(model_chunk, root, "offload_to_cpu", groups)
        for group in groups.values():
            if group.device_flat is None:
                _initial_offload_into_flat(group)
            else:
                _bulk_offload_into_flat(group)

    gc.collect()
    _empty_cuda_cache_if_available()


@torch.no_grad()
def load_megatron_frozen_base_to_gpu(
    models: Sequence[torch.nn.Module],
    *,
    device: torch.device | str | int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Restore frozen-base params/buffers from CPU pinned mirrors back to ``device``.

    When ``stream`` is provided, all per-group H2D copies are issued on that
    CUDA stream. The caller owns the stream's lifetime and is responsible for
    recording an event on it after this function returns and arranging
    downstream consumers (e.g. training-step kernels) to ``wait_event`` on
    that event before touching the restored tensors. When ``stream`` is None
    the copies run on the current stream and this function is fully
    synchronous from the caller's point of view (no extra synchronization
    required).
    """

    target_device = _resolve_load_device(device)
    moved_params = 0
    moved_buffers = 0
    skipped_meta = 0

    stream_ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with stream_ctx:
        for model_chunk in models:
            root = _root_module(model_chunk)
            chunk_id = id(model_chunk)
            groups = _FLAT_GROUPS.get(chunk_id)

            if groups is None:
                params, buffers, meta = _move_unplanned_root_to_device(root, target_device)
                moved_params += params
                moved_buffers += buffers
                skipped_meta += meta
                continue

            for group in groups.values():
                _load_group_from_flat(group, target_device)

    if moved_params or moved_buffers or skipped_meta:
        logger.info(
            "Loaded %d frozen base parameters and %d non-adapter buffers to %s; skipped %d meta tensors",
            moved_params,
            moved_buffers,
            target_device,
            skipped_meta,
        )

    gc.collect()
    _empty_cuda_cache_if_available()


@torch.no_grad()
def load_megatron_adapter_to_gpu(
    models: Sequence[torch.nn.Module],
    *,
    device: torch.device | str | int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> None:
    target_device = _resolve_load_device(device)
    moved_params = 0
    moved_buffers = 0
    skipped_meta = 0

    stream_ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with stream_ctx:
        for model_chunk in models:
            root = _root_module(model_chunk)
            chunk_id = id(model_chunk)
            groups = _ADAPTER_FLAT_GROUPS.get(chunk_id)

            if groups is None:
                megatron_param_ids = _move_trainable_adapter_param_buffers(
                    model_chunk,
                    root,
                    "reload_from_cpu",
                    None,
                )
                params, buffers, meta = _move_unplanned_adapter_root_to_device(
                    root,
                    target_device,
                    exclude_param_ids=megatron_param_ids,
                )
                moved_params += params
                moved_buffers += buffers
                skipped_meta += meta
                continue

            _move_trainable_adapter_param_buffers(model_chunk, root, "reload_from_cpu", groups)
            for group in groups.values():
                _load_group_from_flat(group, target_device)

    if moved_params or moved_buffers or skipped_meta:
        logger.info(
            "Loaded %d adapter parameters and %d adapter buffers to %s; skipped %d meta tensors",
            moved_params,
            moved_buffers,
            target_device,
            skipped_meta,
        )

    gc.collect()
    _empty_cuda_cache_if_available()


__all__ = [
    "load_megatron_adapter_to_gpu",
    "load_megatron_frozen_base_to_gpu",
    "load_megatron_grad_buffers",
    "load_megatron_optimizer",
    "offload_megatron_adapter_to_cpu",
    "offload_megatron_frozen_base_to_cpu",
    "offload_megatron_grad_buffers",
    "offload_megatron_optimizer",
]
