"""Structural audit of PEFT wrapping on real models.

Set ``PEFT_WRAP_AUDIT_DUMP=<base_path>`` before running a launcher to capture
two families of JSONL files:

- ``<base_path>.megatron.tp{T}_pp{P}.jsonl`` — one line per wrapped Megatron
  module with wrapper class, per-slice R buffer shapes, and base-weight
  shape. One file per (TP, PP) shard pair on the training side. DP > 0
  ranks skip the dump (their content is identical to DP rank 0).
- ``<base_path>.sglang.tp{T}.jsonl`` — one line per SGLang ``OFTMemoryPool``
  R buffer entry with shape, stacked_blocks, block_size, and layer index.
  One file per SGLang TP rank.

Each process writes its own rank-suffixed file so multi-rank runs don't
race. ``inspect_peft_wrap_audit.py`` globs both families and aggregates
when interpreting coverage.

Both hooks are no-ops when the env var is unset, so production paths pay
nothing.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_MEGATRON_SUFFIX_FMT = ".megatron.tp{tp}_pp{pp}.jsonl"
_SGLANG_SUFFIX_FMT = ".sglang.tp{tp}.jsonl"
_LOCK = threading.Lock()


def _base_path() -> str | None:
    value = os.getenv("PEFT_WRAP_AUDIT_DUMP", "").strip()
    return value or None


def enabled() -> bool:
    return _base_path() is not None


def _megatron_world() -> dict[str, int] | None:
    """Return distributed world state for the current Megatron process, or
    ``None`` if Megatron's parallel state is not initialized. Lazy import
    keeps this audit module usable in unit tests without Megatron deps.

    Keys: ``tp_rank``, ``tp_size``, ``pp_rank``, ``dp_rank``, ``ep_rank``,
    ``ep_size`` — defaults to 0 for ranks and 1 for sizes on per-call failure.
    """
    try:
        from megatron.core import parallel_state
    except ImportError:
        return None
    try:
        if not parallel_state.is_initialized():
            return None
    except AttributeError:
        # Older Megatron exposes initialization state differently.
        pass

    def _safe(fn, default):
        try:
            return int(fn())
        except (RuntimeError, AssertionError, AttributeError, TypeError):
            return int(default)

    return {
        "tp_rank": _safe(parallel_state.get_tensor_model_parallel_rank, 0),
        "tp_size": _safe(parallel_state.get_tensor_model_parallel_world_size, 1),
        "pp_rank": _safe(parallel_state.get_pipeline_model_parallel_rank, 0),
        "dp_rank": _safe(parallel_state.get_data_parallel_rank, 0),
        "ep_rank": _safe(parallel_state.get_expert_model_parallel_rank, 0),
        "ep_size": _safe(parallel_state.get_expert_model_parallel_world_size, 1),
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _LOCK, path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            n += 1
    return n


def _shape_or_none(tensor) -> list[int] | None:
    if tensor is None:
        return None
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return list(shape)


def _wrapper_class_name(module) -> str:
    return type(module).__name__


def _adapter_R_shape(adapter) -> list[int] | None:
    """Best-effort: extract the R buffer shape from an OFT adapter.

    Megatron-Bridge stores compact weights as ``.oft_r``; the precomputed R
    has shape ``(num_blocks, block_size, block_size)`` after `precompute_oft_r`.
    Different adapter classes may expose the parameter under different
    attribute names — try a few before giving up.
    """
    for attr in ("oft_r", "R", "r", "weight"):
        tensor = getattr(adapter, attr, None)
        if tensor is not None:
            return _shape_or_none(tensor)
    return None


def _walk_megatron_modules(model_chunks) -> Iterable[dict[str, Any]]:
    """Iterate (full_name, wrapped_module) across one or more model chunks."""
    if not isinstance(model_chunks, (list, tuple)):
        model_chunks = [model_chunks]
    for chunk_idx, chunk in enumerate(model_chunks):
        named_modules = getattr(chunk, "named_modules", None)
        if named_modules is None:
            continue
        for module_name, module in named_modules():
            wrapper_cls = _wrapper_class_name(module)
            # OFT wrappers we know about — match on class name so we don't
            # have to import them here (avoids tight coupling to Bridge's
            # private package layout).
            if wrapper_cls not in {
                "OFTLinear",
                "OFTLinearSplitQKV",
                "OFTLinearSplitFC1UpGate",
                "OFTLinearGroupedSplitFC1UpGate",
                "_SplitLNCanonicalOFTQKV",
                "_SplitLNCanonicalOFTFC1",
                "_SplitLNOFTLinear",
            }:
                continue

            record: dict[str, Any] = {
                "side": "megatron",
                "chunk": chunk_idx,
                "module": module_name,
                "wrapper": wrapper_cls,
            }
            # Adapter inspection: try common attribute layouts.
            for adapter_attr in (
                "adapter",  # OFTLinear
                "adapter_q",
                "adapter_k",
                "adapter_v",  # OFTLinearSplitQKV
                "adapter_gate",
                "adapter_up",  # OFTLinearSplitFC1UpGate
            ):
                adapter = getattr(module, adapter_attr, None)
                if adapter is None:
                    continue
                record[f"{adapter_attr}_R_shape"] = _adapter_R_shape(adapter)

            for bank_attr in ("adapter_gate", "adapter_up"):
                bank = getattr(module, bank_attr, None)
                if isinstance(bank, torch.nn.ModuleList):
                    shapes = [_adapter_R_shape(adapter) for adapter in bank]
                    record[f"{bank_attr}_R_shapes"] = shapes

            # Whether this wrapped module routes through the expert TP group.
            # Read from scalar adapter attrs (single OFTRotationModule) and
            # nn.ModuleList banks (grouped wrapper) separately so the grouped
            # wrapper's adapter_gate / adapter_up are not double-counted.
            def _adapter_is_expert(adapter) -> bool:
                return bool(getattr(adapter, "is_expert", False))

            is_expert_flags: list[bool] = []
            for adapter_attr in ("adapter", "adapter_q", "adapter_k", "adapter_v", "adapter_gate", "adapter_up"):
                candidate = getattr(module, adapter_attr, None)
                if candidate is None or isinstance(candidate, torch.nn.ModuleList):
                    continue
                is_expert_flags.append(_adapter_is_expert(candidate))
            for bank_attr in ("adapter_gate", "adapter_up"):
                bank = getattr(module, bank_attr, None)
                if isinstance(bank, torch.nn.ModuleList):
                    is_expert_flags.extend(_adapter_is_expert(sub) for sub in bank)
            record["is_expert"] = bool(is_expert_flags) and all(is_expert_flags)

            # Base weight shape for context.
            base = getattr(module, "base_layer", None) or getattr(module, "_orig_module", None)
            if base is not None:
                weight = getattr(base, "weight", None)
                record["base_weight_shape"] = _shape_or_none(weight)

            yield record


def dump_megatron_audit(model_chunks, *, base_path: str | None = None) -> int:
    """Write one record per OFT-wrapped Megatron module to
    ``<base_path>.megatron.tp{T}_pp{P}.jsonl``. Returns the number of
    records written. No-op when neither the env var nor an explicit
    ``base_path`` is set, AND no-op on DP rank > 0 (whose content is
    identical to DP rank 0's).
    """
    path_str = base_path or _base_path()
    if path_str is None:
        return 0

    world = _megatron_world()
    if world is None:
        world = {"tp_rank": 0, "tp_size": 1, "pp_rank": 0, "dp_rank": 0, "ep_rank": 0, "ep_size": 1}
    if world["dp_rank"] > 0:
        # Redundant; DP replicates the model.
        return 0
    tp, pp = world["tp_rank"], world["pp_rank"]

    records: list[dict[str, Any]] = list(_walk_megatron_modules(model_chunks))

    # Summary line — counters of wrapper classes seen across the whole tree.
    wrapper_counts = Counter(r["wrapper"] for r in records)
    records.append(
        {
            "side": "megatron",
            "summary": True,
            "tp_rank": world["tp_rank"],
            "tp_size": world["tp_size"],
            "pp_rank": world["pp_rank"],
            "ep_rank": world["ep_rank"],
            "ep_size": world["ep_size"],
            "wrapper_counts": dict(wrapper_counts),
            "total_wrapped": sum(wrapper_counts.values()),
        }
    )

    out_path = Path(path_str + _MEGATRON_SUFFIX_FMT.format(tp=tp, pp=pp))
    n = _write_jsonl(out_path, records)
    logger.info("PEFT_WRAP_AUDIT_DUMP wrote %d Megatron records to %s", n, out_path)
    return n


def dump_sglang_audit(memory_pool, *, base_path: str | None = None) -> int:
    """Write one record per ``OFTMemoryPool.R_buffer`` entry to
    ``<base_path>.sglang.tp{T}.jsonl``. Returns the number of records
    written. No-op when neither the env var nor an explicit ``base_path``
    is set.

    Imported lazily by SGLang's ``OFTMemoryPool.init_buffers``; do not
    import SGLang types here.
    """
    path_str = base_path or _base_path()
    if path_str is None:
        return 0

    tp = int(getattr(memory_pool, "tp_rank", 0) or 0)
    records: list[dict[str, Any]] = []

    r_buffer = getattr(memory_pool, "R_buffer", {}) or {}
    for target_module, per_layer in r_buffer.items():
        for layer_idx, tensor in enumerate(per_layer):
            shape = _shape_or_none(tensor)
            stacked_blocks = shape[-3] if shape and len(shape) == 4 else None
            block_size = shape[-1] if shape and len(shape) == 4 else None
            records.append(
                {
                    "side": "sglang",
                    "target": target_module,
                    "layer": layer_idx,
                    "R_buffer_shape": shape,
                    "stacked_blocks": stacked_blocks,
                    "block_size": block_size,
                }
            )

    embedding_buffer = getattr(memory_pool, "embedding_R_buffer", {}) or {}
    for target_module, tensor in embedding_buffer.items():
        records.append(
            {
                "side": "sglang",
                "target": target_module,
                "kind": "embedding",
                "R_buffer_shape": _shape_or_none(tensor),
            }
        )

    lm_head_buffer = getattr(memory_pool, "lm_head_R_buffer", {}) or {}
    for target_module, tensor in lm_head_buffer.items():
        records.append(
            {
                "side": "sglang",
                "target": target_module,
                "kind": "lm_head",
                "R_buffer_shape": _shape_or_none(tensor),
            }
        )

    target_counts = Counter(r["target"] for r in records if "target" in r)
    records.append(
        {
            "side": "sglang",
            "summary": True,
            "tp_rank": tp,
            "tp_size": int(getattr(memory_pool, "tp_size", 1) or 1),
            "target_counts": dict(target_counts),
            "total_records": sum(target_counts.values()),
            "max_ofts_per_batch": getattr(memory_pool, "max_ofts_per_batch", None),
            "num_layer": getattr(memory_pool, "num_layer", None),
        }
    )

    out_path = Path(path_str + _SGLANG_SUFFIX_FMT.format(tp=tp))
    n = _write_jsonl(out_path, records)
    logger.info("PEFT_WRAP_AUDIT_DUMP wrote %d SGLang records to %s", n, out_path)
    return n
