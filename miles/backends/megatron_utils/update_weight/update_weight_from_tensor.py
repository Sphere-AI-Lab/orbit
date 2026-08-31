import hashlib
import logging
import math
import os
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle

from miles.backends.megatron_utils.lora_utils import (
    build_lora_sync_config,
    is_lora_weight_name,
    lora_base_cpu_backup_enabled,
)
from miles.backends.megatron_utils.peft_utils import build_peft_sync_spec
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.lora import LORA_ADAPTER_NAME

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .common import _check_weight_sync_results, begin_weight_update, end_weight_update, weight_update_selector
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed.broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


def _pp_assemble_full_adapter(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Assemble the complete adapter on every PP rank (exporter gathers TP/EP but not PP)."""
    pp_group = get_parallel_state().pp.group
    pp_size = dist.get_world_size(group=pp_group)
    if pp_size == 1:
        return hf_named_tensors
    pp_rank = dist.get_rank(group=pp_group)
    global_ranks = dist.get_process_group_ranks(pp_group)
    device = torch.cuda.current_device()

    local_meta = [(n, tuple(t.shape), t.dtype) for n, t in hf_named_tensors]
    all_meta: list = [None] * pp_size
    dist.all_gather_object(all_meta, local_meta, group=pp_group)

    local_by_name = {n: t for n, t in hf_named_tensors}
    merged: dict[str, torch.Tensor] = {}
    for src_pp, meta in enumerate(all_meta):
        by_dtype: dict = {}
        for n, shape, dtype in meta:
            by_dtype.setdefault(dtype, []).append((n, shape))
        for dtype, entries in by_dtype.items():
            numel = sum(math.prod(shape) for _, shape in entries)
            flat = torch.empty(numel, dtype=dtype, device=device)
            if src_pp == pp_rank:
                off = 0
                for n, shape in entries:
                    k = math.prod(shape)
                    flat[off : off + k].copy_(local_by_name[n].reshape(-1))
                    off += k
            dist.broadcast(flat, src=global_ranks[src_pp], group=pp_group)
            off = 0
            for n, shape in entries:
                k = math.prod(shape)
                merged[n] = flat[off : off + k].view(shape)
                off += k
    return sorted(merged.items())


def _sync_type_label(peft_method: str) -> str:
    """Return the human-readable label for a sync result's source."""
    if peft_method == "lora":
        return "LoRA"
    if peft_method == "oft":
        return "OFT"
    return "Base model"


def _raise_distributed_peft_failure(local_error: Exception | None) -> None:
    """Make a source-only adapter failure visible to every trainer rank."""
    rank = dist.get_rank()
    failure_record = None
    if local_error is not None:
        failure_record = {
            "source_rank": rank,
            "error": f"{type(local_error).__name__}: {local_error}",
        }

    gloo_group = get_gloo_group()
    failure_records = [None] * dist.get_world_size(gloo_group)
    dist.all_gather_object(failure_records, failure_record, group=gloo_group)
    failure_record = next((record for record in failure_records if record is not None), None)
    if failure_record is None:
        return

    message = (
        "PEFT adapter dispatch failed on source rank " f"{failure_record['source_rank']}: {failure_record['error']}"
    )
    if local_error is not None and failure_record["source_rank"] == rank:
        raise RuntimeError(message) from local_error
    raise RuntimeError(message)


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict->GPU) -> broadcast PP/EP(GPU NCCL) -> gather TP(GPU NCCL) -> convert HF(GPU) -> send.
    Colocated: GPU->CPU serialize -> gather_object(Gloo CPU) -> Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        """
        Compute param buckets, create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.is_lora = is_lora
        # Unified PEFT identity: derive from args; legacy is_lora keeps working.
        peft_method = getattr(args, "peft_method", "none") or "none"
        if is_lora and peft_method == "none":
            peft_method = "lora"
        self.peft_method = peft_method
        self._peft_args = args
        if getattr(args, "peft_method", "none") != self.peft_method:
            self._peft_args = Namespace(**vars(args), peft_method=self.peft_method)
        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
            peft_method=self.peft_method,
        )
        if self.is_lora:
            self._lora_config = build_lora_sync_config(args)
            self._lora_loaded = False
            self._lora_base_synced = False

        self._mm_tower_cache: list[tuple[str, torch.Tensor]] | None = None

        self._peft_sync_spec = build_peft_sync_spec(self._peft_args) if self.peft_method != "none" else None
        # Transport is constructed in connect_rollout_engines after use_distribute is known.
        self._peft_transport = None
        self._peft_transport_mode = None
        # Create IPC gather groups within megatron.
        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_gather_layout = tuple(
            (start_rank, self.args.rollout_num_gpus_per_engine)
            for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine)
        )
        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = min(start_rank + self.args.rollout_num_gpus_per_engine, dist.get_world_size())
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

        self._model_update_groups = None
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale: bool = False

    # TODO: avoid dup code during yueming's refactor (temp write this to avoid introducing potentially conflicting base class)
    def is_rollout_engines_fresh(self) -> bool:
        return self.rollout_engines is not None and not self._connection_stale

    def mark_engine_connection_stale(self) -> None:
        self._connection_stale = True

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines
        self._connection_stale = False
        self._all_rollout_engines = list(rollout_engines)
        self.rollout_engines = list(rollout_engines)
        self.distributed_rollout_engines = []

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # In non-colocated launches, rollout offsets are relative to the rollout
        # placement-group slice, so offset zero is still outside the actor GPUs.
        if self._peft_sync_spec is not None and not getattr(self.args, "colocate", True):
            colocate_engine_nums = 0
        else:
            total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
            colocate_engine_nums = 0
            for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
                if gpu_offset + gpu_count > total_actor_gpus:
                    break
                colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                get_parallel_state().intra_dp_cp.rank == 0
                and get_parallel_state().tp.rank == 0
                and get_parallel_state().pp.rank == 0
            )
            self._group_name = "miles"
            if self._is_distributed_src_rank:
                if (g := self._model_update_groups) is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, g, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Determine whether this rank is covered by any colocated engine.
        all_colocated_ranks = set()
        for offset, count in zip(colocate_gpu_offsets, colocate_gpu_counts, strict=True):
            all_colocated_ranks.update(range(offset, offset + count))
        rank_has_engine = dist.get_rank() in all_colocated_ranks

        # Create IPC Gloo gather groups matching actual engine layout.
        # Re-create on first call or when engine layout changes (placeholder ranks
        # that had a group from __init__ but no actual engine need to be reset).
        if rank_has_engine:
            if self._ipc_gather_group is None:
                for i in range(colocate_engine_nums):
                    group_ranks = list(
                        range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i])
                    )
                    new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                    if dist.get_rank() in group_ranks:
                        self._ipc_gather_group = new_group
                        self._ipc_gather_src = colocate_gpu_offsets[i]
        else:
            # Ranks not covered by any engine (e.g. placeholder GPU slots)
            self._ipc_gather_group = None
            self._ipc_gather_src = None

        # Map training ranks to colocated engine actors.
        self._ipc_engine = None
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

        if self._peft_sync_spec is not None and self.use_distribute and colocate_engine_nums:
            raise RuntimeError(
                "Hybrid colocated+distributed PEFT weight sync is not supported yet. "
                "Use fully colocated or fully non-colocated rollout placement."
            )

        if self._peft_sync_spec is not None and self.use_distribute and not self._is_distributed_src_rank:
            if self._peft_transport is not None:
                self._peft_transport.disconnect()
            self._peft_transport = None
            self._peft_transport_mode = "nccl-non-source"
            return

        if self._peft_sync_spec is not None:
            from miles.backends.megatron_utils.peft_transport import build_peft_transport

            desired_mode = "nccl" if self.use_distribute else "ipc"
            if self._peft_transport is None or self._peft_transport_mode != desired_mode:
                if self._peft_transport is not None:
                    self._peft_transport.disconnect()
                self._peft_transport = build_peft_transport(
                    args=self._peft_args,
                    use_distribute=self.use_distribute,
                    ipc_gather_group=self._ipc_gather_group,
                    ipc_gather_src=self._ipc_gather_src,
                )
                self._peft_transport_mode = desired_mode
                if dist.get_rank() == 0:
                    logger.info(self._peft_transport.runtime_mode.log_line())
        if self._peft_transport is not None:
            if self.use_distribute:
                transport_engines = self.distributed_rollout_engines
                transport_gpu_counts = distributed_gpu_counts
            else:
                transport_engines = [self._ipc_engine] if self._ipc_engine is not None else []
                transport_gpu_counts = []
            self._peft_transport.connect(transport_engines, rollout_engine_lock, transport_gpu_counts)

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        if self._peft_sync_spec is not None:
            self._update_peft_weights()
            return

        rank = dist.get_rank()

        # TODO: implement lora weight checker
        colocate_base_persistent = getattr(self.args, "colocate", False) and not getattr(
            self.args, "offload_rollout", True
        )
        skip_base_sync = (
            self.is_lora
            and (self.use_distribute or lora_base_cpu_backup_enabled(self.args) or colocate_base_persistent)
            and not getattr(self.args, "check_weight_update_equal", False)
        )

        if rank == 0:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if not skip_base_sync:
                begin_weight_update(self.rollout_engines, weight_update_selector(self.args))
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        if not skip_base_sync:
            for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights, weight_type="base"
            ):
                refs, long_lived_tensors = self._send_base_params(hf_named_tensors)
                results = ray.get(refs)
                _check_weight_sync_results(results, is_lora=False)
                del long_lived_tensors

            mm_tower_tensors = self._mm_tower_named_tensors()
            if mm_tower_tensors is not None:
                mm_tower_tensors = [
                    (name, tensor.to(torch.cuda.current_device())) for name, tensor in mm_tower_tensors
                ]
                refs, long_lived_tensors = self._send_base_params(mm_tower_tensors)
                results = ray.get(refs)
                _check_weight_sync_results(results, is_lora=False)
                del long_lived_tensors, mm_tower_tensors

        if self.is_lora:
            # SGLang's load_lora_adapter_from_tensors expects the full adapter in
            # one call; drain the bridge's chunker so --update-weight-buffer-size
            # only bounds the base path.
            accumulated_named_tensors: list = []
            for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights, weight_type="lora"
            ):
                accumulated_named_tensors.extend(hf_named_tensors)

            if not accumulated_named_tensors:
                raise RuntimeError(
                    "LoRA weight sync failed: the weight iterator produced zero chunks. "
                    "No adapter weights were sent to the rollout engine. This usually means "
                    "the Megatron-Bridge or SGLang version is incompatible."
                )

            accumulated_named_tensors = _pp_assemble_full_adapter(accumulated_named_tensors)

            refs, long_lived_tensors = self._send_lora_params(accumulated_named_tensors)
            results = ray.get(refs)
            _check_weight_sync_results(results, is_lora=True)
            del long_lived_tensors
            del accumulated_named_tensors
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            if not self._lora_base_synced:
                self._lora_base_synced = True

        dist.barrier(group=get_gloo_group())

        if rank == 0:
            # Skip when no fresh base bytes landed (skip_base_sync).
            if not skip_base_sync:
                end_weight_update(self.rollout_engines)
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _update_peft_weights(self) -> None:
        """Send only the active LoRA/OFT adapter through the selected PEFT transport."""
        rank = dist.get_rank()
        lifecycle_engines = self._all_rollout_engines or list(self.rollout_engines)

        if rank == 0:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in lifecycle_engines])
            ray.get([engine.flush_cache.remote() for engine in lifecycle_engines])
        dist.barrier(group=get_gloo_group())

        weight_chunks = self._hf_weight_iterator.get_hf_weight_chunks(self.weights_getter())
        if self._peft_sync_spec.method == "lora":
            from miles.backends.megatron_utils.peft_transport._gather import coalesce_lora_hf_weight_chunks

            source_chunk_count, weight_chunks = coalesce_lora_hf_weight_chunks(weight_chunks)
        else:
            from miles.backends.megatron_utils.peft_transport._gather import coalesce_oft_hf_weight_chunks

            source_chunk_count, weight_chunks = coalesce_oft_hf_weight_chunks(weight_chunks)

        sync_chunk_count = 0
        for hf_named_tensors in weight_chunks:
            if self.use_distribute:
                chunk_error = None
                try:
                    refs, long_lived_tensors, completed_results = self._send_adapter_params(hf_named_tensors)
                    results = completed_results if completed_results is not None else ray.get(refs)
                    _check_weight_sync_results(results, is_lora=self._peft_sync_spec.method == "lora")
                except Exception as exc:  # noqa: BLE001 -- synchronized below
                    chunk_error = exc
                _raise_distributed_peft_failure(chunk_error)
            else:
                refs, long_lived_tensors, completed_results = self._send_adapter_params(hf_named_tensors)
                results = completed_results if completed_results is not None else ray.get(refs)
                _check_weight_sync_results(results, is_lora=self._peft_sync_spec.method == "lora")
            del long_lived_tensors
            sync_chunk_count += 1

        if sync_chunk_count == 0:
            raise RuntimeError(
                f"{self._peft_sync_spec.method.upper()} weight sync failed: "
                "the weight iterator produced zero chunks."
            )
        if source_chunk_count > 1 and sync_chunk_count != 1:
            raise RuntimeError(
                f"Internal {self._peft_sync_spec.method.upper()} sync error: "
                f"coalesced adapter produced {sync_chunk_count} chunks."
            )

        dist.barrier(group=get_gloo_group())
        if rank == 0:
            ray.get([engine.continue_generation.remote() for engine in lifecycle_engines])
        dist.barrier(group=get_gloo_group())

    def _mm_tower_named_tensors(self) -> list[tuple[str, torch.Tensor]] | None:
        """Frozen vision/audio tower tensors to append to every base sync (see
        __init__ comment). Returns None when the run has no MM towers. EVERY
        gather-group rank contributes the full tower set (read once from its local
        HF checkpoint, the same bytes the engine loaded at boot): the colocated
        send requires homogeneous per-rank bucket counts (num_dtypes is taken from
        rank 0 and indexed into every rank's list), so a src-only contribution
        breaks assembly. The duplicates are ~15MB/rank and load idempotently."""
        provider = getattr(self.args, "custom_model_provider_path", None) or ""
        if "inkling_mm_model_provider" not in provider:
            return None
        if self._mm_tower_cache is None:
            if self._ipc_gather_group is not None:
                import json

                from safetensors import safe_open

                ckpt_dir = self.args.hf_checkpoint
                with open(os.path.join(ckpt_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
                    weight_map = json.load(f)["weight_map"]
                tower_keys = sorted(
                    k
                    for k in weight_map
                    if ".visual." in f".{k}" or ".audio." in f".{k}" or k.startswith(("visual.", "audio."))
                )
                by_shard: dict[str, list[str]] = {}
                for k in tower_keys:
                    by_shard.setdefault(weight_map[k], []).append(k)
                cache = []
                for shard, keys in by_shard.items():
                    with safe_open(os.path.join(ckpt_dir, shard), framework="pt", device="cpu") as f:
                        for k in keys:
                            cache.append((k, f.get_tensor(k)))
                logger.info(
                    "mm tower sync: caching %d tower tensors from %s: %s",
                    len(cache),
                    ckpt_dir,
                    [k for k, _ in cache],
                )
                self._mm_tower_cache = cache
            else:
                self._mm_tower_cache = []
        return self._mm_tower_cache

    def _send_adapter_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any, list[Any] | None]:
        if self.use_distribute and not self._is_distributed_src_rank:
            return [], [], []
        if self._peft_transport is None:
            raise RuntimeError("_send_adapter_params called without a PEFT transport")
        send_result = self._peft_transport.send_adapter(
            hf_named_tensors,
            weight_version=self.weight_version,
        )
        return send_result.refs, [], send_result.results

    def _send_base_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        refs, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors=hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            selector=weight_update_selector(self.args),
            weight_version=self.weight_version,
        )
        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
                selector=weight_update_selector(self.args),
            )
            if refs_distributed:
                refs = (refs or []) + refs_distributed
        return refs or [], long_lived_tensors

    def _send_lora_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        if not any(is_lora_weight_name(n) for n, _ in hf_named_tensors):
            raise RuntimeError(
                "LoRA weight sync failed: chunk contains no LoRA weights "
                "(no lora_A/lora_B names found). Check weight iterator configuration."
            )
        if self.use_distribute and self._is_distributed_src_rank:
            raise NotImplementedError("LoRA weight sync is not yet supported for distributed (non-colocated) engines")

        refs, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors=hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            selector=weight_update_selector(self.args),
            lora_config=self._lora_config,
            lora_name=LORA_ADAPTER_NAME,
            lora_loaded=self._lora_loaded,
            check_equal=getattr(self.args, "check_lora_weight_equal", False),
            repack_lora_for_ipc=getattr(self.args, "offload_train", False),
        )
        self._lora_loaded = True
        return refs or [], long_lived_tensors


def _repack_onto_fresh_storage(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Copy CUDA tensors into freshly allocated flat buffers and return views onto them.

    ``torch_memory_saver.region()`` is a ``torch.cuda.use_mem_pool`` context, so anything
    allocated while building the model -- the LoRA adapter parameters included -- lives in
    a MemPool the preloaded hook backs with cuMem. cuMem allocations cannot be exported
    over the legacy CUDA IPC API that ``MultiprocessingSerializer`` uses, so handing their
    storages straight to the engine fails with "CUDA error: invalid argument" on the first
    sync. Allocating here, outside any region, gets normal caching-allocator memory whose
    handles export fine. (This is also why the FlattenedTensorBucket path works: its
    flattened tensor is likewise allocated at sync time.)

    One buffer per (dtype, device) rather than one clone per tensor: the direct-dict
    transport relies on the pickler memoizing storages so the engine receives a handful of
    IPC handles instead of one per adapter tensor.
    """
    groups: dict[tuple[torch.dtype, torch.device], list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in named_tensors:
        if tensor.is_cuda:
            groups.setdefault((tensor.dtype, tensor.device), []).append((name, tensor))

    views: dict[str, torch.Tensor] = {}
    for (dtype, device), items in groups.items():
        flat = torch.empty(sum(t.numel() for _, t in items), dtype=dtype, device=device)
        offset = 0
        for name, tensor in items:
            view = flat[offset : offset + tensor.numel()].view(tensor.shape)
            view.copy_(tensor)
            views[name] = view
            offset += tensor.numel()

    return {name: views.get(name, tensor) for name, tensor in named_tensors}


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version=None,
    lora_config: dict | None = None,
    lora_name: str | None = None,
    lora_loaded: bool = False,
    check_equal: bool = False,
    selector: str = "all",
    repack_lora_for_ipc: bool = False,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    is_lora = lora_config is not None
    is_gather_src = dist.get_rank() == ipc_gather_src
    long_live_tensors = []

    if is_lora:
        payload = _repack_onto_fresh_storage(hf_named_tensors) if repack_lora_for_ipc else dict(hf_named_tensors)
        long_live_tensors.append(payload)
        converted_named_tensors_by_dtypes = {}
        serialized_lora = MultiprocessingSerializer.serialize(payload, output_str=True)
    elif getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors: list = [serialized_lora] if is_lora else []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": flattened_tensor_bucket.get_metadata(),
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    serialized_named_tensors = [None] * dist.get_world_size(ipc_gather_group) if is_gather_src else None
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if is_gather_src:
        if is_lora:
            try:
                ray.get(ipc_engine.unload_lora_adapter.remote(lora_name=lora_name))
            except Exception as _unload_err:
                logger.debug("lora unload before load skipped: %s", _unload_err)

            expected_checksums = None
            if check_equal:
                expected_checksums = {
                    n: hashlib.sha256(
                        t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    for n, t in hf_named_tensors
                }

            refs.append(
                ipc_engine.load_lora_adapter_from_tensors.remote(
                    lora_name=lora_name,
                    config_dict=lora_config,
                    serialized_named_tensors=[
                        per_rank[0] if per_rank else None for per_rank in serialized_named_tensors
                    ],
                    expected_checksums=expected_checksums,
                )
            )

        else:
            num_dtypes = len(serialized_named_tensors[0])
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                    "load_format": "flattened_bucket",
                    "weight_version": str(weight_version),
                    "selector": selector,
                }
                refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors
