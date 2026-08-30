import hashlib
import logging
import math
import os

# ORBIT-SEAM: `time` is for the orbit pause-window measurement in update_weights below
import time
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

# ORBIT-SEAM: base's sync config is LoRA-only; orbit's PEFT layer (LoRA + OFT) supplies the sync spec
from miles.orbit.megatron.peft_utils import (
    build_peft_sync_spec,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.lora import LORA_ADAPTER_NAME

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .common import _check_weight_sync_results, begin_weight_update, end_weight_update, weight_update_selector
from .hf_weight_iterator_base import HfWeightIteratorBase

# ORBIT-SEAM: weight-sync instrumentation (payload/pause metrics, timeline markers, the
# ORBIT_LOG_WEIGHT_SYNC trace, sync-result labels) lives in the orbit home (P1, Phase 3 slice 3g)
from miles.orbit.megatron.sync_metrics import (
    _barrier_with_logging,
    _log_weight_sync_event,
    _sync_type_label,
    emit_timeline_event,
    emit_update_weights_metrics,
    get_payload_tracker,
    sum_metrics_across_ranks,
)

# ORBIT-SEAM: orbit's adapter/teacher send methods live in the home mixin (P2, Phase 3 slice 3g)
from miles.orbit.transport.update_weight_ext import OrbitUpdateWeightExtensions

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


# ORBIT-SEAM: the mixin carries orbit's added methods (_send_adapter_params, push_teacher_adapter,
# _sync_mode_label); base's own methods keep their names and signatures here
class UpdateWeightFromTensor(OrbitUpdateWeightExtensions):
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
        # ORBIT-SEAM: base's LoRA-only `is_lora: bool = False` widens to orbit's PEFT method
        # (none/lora/oft); is_lora stays accepted so base-shaped callers keep working
        peft_method: str = "none",
        is_lora: bool | None = None,
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
        # ORBIT-SEAM: base stores only `self.is_lora`; orbit additionally stores the resolved PEFT
        # method plus a per-updater args view carrying it (the weight iterator kwarg below follows)
        if is_lora is not None and peft_method == "none":
            peft_method = "lora" if is_lora else "none"
        self.peft_method = peft_method
        self.is_lora = self.peft_method == "lora"

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

        # ORBIT-SEAM: orbit's PEFT sync spec (LoRA + OFT) alongside base's LoRA-only _lora_config,
        # plus the transport slots the home layer fills (adapter transport, its mode, the OPD
        # teacher-slot state)
        self._peft_sync_spec = build_peft_sync_spec(self._peft_args) if self.peft_method != "none" else None
        # Transport is constructed in connect_rollout_engines after use_distribute is known.
        self._peft_transport = None
        self._peft_transport_mode = None
        # Independent loaded-state for the OPD teacher slot (orbit_teacher). The
        # transport's own _peft_loaded flag tracks the STUDENT adapter; the
        # teacher slot fills lazily on first promotion, so it must not be
        # unloaded before it exists.
        self._teacher_slot_loaded = False
        # Create IPC gather groups within megatron.
        # ORBIT-SEAM: record the gather-group layout so connect_rollout_engines can detect a layout
        # change (base rebuilds groups off a rank-has-engine test that placeholder ranks skew)
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
        # ORBIT-SEAM: pre-initialise the engine-split state base only sets in connect_rollout_engines,
        # so update_weights (and its metrics) are safe before the first connect
        self.rollout_engines: Sequence[ActorHandle] = []
        self.distributed_rollout_engines = []
        self._all_rollout_engines = []
        self.use_distribute = False
        self._is_distributed_src_rank = False
        self._connection_stale: bool = False

    # TODO: avoid dup code during yueming's refactor (temp write this to avoid introducing potentially conflicting base class)
    def is_rollout_engines_fresh(self) -> bool:
        # ORBIT-SEAM: orbit pre-initialises rollout_engines to [] in __init__ (base leaves it None),
        # so freshness is "non-empty and not stale" rather than "not None".
        return bool(self.rollout_engines) and not self._connection_stale

    def mark_engine_connection_stale(self) -> None:
        self._connection_stale = True

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        # ORBIT-SEAM: docstring reworded for the src-rank rule change below (a PEFT sync sources
        # from the DP/TP-0 rank of ANY pipeline stage, not only the global PP-0 rank)
        """
        Split colocated/distributed engines. Distributed source ranks create NCCL
        groups; colocated ranks map to their local IPC engine.
        """
        # ORBIT-SEAM: reset the OPD teacher-slot state on (re)connect and keep the full engine list
        # (base only stores the colocated split, which the lifecycle calls now need)
        # (Re)connect means the engine set changed: any new/restarted engine's
        # orbit_teacher slot starts EMPTY (self:* slots are capacity-only, never
        # preloaded). Reset so the next teacher promotion does not try to unload
        # a slot that is not there. First connect: flag is already False, no-op.
        self._teacher_slot_loaded = False
        self._all_rollout_engines = list(rollout_engines)
        self.rollout_engines = list(rollout_engines)
        self.distributed_rollout_engines = []
        self._connection_stale = False

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        # ORBIT-SEAM: base's offset test alone reads a non-colocated PEFT launch as colocated
        # (rollout offsets restart at 0 in the rollout placement group); short-circuit it
        # In async/non-colocated launches, rollout offsets are relative to the rollout
        # placement-group slice, so offset 0 is still outside actor GPUs.
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
        # ORBIT-SEAM: hoisted so the PEFT transport hookup at the end of this method can read it
        distributed_gpu_counts = []

        if self.use_distribute:
            self.rollout_engines = list(rollout_engines[:colocate_engine_nums])
            self.distributed_rollout_engines = list(rollout_engines[colocate_engine_nums:])
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            # ORBIT-SEAM: adapters are replicated across pipeline stages, so a PEFT sync sources
            # from every DP/TP-0 rank; base's PP-0-only rule would ship one stage's shard
            peft_or_base_pp0 = self._peft_sync_spec is not None or get_parallel_state().pp.rank == 0
            self._is_distributed_src_rank = (
                get_parallel_state().intra_dp_cp.rank == 0
                and get_parallel_state().tp.rank == 0
                and peft_or_base_pp0
            )
            self._group_name = "miles"
            # ORBIT-SEAM: PEFT runs skip base's NCCL model-update group; adapters ship over the
            # orbit PeftWeightTransport built below instead
            if self._peft_sync_spec is None and self._is_distributed_src_rank:
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
        # ORBIT-SEAM: clear the src-rank flag on a reconnect that drops the distributed engines
        # (base leaves the previous run's True in place)
        else:
            self._is_distributed_src_rank = False

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # ORBIT-SEAM: base rebuilds the gather groups from a per-rank "do I have an engine" test, so
        # ranks disagree on whether/when to call new_group (a collective) and a layout change on
        # ranks that already hold a group is missed; orbit keys the rebuild on the layout itself,
        # which every rank computes identically
        ipc_gather_layout = tuple(zip(colocate_gpu_offsets, colocate_gpu_counts, strict=True))

        # Create IPC Gloo gather groups matching actual engine layout. All ranks
        # build groups in the same order when the layout changes, including
        # placeholder ranks that are not members of any colocated engine group.
        if self._ipc_gather_layout != ipc_gather_layout:
            self._ipc_gather_group = None
            self._ipc_gather_src = None
            for offset, count in ipc_gather_layout:
                group_ranks = list(range(offset, offset + count))
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if dist.get_rank() in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = offset
            self._ipc_gather_layout = ipc_gather_layout

        # Map training ranks to colocated engine actors.
        self._ipc_engine = None
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

        # ORBIT-SEAM: PEFT transport hookup - the adapter sync path base has no equivalent of
        # (colocated IPC vs NCCL selection, non-source ranks detach, reconnect on a mode change);
        # construction and wire format live in miles.orbit.transport
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
            from miles.orbit.transport import build_peft_transport
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

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear ``update_weight_metrics``. Empty under colocate today; kept symmetric
        with the distributed updaters so the actor can drain unconditionally."""
        out = self.__dict__.pop("update_weight_metrics", {})
        return out

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

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

        # ORBIT-SEAM: instrumentation preamble - the ORBIT_LOG_WEIGHT_SYNC trace, the payload
        # tracker/pause clock feeding perf/update_weights_*, and the full engine list the
        # pause/continue lifecycle must address (base pauses only the colocated split)
        world_size = dist.get_world_size()
        _log_weight_sync_event(
            "update_weights_begin",
            rank=rank,
            world_size=world_size,
            weight_version=self.weight_version,
            peft_method=self.peft_method,
            colocated_engine_count=len(getattr(self, "rollout_engines", [])),
            distributed_engine_count=len(getattr(self, "distributed_rollout_engines", [])),
        )
        lifecycle_rollout_engines = getattr(self, "_all_rollout_engines", None) or self.rollout_engines

        # Sync-cost instrumentation (perf/update_weights_*): every rank tracks
        # the payload it actually ships; rank 0 tracks the engine pause window.
        get_payload_tracker().reset()
        sync_mode = self._sync_mode_label()
        pause_dispatch_started = None
        pause_seconds = 0.0

        if rank == 0:
            mode = self.args.pause_generation_mode
            # ORBIT-SEAM: trace + "update_start" timeline marker + pause-window clock around base's
            # pause/flush dispatch, which now addresses lifecycle_rollout_engines
            _log_weight_sync_event(
                "pause_and_flush_dispatch",
                rank=rank,
                world_size=world_size,
                weight_version=self.weight_version,
                rollout_engine_count=len(lifecycle_rollout_engines),
                pause_generation_mode=mode,
            )
            emit_timeline_event("update_start", weight_version=self.weight_version, mode=sync_mode)
            pause_dispatch_started = time.perf_counter()
            ray.get([engine.pause_generation.remote(mode=mode) for engine in lifecycle_rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in lifecycle_rollout_engines])
            # ORBIT-SEAM: upstream replaced the post_process_weights(restore_weights_before_load=True)
            # call with the begin/end weight-update session; the engine list stays orbit's
            # lifecycle_rollout_engines (base addresses only the colocated split)
            if not skip_base_sync:
                begin_weight_update(lifecycle_rollout_engines, weight_update_selector(self.args))
            _log_weight_sync_event(
                "pause_and_flush_complete",
                rank=rank,
                world_size=world_size,
                weight_version=self.weight_version,
            )
        # ORBIT-SEAM: base's bare dist.barrier, wrapped so the trace brackets it (same group, same
        # collective) - here and at the two barriers below
        _barrier_with_logging(
            "after_pause_and_flush_barrier",
            group=get_gloo_group(),
            rank=rank,
            world_size=world_size,
            weight_version=self.weight_version,
            peft_method=self.peft_method,
        )

        megatron_local_weights = self.weights_getter()

        sync_chunk_count = 0
        source_chunk_count = None

        # ORBIT-SEAM: an orbit PEFT run (LoRA or OFT) never pushes the frozen base; the base pass
        # below is upstream's, gated on _peft_sync_spec in addition to upstream's skip_base_sync.
        if self._peft_sync_spec is None and not skip_base_sync:
            for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights, weight_type="base"
            ):
                refs, long_lived_tensors = self._send_base_params(hf_named_tensors)
                results = ray.get(refs)
                _check_weight_sync_results(results, is_lora=False)
                del long_lived_tensors
                sync_chunk_count += 1

            mm_tower_tensors = self._mm_tower_named_tensors()
            if mm_tower_tensors is not None:
                mm_tower_tensors = [
                    (name, tensor.to(torch.cuda.current_device())) for name, tensor in mm_tower_tensors
                ]
                refs, long_lived_tensors = self._send_base_params(mm_tower_tensors)
                results = ray.get(refs)
                _check_weight_sync_results(results, is_lora=False)
                del long_lived_tensors, mm_tower_tensors
                sync_chunk_count += 1

        # ORBIT-SEAM: upstream's LoRA-only adapter sync (_send_lora_params + _pp_assemble_full_adapter)
        # is superseded by orbit's PEFT transport, which serves LoRA and OFT alike. Chunks are bound
        # first so the per-chunk adapter shards can be coalesced into the single load sglang requires.
        if self._peft_sync_spec is not None:
            weight_chunks = self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights, weight_type="lora"
            )
            if self._peft_sync_spec.method == "lora":
                from miles.orbit.transport._gather import (
                    coalesce_lora_hf_weight_chunks,
                )

                source_chunk_count, weight_chunks = coalesce_lora_hf_weight_chunks(weight_chunks)
                if source_chunk_count > 1 and rank == 0:
                    logger.info(
                        "Coalesced %d LoRA weight chunks into one adapter load.",
                        source_chunk_count,
                    )
            if self._peft_sync_spec.method == "oft":
                from miles.orbit.transport._gather import (
                    coalesce_oft_hf_weight_chunks,
                )

                source_chunk_count, weight_chunks = coalesce_oft_hf_weight_chunks(weight_chunks)
                if source_chunk_count > 1 and rank == 0:
                    logger.info(
                        "Coalesced %d OFT weight chunks into one adapter load.",
                        source_chunk_count,
                    )

            for hf_named_tensors in weight_chunks:
                # ORBIT-SEAM: the home mixin's transport send, which may already hold completed results
                refs, long_lived_tensors, completed_results = self._send_adapter_params(hf_named_tensors)
                results = completed_results if completed_results is not None else ray.get(refs)
                _log_weight_sync_event(
                    "chunk_results_received",
                    rank=rank,
                    world_size=world_size,
                    weight_version=self.weight_version,
                    chunk_idx=sync_chunk_count,
                    results_count=len(results),
                )
                _check_peft_weight_sync_results(results, sync_type=_sync_type_label(self.peft_method))
                del long_lived_tensors
                sync_chunk_count += 1
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            # ORBIT-SEAM: base's LoRA-only zero-chunk guard now covers every PEFT method, plus a
            # second guard that a coalesced adapter really left as one chunk
            if sync_chunk_count == 0:
                raise RuntimeError(
                    f"{self._peft_sync_spec.method.upper()} weight sync failed: "
                    "the weight iterator produced zero chunks. "
                    "No adapter weights were sent to the rollout engine. This usually means "
                    "the Megatron-Bridge or SGLang version is incompatible."
                )
            if source_chunk_count is not None and sync_chunk_count > 1:
                method_label = self._peft_sync_spec.method.upper()
                raise RuntimeError(
                    f"Internal {method_label} sync error: coalesced "
                    f"{method_label} adapter should be sent as one chunk, "
                    f"got {sync_chunk_count} chunks."
                )
            if self.is_lora and not self._lora_base_synced:
                self._lora_base_synced = True

        _barrier_with_logging(
            "after_chunk_sync_barrier",
            group=get_gloo_group(),
            rank=rank,
            world_size=world_size,
            weight_version=self.weight_version,
            sync_chunk_count=sync_chunk_count,
            peft_method=self.peft_method,
        )

        if rank == 0:
            # `post_process_quantization` is related to the `process_weights_after_loading`
            # in the sglang rollout side, which should always be invoked after weight
            # updating.
            # ORBIT-SEAM: trace + "update_end" timeline marker + pause-window close around base's
            # post-process/continue dispatch, which now addresses lifecycle_rollout_engines
            _log_weight_sync_event(
                "post_process_and_continue_dispatch",
                rank=rank,
                world_size=world_size,
                weight_version=self.weight_version,
                rollout_engine_count=len(lifecycle_rollout_engines),
            )
            # ORBIT-SEAM: upstream replaced post_process_weights(post_process_quantization=True)
            # with end_weight_update; the engine list stays orbit's lifecycle_rollout_engines
            # Skip when no fresh base bytes landed (skip_base_sync).
            if not skip_base_sync:
                end_weight_update(lifecycle_rollout_engines)
            ray.get([engine.continue_generation.remote() for engine in lifecycle_rollout_engines])
            if pause_dispatch_started is not None:
                pause_seconds = time.perf_counter() - pause_dispatch_started
            emit_timeline_event("update_end", weight_version=self.weight_version, mode=sync_mode)
            _log_weight_sync_event(
                "post_process_and_continue_complete",
                rank=rank,
                world_size=world_size,
                weight_version=self.weight_version,
            )
        _barrier_with_logging(
            "after_continue_generation_barrier",
            group=get_gloo_group(),
            rank=rank,
            world_size=world_size,
            weight_version=self.weight_version,
            peft_method=self.peft_method,
        )
        # ORBIT-SEAM: base's update ends at the barrier; orbit adds the cross-rank metric reduction
        # and the perf/update_weights_* emission (home layer) plus the closing trace line
        # SUM the per-rank contributions so the perf-logging primary rank
        # (which is the last PP stage, not necessarily rank 0) emits the
        # per-update totals. This is a collective on the same gloo group as the
        # barrier above, which guarantees all ranks reach it in lockstep; local
        # value computation is deterministic and identical on every rank.
        tracker = get_payload_tracker()
        pause_total, payload_bytes_total, payload_tensors_total = sum_metrics_across_ranks(
            [pause_seconds, tracker.payload_bytes, tracker.num_tensors],
            group=get_gloo_group(),
        )
        emit_update_weights_metrics(
            pause_seconds=pause_total,
            payload_bytes=payload_bytes_total,
            num_tensors=payload_tensors_total,
            num_chunks=sync_chunk_count,
        )
        _log_weight_sync_event(
            "update_weights_complete",
            rank=rank,
            world_size=world_size,
            weight_version=self.weight_version,
            sync_chunk_count=sync_chunk_count,
            peft_method=self.peft_method,
        )

    # ORBIT-SEAM: base's LoRA branch (weight filtering, lora_config/lora_name/lora_loaded kwargs,
    # the "not supported for distributed engines" raise) is gone - adapters never reach this method
    # any more, they go through the home mixin's transport send; the rest is base's own send path
    # with the ORBIT_LOG_WEIGHT_SYNC trace threaded through it
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

    # ORBIT-SEAM: base's LoRA branch (weight filtering, lora_config/lora_name/lora_loaded kwargs,
    # the "not supported for distributed engines" raise) is gone - adapters never reach this method
    # any more, they go through the home mixin's transport send; the rest is base's own send path
    # with the ORBIT_LOG_WEIGHT_SYNC trace threaded through it
    def _send_base_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors=hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            selector=weight_update_selector(self.args),
            weight_version=self.weight_version,
        )
        all_refs.extend(refs_colocated)
        _log_weight_sync_event(
            "send_base_params_colocated_dispatched",
            rank=rank,
            world_size=world_size,
            weight_version=self.weight_version,
            peft_method=self.peft_method,
            refs_colocated_count=len(refs_colocated),
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
                all_refs.extend(refs_distributed)
            _log_weight_sync_event(
                "send_base_params_distributed_dispatched",
                rank=rank,
                world_size=world_size,
                weight_version=self.weight_version,
                refs_distributed_count=len(refs_distributed),
            )

        _log_weight_sync_event(
            "send_base_params_complete",
            rank=rank,
            world_size=world_size,
            weight_version=self.weight_version,
            peft_method=self.peft_method,
            refs_total_count=len(all_refs),
        )

        return all_refs, long_lived_tensors

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
    # ORBIT-SEAM: orbit's PEFT adapter path never reaches this function (it goes through the home
    # mixin's transport send); upstream's lora_* parameters below stay for base's own LoRA path
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

    # ORBIT-SEAM: record what this rank actually puts on the wire (perf/update_weights_payload_*)
    # Payload accounting: the engine deserializes every gather-group rank's
    # flattened bucket(s), so each rank records its own flat tensor(s). Byte
    # computation happens inside record()'s never-raise guard.
    get_payload_tracker().record([data.get("flattened_tensor") for data in long_live_tensors])

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


# ORBIT-SEAM: base derives the label inline from `is_lora`; with three PEFT methods orbit's adapter
# path passes the label in (miles.orbit.megatron.sync_metrics._sync_type_label). Named apart from
# `.common._check_weight_sync_results`, which base's own call sites still use with is_lora=.
def _check_peft_weight_sync_results(results: list, *, sync_type: str) -> None:
    """Validate return values from rollout engine weight-sync RPCs.

    Raises RuntimeError if any engine reports failure, preventing silent
    failures when SGLang versions are incompatible.
    """
    for result in results:
        if isinstance(result, Mapping):
            success = result.get("success")
            error_msg = result.get("error_message") or result.get("error") or "unknown error"
        elif hasattr(result, "success"):
            success = result.success
            error_msg = getattr(result, "error_message", "unknown error")
        else:
            continue

        if success is False:
            raise RuntimeError(
                f"{sync_type} weight sync failed on rollout engine: {error_msg}. "
                f"Check SGLang version compatibility."
            )
