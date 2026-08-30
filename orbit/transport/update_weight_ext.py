"""Orbit's ``UpdateWeightFromTensor`` methods.

Home mixin for the methods lifted out of
miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py: the
updater's construction and engine-connection lifecycle, the instrumented
weight-sync orchestration, the base-model send, the PEFT transport send path,
the one-shot teacher-slot push, and the sync-mode label the metrics/timeline
markers carry.

``UpdateWeightFromTensor`` in the miles file lists ``OrbitUpdateWeightExtensions``
as its ONLY base, so this mixin is not an override layer sitting in front of an
upstream implementation -- the vendored class no longer defines these methods at
all and ``super()`` from here resolves straight to ``object``. Every method below
therefore carries its whole body; there is no upstream behaviour to delegate to.
Upstream's own remaining methods (``mark_engine_connection_stale``,
``pop_metrics``, ``_mm_tower_named_tensors``, ``_send_lora_params``) stay in the
miles file and reach this mixin's state the normal attribute-lookup way.

Import direction: orbit may import miles freely, so the miles helpers these
bodies need are imported at module level below. The two exceptions are
``_send_to_colocated_engine`` and ``_check_peft_weight_sync_results``, which live
in the miles updater module itself -- and that module imports THIS one to name
its base class, so those two are imported at call time to break the cycle.
"""

from __future__ import annotations

import dataclasses
import logging
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
    lora_base_cpu_backup_enabled,
)
from miles.backends.megatron_utils.update_weight.common import (
    _check_weight_sync_results,
    begin_weight_update,
    end_weight_update,
    weight_update_selector,
)
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_base import (
    HfWeightIteratorBase,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group

from orbit.megatron.peft_utils import build_peft_sync_spec
from orbit.megatron.sync_metrics import (
    _barrier_with_logging,
    _log_weight_sync_event,
    _sync_type_label,
    emit_timeline_event,
    emit_update_weights_metrics,
    get_payload_tracker,
    sum_metrics_across_ranks,
)
from orbit.transport.slots import (
    MutationPurpose,
    authorize_adapter_destination,
)

logger = logging.getLogger(__name__)


class OrbitUpdateWeightExtensions:
    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        # base's LoRA-only `is_lora: bool = False` widens to orbit's PEFT method
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
        # base stores only `self.is_lora`; orbit additionally stores the resolved PEFT
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

        # orbit's PEFT sync spec (LoRA + OFT) alongside base's LoRA-only _lora_config,
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
        # record the gather-group layout so connect_rollout_engines can detect a layout
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
        # pre-initialise the engine-split state base only sets in connect_rollout_engines,
        # so update_weights (and its metrics) are safe before the first connect
        self.rollout_engines: Sequence[ActorHandle] = []
        self.distributed_rollout_engines = []
        self._all_rollout_engines = []
        self.use_distribute = False
        self._is_distributed_src_rank = False
        self._connection_stale: bool = False

    def is_rollout_engines_fresh(self) -> bool:
        # orbit pre-initialises rollout_engines to [] in __init__ (base leaves it None),
        # so freshness is "non-empty and not stale" rather than "not None".
        return bool(self.rollout_engines) and not self._connection_stale

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        # docstring reworded for the src-rank rule change below (a PEFT sync sources
        # from the DP/TP-0 rank of ANY pipeline stage, not only the global PP-0 rank)
        """
        Split colocated/distributed engines. Distributed source ranks create NCCL
        groups; colocated ranks map to their local IPC engine.
        """
        # reset the OPD teacher-slot state on (re)connect and keep the full engine list
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
        # base's offset test alone reads a non-colocated PEFT launch as colocated
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
        # hoisted so the PEFT transport hookup at the end of this method can read it
        distributed_gpu_counts = []

        if self.use_distribute:
            self.rollout_engines = list(rollout_engines[:colocate_engine_nums])
            self.distributed_rollout_engines = list(rollout_engines[colocate_engine_nums:])
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            # adapters are replicated across pipeline stages, so a PEFT sync sources
            # from every DP/TP-0 rank; base's PP-0-only rule would ship one stage's shard
            peft_or_base_pp0 = self._peft_sync_spec is not None or get_parallel_state().pp.rank == 0
            self._is_distributed_src_rank = (
                get_parallel_state().intra_dp_cp.rank == 0
                and get_parallel_state().tp.rank == 0
                and peft_or_base_pp0
            )
            self._group_name = "miles"
            # PEFT runs skip base's NCCL model-update group; adapters ship over the
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
        # clear the src-rank flag on a reconnect that drops the distributed engines
        # (base leaves the previous run's True in place)
        else:
            self._is_distributed_src_rank = False

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # base rebuilds the gather groups from a per-rank "do I have an engine" test, so
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

        # PEFT transport hookup - the adapter sync path base has no equivalent of
        # (colocated IPC vs NCCL selection, non-source ranks detach, reconnect on a mode change);
        # construction and wire format live in orbit.transport
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
            from orbit.transport import build_peft_transport
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
        # Orbit's PEFT sync-result check; the miles updater module is its only home
        # (and it imports this module, so the import has to stay call-time).
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import (
            _check_peft_weight_sync_results,
        )

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

        # instrumentation preamble - the ORBIT_LOG_WEIGHT_SYNC trace, the payload
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
            # trace + "update_start" timeline marker + pause-window clock around base's
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
            # upstream replaced the post_process_weights(restore_weights_before_load=True)
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
        # base's bare dist.barrier, wrapped so the trace brackets it (same group, same
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

        # an orbit PEFT run (LoRA or OFT) never pushes the frozen base; the base pass
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

        # upstream's LoRA-only adapter sync (_send_lora_params + _pp_assemble_full_adapter)
        # is superseded by orbit's PEFT transport, which serves LoRA and OFT alike. Chunks are bound
        # first so the per-chunk adapter shards can be coalesced into the single load sglang requires.
        if self._peft_sync_spec is not None:
            weight_chunks = self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights, weight_type="lora"
            )
            if self._peft_sync_spec.method == "lora":
                from orbit.transport._gather import (
                    coalesce_lora_hf_weight_chunks,
                )

                source_chunk_count, weight_chunks = coalesce_lora_hf_weight_chunks(weight_chunks)
                if source_chunk_count > 1 and rank == 0:
                    logger.info(
                        "Coalesced %d LoRA weight chunks into one adapter load.",
                        source_chunk_count,
                    )
            if self._peft_sync_spec.method == "oft":
                from orbit.transport._gather import (
                    coalesce_oft_hf_weight_chunks,
                )

                source_chunk_count, weight_chunks = coalesce_oft_hf_weight_chunks(weight_chunks)
                if source_chunk_count > 1 and rank == 0:
                    logger.info(
                        "Coalesced %d OFT weight chunks into one adapter load.",
                        source_chunk_count,
                    )

            for hf_named_tensors in weight_chunks:
                # the home mixin's transport send, which may already hold completed results
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

            # base's LoRA-only zero-chunk guard now covers every PEFT method, plus a
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
            # trace + "update_end" timeline marker + pause-window close around base's
            # post-process/continue dispatch, which now addresses lifecycle_rollout_engines
            _log_weight_sync_event(
                "post_process_and_continue_dispatch",
                rank=rank,
                world_size=world_size,
                weight_version=self.weight_version,
                rollout_engine_count=len(lifecycle_rollout_engines),
            )
            # upstream replaced post_process_weights(post_process_quantization=True)
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
        # base's update ends at the barrier; orbit adds the cross-rank metric reduction
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

    # base's LoRA branch (weight filtering, lora_config/lora_name/lora_loaded kwargs,
    # the "not supported for distributed engines" raise) is gone - adapters never reach this method
    # any more, they go through the transport send below; the rest is base's own send path
    # with the ORBIT_LOG_WEIGHT_SYNC trace threaded through it
    def _send_base_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        # Upstream's colocated send helper; the miles updater module is its only home
        # (and it imports this module, so the import has to stay call-time).
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import (
            _send_to_colocated_engine,
        )

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

    def _sync_mode_label(self) -> str:
        """Human-readable sync mode for metrics/timeline markers.

        "full" for base-model sync, "adapter_double_buffer" when the PEFT
        transport stages into an inactive slot, "adapter_single_slot"
        otherwise. Only consulted for rank-0 emission; on distributed non-src
        PEFT ranks (transport is None) the label falls back to single-slot.
        """
        if self._peft_sync_spec is None:
            return "full"
        runtime_mode = getattr(self._peft_transport, "runtime_mode", None)
        if getattr(runtime_mode, "adapter_double_buffer", False):
            return "adapter_double_buffer"
        return "adapter_single_slot"

    def _send_adapter_params(
        self,
        hf_named_tensors,
        adapter_name: str | None = None,
        *,
        purpose: MutationPurpose = MutationPurpose.STUDENT_SYNC,
    ) -> tuple[list[ObjectRef], Any, list[Any] | None]:
        authorized_name = authorize_adapter_destination(
            self._peft_args,
            requested_name=adapter_name,
            purpose=purpose,
        )
        if self.use_distribute and not self._is_distributed_src_rank:
            return [], [], []
        if self._peft_transport is None:
            raise RuntimeError("_send_adapter_params called without a PEFT transport")

        transport = self._peft_transport
        original_sync_spec = transport.sync_spec
        transport.sync_spec = dataclasses.replace(
            original_sync_spec,
            adapter_name=authorized_name,
        )
        # The IPC/Ray LoRA transport unloads the previous adapter before
        # reloading, gated on transport._peft_loaded — which tracks the STUDENT
        # slot. Point it at the teacher slot's own loaded-state so the first
        # promotion does not unload an orbit_teacher that is not there yet.
        # (NcclBackend has no _peft_loaded and stages/activates instead, so the
        # getattr returns None and this is a no-op there.)
        student_peft_loaded = (
            getattr(transport, "_peft_loaded", None)
            if purpose is MutationPurpose.LEGACY_SELF_TEACHER_PROMOTION
            else None
        )
        if student_peft_loaded is not None:
            transport._peft_loaded = self._teacher_slot_loaded
        try:
            send_result = transport.send_adapter(
                hf_named_tensors,
                weight_version=self.weight_version,
            )
        finally:
            if student_peft_loaded is not None:
                self._teacher_slot_loaded = transport._peft_loaded
                transport._peft_loaded = student_peft_loaded
            transport.sync_spec = original_sync_spec
        return send_result.refs, [], send_result.results

    def push_teacher_adapter(self) -> None:
        """One-shot push of the CURRENT adapter params to the orbit_teacher slot.

        Mirrors ``update_weights``' gather/coalesce/send for a single sync but
        targets the reserved teacher slot instead of the student adapter, and
        skips the pause/flush/continue-generation lifecycle (this fills an
        inactive scoring slot, not the live generation adapter). The caller
        (``actor._promote_self_teacher``) has swapped the EMA/lag buffer into
        the live adapter params, so the existing Megatron->HF adapter export
        picks up the teacher tensors unchanged.
        """
        # Orbit's PEFT sync-result check; the miles updater module is its only home
        # (and it imports this module, so the import has to stay call-time).
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import (
            _check_peft_weight_sync_results,
        )
        from orbit.opd.opd_teacher_spec import OPD_TEACHER_ADAPTER_NAME

        authorized_name = authorize_adapter_destination(
            self._peft_args,
            requested_name=OPD_TEACHER_ADAPTER_NAME,
            purpose=MutationPurpose.LEGACY_SELF_TEACHER_PROMOTION,
        )
        if self._peft_sync_spec is None:
            raise RuntimeError("push_teacher_adapter requires a PEFT (LoRA/OFT) run.")

        megatron_local_weights = self.weights_getter()
        weight_chunks = self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights)
        if self._peft_sync_spec.method == "lora":
            from orbit.transport._gather import (
                coalesce_lora_hf_weight_chunks,
            )
            _source_chunk_count, weight_chunks = coalesce_lora_hf_weight_chunks(weight_chunks)
        elif self._peft_sync_spec.method == "oft":
            from orbit.transport._gather import (
                coalesce_oft_hf_weight_chunks,
            )
            _source_chunk_count, weight_chunks = coalesce_oft_hf_weight_chunks(weight_chunks)

        sync_chunk_count = 0
        for hf_named_tensors in weight_chunks:
            refs, _long_lived, completed_results = self._send_adapter_params(
                hf_named_tensors,
                adapter_name=authorized_name,
                purpose=MutationPurpose.LEGACY_SELF_TEACHER_PROMOTION,
            )
            results = completed_results if completed_results is not None else ray.get(refs)
            _check_peft_weight_sync_results(results, sync_type=_sync_type_label(self.peft_method))
            sync_chunk_count += 1

        if sync_chunk_count == 0:
            raise RuntimeError(
                f"{self._peft_sync_spec.method.upper()} teacher promotion failed: the weight "
                "iterator produced zero chunks; the orbit_teacher slot was not filled."
            )


__all__ = ["OrbitUpdateWeightExtensions"]
