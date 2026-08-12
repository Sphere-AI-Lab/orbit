"""IPC backend — colocate path. Wraps existing _sync_peft_adapter logic from
update_weight_from_tensor.py. Wire format unchanged from today.
"""
from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterable, Sequence

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle

from orbit.backends.megatron_utils.peft_utils import PeftSyncSpec
from orbit.utils.distributed_utils import get_gloo_group

from .._gather import peft_adapter_preloaded, validate_adapter_weight_chunk
from ..interface import PeftSendResult, PeftWeightTransport
from ..registry import PeftMethodSpec
from ..runtime import PeftRuntimeMode, resolve_peft_runtime_mode


class IpcBackend(PeftWeightTransport):
    def __init__(
        self,
        *,
        args: Namespace,
        method_spec: PeftMethodSpec,
        sync_spec: PeftSyncSpec,
        ipc_gather_group,        # torch.distributed.ProcessGroup
        ipc_gather_src: int,
        runtime_mode: PeftRuntimeMode | None = None,
    ) -> None:
        self.args = args
        self.method_spec = method_spec
        self.sync_spec = sync_spec
        self.ipc_gather_group = ipc_gather_group
        self.ipc_gather_src = ipc_gather_src
        self._engines: Sequence[ActorHandle] = ()
        self._lock: ActorHandle | None = None
        self._peft_loaded = peft_adapter_preloaded(args, sync_spec.method)
        # Fallback resolves the mode when the backend is constructed directly in tests;
        # production code always supplies runtime_mode via build_peft_transport().
        self.runtime_mode = runtime_mode or resolve_peft_runtime_mode(args, use_distribute=False)

    def connect(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
    ) -> None:
        # IPC path needs only the engines + lock; gather group was set at construct.
        self._engines = rollout_engines
        self._lock = rollout_engine_lock
        # engine_gpu_counts is unused in the IPC path (no NCCL group sizing).

    def send_adapter(
        self,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        weight_version: int,
    ) -> PeftSendResult:
        weight_tensors = validate_adapter_weight_chunk(named_tensors, self.method_spec)
        rank = dist.get_rank()
        world_size = dist.get_world_size(self.ipc_gather_group)
        is_src = rank == self.ipc_gather_src

        if self.method_spec.payload_shaper is not None:
            # The registry holds the shaper; use it.
            payload = self.method_spec.payload_shaper(weight_tensors)
            # CUDA IPC handle reconstruction needs pidfd_getfd, which is blocked
            # by some schedulers' security profiles. Gather small CPU adapter
            # payloads instead, then let the SGLang parent actor serialize each
            # TP shard for its own scheduler child.
            rank_payload = (
                payload.flat_tensor.detach().to(device="cpu", copy=True).contiguous(),
                payload.metadata,
                payload.extra["entries"],
            )
            gathered = [None] * world_size if is_src else None
            dist.gather_object(
                rank_payload,
                object_gather_list=gathered,
                dst=self.ipc_gather_src,
                group=self.ipc_gather_group,
            )
            source_record = None
            source_error: Exception | None = None
            if is_src:
                try:
                    engine = self._engines[0]
                    load_ref = engine.update_oft_adapter_from_rank_tensors.remote(
                        rank_payloads=gathered,
                        load_format=self.method_spec.sglang_load_format,
                        adapter_config=self.sync_spec.adapter_config,
                        adapter_name=self.sync_spec.adapter_name,
                    )
                    send_result = self._record_weight_version_after_load(
                        engine, load_ref, weight_version
                    )
                    source_record = {
                        "source_rank": rank,
                        "results": send_result.results,
                        "error": None,
                    }
                except Exception as exc:  # noqa: BLE001 -- synchronized below
                    source_error = exc
                    source_record = {
                        "source_rank": rank,
                        "results": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

            # Local gather groups can represent separate colocated engines, so
            # only the global Gloo group can make source RPC failures and
            # completed SGLang results visible to every trainer rank.
            gloo_group = get_gloo_group()
            source_records = [None] * dist.get_world_size(gloo_group)
            dist.all_gather_object(source_records, source_record, group=gloo_group)

            error_record = next(
                (
                    record
                    for record in source_records
                    if record is not None and record["error"] is not None
                ),
                None,
            )
            if error_record is not None:
                message = (
                    "OFT IPC adapter dispatch failed on source rank "
                    f"{error_record['source_rank']}: {error_record['error']}"
                )
                if source_error is not None and error_record["source_rank"] == rank:
                    raise RuntimeError(message) from source_error
                raise RuntimeError(message)

            results = [
                result
                for record in source_records
                if record is not None
                for result in record["results"]
            ]
            return PeftSendResult(refs=[], results=results)

        # LoRA path: unload existing adapter before loading new weights so SGLang
        # doesn't layer new tensors on top of stale state.
        #
        # CPU copies through the engine actor's from_ray_tensors path -- never
        # CUDA IPC handles, and never trainer-side CPU serialization. Both were
        # measured failures on B200 (2026-08-04 smokes):
        #   - CUDA handles: the scheduler's cross-device cudaIpcOpenMemHandle
        #     fails with "invalid argument" -- deterministically for raw
        #     Megatron param-buffer views, and still intermittently for fresh
        #     clones (one engine in four at the first smoke push).
        #   - Trainer-side CPU serialize: ForkingPickler ships storages as
        #     multiprocessing resource-sharer fds, redeemable only by the
        #     serializer's descendants; the scheduler is not one
        #     (AuthenticationError: digest sent was rejected).
        # The engine actor IS the server process's parent, so it re-serializes
        # legitimately -- the same reasoning, and the same receive path, as the
        # distributed RayObjectBackend.
        send_result: PeftSendResult | None = None
        load_error: Exception | None = None
        if is_src:
            cpu_tensors = {
                name: tensor.detach().to(device="cpu", copy=True).contiguous()
                for name, tensor in weight_tensors
            }
            engine = self._engines[0]
            try:
                if self._peft_loaded:
                    ray.get(engine.unload_lora_adapter.remote(lora_name=self.sync_spec.adapter_name))
                load_ref = engine.load_lora_adapter_from_ray_tensors.remote(
                    lora_name=self.sync_spec.adapter_name,
                    tensors=cpu_tensors,
                    config_dict=self.sync_spec.adapter_config,
                )
                self._peft_loaded = True
                send_result = self._record_weight_version_after_load(engine, load_ref, weight_version)
            except Exception as exc:  # noqa: BLE001  -- re-raised after the barrier
                load_error = exc
        # The old gather_object was a rendezvous as well as a (redundant, only
        # gathered[0] was read) data move. Peers returning before the src rank
        # finishes its RPCs is a timing change this fix must not smuggle in,
        # so the collective stays as a barrier.
        dist.barrier(group=self.ipc_gather_group)
        if load_error is not None:
            raise load_error
        if send_result is not None:
            return send_result
        return PeftSendResult(refs=[])

    def _record_weight_version_after_load(self, engine, load_ref: ObjectRef, weight_version: int) -> PeftSendResult:
        """Wait for the IPC adapter load, then propagate weight_version to SGLang.

        Returns both refs and concatenated results so the caller's downstream
        validation (`_check_weight_sync_results`) covers both operations and
        does not re-issue `ray.get(refs)`.
        """
        load_results = ray.get([load_ref])
        version_ref = engine.update_weight_version.remote(weight_version=str(weight_version))
        version_results = ray.get([version_ref])
        return PeftSendResult(refs=[load_ref, version_ref], results=load_results + version_results)

    def disconnect(self) -> None:
        self._engines = ()
        self._lock = None
