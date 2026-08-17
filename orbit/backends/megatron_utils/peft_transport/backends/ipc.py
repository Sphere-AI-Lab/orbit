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
from orbit.backends.megatron_utils.sglang import MultiprocessingSerializer
from orbit.backends.megatron_utils.update_weight.sync_metrics import get_payload_tracker

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
            # The wire format inherited from verl: outer pickle wraps inner
            # IPC-handle-bearing serialization of the flat tensor. See
            # update_weight_from_tensor.py:488-525 for provenance. The tag is
            # per-method: sglang's normalize_{oft,lora}_weight_payload asserts on
            # "flattened_oft_payload" / "flattened_lora_payload" respectively.
            inner = (
                f"flattened_{self.method_spec.name}_payload",
                MultiprocessingSerializer.serialize(payload.flat_tensor),
                payload.metadata,
                payload.extra["entries"],
            )
            serialized = MultiprocessingSerializer.serialize(inner, output_str=True)
            # Payload accounting: the engine deserializes every gather-group
            # rank's flat tensor, so each rank records its own.
            get_payload_tracker().record([payload.flat_tensor])
            gathered = [None] * world_size if is_src else None
            dist.gather_object(
                serialized,
                object_gather_list=gathered,
                dst=self.ipc_gather_src,
                group=self.ipc_gather_group,
            )
            send_result: PeftSendResult | None = None
            load_error: Exception | None = None
            if is_src:
                engine = self._engines[0]
                load_ref = engine.update_weights_from_tensor.remote(
                    serialized_named_tensors=gathered,
                    load_format=self.method_spec.sglang_load_format,
                    adapter_config=self.sync_spec.adapter_config,
                    adapter_name=self.sync_spec.adapter_name,
                )
                try:
                    send_result = self._record_weight_version_after_load(engine, load_ref, weight_version)
                except Exception as exc:
                    load_error = exc

            # Every rank serialized a CUDA IPC handle to its local flat tensor.
            # Keep those local tensors alive until the source rank has finished
            # the SGLang load/version RPCs; otherwise peer ranks can return and
            # free the storage before SGLang deserializes their handles.
            dist.barrier(group=self.ipc_gather_group)
            if load_error is not None:
                raise load_error
            if send_result is not None:
                return send_result
            return PeftSendResult(refs=[])

        # LoRA path: unload existing adapter before loading new weights so SGLang
        # doesn't layer new tensors on top of stale state.
        serialized = MultiprocessingSerializer.serialize(
            dict(weight_tensors), output_str=True
        )
        gathered = [None] * world_size if is_src else None
        dist.gather_object(
            serialized,
            object_gather_list=gathered,
            dst=self.ipc_gather_src,
            group=self.ipc_gather_group,
        )
        if is_src:
            # Payload accounting: this path loads only gathered[0] (the source
            # rank's serialized tensors) into SGLang, so only the source rank
            # records its payload.
            get_payload_tracker().record(weight_tensors)
            engine = self._engines[0]
            if self._peft_loaded:
                ray.get(engine.unload_lora_adapter.remote(lora_name=self.sync_spec.adapter_name))
            load_ref = engine.load_lora_adapter_from_tensors.remote(
                lora_name=self.sync_spec.adapter_name,
                serialized_tensors=gathered[0],
                config_dict=self.sync_spec.adapter_config,
            )
            self._peft_loaded = True
            return self._record_weight_version_after_load(engine, load_ref, weight_version)
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
