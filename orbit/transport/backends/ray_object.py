"""Ray object-store backend for distributed PEFT adapter sync.

This path is intended for adapter-only updates where robustness matters more
than NCCL throughput. It serializes CPU copies of LoRA/OFT adapter tensors and
uses the same SGLang tensor-loading endpoints as the colocated IPC backend.
"""
from __future__ import annotations

import time
from argparse import Namespace
from collections.abc import Iterable, Sequence

import ray
import torch
from ray import ObjectRef
from ray.actor import ActorHandle

from orbit.megatron.peft_utils import PeftSyncSpec
from orbit.megatron.sync_metrics import get_payload_tracker

from orbit.transport._gather import peft_adapter_preloaded, validate_adapter_weight_chunk
from orbit.transport.interface import PeftSendResult, PeftWeightTransport
from orbit.transport.registry import PeftMethodSpec
from orbit.transport.runtime import PeftRuntimeMode, resolve_peft_runtime_mode


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", copy=True).contiguous()


class RayObjectBackend(PeftWeightTransport):
    def __init__(
        self,
        *,
        args: Namespace,
        method_spec: PeftMethodSpec,
        sync_spec: PeftSyncSpec,
        runtime_mode: PeftRuntimeMode | None = None,
    ) -> None:
        self.args = args
        self.method_spec = method_spec
        self.sync_spec = sync_spec
        self.runtime_mode = runtime_mode or resolve_peft_runtime_mode(args, use_distribute=True)
        self._engines: Sequence[ActorHandle] = ()
        self._lock: ActorHandle | None = None
        self._peft_loaded = peft_adapter_preloaded(args, sync_spec.method)

    def connect(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
    ) -> None:
        self._engines = rollout_engines
        self._lock = rollout_engine_lock

    def send_adapter(
        self,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        weight_version: int,
    ) -> PeftSendResult:
        if self._lock is None:
            raise RuntimeError("RayObjectBackend.send_adapter called before connect().")

        weight_tensors = validate_adapter_weight_chunk(named_tensors, self.method_spec)
        while not ray.get(self._lock.acquire.remote()):
            time.sleep(0.1)
        try:
            if self.method_spec.payload_shaper is not None:
                return self._send_shaped_adapter(weight_tensors, weight_version)
            return self._send_unshaped_lora_adapter(weight_tensors, weight_version)
        finally:
            ray.get(self._lock.release.remote())

    def _send_unshaped_lora_adapter(
        self,
        weight_tensors: list[tuple[str, torch.Tensor]],
        weight_version: int,
    ) -> PeftSendResult:
        """Per-tensor LoRA load, for a method registered without a payload_shaper.

        Both registry entries currently define one, so this is unreachable today;
        it is kept as the fallback for any method registered without a shaper.
        """
        tensors = {name: _cpu_tensor(tensor) for name, tensor in weight_tensors}
        # Payload accounting: logical adapter payload, counted once per update
        # (per-engine fan-out is not multiplied).
        get_payload_tracker().record(list(tensors.values()))
        results: list = []
        refs: list[ObjectRef] = []

        if self._peft_loaded:
            unload_refs = [
                engine.unload_lora_adapter.remote(lora_name=self.sync_spec.adapter_name)
                for engine in self._engines
            ]
            results.extend(ray.get(unload_refs))

        load_refs = [
            engine.load_lora_adapter_from_ray_tensors.remote(
                lora_name=self.sync_spec.adapter_name,
                tensors=tensors,
                config_dict=self.sync_spec.adapter_config,
            )
            for engine in self._engines
        ]
        results.extend(ray.get(load_refs))
        refs.extend(load_refs)
        self._peft_loaded = True

        version_refs, version_results = self._update_weight_version(weight_version)
        refs.extend(version_refs)
        results.extend(version_results)
        return PeftSendResult(refs=refs, results=results)

    def _send_shaped_adapter(
        self,
        weight_tensors: list[tuple[str, torch.Tensor]],
        weight_version: int,
    ) -> PeftSendResult:
        """Flattened-payload load, taken by every method that has a shaper.

        LoRA has one too, so the wire tag must follow method_spec.name: sglang's
        normalize_lora_weight_payload asserts payload[0] == "flattened_lora_payload"
        and would reject an OFT-tagged payload outright.
        """
        payload = self.method_spec.payload_shaper(weight_tensors)
        # Payload accounting: the ONE flat tensor, counted once per update
        # (per-engine fan-out is not multiplied).
        get_payload_tracker().record([payload.flat_tensor])
        load_refs = [
            engine.update_adapter_from_ray_tensor.remote(
                flat_tensor=_cpu_tensor(payload.flat_tensor),
                metadata=payload.metadata,
                entries=payload.extra["entries"],
                payload_tag=f"flattened_{self.method_spec.name}_payload",
                load_format=self.method_spec.sglang_load_format,
                adapter_config=self.sync_spec.adapter_config,
                adapter_name=self.sync_spec.adapter_name,
            )
            for engine in self._engines
        ]
        results = list(ray.get(load_refs))
        version_refs, version_results = self._update_weight_version(weight_version)
        results.extend(version_results)
        return PeftSendResult(refs=load_refs + version_refs, results=results)

    def _update_weight_version(self, weight_version: int) -> tuple[list[ObjectRef], list]:
        version_refs = [
            engine.update_weight_version.remote(weight_version=str(weight_version))
            for engine in self._engines
        ]
        return version_refs, list(ray.get(version_refs))

    def disconnect(self) -> None:
        self._engines = ()
        self._lock = None
