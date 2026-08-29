"""NCCL backend — distributed path. Mirrors UpdateWeightFromDistributed
(miles/backends/megatron_utils/update_weight/update_weight_from_distributed/broadcast.py)
for adapter-only transport.
"""
from __future__ import annotations

import time
from argparse import Namespace
from collections.abc import Iterable, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from orbit.megatron.peft_utils import PeftSyncSpec
from orbit.megatron.sync_metrics import get_payload_tracker

from orbit.transport.interface import PeftSendResult, PeftWeightTransport
from orbit.transport.registry import PeftMethodSpec
from orbit.transport._gather import validate_adapter_weight_chunk
from orbit.transport.runtime import PeftRuntimeMode, resolve_peft_runtime_mode


def _flatten_meta_to_json(meta) -> dict:
    """Convert a FlattenedTensorMetadata to a JSON-safe dict.

    FlattenedTensorMetadata carries torch.dtype (not JSON-serializable) and
    torch.Size (a tuple subclass, safe but explicit). This helper converts both
    to primitive Python types so json.dumps succeeds without a custom encoder.
    """
    return {
        "name": meta.name,
        "shape": list(meta.shape),
        "dtype": str(meta.dtype).replace("torch.", ""),
        "start_idx": int(meta.start_idx),
        "end_idx": int(meta.end_idx),
        "numel": int(meta.numel),
    }


def _as_version(value) -> str | None:
    return None if value is None else str(value)


def _validate_adapter_aliases(result: dict, requested_version: str) -> None:
    adapter_version = _as_version(result.get("adapter_version"))
    weight_version = _as_version(result.get("weight_version"))
    if adapter_version is not None and weight_version is not None and adapter_version != weight_version:
        raise RuntimeError(
            f"SGLang returned mismatched adapter_version={adapter_version} and weight_version={weight_version}"
        )
    for field in ("adapter_version", "weight_version"):
        value = _as_version(result.get(field))
        if value is not None and value != requested_version:
            raise RuntimeError(f"SGLang returned {field}={value}, expected {requested_version}")


def _validate_staged_results(results: list[dict], requested_version: str) -> None:
    for result in results:
        if not result.get("success", False):
            error = result.get("error")
            in_flight = result.get("in_flight_requests_on_retiring_slot")
            if error == "inactive_slot_busy":
                raise RuntimeError(
                    f"NCCL double-buffer: SGLang reports inactive slot busy "
                    f"(error={error!r}, in_flight_requests={in_flight!r}). "
                    "Previous adapter version is still draining; reduce rollout "
                    "concurrency or wait longer between adapter updates."
                )
            raise RuntimeError(f"SGLang adapter update failed: {result}")
        _validate_adapter_aliases(result, requested_version)
        staged_raw = result.get("staged_adapter_version",
                                result.get("adapter_version",
                                           result.get("weight_version")))
        staged = _as_version(staged_raw)
        if staged is None:
            raise RuntimeError(
                f"SGLang response missing staged_adapter_version (and fallback fields); "
                f"SGLang version may be incompatible. Full result: {result}"
            )
        if staged != requested_version:
            raise RuntimeError(
                f"SGLang staged_adapter_version={staged}, expected {requested_version}"
            )


def _validate_active_results(results: list[dict], requested_version: str) -> None:
    for result in results:
        if not result.get("success", False):
            raise RuntimeError(f"SGLang adapter activation failed: {result}")
        active = _as_version(result.get("active_adapter_version"))
        if active != requested_version:
            raise RuntimeError(f"SGLang active_adapter_version={active}, expected {requested_version}")


def connect_rollout_engines_from_distributed(args, group_name, rollout_engines, engine_gpu_counts=None):
    """Lazy-imported wrapper — kept at module level so tests can monkeypatch the name."""
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
        connect_rollout_engines_from_distributed as _impl,
    )
    return _impl(args, group_name, rollout_engines, engine_gpu_counts)


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_group, rollout_engines):
    """Lazy-imported wrapper — kept at module level so tests can monkeypatch the name."""
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
        disconnect_rollout_engines_from_distributed as _impl,
    )
    return _impl(args, group_name, model_update_group, rollout_engines)


class NcclBackend(PeftWeightTransport):
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
        self._engines: Sequence[ActorHandle] = ()
        self._lock: ActorHandle | None = None
        self._group_name: str | None = None
        self._model_update_group = None
        # Fallback resolves the mode when the backend is constructed directly in tests;
        # production code always supplies runtime_mode via build_peft_transport().
        self.runtime_mode = runtime_mode or resolve_peft_runtime_mode(args, use_distribute=True)

    def connect(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
    ) -> None:
        if self._model_update_group is not None:
            self.disconnect()
        self._engines = rollout_engines
        self._lock = rollout_engine_lock
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        self._group_name = f"orbit-peft-pp_{pp_rank}"
        self._model_update_group = connect_rollout_engines_from_distributed(
            self.args, self._group_name, rollout_engines, engine_gpu_counts
        )

    def send_adapter(
        self,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        weight_version: int,
    ) -> PeftSendResult:
        weight_tensors = validate_adapter_weight_chunk(named_tensors, self.method_spec)
        # Acquire the lock — same pattern as broadcast.py:84-97.
        while not ray.get(self._lock.acquire.remote()):
            time.sleep(0.1)
        refs: list[ObjectRef] = []
        results = None
        try:
            payload_metadata = None
            tensors_to_broadcast: list[tuple[str, torch.Tensor]] = weight_tensors
            if self.method_spec.payload_shaper is not None:
                payload = self.method_spec.payload_shaper(weight_tensors)
                tensors_to_broadcast = [("__flattened__", payload.flat_tensor)]
                payload_metadata = {
                    "metadata": [_flatten_meta_to_json(m) for m in payload.metadata],
                    "extra": {
                        "entries": [
                            # entries is list[tuple[str, int]] — convert tuple to list
                            # so json.dumps can serialize without a custom encoder.
                            [name, int(idx)]
                            for name, idx in payload.extra["entries"]
                        ],
                    },
                }

            names, dtypes, shapes = [], [], []
            for name, tensor in tensors_to_broadcast:
                names.append(name)
                dtypes.append(str(tensor.dtype).removeprefix("torch."))
                shapes.append(list(tensor.shape))

            # Payload accounting: only the broadcast source rank runs
            # send_adapter, so the wire payload (the ONE flat tensor on the
            # shaped path) is counted exactly once per update.
            get_payload_tracker().record(tensors_to_broadcast)

            # Engines must allocate staging buffers before the broadcast begins,
            # so dispatch the metadata RPC first; broadcast and engine network
            # round-trips then overlap until ray.get(refs) at the end.
            refs = [
                engine.update_adapter_from_distributed.remote(
                    names=names,
                    dtypes=dtypes,
                    shapes=shapes,
                    group_name=self._group_name,
                    weight_version=str(weight_version),
                    adapter_version=str(weight_version),
                    double_buffer=self.runtime_mode.adapter_double_buffer,
                    load_format=self.method_spec.sglang_load_format,
                    adapter_config=self.sync_spec.adapter_config,
                    adapter_name=self.sync_spec.adapter_name,
                    payload_metadata=payload_metadata,
                )
                for engine in self._engines
            ]
            handles = [
                dist.broadcast(t.data, 0, group=self._model_update_group, async_op=True)
                for _, t in tensors_to_broadcast
            ]
            for h in handles:
                h.wait()
            results = ray.get(refs)
            requested_version = str(weight_version)
            _validate_staged_results(results, requested_version)
            if self.runtime_mode.adapter_double_buffer:
                activate_refs = [
                    engine.activate_adapter_version.remote(
                        adapter_name=self.sync_spec.adapter_name,
                        adapter_version=requested_version,
                        weight_version=requested_version,
                        load_format=self.method_spec.sglang_load_format,
                    )
                    for engine in self._engines
                ]
                activate_results = ray.get(activate_refs)
                _validate_active_results(activate_results, requested_version)
                results = results + activate_results
                # activate_refs intentionally not appended to refs; they are
                # already resolved (we awaited them via ray.get above) and the
                # PeftSendResult.refs contract is "refs the caller may need to
                # wait on." Keeping activate_results in `results` preserves the
                # validation/audit trail.
            else:
                _validate_active_results(results, requested_version)
        finally:
            ray.get(self._lock.release.remote())
        return PeftSendResult(refs=refs, results=results)

    def disconnect(self) -> None:
        if self._model_update_group is not None:
            disconnect_rollout_engines_from_distributed(
                self.args, self._group_name, self._model_update_group, self._engines
            )
            self._model_update_group = None
            self._group_name = None
        self._engines = ()
        self._lock = None
