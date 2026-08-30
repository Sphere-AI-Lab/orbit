import hashlib
import logging
import math
import os
from collections.abc import Mapping
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef

from miles.backends.megatron_utils.lora_utils import (
    is_lora_weight_name,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.lora import LORA_ADAPTER_NAME

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .common import weight_update_selector

# ORBIT-SEAM: weight-sync payload accounting (perf/update_weights_payload_*) lives in the
# orbit home (P1, Phase 3 slice 3g)
from orbit.megatron.sync_metrics import (
    get_payload_tracker,
)

# ORBIT-SEAM: orbit's updater methods live in the home mixin (P2, Phase 3 slice 3g)
from orbit.transport.update_weight_ext import OrbitUpdateWeightExtensions

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


# ORBIT-SEAM: the mixin is this class's ONLY base and carries orbit's whole updater surface
# (__init__, is_rollout_engines_fresh, connect_rollout_engines, update_weights,
# _send_base_params, _send_adapter_params, push_teacher_adapter, _sync_mode_label). Nothing
# here may redefine one of those names: a definition on this class would win the MRO and
# silently shadow orbit's. Base's remaining methods keep their names and signatures here.
class UpdateWeightFromTensor(OrbitUpdateWeightExtensions):
    """
    Update rollout engines from tensor dict:
    load(dict->GPU) -> broadcast PP/EP(GPU NCCL) -> gather TP(GPU NCCL) -> convert HF(GPU) -> send.
    Colocated: GPU->CPU serialize -> gather_object(Gloo CPU) -> Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    # TODO: avoid dup code during yueming's refactor (temp write this to avoid introducing potentially conflicting base class)
    def mark_engine_connection_stale(self) -> None:
        self._connection_stale = True

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear ``update_weight_metrics``. Empty under colocate today; kept symmetric
        with the distributed updaters so the actor can drain unconditionally."""
        out = self.__dict__.pop("update_weight_metrics", {})
        return out

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
# path passes the label in (orbit.megatron.sync_metrics._sync_type_label). Named apart from
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
