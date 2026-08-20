import socket
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group, init_process_group
from miles.utils.lora import LORA_ADAPTER_NAME
from miles.utils.timer import timer

from ..common import _check_weight_sync_results
from ..hf_weight_iterator_base import HfWeightIteratorBase
from .mixin import DistBucketedWeightUpdateMixin


class UpdateWeightFromDistributed(DistBucketedWeightUpdateMixin):
    """
    Update distributed engines via NCCL. Each PP rank: group "miles-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
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
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        self.weights_getter = weights_getter
        # Bridge mode delegates Megatron -> HF export to Megatron-Bridge
        # AutoBridge. This covers model families such as Qwen3-VL whose vision
        # and merger weights are not handled by the legacy convert_to_hf table;
        # this class still owns the NCCL delivery to rollout engines.
        self._bridge_mode = args.megatron_to_hf_mode == "bridge"
        if self._bridge_mode:
            self._hf_weight_iterator = HfWeightIteratorBase.create(
                args=args,
                model=model,
                model_name=model_name,
                quantization_config=quantization_config,
                is_lora=is_lora,
            )
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale: bool = False
        self._init_lora(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
            is_lora=is_lora,
        )

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
        Create NCCL "miles-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.

        Raw mode creates one NCCL group per PP rank because each PP stage owns
        different params. Bridge mode uses a single NCCL group ``miles-bridge``
        rooted at the global source rank (PP=TP=DP=0) because
        ``AutoBridge.export_hf_weights`` already gathers across PP and yields a
        full HF weight stream.
        """
        self.rollout_engines = rollout_engines
        self._connection_stale = False
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        if self._is_source:
            if self._bridge_mode:
                self._group_name = "miles-bridge"
            else:
                pp_rank = get_parallel_state().pp.rank
                self._group_name = f"miles-pp_{pp_rank}"

        if self._is_source:
            disconnect_rollout_engines_from_distributed(
                self.args, self._group_name, self._model_update_groups, self.rollout_engines
            )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args,
                self._group_name,
                rollout_engines,
                engine_gpu_counts=engine_gpu_counts,
            )

    @property
    def _is_source(self):
        """Whether this rank broadcasts weights to rollout engines."""
        pstate = get_parallel_state()
        is_source = pstate.intra_dp_cp.rank == 0 and pstate.tp.rank == 0
        if self._bridge_mode:
            is_source = is_source and pstate.pp.rank == 0
        return is_source

    @property
    def _is_lora_source(self) -> bool:
        """The single rank holding the full adapter (DP=TP=PP=0). At PP=1 (enforced
        for LoRA) this coincides with ``_is_source``."""
        ps = get_parallel_state()
        return ps.pp.rank == 0 and ps.tp.rank == 0 and ps.intra_dp_cp.rank == 0

    def _update_weight_implementation(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Serialize NCCL broadcasts and always release the rollout lock."""
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        try:
            refs = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.rollout_engines,
                converted_named_tensors,
                selector=self._weight_update_selector,
            )
            ray.get(refs)
            converted_named_tensors.clear()
        finally:
            # Leaking this lock makes the next weight sync poll forever, so the
            # release must run after both successful and failed broadcasts.
            ray.get(self.rollout_engine_lock.release.remote())
        if pbar:
            pbar.update(1)

    @torch.no_grad()
    def update_weights(self) -> None:
        """Run the local full-weight bridge export or delegate to the synced mixin."""
        # The synced mixin owns both single- and multi-adapter LoRA updates.
        # Keep this override only for the fork's full-weight AutoBridge export.
        if self.is_lora or not self._bridge_mode:
            super().update_weights()
            return

        self.weight_version += 1

        self._pause_and_prepare_engines()
        dist.barrier(group=get_gloo_group())

        with timer("update_weights_implementation"):
            pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_source else None
            megatron_local_weights = self.weights_getter()
            is_source = self._is_source
            for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights, weight_type="base"
            ):
                if is_source:
                    self._update_weight_implementation(list(hf_named_tensors), pbar)
                # Non-source ranks still drain AutoBridge collectives.
                del hf_named_tensors
            dist.barrier(group=get_gloo_group())

        with timer("finalize_and_resume_engines"):
            self._finalize_and_resume_engines()
            dist.barrier(group=get_gloo_group())

    def _update_lora_weight_implementation(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Send adapter metadata over Ray, then broadcast the tensors (src=0).

        Reuses the base broadcast group (``self._model_update_groups`` /
        ``self._group_name``); base and adapter syncs are strictly sequential, so
        sharing the NCCL communicator is safe. No CUDA IPC, so it works across
        nodes: the engine allocates buffers from the metadata and broadcast-receives
        in order.
        """
        names = [name for name, _ in named_tensors]
        dtypes = [param.dtype for _, param in named_tensors]
        shapes = [list(param.shape) for _, param in named_tensors]

        refs = [
            engine.load_lora_adapter_from_distributed.remote(
                lora_name=LORA_ADAPTER_NAME,
                config_dict=self._lora_config,
                names=names,
                dtypes=dtypes,
                shapes=shapes,
                group_name=self._group_name,
            )
            for engine in self.rollout_engines
        ]
        contiguous_tensors = [
            param.data if param.data.is_contiguous() else param.data.contiguous() for _, param in named_tensors
        ]
        handles = [
            dist.broadcast(tensor, 0, group=self._model_update_groups, async_op=True) for tensor in contiguous_tensors
        ]
        for handle in handles:
            handle.wait()

        _check_weight_sync_results(ray.get(refs), is_lora=True)

    def _update_multi_lora_weight_implementation(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        *,
        lora_name: str,
        lora_config: dict,
    ) -> None:
        """Multi-LoRA variant of ``_update_lora_weight_implementation``: same transport, but with a
        per-adapter slot name/config and an upsert RPC (in-place insert-or-overwrite, no unload/register)."""
        names = [name for name, _ in named_tensors]
        dtypes = [param.dtype for _, param in named_tensors]
        shapes = [list(param.shape) for _, param in named_tensors]

        refs = [
            engine.load_lora_adapter_from_distributed.remote(
                lora_name=lora_name,
                config_dict=lora_config,
                names=names,
                dtypes=dtypes,
                shapes=shapes,
                group_name=self._group_name,
                upsert=True,
            )
            for engine in self.rollout_engines
        ]
        # NCCL needs contiguous buffers (lora_B slices are strided); the list keeps them alive
        # until the async broadcasts complete.
        broadcast_tensors = [param.data.contiguous() for _, param in named_tensors]
        handles = [
            dist.broadcast(tensor, 0, group=self._model_update_groups, async_op=True) for tensor in broadcast_tensors
        ]
        for handle in handles:
            handle.wait()

        _check_weight_sync_results(ray.get(refs), is_lora=True)


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1

    refs = []
    rank_cursor = 1
    for i, engine in enumerate(rollout_engines):
        refs.append(
            engine.init_weights_update_group.remote(
                master_address,
                master_port,
                rank_cursor,
                world_size,
                group_name,
                backend="nccl",
            )
        )
        rank_cursor += engine_gpu_counts[i]
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    try:
        if model_update_groups is not None:
            dist.destroy_process_group(model_update_groups)
    finally:
        ray.get(refs)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    selector: str = "all",
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    """
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            selector=selector,
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine in rollout_engines
    ]

    contiguous_tensors = [
        param.data if param.data.is_contiguous() else param.data.contiguous() for _, param in converted_named_tensors
    ]
    handles = []
    for tensor in contiguous_tensors:
        handles.append(dist.broadcast(tensor, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs
