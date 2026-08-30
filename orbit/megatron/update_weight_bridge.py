"""Bridge-aware disaggregated weight sync (design doc
docs/plans/2026-07-07-bridge-aware-disagg-weight-sync.md).

The name-based ``UpdateWeightFromDistributed`` converts per-param via the
``megatron_to_hf`` name dispatch, which has no entry for bridge-loaded models
(Nemotron-H, Gemma-4) — hence the historical ``--colocate`` requirement.
Here the megatron-bridge export produces the ``(hf_name, tensor)`` stream
instead: ``AutoBridge.export_hf_weights`` is a collective across TP/PP/EP
that yields FULL tensors on every rank (already gathered), so

- every training rank drains the chunk iterator in lockstep (the export's
  internal collectives require all ranks to participate), and
- only global rank 0 holds the NCCL group with the engines and broadcasts
  each chunk. No per-PP-stage source groups: PP is gathered inside the
  export.

v1 scope: full finetuning only (PEFT routes to ``UpdateWeightFromTensor``
unconditionally), no quantized checkpoints, ``broadcast`` transfer mode.
"""

import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from tqdm import tqdm

from miles.utils.distributed_utils import get_gloo_group

from miles.backends.megatron_utils.update_weight.hf_weight_iterator_base import HfWeightIteratorBase
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin import DistBucketedWeightUpdateMixin


class UpdateWeightFromDistributedBridge(DistBucketedWeightUpdateMixin):
    """Disaggregated NCCL weight sync fed by the megatron-bridge HF export."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict | None,
        is_lora: bool = False,
    ) -> None:
        if is_lora:
            raise ValueError(
                "UpdateWeightFromDistributedBridge does not support PEFT adapters; "
                "PEFT weight sync routes through UpdateWeightFromTensor."
            )
        if quantization_config is not None:
            raise ValueError(
                "UpdateWeightFromDistributedBridge does not support quantized checkpoints "
                "(the bridge weight iterator has no quantization support)."
            )
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._group_name = "orbit-bridge-sync"
        self._model_update_groups = None
        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
            peft_method="none",
        )

    @property
    def _is_source(self) -> bool:
        """Single global source: the bridge export yields full tensors everywhere."""
        return dist.get_rank() == 0

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        if self._is_source:
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args, self._group_name, self._model_update_groups, self.rollout_engines
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self._group_name, rollout_engines, engine_gpu_counts
            )

    def _update_weight_implementation(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Lock -> broadcast one chunk -> unlock (same shape as the name-based path)."""
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        refs = update_weights_from_distributed(
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_engines,
            converted_named_tensors,
        )
        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())
        if pbar:
            pbar.update(1)

    @torch.no_grad()
    def update_weights(self) -> None:
        """Pause -> drain bridge export chunks (all ranks) -> broadcast (source) -> resume."""
        self.weight_version += 1

        self._pause_and_prepare_engines()
        dist.barrier(group=get_gloo_group())

        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_source else None

        megatron_local_weights = self.weights_getter()
        for chunk in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            # Every rank must consume every chunk — the export's collectives
            # run inside the generator. Only the source ships it out.
            if self._is_source:
                self._update_weight_implementation(list(chunk), pbar)

        dist.barrier(group=get_gloo_group())
        self._finalize_and_resume_engines()
        dist.barrier(group=get_gloo_group())
