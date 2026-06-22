import logging
import os
import re
from pathlib import Path

import torch.distributed as dist

# Follow-up: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

from orbit.utils import megatron_bridge_utils

from .low_precision_bootstrap import (
    is_distributed_checkpoint,
    is_legacy_megatron_checkpoint,
    load_dist_checkpoint,
    validate_low_precision_bootstrap_args,
)
from .peft_utils import is_peft_enabled, is_peft_model, load_peft_adapter, save_peft_checkpoint

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # Follow-up: find a less hacky way to do this.
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint", "save_checkpoint_with_peft", "load_checkpoint"]


def load_checkpoint(
    ddp_model,
    optimizer,
    opt_param_scheduler,
    checkpointing_context,
    skip_load_to_model_and_opt,
    *,
    is_value_model: bool = False,
):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    if getattr(args, "megatron_to_hf_mode", None) == "bridge":
        validate_low_precision_bootstrap_args(args)
    load_path = args.load

    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if is_distributed_checkpoint(load_path):
        result = _load_checkpoint_dist(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
            is_value_model=is_value_model,
        )
    elif _is_megatron_checkpoint(load_path):
        result = _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        result = _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )

    # Load PEFT adapter weights if available
    if is_peft_enabled(args):
        adapter_path = (
            getattr(args, "peft_adapter_path", None)
            or getattr(args, "lora_adapter_path", None)
            or getattr(args, "oft_adapter_path", None)
        )
        if adapter_path is not None:
            loaded, iteration = load_peft_adapter(
                ddp_model,
                args,
                adapter_path,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
            )
            if loaded:
                logger.info(f"Successfully loaded PEFT adapter from {adapter_path}")
                if iteration is not None:
                    result = (iteration, result[1])
            else:
                logger.warning(
                    f"PEFT is enabled and adapter_path={adapter_path} was specified, "
                    f"but adapter weights could not be loaded. "
                    f"Training will start with freshly initialized adapter weights."
                )

    return result


def save_checkpoint_with_peft(iteration, model, optimizer, opt_param_scheduler):
    """Extended save that handles PEFT adapters separately."""
    args = get_args()

    if is_peft_model(model):
        save_dir = Path(args.save) / f"iter_{iteration:07d}" / "adapter"
        logger.info(f"Saving PEFT checkpoint to {save_dir}")
        save_peft_checkpoint(
            model,
            args,
            str(save_dir),
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            iteration=iteration,
        )
    else:
        save_checkpoint(iteration, model, optimizer, opt_param_scheduler)


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return is_legacy_megatron_checkpoint(path)


def _load_checkpoint_dist(ddp_model, optimizer, args, load_path: str, *, is_value_model: bool = False):
    logger.info("Load checkpoint from Megatron distributed checkpoint (path=%s)", load_path)

    load_dist_checkpoint(ddp_model, load_path, is_value_model=is_value_model)

    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
