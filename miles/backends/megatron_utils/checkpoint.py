import logging
import os
# ORBIT-SEAM: removed base's `import re`: the legacy-checkpoint predicate below now delegates to
# orbit.megatron.low_precision_bootstrap, so this module no longer matches iter_ names itself
from pathlib import Path

import torch.distributed as dist

# ORBIT-SEAM: repo-wide comment-style pass (TODO -> Follow-up), no functional change
# Follow-up: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
# ORBIT-SEAM: base re-exports megatron's save_checkpoint as-is; orbit wraps it below, so the
# upstream function is bound under a private name here
from megatron.training.checkpointing import save_checkpoint as _save_checkpoint_megatron
from megatron.training.global_vars import get_args

from miles.utils import megatron_bridge_utils

# ORBIT-SEAM: base's .lora_utils helpers are orbit's PEFT layer (LoRA + OFT), and the orbit-added
# checkpoint subsystem (orbit.training_checkpoint marker format, rank-consensus legacy preflight,
# torch_dist/legacy/HF load dispatch, PEFT adapter-state load) lives in orbit.megatron.checkpointing
from orbit.megatron import checkpointing as orbit_checkpointing
from orbit.megatron.low_precision_bootstrap import is_distributed_checkpoint, is_legacy_megatron_checkpoint
from orbit.megatron.peft_utils import is_peft_enabled, is_peft_model, save_peft_checkpoint

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # ORBIT-SEAM: repo-wide comment-style pass (TODO -> Follow-up), no functional change
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

__all__ = ["save_checkpoint", "save_checkpoint_with_lora", "load_checkpoint"]


# ORBIT-SEAM: base imports megatron's save_checkpoint verbatim; orbit wraps it so a save Megatron
# accepted is also stamped with the orbit.training_checkpoint marker that makes it recognizable as
# a resumable training checkpoint (marker write lives in the home layer)
def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, *args, **kwargs):
    runtime_args = get_args()
    result = _save_checkpoint_megatron(iteration, model, optimizer, opt_param_scheduler, *args, **kwargs)
    orbit_checkpointing.record_training_checkpoint_marker(
        runtime_args,
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        release=bool(kwargs.get("release", False)),
    )
    return result


# ORBIT-SEAM: two orbit-only load modes added to the base signature - the critic value-model
# bootstrap, and an opt-in training-state resume (this loader also serves reference policies and
# base/converted checkpoints, so full-state restore must never be the default)
def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt, *, is_value_model: bool = False, load_training_state: bool = False):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    # ORBIT-SEAM: orbit load preamble - low-precision bridge-bootstrap validation plus the per-load
    # resume-orchestration flags the actor/model layer reads afterwards (home layer)
    orbit_checkpointing.prepare_load_checkpoint(args)

    load_path = args.load

    has_local_checkpoint_manager = "local_checkpoint_manager" in (checkpointing_context or {})
    if has_local_checkpoint_manager:
        logger.info("Skipping disk path validation: using in-memory checkpoint via local_checkpoint_manager")
    else:
        assert Path(load_path).exists() and _is_dir_nonempty(
            load_path
        ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    # ORBIT-SEAM: reject a partially written torch_dist directory before legacy detection claims it
    # (skipped for the in-memory local_checkpoint_manager path, which has no on-disk load_path)
    if not has_local_checkpoint_manager:
        orbit_checkpointing._raise_if_incomplete_direct_distributed_checkpoint(load_path)

    # ORBIT-SEAM: torch_dist sources are an orbit-added branch ahead of base's legacy check; the
    # marker-aware training-resume vs model-only dispatch lives in the home layer
    if not has_local_checkpoint_manager and is_distributed_checkpoint(load_path):
        result = orbit_checkpointing.load_distributed_checkpoint(
            args,
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
            is_value_model=is_value_model,
            load_training_state=load_training_state,
            megatron_load_checkpoint=_load_checkpoint_megatron,
        )
    # ORBIT-SEAM: base calls megatron's loader directly here; orbit first preflights the legacy
    # checkpoint across ranks (home layer) so an ambiguous source degrades to a model-only bootstrap
    # instead of a half-restored resume, then runs that same loader
    elif has_local_checkpoint_manager or _is_megatron_checkpoint(load_path):
        result = orbit_checkpointing.load_legacy_megatron_checkpoint(
            args,
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
            load_training_state=load_training_state,
            megatron_load_checkpoint=_load_checkpoint_megatron,
        )
    else:
        result = _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )

    # ORBIT-SEAM: base's LoRA-only adapter load is orbit's PEFT (LoRA + OFT) adapter load plus the
    # adapter-resume bookkeeping the actor/model layer reads; the body lives in the home layer while
    # the "is PEFT in play" predicates stay here
    result = orbit_checkpointing.load_peft_adapter_state(
        args,
        ddp_model,
        optimizer,
        opt_param_scheduler,
        result,
        peft_active=is_peft_enabled(args) and is_peft_model(ddp_model),
    )

    return result


# ORBIT-SEAM: base's LoRA-only save is orbit's PEFT save (LoRA + OFT adapters, plus the self-teacher
# sidecar threaded through save_peft_checkpoint); the base name is kept and orbit's
# save_checkpoint_with_peft aliases it below, so both call sites resolve to one function
def save_checkpoint_with_lora(iteration, model, optimizer, opt_param_scheduler, *, self_teacher=None):
    """Extended save that handles PEFT adapters separately."""
    args = get_args()

    # ORBIT-SEAM: PEFT (LoRA + OFT) model predicate replaces base's LoRA-only is_lora_model
    if is_peft_model(model):
        save_dir = Path(args.save) / f"iter_{iteration:07d}" / "adapter"
        # ORBIT-SEAM: orbit's PEFT adapter writer replaces base's save_lora_checkpoint (same call shape)
        logger.info(f"Saving PEFT checkpoint to {save_dir}")
        save_peft_checkpoint(
            model,
            args,
            str(save_dir),
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            iteration=iteration,
            # ORBIT-SEAM: self-teacher sidecar is written beside the adapter (orbit OPD self-distillation)
            self_teacher=self_teacher,
        )
    else:
        save_checkpoint(iteration, model, optimizer, opt_param_scheduler)


save_checkpoint_with_peft = save_checkpoint_with_lora


# ORBIT-SEAM: legacy-vs-torch_dist detection is shared with orbit's bootstrap layer (base's inline
# predicate is that helper's body, plus a guard for a falsy path)
def _is_megatron_checkpoint(path: str | Path) -> bool:
    return is_legacy_megatron_checkpoint(path)


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
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
