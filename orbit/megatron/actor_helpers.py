"""Orbit's module-level helpers for the Megatron train actor.

Home for the orbit-added module functions lifted out of
miles/backends/megatron_utils/actor.py (Phase 3 isolation, slice 3g):
weight-updater selection plus its constructor kwargs, the offload/role
validation, and the checkpoint-iteration -> start-rollout-id translation. The
miles file keeps one stamped import seam and calls these unchanged; the bodies
are verbatim apart from the call-time imports noted below.

Import direction: nothing here imports miles at module level. The four
weight-updater classes are read back off the miles actor module at call time --
that module is the single place where ``UpdateWeightP2P`` is guarded down to
``None`` on older sglang builds (no ParallelismContext), and reading them there
keeps that guard the one source of truth instead of duplicating it here. It
also keeps the identity comparison in :func:`_get_weight_updater_kwargs`
against the very class object the actor selected.

``_should_offload_frozen_base`` came out of the same block but lives next to
the offload routine it guards, in ``orbit.megatron.peft_offload``.
"""

from argparse import Namespace

from orbit.megatron.peft_utils import get_peft_method


def _get_weight_updater_kwargs(args: Namespace, update_weight_cls: type) -> dict[str, object]:
    from miles.backends.megatron_utils.actor import UpdateWeightFromTensor
    from miles.backends.megatron_utils.lora_utils import is_lora_enabled

    if update_weight_cls is UpdateWeightFromTensor:
        return {"peft_method": get_peft_method(args)}
    return {"is_lora": is_lora_enabled(args)}


def _select_update_weight_cls(args: Namespace) -> type:
    """Pick the weight updater for this run's (colocate, PEFT, transfer, to-hf) combo."""
    from miles.backends.megatron_utils.actor import (
        UpdateWeightFromDistributed,
        UpdateWeightFromDistributedBridge,
        UpdateWeightFromTensor,
        UpdateWeightP2P,
    )

    if args.colocate or get_peft_method(args) != "none":
        # PEFT (LoRA/OFT) routes through UpdateWeightFromTensor regardless
        # of colocate, so the unified PeftWeightTransport (IPC for colocate,
        # NCCL for async) is the only adapter sync path.
        return UpdateWeightFromTensor
    if args.update_weight_transfer_mode == "broadcast":
        # Bridge-loaded models (Nemotron-H, Gemma-4) have no megatron_to_hf
        # name mapping; their disaggregated sync streams the bridge export.
        if args.megatron_to_hf_mode == "bridge":
            return UpdateWeightFromDistributedBridge
        return UpdateWeightFromDistributed
    return UpdateWeightP2P


def _validate_train_offload_role(args: Namespace, role: str) -> None:
    if role == "critic" and getattr(args, "offload_train", False):
        raise NotImplementedError(
            "Megatron critic --offload-train is not supported. Critic models are full-model even when "
            "actor PEFT is enabled, so they cannot use the PEFT frozen-base offload path."
        )


def _start_rollout_id_from_checkpoint(args: Namespace, loaded_iteration: int) -> int:
    """Translate checkpoint iteration into the next rollout id.

    Bridge startup commonly loads a model-only HF/distributed base checkpoint,
    whose synthetic iteration is zero.  That is initialization, not an Orbit
    training resume, and must start at rollout zero.  The checkpoint loader sets
    ``_orbit_training_checkpoint_loaded`` only when it restored actual training
    state (a Megatron actor/critic checkpoint or PEFT training sidecar).
    """
    if getattr(args, "_orbit_training_checkpoint_loaded", False):
        return loaded_iteration + 1
    return 0


__all__ = [
    "_get_weight_updater_kwargs",
    "_select_update_weight_cls",
    "_start_rollout_id_from_checkpoint",
    "_validate_train_offload_role",
]
