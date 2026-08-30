"""Megatron optimizer + LR/WD scheduler construction for the training backends.

Lifted verbatim out of ``miles/backends/megatron_utils/model.py`` (Phase-3
slice 3f, P1 lift-out); that module re-exports ``_build_optimizer_and_scheduler``
behind a stamped ``# ORBIT-SEAM`` hook, so its call sites and
``tests/test_pion_optimizer.py`` (which reads the dispatch back off
``model._build_optimizer_and_scheduler``) are unchanged.

The orbit delta this owns is the Pion / Pion-msign dispatch: those optimizers
ship their own Megatron getters (own sharding), while Muon/Adam/SGD are
dispatched inside ``get_megatron_optimizer`` itself. It also carries orbit's
argument rename for the gloo switch (base reads ``args.enable_gloo_process_groups``,
orbit's argument set calls it ``--use-gloo-process-groups``).
"""

import dataclasses
from argparse import Namespace

from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


def _build_optimizer_and_scheduler(
    args: Namespace, model: list[DDP]
) -> tuple[MegatronOptimizer, OptimizerParamScheduler]:
    # miles owns the base scheduler builder; imported at call time so the home
    # layer keeps no module-level miles dependency (and so a caller that
    # rebinds it on the miles module still drives what this builder uses).
    from miles.backends.megatron_utils.model import get_optimizer_param_scheduler

    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None
    # Pion has its own getters (own sharding; ZeRO already disabled upstream by
    # the arguments shim). Muon/Adam/SGD are dispatched inside
    # get_megatron_optimizer, so they fall through here. Mirrors the Sphere-AI
    # pion fork's training.py dispatch.
    optimizer_type = (config.optimizer or "").lower()
    if "pion" in optimizer_type:
        if optimizer_type == "pion_msign":
            from megatron.core.optimizer.pion_msign import get_megatron_pion_ortho_exp_optimizer

            optimizer = get_megatron_pion_ortho_exp_optimizer(
                config, model, use_gloo_process_groups=args.use_gloo_process_groups
            )
        else:
            from megatron.core.optimizer.pion import get_megatron_pion_optimizer

            optimizer = get_megatron_pion_optimizer(
                config, model, use_gloo_process_groups=args.use_gloo_process_groups
            )
    else:
        optimizer = get_megatron_optimizer(
            config=config,
            model_chunks=model,
            use_gloo_process_groups=args.use_gloo_process_groups,
        )
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    return optimizer, opt_param_scheduler


# Orbit's optimizer-family classifier. Lifted out of
# ``miles/backends/megatron_utils/arguments.py`` (now byte-pristine again), where
# these predicates were dead in production -- upstream's Adam-only allow-list
# subsumed the muon/pion case, so nothing there called them. They live here
# beside the pion/muon dispatch they describe; ``tests/test_pion_optimizer.py``
# imports them, and ``model.py`` still carries its own mirrored copy.
def _is_muon_optimizer(optimizer: str | None) -> bool:
    return optimizer is not None and "muon" in optimizer.lower()


def _is_pion_optimizer(optimizer: str | None) -> bool:
    return optimizer is not None and "pion" in optimizer.lower()
