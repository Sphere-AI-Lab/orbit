"""One-trunk PPO critic (--critic-mode adapter): build/alias helpers.

The adapter-mode critic is a normal PEFT model (adapters + scalar value head)
whose frozen trunk parameters alias the actor's tensors, so PPO pays for no
second trunk copy. Frozen params are trunk by definition of PEFT; trainable
params (adapters, value head) are role-owned and never aliased. See
docs/superpowers/specs/2026-07-27-one-trunk-ppo-design.md (clthegoat docs).
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_OPTIMIZER_PARAMETER_STATE_PREFIX = "optimizer_parameter_state_rank"


def _named_params(model) -> dict[str, torch.nn.Parameter]:
    params = {}
    for chunk_id, chunk in enumerate(model):
        for name, param in chunk.named_parameters():
            params[f"{chunk_id}:{name}"] = param
    return params


def alias_trunk_storage(critic_model, actor_model) -> int:
    """Re-point every frozen critic parameter at the actor's tensor.

    Returns the number of aliased parameters. Fails loud on any mismatch: the
    two instances are built from the same args on the same parallel layout, so
    every frozen critic param must exist in the actor with an identical shape.
    """
    actor_params = _named_params(actor_model)
    aliased = 0
    for name, critic_param in _named_params(critic_model).items():
        if critic_param.requires_grad:
            continue
        actor_param = actor_params.get(name)
        if actor_param is None:
            raise RuntimeError(f"trunk alias: {name} missing from actor model")
        if actor_param.shape != critic_param.shape:
            raise RuntimeError(
                f"trunk alias: {name} shape mismatch "
                f"actor={tuple(actor_param.shape)} critic={tuple(critic_param.shape)}"
            )
        critic_param.data = actor_param.data
        aliased += 1
    if aliased == 0:
        raise RuntimeError("trunk alias: no frozen critic parameters found; is PEFT enabled?")
    return aliased


def assert_trunk_aliased(critic_model, actor_model) -> None:
    """Guard that no frozen critic param silently materialized its own storage."""
    actor_params = _named_params(actor_model)
    for name, critic_param in _named_params(critic_model).items():
        if critic_param.requires_grad:
            continue
        if critic_param.data.data_ptr() != actor_params[name].data.data_ptr():
            raise RuntimeError(f"trunk alias broken: {name} has its own storage")


def _trainable_named_tensors(model) -> dict[str, torch.nn.Parameter]:
    return {name: p for name, p in _named_params(model).items() if p.requires_grad}


def _critic_checkpoint_dir(root: str, iteration: int) -> Path:
    return Path(root) / f"iter_{iteration:07d}"


def _global_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _contains_tensor(value: Any) -> bool:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, torch.Tensor):
            return True
        if type(item) is dict:
            pending.extend(item.values())
        elif type(item) in (list, tuple):
            pending.extend(item)
    return False


def _contains_inline_optimizer_tensor(value: Any) -> bool:
    """Match Megatron's split optimizer-checkpoint convention.

    DistributedOptimizer.state_dict() deliberately excludes parameter-dependent
    tensors (main parameters and moments); those must be persisted through
    save_parameter_state()/load_parameter_state(). Other Megatron optimizers
    carry those tensors inline in their state dict.
    """
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if "state" in item and _contains_tensor(item["state"]):
                return True
            if "optimizer" in item:
                pending.append(item["optimizer"])
            if "fp32_from_fp16_params" in item and _contains_tensor(item["fp32_from_fp16_params"]):
                return True
        elif type(item) in (list, tuple):
            pending.extend(item)
    return False


def _contains_megatron_optimizer_wrapper(value: Any) -> bool:
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if "optimizer" in item:
                return True
            pending.extend(item.values())
        elif type(item) in (list, tuple):
            pending.extend(item)
    return False


def _uses_external_parameter_state(optimizer_state: Any, transfer_fn: Any) -> bool:
    # The wrapper-key check distinguishes DistributedOptimizer's split state
    # dict from an unstepped plain/FP32 optimizer whose empty ``state`` also
    # contains no tensors. ChainedOptimizer exposes the transfer methods even
    # when its sole child is not distributed.
    return (
        not _contains_inline_optimizer_tensor(optimizer_state)
        and _contains_megatron_optimizer_wrapper(optimizer_state)
        and callable(transfer_fn)
    )


def _optimizer_parameter_state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"{_OPTIMIZER_PARAMETER_STATE_PREFIX}{_global_rank()}.pt"


def _reload_optimizer_model_params(optimizer) -> None:
    """Synchronize optimizer-owned main parameters after direct model copies."""
    reload_model_params = getattr(optimizer, "reload_model_params", None)
    if callable(reload_model_params):
        reload_model_params()


def save_critic_checkpoint(
    args,
    iteration: int,
    critic_model,
    optimizer=None,
    opt_param_scheduler=None,
) -> str:
    """Save the adapter+value-head critic per rank (trainable tensors + optimizer state).

    Files are tagged by global rank; loading requires the same world layout.
    """
    save_root = getattr(args, "critic_save", None)
    if not save_root:
        raise ValueError("critic_save is required to save an adapter critic checkpoint")

    ckpt_dir = _critic_checkpoint_dir(save_root, iteration)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer_state = None
    optimizer_parameter_state = False
    scheduler_state = None
    parameter_state_path = _optimizer_parameter_state_path(ckpt_dir)
    if optimizer is not None and not getattr(args, "no_save_optim", False):
        optimizer_state = optimizer.state_dict()
        save_parameter_state = getattr(optimizer, "save_parameter_state", None)
        optimizer_parameter_state = _uses_external_parameter_state(optimizer_state, save_parameter_state)
        if optimizer_parameter_state:
            save_parameter_state(str(parameter_state_path))
        if opt_param_scheduler is not None:
            scheduler_state = opt_param_scheduler.state_dict()
    if not optimizer_parameter_state:
        # A repeated save at the same iteration must not retain stale optimizer
        # shards when --no-save-optim (or an inline-state optimizer) is used.
        parameter_state_path.unlink(missing_ok=True)

    payload = {
        "tensors": {k: v.detach().cpu() for k, v in _trainable_named_tensors(critic_model).items()},
        "optimizer": optimizer_state,
        "optimizer_parameter_state": optimizer_parameter_state,
        "opt_param_scheduler": scheduler_state,
        "iteration": iteration,
    }
    torch.save(payload, ckpt_dir / f"critic_rank{_global_rank()}.pt")
    if dist.is_initialized():
        dist.barrier()
    if _global_rank() == 0:
        (Path(save_root) / "latest_checkpointed_iteration.txt").write_text(str(iteration))
    return str(ckpt_dir)


def load_critic_checkpoint(args, critic_model, optimizer=None, opt_param_scheduler=None) -> int | None:
    """Restore trainable critic tensors saved by save_critic_checkpoint.

    Returns None on fresh start (no checkpoint), else the loaded iteration.
    Never touches frozen params, so a load can never materialize a trunk copy.
    """
    load_root = getattr(args, "critic_load", None)
    if not load_root:
        return None
    latest_path = Path(load_root) / "latest_checkpointed_iteration.txt"
    if not latest_path.is_file():
        raise FileNotFoundError(f"--critic-load does not contain a critic checkpoint marker: {latest_path}")
    iteration = int(latest_path.read_text().strip())
    checkpoint_dir = _critic_checkpoint_dir(load_root, iteration)
    path = checkpoint_dir / f"critic_rank{_global_rank()}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("iteration") != iteration:
        raise RuntimeError(
            f"critic checkpoint iteration mismatch: marker={iteration} payload={payload.get('iteration')}"
        )
    params = _trainable_named_tensors(critic_model)
    saved = payload["tensors"]
    if set(saved) != set(params):
        raise RuntimeError(
            "critic checkpoint mismatch: "
            f"missing={sorted(set(params) - set(saved))} extra={sorted(set(saved) - set(params))}"
        )
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(saved[name].to(device=param.device, dtype=param.dtype))

    if optimizer is not None:
        # The tensors above were copied after optimizer construction. Refresh
        # optimizer-owned FP32/main parameters before optionally replacing them
        # with their higher-precision checkpointed values below.
        _reload_optimizer_model_params(optimizer)

        optimizer_state = payload.get("optimizer")
        if optimizer_state is None:
            raise RuntimeError(
                "critic checkpoint has no optimizer state (it may have been saved with --no-save-optim); "
                "training resume is not possible"
            )

        load_parameter_state = getattr(optimizer, "load_parameter_state", None)
        external_parameter_state = payload.get("optimizer_parameter_state") is True
        requires_external_parameter_state = _uses_external_parameter_state(optimizer_state, load_parameter_state)
        if requires_external_parameter_state and not external_parameter_state:
            raise RuntimeError("critic checkpoint is missing distributed optimizer parameter state")
        if external_parameter_state:
            parameter_state_path = _optimizer_parameter_state_path(checkpoint_dir)
            if not callable(load_parameter_state):
                raise RuntimeError("critic checkpoint requires distributed optimizer parameter state")

        optimizer.load_state_dict(optimizer_state)
        if external_parameter_state:
            # Megatron writes/reads this file only on DP rank zero; other ranks
            # still participate in load_parameter_state's collectives and do
            # not have a local file to pre-validate.
            try:
                load_parameter_state(str(parameter_state_path))
            except FileNotFoundError as exc:
                raise RuntimeError(f"critic optimizer parameter state is missing: {parameter_state_path}") from exc

    if opt_param_scheduler is not None:
        scheduler_state = payload.get("opt_param_scheduler")
        if scheduler_state is None:
            raise RuntimeError("critic checkpoint has no optimizer scheduler state; training resume is not possible")
        opt_param_scheduler.load_state_dict(scheduler_state)
    return iteration


def _check_resume_iteration(
    loaded: int | None,
    expected: int | None,
    *,
    require_checkpoint: bool = False,
) -> None:
    """Fail loud if the critic and actor resumed at different iterations.

    Raises when an actor resume requires critic state but none was loaded, and
    when both iterations are known but disagree. Otherwise unknown state means
    a fresh critic start.
    """
    if loaded is None:
        if require_checkpoint:
            raise RuntimeError(
                "actor resumed from a PEFT checkpoint but no matching adapter critic checkpoint was loaded; "
                "set --critic-load to the corresponding critic checkpoint root"
            )
        return
    if expected is None:
        return
    if loaded != expected:
        raise RuntimeError(
            f"critic/actor checkpoint iteration mismatch: critic resumed at iteration {loaded}, "
            f"actor resumed at iteration {expected}"
        )


@contextmanager
def _critic_build_args(args):
    """Temporarily rewrite the global args the way the separate critic worker does at
    init (actor.py role=="critic" branch), except `load`: the trunk arrives via
    aliasing and the adapters/head resume through load_critic_checkpoint, so the
    Megatron trunk-checkpoint load is skipped entirely.
    """
    saved = {key: getattr(args, key) for key in ("load", "save", "lr", "lr_warmup_iters")}
    args.load = None
    args.save = args.critic_save
    args.lr = args.critic_lr
    args.lr_warmup_iters = args.critic_lr_warmup_iters
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(args, key, value)


def build_critic_instance(args, actor_model, expected_iteration: int | None = None):
    """Build the one-trunk critic: PEFT model + value head, trunk aliased to the actor.

    Known V1 cost: the bridge build loads base weights before aliasing frees
    them, so init transiently holds a second trunk until clear_memory().

    ``expected_iteration``, when given, must match the critic's resumed
    iteration (see ``_check_resume_iteration``) so the actor and critic never
    silently train from different points in the run.
    """
    from .model import clear_memory, initialize_model_and_optimizer

    with _critic_build_args(args):
        model, optimizer, opt_param_scheduler, _ = initialize_model_and_optimizer(args, role="critic")
    aliased = alias_trunk_storage(model, actor_model)
    clear_memory()
    resumed_iteration = load_critic_checkpoint(
        args,
        model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
    )
    _check_resume_iteration(
        resumed_iteration,
        expected_iteration,
        require_checkpoint=getattr(args, "_peft_resume_adapter_dir", None) is not None,
    )
    assert_trunk_aliased(model, actor_model)
    logger.info("adapter critic ready: %d trunk params aliased, resumed_iteration=%s", aliased, resumed_iteration)
    return model, optimizer, opt_param_scheduler


@contextmanager
def value_loss_phase(args):
    """Route train() to the value loss for the critic phase, restoring afterwards.

    train() reads loss_type from the global Megatron args (model.py train() ->
    get_args()), which is the same Namespace the actor holds, so a scoped
    mutation is the faithful in-process equivalent of train_critic's assignment.
    """
    saved = args.loss_type
    args.loss_type = "value_loss"
    try:
        yield
    finally:
        args.loss_type = saved
