"""One-trunk PPO critic (--critic-mode adapter): build/alias helpers.

The adapter-mode critic is a normal PEFT model (adapters + scalar value head)
whose frozen trunk parameters alias the actor's tensors, so PPO pays for no
second trunk copy. Frozen params are trunk by definition of PEFT; trainable
params (adapters, value head) are role-owned and never aliased. See
docs/superpowers/specs/2026-07-27-one-trunk-ppo-design.md (clthegoat docs).
"""

import logging
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


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


def _critic_checkpoint_dir(save_root: str, iteration: int) -> Path:
    return Path(save_root) / f"iter_{iteration:07d}"


def _global_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def save_critic_checkpoint(args, iteration: int, critic_model, optimizer=None) -> str:
    """Save the adapter+value-head critic per rank (trainable tensors + optimizer state).

    Files are tagged by global rank; loading requires the same world layout.
    """
    ckpt_dir = _critic_checkpoint_dir(args.critic_save, iteration)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tensors": {k: v.detach().cpu() for k, v in _trainable_named_tensors(critic_model).items()},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "iteration": iteration,
    }
    torch.save(payload, ckpt_dir / f"critic_rank{_global_rank()}.pt")
    if dist.is_initialized():
        dist.barrier()
    if _global_rank() == 0:
        (Path(args.critic_save) / "latest_checkpointed_iteration.txt").write_text(str(iteration))
    return str(ckpt_dir)


def load_critic_checkpoint(args, critic_model, optimizer=None) -> bool:
    """Restore trainable critic tensors saved by save_critic_checkpoint.

    Returns False on fresh start (no checkpoint). Never touches frozen params,
    so a load can never materialize a trunk copy.
    """
    save_root = getattr(args, "critic_save", None)
    if not save_root or not (Path(save_root) / "latest_checkpointed_iteration.txt").is_file():
        return False
    iteration = int((Path(save_root) / "latest_checkpointed_iteration.txt").read_text().strip())
    path = _critic_checkpoint_dir(save_root, iteration) / f"critic_rank{_global_rank()}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
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
    if optimizer is not None and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return True


from contextlib import contextmanager


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


def build_critic_instance(args, actor_model):
    """Build the one-trunk critic: PEFT model + value head, trunk aliased to the actor.

    Known V1 cost: the bridge build loads base weights before aliasing frees
    them, so init transiently holds a second trunk until clear_memory().
    """
    from .model import clear_memory, initialize_model_and_optimizer

    with _critic_build_args(args):
        model, optimizer, opt_param_scheduler, _ = initialize_model_and_optimizer(args, role="critic")
    aliased = alias_trunk_storage(model, actor_model)
    clear_memory()
    resumed = load_critic_checkpoint(args, model, optimizer=optimizer)
    assert_trunk_aliased(model, actor_model)
    logger.info("adapter critic ready: %d trunk params aliased, resumed=%s", aliased, resumed)
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
