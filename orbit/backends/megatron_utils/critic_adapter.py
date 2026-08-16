"""One-trunk PPO critic (--critic-mode adapter): build/alias helpers.

The adapter-mode critic is a normal PEFT model (adapters + scalar value head)
whose frozen trunk parameters alias the actor's tensors, so PPO pays for no
second trunk copy. Frozen params are trunk by definition of PEFT; trainable
params (adapters, value head) are role-owned and never aliased. See
docs/superpowers/specs/2026-07-27-one-trunk-ppo-design.md (clthegoat docs).
"""

import logging
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from . import peft_utils

logger = logging.getLogger(__name__)

_OPTIMIZER_PARAMETER_STATE_PREFIX = "optimizer_parameter_state_rank"
_MAX_CHECKPOINT_COUNTER = 2**63 - 1


def _named_params(model) -> dict[str, torch.nn.Parameter]:
    params = {}
    for chunk_id, chunk in enumerate(model):
        for name, param in chunk.named_parameters():
            params[f"{chunk_id}:{name}"] = param
    return params


def prepare_head_critic(critic_model) -> int:
    """Freeze every critic parameter except the value head (--critic-mode head).

    The plain (non-PEFT) builder produces an all-trainable model; the head-mode
    contract is: value head trainable, everything else frozen so
    ``alias_trunk_storage`` can re-point the trunk at the actor's tensors and a
    value backward produces no trunk gradients — safe even when the actor
    full-finetunes the shared storage. The value head is the ``output_layer``
    module (the same convention ``model._iter_critic_output_layers`` keys on).

    Returns the number of parameters newly frozen.
    """
    frozen = 0
    for chunk in critic_model:
        head_params = {id(p) for name, p in chunk.named_parameters() if "output_layer" in name}
        for _name, param in chunk.named_parameters():
            if id(param) in head_params:
                continue
            if param.requires_grad:
                param.requires_grad_(False)
                frozen += 1
    return frozen


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


def _coordinated_critic_checkpoint_call(label: str, fn):
    """Coordinate rank-local failures before a later optimizer collective.

    Keep the original exception type in a single-process job for compatibility
    with the pre-distributed critic checkpoint implementation.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return fn()
    return peft_utils._coordinated_checkpoint_call(f"adapter critic {label}", fn)


def _validate_checkpoint_binding(root: str, iteration: int | None = None) -> tuple[str, int | None]:
    if iteration is not None and (type(iteration) is not int or not 0 <= iteration <= _MAX_CHECKPOINT_COUNTER):
        raise ValueError("adapter critic checkpoint iteration must be a bounded nonnegative integer")
    return str(Path(root).expanduser().resolve(strict=False)), iteration


def _require_checkpoint_binding_consensus(binding: tuple[str, int | None], *, operation: str) -> None:
    bindings = peft_utils._all_gather_checkpoint_object(binding)
    if len(set(bindings)) != 1:
        raise RuntimeError(f"adapter critic checkpoint {operation} binding differs across ranks: {bindings}")


def _read_bound_checkpoint_text(path: Path, binding) -> str:
    if binding is None:
        raise FileNotFoundError(path)
    absolute_path = peft_utils._absolute_checkpoint_path(path)
    if str(absolute_path) != binding.path:
        raise RuntimeError(f"checkpoint binding was for {binding.path}, not {absolute_path}")
    if binding.fingerprint.size > 64:
        raise RuntimeError(f"adapter critic checkpoint marker is too large: {absolute_path}")
    try:
        checkpoint_file = peft_utils._open_checkpoint_file(absolute_path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"checkpoint file disappeared after preflight: {absolute_path}") from exc
    with checkpoint_file:
        before = peft_utils._regular_file_fingerprint(os.fstat(checkpoint_file.fileno()))
        if before != binding.fingerprint:
            raise RuntimeError(f"checkpoint file changed after preflight: {absolute_path}")
        raw = checkpoint_file.read(65)
        after = peft_utils._regular_file_fingerprint(os.fstat(checkpoint_file.fileno()))
    peft_utils._verify_checkpoint_file_binding(absolute_path, binding)
    if after != binding.fingerprint or len(raw) > 64:
        raise RuntimeError(f"checkpoint file changed while it was being loaded: {absolute_path}")
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"adapter critic checkpoint marker is invalid: {absolute_path}") from exc


def _optimizer_parameter_state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"{_OPTIMIZER_PARAMETER_STATE_PREFIX}{_global_rank()}.pt"


def _reload_optimizer_model_params(optimizer) -> None:
    """Synchronize optimizer-owned main parameters after direct model copies."""
    reload_model_params = getattr(optimizer, "reload_model_params", None)
    if callable(reload_model_params):
        reload_model_params()


def _validate_critic_payload(
    payload: Any,
    *,
    iteration: int,
    critic_model,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
    optimizer_parameter_state_binding,
) -> dict[str, Any]:
    """Validate every rank-local value used before checkpoint mutation."""
    if type(payload) is not dict:
        raise RuntimeError("adapter critic checkpoint payload is invalid")
    if payload.get("iteration") != iteration:
        raise RuntimeError(
            f"critic checkpoint iteration mismatch: marker={iteration} payload={payload.get('iteration')}"
        )

    params = _trainable_named_tensors(critic_model)
    saved = payload.get("tensors")
    if not isinstance(saved, Mapping):
        raise RuntimeError("adapter critic checkpoint tensor payload is invalid")
    if set(saved) != set(params):
        raise RuntimeError(
            "critic checkpoint mismatch: "
            f"missing={sorted(set(params) - set(saved))} extra={sorted(set(saved) - set(params))}"
        )
    for name, param in params.items():
        tensor = saved[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.device.type == "meta"
            or tensor.is_quantized
            or not tensor.is_floating_point()
            or tuple(tensor.shape) != tuple(param.shape)
        ):
            raise RuntimeError(f"adapter critic checkpoint tensor {name!r} is incompatible")

    external_parameter_state = payload.get("optimizer_parameter_state", False)
    if type(external_parameter_state) is not bool:
        raise RuntimeError("adapter critic checkpoint optimizer-parameter-state marker is invalid")
    if not external_parameter_state and optimizer_parameter_state_binding is not None:
        raise RuntimeError(
            "adapter critic checkpoint has an optimizer parameter-state file but its payload marker is false"
        )
    optimizer_state = payload.get("optimizer")
    if optimizer_state is not None:
        peft_utils._validate_no_embedded_distributed_parameter_state(optimizer_state)

    if optimizer is not None:
        if optimizer_state is None:
            raise RuntimeError(
                "critic checkpoint has no optimizer state (it may have been saved with --no-save-optim); "
                "training resume is not possible"
            )
        load_parameter_state = getattr(optimizer, "load_parameter_state", None)
        requires_external_parameter_state = peft_utils._uses_external_parameter_state(
            optimizer,
            optimizer_state,
            load_parameter_state,
        )
        if requires_external_parameter_state and not external_parameter_state:
            raise RuntimeError("critic checkpoint is missing distributed optimizer parameter state")
        if external_parameter_state and not requires_external_parameter_state:
            raise RuntimeError("critic checkpoint external optimizer state does not match the current optimizer")
        if external_parameter_state and not callable(load_parameter_state):
            raise RuntimeError("critic checkpoint requires distributed optimizer parameter state")

    if opt_param_scheduler is not None and payload.get("opt_param_scheduler") is None:
        raise RuntimeError("critic checkpoint has no optimizer scheduler state; training resume is not possible")
    return payload


def _build_external_parameter_state_plan(optimizer: Any, parameter_state_path: Path, parameter_state_binding):
    try:
        return peft_utils._build_external_parameter_state_plan(
            optimizer,
            parameter_state_path,
            parameter_state_binding,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"critic optimizer parameter state is missing: {parameter_state_path}") from exc
    except RuntimeError as exc:
        if "checkpoint file was absent during preflight" in str(exc):
            raise RuntimeError(f"critic optimizer parameter state is missing: {parameter_state_path}") from exc
        raise


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

    binding = _coordinated_critic_checkpoint_call(
        "save binding validation",
        lambda: _validate_checkpoint_binding(save_root, iteration),
    )
    _require_checkpoint_binding_consensus(binding, operation="save")

    ckpt_dir = _critic_checkpoint_dir(save_root, iteration)
    _coordinated_critic_checkpoint_call(
        "directory creation",
        lambda: ckpt_dir.mkdir(parents=True, exist_ok=True),
    )

    optimizer_state = None
    optimizer_parameter_state = False
    scheduler_state = None
    parameter_state_path = _optimizer_parameter_state_path(ckpt_dir)
    save_optimizer = optimizer is not None and not getattr(args, "no_save_optim", False)
    _coordinated_critic_checkpoint_call(
        "distributed optimizer state initialization",
        lambda: peft_utils.prepare_distributed_optimizer_state_for_save(optimizer) if save_optimizer else None,
    )
    optimizer_state = _coordinated_critic_checkpoint_call(
        "optimizer state serialization",
        optimizer.state_dict if save_optimizer else lambda: None,
    )
    _coordinated_critic_checkpoint_call(
        "optimizer state validation",
        lambda: peft_utils._validate_no_embedded_distributed_parameter_state(optimizer_state)
        if optimizer_state is not None
        else None,
    )
    save_parameter_state = getattr(optimizer, "save_parameter_state", None) if optimizer is not None else None
    optimizer_parameter_state = _coordinated_critic_checkpoint_call(
        "external optimizer layout validation",
        lambda: peft_utils._uses_external_parameter_state(
            optimizer,
            optimizer_state,
            save_parameter_state,
        )
        if save_optimizer
        else False,
    )
    _coordinated_critic_checkpoint_call(
        "distributed optimizer source validation",
        lambda: peft_utils.validate_distributed_optimizer_sources_for_save(optimizer)
        if optimizer_parameter_state
        else None,
    )
    _coordinated_critic_checkpoint_call(
        "optimizer parameter-state materialization",
        lambda: save_parameter_state(str(parameter_state_path))
        if optimizer_parameter_state
        else parameter_state_path.unlink(missing_ok=True),
    )
    scheduler_state = _coordinated_critic_checkpoint_call(
        "optimizer scheduler state serialization",
        opt_param_scheduler.state_dict if save_optimizer and opt_param_scheduler is not None else lambda: None,
    )
    trainable_tensors = _coordinated_critic_checkpoint_call(
        "model tensor serialization",
        lambda: {k: v.detach().cpu() for k, v in _trainable_named_tensors(critic_model).items()},
    )

    payload = {
        "tensors": trainable_tensors,
        "optimizer": optimizer_state,
        "optimizer_parameter_state": optimizer_parameter_state,
        "opt_param_scheduler": scheduler_state,
        "iteration": iteration,
    }
    _coordinated_critic_checkpoint_call(
        "payload save",
        lambda: torch.save(payload, ckpt_dir / f"critic_rank{_global_rank()}.pt"),
    )
    _coordinated_critic_checkpoint_call(
        "latest marker publication",
        lambda: (Path(save_root) / "latest_checkpointed_iteration.txt").write_text(str(iteration))
        if _global_rank() == 0
        else None,
    )
    return str(ckpt_dir)


def load_critic_checkpoint(args, critic_model, optimizer=None, opt_param_scheduler=None) -> int | None:
    """Restore trainable critic tensors saved by save_critic_checkpoint.

    Returns None on fresh start (no checkpoint), else the loaded iteration.
    Never touches frozen params, so a load can never materialize a trunk copy.
    """
    load_root = getattr(args, "critic_load", None)
    normalized_root = _coordinated_critic_checkpoint_call(
        "load binding validation",
        lambda: None if not load_root else _validate_checkpoint_binding(load_root)[0],
    )
    load_roots = peft_utils._all_gather_checkpoint_object(normalized_root)
    if len(set(load_roots)) != 1:
        raise RuntimeError(f"adapter critic checkpoint load roots differ across ranks: {load_roots}")
    if normalized_root is None:
        return None

    latest_path = Path(load_root) / "latest_checkpointed_iteration.txt"
    latest_binding = _coordinated_critic_checkpoint_call(
        "latest marker snapshot capture",
        lambda: peft_utils._capture_checkpoint_file_binding(latest_path),
    )

    def read_latest_iteration() -> int:
        if latest_binding is None:
            raise FileNotFoundError(f"--critic-load does not contain a critic checkpoint marker: {latest_path}")
        try:
            loaded_iteration = int(_read_bound_checkpoint_text(latest_path, latest_binding).strip())
        except ValueError as exc:
            raise RuntimeError(f"adapter critic checkpoint marker is invalid: {latest_path}") from exc
        _validate_checkpoint_binding(load_root, loaded_iteration)
        return loaded_iteration

    iteration = _coordinated_critic_checkpoint_call("latest marker validation", read_latest_iteration)
    _require_checkpoint_binding_consensus((normalized_root, iteration), operation="load")
    checkpoint_dir = _critic_checkpoint_dir(load_root, iteration)
    path = checkpoint_dir / f"critic_rank{_global_rank()}.pt"
    parameter_state_path = _optimizer_parameter_state_path(checkpoint_dir)
    payload_binding, parameter_state_binding = _coordinated_critic_checkpoint_call(
        "checkpoint snapshot capture",
        lambda: (
            peft_utils._capture_checkpoint_file_binding(path),
            peft_utils._capture_checkpoint_file_binding(parameter_state_path),
        ),
    )

    payload = _coordinated_critic_checkpoint_call(
        "payload parse/validation",
        lambda: _validate_critic_payload(
            peft_utils._load_bound_torch_checkpoint(
                path,
                payload_binding,
                map_location="cpu",
                weights_only=False,
            ),
            iteration=iteration,
            critic_model=critic_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            optimizer_parameter_state_binding=parameter_state_binding,
        ),
    )
    params = _trainable_named_tensors(critic_model)
    saved = payload["tensors"]

    optimizer_state = payload.get("optimizer")
    external_parameter_state = payload.get("optimizer_parameter_state") is True
    external_parameter_state_plan = _coordinated_critic_checkpoint_call(
        "optimizer parameter-state preflight",
        lambda: _build_external_parameter_state_plan(
            optimizer,
            parameter_state_path,
            parameter_state_binding,
        )
        if optimizer is not None and external_parameter_state
        else None,
    )
    _coordinated_critic_checkpoint_call(
        "checkpoint snapshot validation",
        lambda: (
            peft_utils._verify_checkpoint_file_binding(path, payload_binding),
            peft_utils._verify_checkpoint_file_binding(parameter_state_path, parameter_state_binding),
        ),
    )

    def restore_model_tensors() -> None:
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(saved[name].to(device=param.device, dtype=param.dtype))

    _coordinated_critic_checkpoint_call("model tensor restore", restore_model_tensors)

    # The tensors above were copied after optimizer construction. Refresh
    # optimizer-owned FP32/main parameters before optionally replacing them
    # with their higher-precision checkpointed values below.
    _coordinated_critic_checkpoint_call(
        "optimizer model-parameter reload",
        lambda: _reload_optimizer_model_params(optimizer) if optimizer is not None else None,
    )
    _coordinated_critic_checkpoint_call(
        "optimizer state restore",
        lambda: optimizer.load_state_dict(optimizer_state) if optimizer is not None else None,
    )
    _coordinated_critic_checkpoint_call(
        "optimizer parameter-state destination validation",
        lambda: peft_utils._validate_external_parameter_state_destinations(external_parameter_state_plan)
        if external_parameter_state_plan is not None
        else None,
    )

    def restore_external_parameter_state() -> None:
        if optimizer is None or not external_parameter_state:
            return
        if external_parameter_state_plan is None:
            raise RuntimeError("adapter critic external optimizer state has no validated restore plan")
        peft_utils._dispatch_external_parameter_state(
            external_parameter_state_plan,
            custom_load_parameter_state=getattr(optimizer, "load_parameter_state", None),
        )

    _coordinated_critic_checkpoint_call(
        "optimizer parameter-state restore",
        restore_external_parameter_state,
    )
    _coordinated_critic_checkpoint_call(
        "optimizer scheduler restore",
        lambda: opt_param_scheduler.load_state_dict(payload["opt_param_scheduler"])
        if opt_param_scheduler is not None
        else None,
    )
    return iteration


def _check_resume_iteration(
    loaded: int | None,
    expected: int | None,
    *,
    require_checkpoint: bool = False,
) -> None:
    """Fail loud if the critic and actor resumed at different iterations.

    Raises when only one role restored training state, or when both restored
    iterations are known but disagree. Both unknown means a fresh PPO run.
    """
    if expected is None:
        if loaded is not None:
            raise RuntimeError(
                f"adapter critic checkpoint resumed at iteration {loaded}, but the actor loaded no training "
                "checkpoint; remove --critic-load for a fresh run"
            )
        return
    if loaded is None:
        if require_checkpoint:
            raise RuntimeError(
                "actor resumed from a PEFT checkpoint but no matching adapter critic checkpoint was loaded; "
                "set --critic-load to the corresponding critic checkpoint root"
            )
        return
    if loaded != expected:
        raise RuntimeError(
            f"critic/actor checkpoint iteration mismatch: critic resumed at iteration {loaded}, "
            f"actor resumed at iteration {expected}"
        )


def _expected_critic_resume_iteration(args, loaded_iteration: int) -> int | None:
    """Bind critic resume only to actual actor training state, not model bootstrap."""
    if getattr(args, "_orbit_training_checkpoint_loaded", False):
        return loaded_iteration
    return None


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
        require_checkpoint=expected_iteration is not None,
    )
    assert_trunk_aliased(model, actor_model)
    logger.info("one-trunk critic ready: %d trunk params aliased, resumed_iteration=%s", aliased, resumed_iteration)
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
