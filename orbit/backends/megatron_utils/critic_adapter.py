"""One-trunk PPO critic (--critic-mode adapter): build/alias helpers.

The adapter-mode critic is a normal PEFT model (adapters + scalar value head)
whose frozen trunk parameters alias the actor's tensors, so PPO pays for no
second trunk copy. Frozen params are trunk by definition of PEFT; trainable
params (adapters, value head) are role-owned and never aliased. See
docs/superpowers/specs/2026-07-27-one-trunk-ppo-design.md (clthegoat docs).
"""

import logging

import torch

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
