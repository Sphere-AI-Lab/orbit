"""Critic value head: build it, freeze around it, reinitialize it after a load.

Lifted verbatim out of ``miles/backends/megatron_utils/model_provider.py`` and
``miles/backends/megatron_utils/model.py`` (Phase-3 slice 3f, P1 lift-out).
Both miles modules re-export the names they still call behind stamped
``# ORBIT-SEAM`` hooks, so importers
(``orbit.megatron.bridge_peft_helpers``, ``tests/test_model_provider.py``) and
the tests' monkeypatch surface (``model._head_critic_provider``,
``model._critic_output_layer_needs_reinit``) are unchanged.

A critic replaces the LM output layer with a scalar value head. ``--critic-mode
head`` additionally freezes everything else inside the model provider, before
the DDP wrap, so grad buffers and optimizer state cover the head only. Any mode
may need the head reinitialized after a checkpoint load whose stored head does
not match the runtime shape (a base/actor checkpoint has no value head at all).

This module is the value-head layer itself; ``orbit.critic.critic_adapter``
owns the one-trunk critic build/alias and critic checkpoint I/O around it.
"""

import logging
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path

import torch
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from megatron.core.pipeline_parallel.utils import unwrap_model
except ImportError:
    from megatron.core.utils import unwrap_model

logger = logging.getLogger(__name__)


def replace_output_layer_with_value_head(model: torch.nn.Module, config: TransformerConfig) -> torch.nn.Module:
    # LinearForLastLayer is miles' own (base) class; imported at call time so
    # the home layer keeps no module-level miles dependency.
    from miles.backends.megatron_utils.model_provider import LinearForLastLayer

    model.output_layer = LinearForLastLayer(input_size=config.hidden_size, output_size=1, config=config)
    return model


def _head_critic_provider(provider):
    """Freeze all-but-value-head inside the provider, BEFORE the DDP wrap, so
    grad buffers and optimizer state cover only the value head (the trunk is
    later re-pointed at the actor's storage via ``alias_trunk_storage``)."""

    def wrapped(*p_args, **p_kwargs):
        from orbit.critic.critic_adapter import prepare_head_critic

        module = provider(*p_args, **p_kwargs)
        prepare_head_critic([module])
        return module

    return wrapped


def _iter_critic_output_layers(model: Sequence[DDP]):
    for chunk_id, module in enumerate(unwrap_model(model)):
        output_layer = getattr(module, "output_layer", None)
        if output_layer is not None:
            yield chunk_id, output_layer


def _critic_output_layer_needs_reinit(args: Namespace, model: Sequence[DDP], role: str) -> bool:
    if role != "critic" or args.load is None:
        return False

    from megatron.core.dist_checkpointing.serialization import load_tensors_metadata
    from megatron.training.checkpointing import get_load_checkpoint_path_by_args

    checkpoint_path = Path(get_load_checkpoint_path_by_args(args))
    if not (checkpoint_path / ".metadata").is_file():
        return False

    checkpoint_metadata = load_tensors_metadata(str(checkpoint_path))
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        for name in ("weight", "bias"):
            param = getattr(output_layer, name, None)
            if param is None:
                continue

            param_name = f"output_layer.{name}"
            ckpt_tensor_metadata = next(
                (
                    tensor_metadata
                    for key, tensor_metadata in checkpoint_metadata.items()
                    if key == param_name or key.endswith(f".{param_name}")
                ),
                None,
            )
            expected_shape = tuple(param.shape)
            checkpoint_shape = tuple(ckpt_tensor_metadata.global_shape) if ckpt_tensor_metadata is not None else None
            if checkpoint_shape == expected_shape:
                continue

            reason = (
                "missing from checkpoint metadata"
                if checkpoint_shape is None
                else f"shape mismatch checkpoint={checkpoint_shape} runtime={expected_shape}"
            )
            logger.warning(
                "Will reinitialize critic %s after checkpoint load because it is %s",
                param_name,
                reason,
            )
            return True

    return False


@torch.no_grad()
def _reinitialize_critic_output_layer(model: Sequence[DDP]) -> None:
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        output_layer.weight.data.normal_(mean=0.0, std=0.02)
        if output_layer.bias is not None:
            output_layer.bias.data.zero_()
