"""Adapter-parameter name predicates and enumeration for Megatron models.

Home for the PEFT adapter-tensor helpers lifted out of
miles/backends/megatron_utils/update_weight/common.py: deciding whether a
named parameter/buffer belongs to a LoRA/OFT adapter, and enumerating just
the adapter tensors of a (possibly virtual-pipeline-sharded) model.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from miles.orbit.megatron.tensor_semantics import is_dsv4_grouped_moe_oft_param_name


def is_named_adapter_tensor(name: str) -> bool:
    from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix

    stripped_name = strip_param_name_prefix(name)
    return (
        "lora_" in stripped_name
        or ".adapter." in stripped_name
        or stripped_name.startswith("adapter.")
        or ".oft_" in stripped_name
        or stripped_name.startswith("oft_")
        or is_dsv4_grouped_moe_oft_param_name(stripped_name)
    )


def named_adapter_params_and_buffers(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix

    use_vp_prefix = len(model) > 1

    for vp_stage, model_module in enumerate(model):
        def _compute_name(name: str, vp_stage=vp_stage) -> str:
            stripped_name = strip_param_name_prefix(name)
            if use_vp_prefix:
                return f"vp_stages.{vp_stage}.{stripped_name}"
            return stripped_name

        for name, param in model_module.named_parameters():
            if not is_named_adapter_tensor(name):
                continue
            yield _compute_name(name), param

        for name, buffer in model_module.named_buffers():
            if not is_named_adapter_tensor(name):
                continue
            yield _compute_name(name), buffer


def named_adapter_params(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    for name, tensor in named_adapter_params_and_buffers(model):
        if isinstance(tensor, torch.nn.Parameter):
            yield name, tensor
