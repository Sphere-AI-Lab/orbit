"""Orbit dense Qwen3 bridge that preserves FP8 weights during HF -> Megatron import."""

from __future__ import annotations

import contextlib
import logging
from typing import List, Mapping, Optional

import torch

from megatron.bridge.models.conversion.param_mapping import (
    GatedMLPMapping,
    QKVMapping,
    RowParallelMapping,
)
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge
from megatron.bridge.orbit.quant.fp8_utils import merge_gated_mlp_scale_inv, merge_qkv_scale_inv


logger = logging.getLogger(__name__)


class OrbitQwen3FP8Bridge(Qwen3Bridge):
    """Dense Qwen3 bridge that keeps FP8 weights in FP8 throughout conversion."""

    def load_weights_hf_to_megatron(
        self,
        hf_pretrained,
        megatron_model,
        allowed_mismatched_params: Optional[List[str]] = None,
    ):
        del allowed_mismatched_params

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        with contextlib.ExitStack() as stack:
            if hasattr(megatron_model[0], "hide_teacher_model"):
                stack.enter_context(megatron_model[0].hide_teacher_model())
            if hasattr(megatron_model[0], "hide_loss_modules"):
                stack.enter_context(megatron_model[0].hide_loss_modules())
            hf_to_megatron_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)

        hf_state_dict: Mapping[str, torch.Tensor] = (
            hf_pretrained.state if hasattr(hf_pretrained, "state") else {}
        )

        description = f"Loading FP8 from {hf_pretrained.model_name_or_path}"
        for task in self._with_progress_tracking(hf_to_megatron_tasks, description):
            if task is None or task.megatron_module is None:
                continue

            hf_weights = self.maybe_modify_loaded_hf_weight(task.mapping.hf_param, hf_state_dict)
            is_fp8 = _is_fp8(hf_weights)

            if is_fp8 and task.param_weight is not None:
                task.param_weight.data = torch.empty(
                    task.param_weight.shape,
                    dtype=torch.float8_e4m3fn,
                    device=task.param_weight.device,
                )

            converted_weights = task.mapping.hf_to_megatron(hf_weights, task.megatron_module)

            if converted_weights is not None:
                assert task.param_weight is not None

                if converted_weights.shape != task.param_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {task.mapping.megatron_param}: "
                        f"expected {task.param_weight.shape}, got {converted_weights.shape}"
                    )

                if is_fp8:
                    task.param_weight.data = converted_weights
                    self._store_scale_inv(task, hf_state_dict)
                else:
                    task.param_weight.data.copy_(converted_weights)

        self._broadcast_shared_embeddings(megatron_model)
        return megatron_model

    def _store_scale_inv(self, task, hf_state_dict: Mapping[str, torch.Tensor]):
        """Load, merge/split, and register ``weight_scale_inv`` for one task."""
        mapping = task.mapping
        tp_rank = mapping.tp_rank
        tp_size = mapping.tp_size
        hf_param = mapping.hf_param

        if isinstance(mapping, QKVMapping):
            q_scale = _load_scale(hf_param["q"], hf_state_dict)
            k_scale = _load_scale(hf_param["k"], hf_state_dict)
            v_scale = _load_scale(hf_param["v"], hf_state_dict)
            if q_scale is None:
                return
            config = mapping._get_config(task.megatron_module)
            merged = merge_qkv_scale_inv(config, q_scale, k_scale, v_scale)
            scale = _tp_chunk(merged, tp_size, tp_rank, dim=0)
        elif isinstance(mapping, GatedMLPMapping):
            gate_scale = _load_scale(hf_param["gate"], hf_state_dict)
            up_scale = _load_scale(hf_param["up"], hf_state_dict)
            if gate_scale is None:
                return
            merged = merge_gated_mlp_scale_inv(gate_scale, up_scale)
            scale = _tp_chunk(merged, tp_size, tp_rank, dim=0)
        elif isinstance(hf_param, str):
            raw_scale = _load_scale(hf_param, hf_state_dict)
            if raw_scale is None:
                return
            if isinstance(mapping, RowParallelMapping) or _is_row_parallel(mapping):
                scale = _tp_chunk(raw_scale, tp_size, tp_rank, dim=1)
            else:
                scale = _tp_chunk(raw_scale, tp_size, tp_rank, dim=0)
        else:
            return

        module, _ = get_module_and_param_from_name(task.megatron_module, task.param_name)
        module.register_buffer("weight_scale_inv", scale.contiguous(), persistent=False)
        logger.debug(
            "Registered weight_scale_inv on %s: shape=%s", task.param_name, list(scale.shape)
        )


def _is_fp8(hf_weights) -> bool:
    if isinstance(hf_weights, dict):
        return any(weight.dtype == torch.float8_e4m3fn for weight in hf_weights.values())
    return hf_weights.dtype == torch.float8_e4m3fn


def _load_scale(weight_key: str, hf_state_dict: Mapping[str, torch.Tensor]):
    key = weight_key + "_scale_inv"
    if key in hf_state_dict:
        return hf_state_dict[key]

    key = weight_key + "_scale"
    if key in hf_state_dict:
        return 1.0 / hf_state_dict[key].float()

    return None


def _tp_chunk(tensor: torch.Tensor, tp_size: int, tp_rank: int, dim: int) -> torch.Tensor:
    if tp_size <= 1:
        return tensor
    return torch.chunk(tensor, tp_size, dim=dim)[tp_rank]


def _is_row_parallel(mapping) -> bool:
    inner = getattr(mapping, "_mapping", None)
    return isinstance(inner, RowParallelMapping)
