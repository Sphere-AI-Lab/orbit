from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TensorSemantics:
    quantization: str
    is_placeholder: bool
    skip_for_tracking: bool
    skip_for_meta_validation: bool


_EXPERT_TRIPLET_RE = re.compile(r"^weight\d+_(packed|scale|shape)$")
_NVFP4_EXPERT_WEIGHT_RE = re.compile(r"^weight(?P<idx>\d+)$")
_NVFP4_EXPERT_PACKED_RE = re.compile(r"^weight(?P<idx>\d+)(?:_[wv])?_packed$")
_NVFP4_EXPERT_SCALE_RE = re.compile(r"^weight_(?P<suffix>scale|double_scale)\d+$")


def _resolve_owner(module: nn.Module, name: str) -> tuple[nn.Module, str, str]:
    if "." in name:
        parent_name, leaf_name = name.rsplit(".", 1)
        owner = module.get_submodule(parent_name)
    else:
        parent_name = ""
        owner = module
        leaf_name = name
    return owner, parent_name, leaf_name


def _candidate_owners(module: nn.Module, name: str) -> tuple[list[nn.Module], str]:
    owner, parent_name, leaf_name = _resolve_owner(module, name)
    owners = [owner]
    if parent_name.endswith(".to_wrap"):
        wrapper_name = parent_name[: -len(".to_wrap")]
        if wrapper_name:
            owners.insert(0, module.get_submodule(wrapper_name))
    return owners, leaf_name


def _has_triplet_buffers(owner: nn.Module, base_name: str) -> bool:
    return all(isinstance(getattr(owner, f"{base_name}_{suffix}", None), torch.Tensor) for suffix in ("packed", "scale", "shape"))


def _has_any_expert_triplets(owner: nn.Module) -> bool:
    suffixes_by_base: dict[str, set[str]] = {}
    for buffer_name in owner._buffers:
        match = _EXPERT_TRIPLET_RE.match(buffer_name)
        if match is None:
            continue
        base_name, suffix = buffer_name.rsplit("_", 1)
        suffixes_by_base.setdefault(base_name, set()).add(suffix)
    return any(suffixes >= {"packed", "scale", "shape"} for suffixes in suffixes_by_base.values())


def _has_nvfp4_expert_buffers(owner: nn.Module, base_name: str) -> bool:
    match = _NVFP4_EXPERT_WEIGHT_RE.match(base_name)
    if match is None:
        return False

    idx = match.group("idx")
    has_packed_weight = isinstance(getattr(owner, f"weight{idx}_packed", None), torch.Tensor)
    has_split_packed_weight = all(
        isinstance(getattr(owner, f"weight{idx}_{suffix}_packed", None), torch.Tensor)
        for suffix in ("w", "v")
    )
    has_scales = all(
        isinstance(getattr(owner, f"weight_{suffix}{idx}", None), torch.Tensor)
        for suffix in ("scale", "double_scale")
    )
    return (has_packed_weight or has_split_packed_weight) and has_scales


def _has_any_nvfp4_expert_buffers(owner: nn.Module) -> bool:
    has_packed = False
    scale_suffixes = set()
    for buffer_name in owner._buffers:
        if _NVFP4_EXPERT_PACKED_RE.match(buffer_name) is not None:
            has_packed = True
            continue
        scale_match = _NVFP4_EXPERT_SCALE_RE.match(buffer_name)
        if scale_match is not None:
            scale_suffixes.add(scale_match.group("suffix"))
    return has_packed and scale_suffixes >= {"scale", "double_scale"}


def _dtype_quantization_name(tensor: torch.Tensor) -> str:
    if tensor.device.type == "meta":
        return "meta"
    if tensor.dtype == torch.bfloat16:
        return "bf16"
    if tensor.dtype == torch.float16:
        return "fp16"
    if tensor.dtype == torch.float32:
        return "fp32"
    return str(tensor.dtype).replace("torch.", "")


def describe_named_tensor(module: nn.Module, name: str, tensor: torch.Tensor) -> TensorSemantics:
    owners, leaf_name = _candidate_owners(module, name)

    for owner in owners:
        if leaf_name.startswith("weight") and _has_triplet_buffers(owner, leaf_name):
            is_placeholder = tensor.device.type == "meta" or tensor.numel() == 0
            return TensorSemantics(
                quantization="int4",
                is_placeholder=is_placeholder,
                skip_for_tracking=is_placeholder,
                skip_for_meta_validation=is_placeholder,
            )

        if leaf_name == "weight" and _has_any_expert_triplets(owner):
            is_placeholder = tensor.device.type == "meta" or tensor.numel() == 0
            return TensorSemantics(
                quantization="int4",
                is_placeholder=is_placeholder,
                skip_for_tracking=is_placeholder,
                skip_for_meta_validation=is_placeholder,
            )

        if leaf_name.startswith("weight") and _has_nvfp4_expert_buffers(owner, leaf_name):
            is_placeholder = tensor.device.type == "meta" or tensor.numel() == 0
            return TensorSemantics(
                quantization="nvfp4",
                is_placeholder=is_placeholder,
                skip_for_tracking=is_placeholder,
                skip_for_meta_validation=is_placeholder,
            )

        if (leaf_name == "weight" or _NVFP4_EXPERT_WEIGHT_RE.match(leaf_name)) and _has_any_nvfp4_expert_buffers(owner):
            is_placeholder = tensor.device.type == "meta" or tensor.numel() == 0
            return TensorSemantics(
                quantization="nvfp4",
                is_placeholder=is_placeholder,
                skip_for_tracking=is_placeholder,
                skip_for_meta_validation=is_placeholder,
            )

        if leaf_name == "weight" and all(
            isinstance(getattr(owner, attr, None), torch.Tensor) for attr in ("weight_scale", "weight_scale_2")
        ):
            return TensorSemantics(
                quantization="nvfp4",
                is_placeholder=False,
                skip_for_tracking=False,
                skip_for_meta_validation=False,
            )

        if leaf_name == "weight" and any(
            isinstance(getattr(owner, attr, None), torch.Tensor) for attr in ("weight_scale_inv", "input_scale")
        ):
            return TensorSemantics(
                quantization="fp8",
                is_placeholder=False,
                skip_for_tracking=False,
                skip_for_meta_validation=False,
            )

    return TensorSemantics(
        quantization=_dtype_quantization_name(tensor),
        is_placeholder=False,
        skip_for_tracking=False,
        skip_for_meta_validation=False,
    )


def should_skip_named_tensor_for_tracking(module: nn.Module, name: str, tensor: torch.Tensor) -> bool:
    return describe_named_tensor(module, name, tensor).skip_for_tracking


def should_skip_named_tensor_for_meta_validation(module: nn.Module, name: str, tensor: torch.Tensor) -> bool:
    return describe_named_tensor(module, name, tensor).skip_for_meta_validation


DSV4_GROUPED_MOE_OFT_PARAM_NAMES = frozenset({"w1_oft_r", "w2_oft_r", "w3_oft_r"})
DSV4_SHARED_EXPERT_TP_PARAM_FRAGMENTS = (
    ".mlp.shared_experts.w1.",
    ".mlp.shared_experts.w2.",
    ".mlp.shared_experts.w3.",
    ".mlp.shared_experts.linear_fc1.",
    ".mlp.shared_experts.linear_fc2.",
    ".ffn.shared_experts.w1.",
    ".ffn.shared_experts.w2.",
    ".ffn.shared_experts.w3.",
    ".ffn.shared_experts.linear_fc1.",
    ".ffn.shared_experts.linear_fc2.",
)


def is_dsv4_grouped_moe_oft_param_name(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in DSV4_GROUPED_MOE_OFT_PARAM_NAMES


def uses_expert_tensor_parallel_group(name: str) -> bool:
    matchable = name.replace("._orig_module.", ".").replace(".to_wrap.", ".")
    return (
        ".experts." in matchable
        or any(
            fragment in matchable or matchable.endswith(fragment[:-1])
            for fragment in DSV4_SHARED_EXPERT_TP_PARAM_FRAGMENTS
        )
        or is_dsv4_grouped_moe_oft_param_name(matchable)
    )
