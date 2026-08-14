"""Small helpers for keeping Miles and Megatron CP settings in sync."""

from __future__ import annotations

import math
import sys
from argparse import Namespace
from collections.abc import Sequence


_CP_COMM_TYPES = {"p2p", "a2a", "all_gather", "a2a+p2p"}
_CP_COMM_ALIASES = {"allgather": "all_gather"}
_QWEN3_VL_MODEL_TYPES = {"qwen3_vl", "qwen3_vl_moe"}


def cp_comm_type_was_explicit(args: Namespace, argv: Sequence[str] | None = None) -> bool:
    """Return whether the runtime CP transport came from an explicit user choice."""

    marker = getattr(args, "cp_comm_type_explicit", None)
    if marker is not None:
        return bool(marker)

    tokens = sys.argv[1:] if argv is None else argv
    return any(token == "--cp-comm-type" or token.startswith("--cp-comm-type=") for token in tokens)


def canonicalize_cp_comm_type(value, *, default_to_p2p: bool = True) -> str | None:
    """Collapse Megatron's string/per-layer representation to one transport.

    Miles currently has one data/loss layout decision for the whole model, so a
    per-layer mixture would be ambiguous. Repeated identical values are safe to
    collapse; a real mixture is rejected.
    """

    if value is None:
        return "p2p" if default_to_p2p else None

    if isinstance(value, str):
        normalized = _CP_COMM_ALIASES.get(value.strip().lower(), value.strip().lower())
        if normalized not in _CP_COMM_TYPES:
            raise ValueError(f"Unsupported cp_comm_type={value!r}; expected one of {sorted(_CP_COMM_TYPES)}")
        return normalized

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError("cp_comm_type must not be an empty list")
        normalized_values = [canonicalize_cp_comm_type(item, default_to_p2p=False) for item in value]
        if any(item is None for item in normalized_values):
            raise ValueError("cp_comm_type contains an empty per-layer value")
        unique_values = set(normalized_values)
        if len(unique_values) != 1:
            raise ValueError("Mixed per-layer cp_comm_type is not supported by Miles yet: " f"{list(value)!r}")
        return normalized_values[0]

    raise ValueError(f"Unsupported cp_comm_type representation: {type(value).__name__}")


def normalize_cp_contract(args: Namespace, model_type: str | None = None) -> None:
    """Normalize the startup CP contract after all Miles config overrides."""

    cp_size = int(getattr(args, "context_parallel_size", 1) or 1)
    canonical = canonicalize_cp_comm_type(getattr(args, "cp_comm_type", None))

    # Megatron's argument validation expects the per-layer-capable list shape.
    args.cp_comm_type = [canonical]
    # Miles consumers should never need to guess whether they received a list.
    args.cp_comm_type_canonical = canonical
    args.cp_token_layout = (
        "inactive" if cp_size <= 1 else ("contiguous" if getattr(args, "allgather_cp", False) else "zigzag")
    )

    if cp_size > 1 and canonical == "a2a+p2p":
        hierarchy = getattr(args, "hierarchical_context_parallel_sizes", None)
        if not isinstance(hierarchy, (list, tuple)) or len(hierarchy) != 2:
            raise ValueError("cp_comm_type=a2a+p2p requires two hierarchical context-parallel sizes")
        if any(not isinstance(size, int) or size <= 0 for size in hierarchy):
            raise ValueError("hierarchical context-parallel sizes must be positive integers")
        if math.prod(hierarchy) != cp_size:
            raise ValueError(
                "hierarchical context-parallel sizes must multiply to context_parallel_size: "
                f"hierarchy={list(hierarchy)}, cp_size={cp_size}"
            )
        args.hierarchical_context_parallel_sizes = list(hierarchy)

    normalized_model_type = (model_type or "").lower().replace("-", "_")
    if (
        normalized_model_type in _QWEN3_VL_MODEL_TYPES
        and cp_size > 1
        and getattr(args, "qkv_format", "thd") == "thd"
        and args.cp_token_layout == "contiguous"
    ):
        raise ValueError(
            "Qwen3-VL with THD and CP>1 does not yet support --allgather-cp: "
            "its packed mRoPE path currently requires the default zigzag token layout"
        )
