#!/usr/bin/env python3
"""Orbit-owned import checkpoint entrypoint built on upstream Megatron-Bridge."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import torch

from megatron.bridge import AutoBridge


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert a CLI dtype string into a torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtype_map[dtype_str]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_str}") from exc


def import_hf_to_megatron(
    hf_model: str,
    megatron_path: str,
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    trust_remote_code: bool = False,
) -> None:
    """Import a HuggingFace checkpoint into Megatron checkpoint format."""
    kwargs = {}
    if torch_dtype:
        kwargs["torch_dtype"] = get_torch_dtype(torch_dtype)
    if device_map:
        kwargs["device_map"] = device_map
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    AutoBridge.import_ckpt(
        hf_model_id=hf_model,
        megatron_path=megatron_path,
        **kwargs,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HuggingFace checkpoints into Megatron format")
    subparsers = parser.add_subparsers(dest="command", help="Conversion direction")

    import_parser = subparsers.add_parser("import", help="Import HuggingFace model to Megatron checkpoint format")
    import_parser.add_argument("--hf-model", required=True, help="HuggingFace model ID or path to model directory")
    import_parser.add_argument(
        "--megatron-path",
        required=True,
        help="Directory path where the Megatron checkpoint will be saved",
    )
    import_parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], help="Model precision")
    import_parser.add_argument("--device-map", help='Device placement strategy (e.g. "auto", "cuda:0")')
    import_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code execution")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the import-only checkpoint conversion path."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "import":
        parser.print_help()
        return 1

    import_hf_to_megatron(
        hf_model=args.hf_model,
        megatron_path=args.megatron_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
