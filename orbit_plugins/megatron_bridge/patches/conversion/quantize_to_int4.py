#!/usr/bin/env python3
"""Quantize expert weights in a BF16 HF checkpoint into native INT4 format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from megatron.bridge.models.conversion.low_precision.int4 import quantize_to_int4


def should_quantize(key: str) -> bool:
    """Check if a weight key is an expert MLP weight that should be INT4."""
    if not key.endswith(".weight"):
        return False
    if "experts." not in key:
        return False
    if "shared_expert" in key:
        return False
    return any(proj in key for proj in ("gate_proj", "up_proj", "down_proj"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to BF16 HF model")
    parser.add_argument("--output", required=True, help="Output path for INT4 model")
    parser.add_argument("--group-size", type=int, default=32)
    args = parser.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)

    for source in Path(args.input).iterdir():
        if source.suffix != ".safetensors" and source.name != "model.safetensors.index.json":
            destination = Path(args.output) / source.name
            if not destination.exists():
                if source.is_dir():
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)

    shard_files = sorted(Path(args.input).glob("model*.safetensors"))
    new_weight_map = {}
    total_quantized = 0
    total_kept = 0
    total_size = 0

    for shard_path in shard_files:
        print(f"Processing {shard_path.name}...")
        new_tensors = {}

        with safe_open(str(shard_path), framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)

                if should_quantize(key):
                    packed, scale, shape = quantize_to_int4(tensor, group_size=args.group_size)
                    base = key[: -len(".weight")]
                    new_tensors[f"{base}.weight_packed"] = packed
                    new_tensors[f"{base}.weight_scale"] = scale
                    new_tensors[f"{base}.weight_shape"] = shape
                    total_quantized += 1
                else:
                    new_tensors[key] = tensor
                    total_kept += 1

        output_path = Path(args.output) / shard_path.name
        save_file(new_tensors, str(output_path))
        total_size += sum(tensor.numel() * tensor.element_size() for tensor in new_tensors.values())

        for key in new_tensors:
            new_weight_map[key] = shard_path.name

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }
    with open(Path(args.output) / "model.safetensors.index.json", "w") as handle:
        json.dump(index, handle, indent=2)

    print(f"\nDone. Quantized {total_quantized} expert weights, kept {total_kept} as BF16.")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
