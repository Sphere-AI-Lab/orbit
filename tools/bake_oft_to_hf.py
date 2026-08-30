#!/usr/bin/env python
"""Bake a single orbit OFT adapter into a standalone dense HF model.

Thin CLI over miles.orbit.merge.bake_hf.bake_hf_model for one adapter directory
(`iter_*/adapter` or `merged_adapter`). The merge tool's `--save-hf` covers
merged outputs; this covers individual adapters for evaluation or deployment
paths that expect dense Hugging Face weights.

    python tools/bake_oft_to_hf.py \
        --base /path/Qwen2.5-0.5B-Instruct \
        --adapter orbit_ckpts/RUN/iter_0000059/adapter \
        --output baked/A1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from miles.orbit.merge.bake_hf import bake_hf_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base HF model path")
    parser.add_argument("--adapter", required=True, help="adapter dir with adapter_model.safetensors")
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-size", type=int, default=None,
                        help="OFT block size; default: oft_block_size from adapter_config.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    block_size = args.block_size
    if block_size is None:
        cfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
        block_size = int(cfg["oft_block_size"])

    baked = bake_hf_model(
        base_model_path=args.base,
        merged_adapter_dir=args.adapter,
        block_size=block_size,
        output_dir=args.output,
        device=args.device,
    )
    print(f"baked {baked} linears (block_size={block_size}) -> {args.output}")


if __name__ == "__main__":
    main()
