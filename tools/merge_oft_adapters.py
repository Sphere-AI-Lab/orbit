#!/usr/bin/env python
"""Merge N orbit OFT adapters into one (OrthoMerge magnitude-corrected Lie-algebra merge)."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
sys.path = [p for p in sys.path if p != _repo_root]
sys.path.insert(0, _repo_root)

import torch
from safetensors.torch import load_file, save_file

from orbit.peft.merge import get_strategy  # light: pulls only torch
from orbit.utils.logging_utils import configure_logger

_COMPAT_KEYS = ("oft_type", "oft_block_size", "target_modules", "base_model_name_or_path")


def read_oft_config(adapter_dir: str) -> dict:
    """Read adapter_config.json and assert it is an OFT adapter; return the config.

    Reimplemented locally (NOT orbit.backends.megatron_utils) to keep this tool
    CPU-only: that module's import chain pulls deep_ep, which requires CUDA.
    """
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        raise ValueError(f"missing adapter_config.json at {adapter_dir}")
    cfg = json.loads(cfg_path.read_text())
    if (cfg.get("peft_type") or "").upper() != "OFT":
        raise ValueError(
            f"adapter at {adapter_dir} has peft_type={cfg.get('peft_type')!r}, expected OFT"
        )
    return cfg


def validate_adapters(adapter_dirs: list[str]) -> dict:
    """Validate that all adapters are OFT and mutually compatible. Returns the shared config."""
    if len(adapter_dirs) < 2:
        raise ValueError("merging requires at least 2 adapters")
    configs = [read_oft_config(d) for d in adapter_dirs]
    ref = configs[0]
    for d, cfg in zip(adapter_dirs[1:], configs[1:], strict=True):
        for key in _COMPAT_KEYS:
            got = cfg.get(key)
            expected = ref.get(key)
            if key == "target_modules" and isinstance(got, list) and isinstance(expected, list):
                compatible = Counter(got) == Counter(expected)
            else:
                compatible = got == expected
            if not compatible:
                raise ValueError(
                    f"adapter {d} differs on {key}: {got!r} != {expected!r}"
                )
    return ref


def load_adapters(adapter_dirs: list[str]) -> list[dict[str, torch.Tensor]]:
    out = []
    for d in adapter_dirs:
        f = Path(d) / "adapter_model.safetensors"
        if not f.exists():
            raise FileNotFoundError(f"no adapter_model.safetensors in {d}")
        out.append(load_file(str(f)))
    return out


def write_merged_adapter(merged: dict[str, torch.Tensor], src_config_dir: str, output_dir: str) -> str:
    merged_dir = Path(output_dir) / "merged_adapter"
    merged_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(merged_dir / "adapter_model.safetensors"))
    shutil.copyfile(Path(src_config_dir) / "adapter_config.json", merged_dir / "adapter_config.json")
    return str(merged_dir)


def main(argv: list[str] | None = None) -> int:
    configure_logger()
    p = argparse.ArgumentParser(description="Merge N orbit OFT adapters (OrthoMerge).")
    p.add_argument("--adapters", nargs="+", required=True, help="paths to OFT adapter dirs (>=2)")
    p.add_argument("--output", required=True, help="output dir; writes <output>/merged_adapter/")
    p.add_argument("--method", default="oft", help="merge strategy (default: oft)")
    p.add_argument("--weights", nargs="+", type=float, default=None, help="per-adapter weights (default: equal)")
    p.add_argument("--base", default=None, help="optional: assert recorded base_model_name_or_path matches")
    p.add_argument("--save-megatron", action="store_true",
                   help="also merge the Megatron-native shards -> <output>/merged_megatron/")
    p.add_argument("--save-hf", action="store_true",
                   help="bake the merged rotation into dense HF weights -> <output>/merged_model_hf/")
    p.add_argument("--device", default="cpu",
                   help="device for the --save-hf bake (e.g. cuda:0); default cpu")
    args = p.parse_args(argv)

    cfg = validate_adapters(args.adapters)
    if args.base is not None and cfg.get("base_model_name_or_path") != args.base:
        raise ValueError(f"--base {args.base!r} != recorded {cfg.get('base_model_name_or_path')!r}")
    if args.weights is not None and len(args.weights) != len(args.adapters):
        raise ValueError(f"{len(args.weights)} weights for {len(args.adapters)} adapters")

    state_dicts = load_adapters(args.adapters)
    merged = get_strategy(args.method).merge(state_dicts, args.weights)
    merged_dir = write_merged_adapter(merged, args.adapters[0], args.output)
    print(f"[merge] {len(args.adapters)} adapters -> {merged_dir} ({len(merged)} tensors)")

    if args.save_megatron:
        from orbit.peft.merge.megatron_io import merge_megatron_adapters, write_megatron_adapter
        merged_meg = merge_megatron_adapters(args.adapters, args.weights, args.method)
        meg_dir = write_megatron_adapter(merged_meg, args.adapters[0], str(Path(args.output) / "merged_megatron"))
        print(f"[merge] Megatron-native adapter -> {meg_dir} ({len(merged_meg)} shard(s))")

    if args.save_hf:
        from orbit.peft.merge.bake_hf import bake_hf_model
        base = args.base or cfg.get("base_model_name_or_path")
        if not base:
            raise ValueError("--save-hf needs a base model: pass --base or ensure adapter_config has base_model_name_or_path")
        hf_dir = str(Path(args.output) / "merged_model_hf")
        n = bake_hf_model(base, merged_dir, int(cfg["oft_block_size"]), hf_dir, args.device, adapter=merged)
        print(f"[merge] baked dense HF model ({n} linears) -> {hf_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
