"""Merge orbit Megatron-native OFT adapter shards (adapter_megatron_tp{tp}_pp{pp}.pt).

The shards are torch state dicts keyed by ``(VPP chunk, Megatron param name)``;
legacy plain-name shards remain supported. We merge them with the same
OFTLieAlgebraMerge core used for HF adapters, per (tp,pp) shard.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch

from orbit.peft.merge import get_strategy
from orbit.peft.merge.strategy import StateDict

_SHARD_GLOB = "adapter_megatron_tp*_pp*.pt"


def list_megatron_shards(adapter_dir: str) -> list[str]:
    shards = sorted(p.name for p in Path(adapter_dir).glob(_SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(f"no {_SHARD_GLOB} in {adapter_dir}")
    return shards


def merge_megatron_adapters(
    adapter_dirs: list[str],
    weights: list[float] | None = None,
    method: str = "oft",
) -> dict[str, StateDict]:
    """Merge the Megatron-native shards of N adapters, one (tp,pp) shard at a time.

    All adapters must expose the identical set of shard filenames. Returns
    {shard_filename: merged_state_dict}.
    """
    if len(adapter_dirs) < 2:
        raise ValueError("merging requires at least 2 adapters")
    shards_per_adapter = [list_megatron_shards(d) for d in adapter_dirs]  # each already sorted
    if any(s != shards_per_adapter[0] for s in shards_per_adapter[1:]):
        raise ValueError(f"adapters expose different Megatron shard sets: {shards_per_adapter}")
    strategy = get_strategy(method)
    merged: dict[str, StateDict] = {}
    for shard in shards_per_adapter[0]:
        state_dicts = [
            torch.load(Path(d) / shard, map_location="cpu", weights_only=True)
            for d in adapter_dirs
        ]
        merged[shard] = strategy.merge(state_dicts, weights)
    return merged


def write_megatron_adapter(
    merged_shards: dict[str, StateDict],
    src_config_dir: str,
    output_dir: str,
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for shard, state_dict in merged_shards.items():
        torch.save(state_dict, out / shard)
    shutil.copyfile(Path(src_config_dir) / "adapter_config.json", out / "adapter_config.json")
    return str(out)
