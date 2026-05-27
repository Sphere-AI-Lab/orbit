#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distributed HF -> Megatron torch_dist converter for DeepSeek V4.

Launch with ``torchrun`` so multi-rank TP/EP works for the 800 GB DSV4-Pro
release. The companion shell wrapper is
``scripts/conversion/convert_dsv4_hf_to_megatron.sh``.

Direct invocation:

  torchrun --nproc-per-node=$N tools/convert_dsv4_hf_to_megatron.py \\
      --hf-path /path/to/hf/dsv4_ckpt \\
      --megatron-path /path/to/megatron_out \\
      --tp $TP --ep $EP --pp $PP --cp $CP --etp $ETP

For the 6-layer DSV4-Flash debug ckpt (~25 GB) a single rank suffices
(TP=EP=PP=CP=ETP=1). Pass the resulting torch_dist output directory to
Orbit launchers as ``MEGATRON_LOAD``.

The flow is the canonical bridge path used by orbit training startup:

  1. ``AutoBridge.from_hf_pretrained(...)`` registers ``DeepSeekV4Bridge``
     against ``DeepseekV4ForCausalLM``.
  2. ``auto_bridge.to_megatron_model(use_cpu_initialization=...)`` builds the
     DSV4 model. By default it builds on CUDA; set
     ``DSV4_MEGATRON_USE_CPU_INIT=1`` or pass ``--use-cpu-initialization`` for
     large Pro conversions where the temporary vanilla TE MoE block OOMs before
     the DSV4 decoder swap can replace it. The
     bridge's ``provider_bridge`` wires the experimental DSV4 attention spec,
     and the registered pre-wrap hook swaps the vanilla TransformerBlock for
     ``DeepSeekV4TransformerBlock``
     (``deepseek_v4_bridge.py:_swap_decoder_to_dsv4``) before
     ``load_weights_hf_to_megatron`` fills the parameters. ``Float16Module``
     is skipped because its ``module.bfloat16()`` call hits the
     ``Float4_e2m1fn_x2`` packed expert weights, which TE's cast hook
     rejects. The driver defaults ``DSV4_STREAM_SAFETENSORS=1`` so Bridge reads
     only requested tensors from the official single-shard Pro stage instead of
     materializing the full 810 GB safetensors file in each rank.
  3. ``auto_bridge.save_megatron_model(model, megatron_path,
     low_memory_save=False)`` writes a torch_dist checkpoint. The
     low-memory factory-expansion path is incompatible with DSV4's FP4
     experts (it produces ShardedTensors whose ``local_shape`` disagrees
     with ``data.size()`` and trips the torch_dist planner assertion); the
     standard save path holds the full state once.

The HF ckpt must have HF-shim config fields (``num_hidden_layers``,
``hidden_size``, ``model_type=deepseek_v4``, ``auto_map``). If your source
checkpoint is a native-only DeepSeek inference directory, run
``v4_plan/make_dsv4_hf_shim_config.py`` first.
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

import torch


_SAFETENSORS_HANDLE_CACHE: dict[str, object] = {}
_SAFETENSORS_CONTEXT_CACHE: dict[str, object] = {}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got: {value!r}")


def _close_safetensors_handles() -> None:
    for context in list(_SAFETENSORS_CONTEXT_CACHE.values()):
        exit_fn = getattr(context, "__exit__", None)
        if exit_fn is not None:
            exit_fn(None, None, None)
    _SAFETENSORS_CONTEXT_CACHE.clear()
    _SAFETENSORS_HANDLE_CACHE.clear()


atexit.register(_close_safetensors_handles)


def _get_safetensors_handle(file_path: Path):
    from safetensors import safe_open

    cache_key = str(file_path)
    handle = _SAFETENSORS_HANDLE_CACHE.get(cache_key)
    if handle is not None:
        return handle

    context = safe_open(cache_key, framework="pt", device="cpu")
    enter_fn = getattr(context, "__enter__", None)
    handle = enter_fn() if enter_fn is not None else context
    _SAFETENSORS_CONTEXT_CACHE[cache_key] = context
    _SAFETENSORS_HANDLE_CACHE[cache_key] = handle
    return handle


def _stream_safetensors_load_tensors(source, keys_to_load: list[str]) -> dict[str, torch.Tensor]:
    """Load only requested tensors from safetensors files.

    Megatron-Bridge's default loader reads an entire safetensors shard into RAM
    to avoid many mmap handles. DeepSeek V4 Pro official staging creates one
    ~810 GB shard, so that behavior can OOM when repeated across ranks.
    """
    if not keys_to_load:
        return {}

    loaded_tensors: dict[str, torch.Tensor] = {}
    remaining_keys = set(keys_to_load)
    key_to_filename_map = source.key_to_filename_map

    if key_to_filename_map:
        file_to_keys_map: dict[str, list[str]] = defaultdict(list)
        for key in list(remaining_keys):
            if key in key_to_filename_map:
                file_to_keys_map[key_to_filename_map[key]].append(key)

        for filename, keys_in_file in file_to_keys_map.items():
            file_path = Path(source.path) / filename
            if not file_path.exists():
                continue
            handle = _get_safetensors_handle(file_path)
            available_keys = set(handle.keys())
            for key in keys_in_file:
                if key in available_keys:
                    loaded_tensors[key] = handle.get_tensor(key)
                    remaining_keys.discard(key)

    if remaining_keys:
        for safetensor_file_path in glob(str(Path(source.path) / "*.safetensors")):
            if not remaining_keys:
                break
            handle = _get_safetensors_handle(Path(safetensor_file_path))
            available_keys = set(handle.keys())
            for key in list(remaining_keys):
                if key in available_keys:
                    loaded_tensors[key] = handle.get_tensor(key)
                    remaining_keys.remove(key)

    if remaining_keys:
        raise KeyError(f"Keys not found in safetensors from {source.model_name_or_path}: {remaining_keys}")

    return loaded_tensors


def install_streaming_safetensors_loader() -> bool:
    """Patch Bridge's SafeTensorsStateSource to avoid full-shard RAM loads."""
    from megatron.bridge.models.hf_pretrained import state as hf_state

    current_loader = hf_state.SafeTensorsStateSource.load_tensors
    if getattr(current_loader, "_dsv4_streaming_safe_open", False):
        return False

    _stream_safetensors_load_tensors._dsv4_streaming_safe_open = True
    hf_state.SafeTensorsStateSource.load_tensors = _stream_safetensors_load_tensors
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hf-path", required=True, type=Path, help="HF DSV4 ckpt directory")
    parser.add_argument(
        "--megatron-path", required=True, type=Path, help="Output Megatron torch_dist directory"
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor model parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline model parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert model parallel size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallel size")
    parser.add_argument(
        "--use-cpu-initialization",
        action="store_true",
        help=(
            "Build/load the Megatron model on CPU before saving. Also enabled by "
            "DSV4_MEGATRON_USE_CPU_INIT=1; useful for DSV4-Pro when the temporary "
            "vanilla TE MoE block OOMs on GPU before the DSV4 swap."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    base_model_size = args.tp * args.pp * args.cp
    if world_size % base_model_size != 0:
        if rank == 0:
            print(
                f"ERROR: WORLD_SIZE={world_size} is not divisible by "
                f"tp*pp*cp={base_model_size}.",
                file=sys.stderr,
                flush=True,
            )
        return 2

    expert_model_size = args.etp * args.ep * args.pp
    if world_size % expert_model_size != 0:
        if rank == 0:
            print(
                f"ERROR: WORLD_SIZE={world_size} is not divisible by "
                f"etp*ep*pp={expert_model_size}.",
                file=sys.stderr,
                flush=True,
            )
        return 2

    if rank == 0:
        print(
            f"[rank0] world_size={world_size} tp={args.tp} pp={args.pp} cp={args.cp} "
            f"dp={world_size // base_model_size} ep={args.ep} etp={args.etp} "
            f"edp={world_size // expert_model_size}",
            flush=True,
        )
        print(f"[rank0] hf_path:       {args.hf_path}", flush=True)
        print(f"[rank0] megatron_path: {args.megatron_path}", flush=True)

    if not args.hf_path.is_dir():
        print(f"ERROR: HF path not a directory: {args.hf_path}", file=sys.stderr, flush=True)
        return 2

    cpu_only = env_flag("DSV4_CPU_ONLY", default=False)
    use_cpu_initialization = cpu_only or args.use_cpu_initialization or env_flag(
        "DSV4_MEGATRON_USE_CPU_INIT", default=False
    )
    stream_safetensors = env_flag("DSV4_STREAM_SAFETENSORS", default=True)

    dist_backend = os.environ.get("DSV4_DIST_BACKEND", "gloo" if cpu_only else "nccl").strip().lower()
    if dist_backend not in {"nccl", "gloo"}:
        raise ValueError(f"DSV4_DIST_BACKEND must be 'nccl' or 'gloo', got: {dist_backend!r}")
    if not cpu_only:
        torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=dist_backend)

    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=args.etp,
        context_parallel_size=args.cp,
    )
    if not cpu_only:
        from megatron.core.tensor_parallel.random import (
            get_cuda_rng_tracker,
            model_parallel_cuda_manual_seed,
        )

        if "model-parallel-rng" not in get_cuda_rng_tracker().get_states():
            model_parallel_cuda_manual_seed(0)

    torch.set_default_dtype(torch.bfloat16)
    os.environ.setdefault("MEGATRON_USE_KV_QAT", "1")

    from megatron.bridge import AutoBridge

    if rank == 0:
        print(
            f"[rank0] DSV4_CPU_ONLY={int(cpu_only)} dist_backend={dist_backend} "
            f"use_cpu_initialization={use_cpu_initialization}",
            flush=True,
        )

    if stream_safetensors:
        install_streaming_safetensors_loader()
        if rank == 0:
            print(
                "[rank0] DSV4_STREAM_SAFETENSORS=1: using per-key safe_open loader",
                flush=True,
            )

    if rank == 0:
        t0 = time.monotonic()
        print(f"[rank0] AutoBridge.from_hf_pretrained({args.hf_path}) ...", flush=True)
    auto_bridge = AutoBridge.from_hf_pretrained(str(args.hf_path), trust_remote_code=True)
    if rank == 0:
        print(f"[rank0]   done in {time.monotonic() - t0:.1f}s", flush=True)

    if rank == 0:
        t0 = time.monotonic()
        print(
            f"[rank0] auto_bridge.to_megatron_model(use_cpu_initialization={use_cpu_initialization}, "
            "mixed_precision_wrapper=None) ... (DSV4 experimental block + "
            "load_weights_hf_to_megatron)",
            flush=True,
        )
    model = auto_bridge.to_megatron_model(
        wrap_with_ddp=False,
        use_cpu_initialization=use_cpu_initialization,
        mixed_precision_wrapper=None,
    )
    if rank == 0:
        print(f"[rank0]   model built + HF weights loaded in {time.monotonic() - t0:.1f}s", flush=True)

    if rank == 0:
        t0 = time.monotonic()
        print(
            f"[rank0] auto_bridge.save_megatron_model(model, {args.megatron_path}) "
            "(torch_dist, low_memory_save=False) ...",
            flush=True,
        )
    auto_bridge.save_megatron_model(
        model,
        str(args.megatron_path),
        hf_tokenizer_path=str(args.hf_path),
        hf_tokenizer_kwargs={"trust_remote_code": True},
        low_memory_save=False,
    )
    if rank == 0:
        print(f"[rank0]   saved in {time.monotonic() - t0:.1f}s", flush=True)
        print(f"[rank0] DONE -> {args.megatron_path}", flush=True)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
