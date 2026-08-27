#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Direct-write INT4 checkpoint converter for large DeepSeek/Kimi-style models.

This path never instantiates a full BF16 Megatron model. It builds a
single-rank Megatron model on the meta device only for name resolution and
checkpoint metadata, then streams HF tensors directly into a prebuilt Megatron
distributed checkpoint state dict.

Expert INT4 tensors are processed one parameter at a time:

    HF INT4 -> BF16 scratch -> Megatron merge/concat -> INT4 buffers

This still uses temporary BF16 scratch for one weight at a time, but avoids the
catastrophic memory blow-up of allocating the entire Megatron model in BF16.

Current scope:
    - DeepSeek-V3 / Moonlight INT4 checkpoints
    - Kimi-K2.5 INT4 checkpoints
    - Single-rank direct conversion (TP=PP=EP=1 checkpoint output)

The produced checkpoint omits BF16 expert-weight placeholders and writes only
the INT4 triplets (``*_packed``, ``*_scale``, ``*_shape``) for expert linears.
"""

import argparse
import sys
import time
from contextlib import contextmanager
from typing import Any

from megatron.bridge import AutoBridge
from megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge import KimiK25VLBridge
from megatron.bridge.orbit.low_precision.common import (
    build_single_rank_meta_provider,
    patch_meta_init_for_te_modules,
)
from megatron.bridge.orbit.low_precision.int4 import build_int4_direct_model_state_dict
from megatron.bridge.orbit.model_bridges.deepseek_v3_int4_bridge import DeepSeekV3INT4Bridge
from megatron.bridge.orbit.model_bridges.llama_int4_bridge import LlamaINT4Bridge
from megatron.bridge.orbit.model_bridges.qwen3_int4_bridge import Qwen3INT4Bridge, Qwen3MoEINT4Bridge
from megatron.bridge.training.checkpointing import (
    get_checkpoint_name,
    save_checkpoint,
    save_tokenizer_assets,
)
from megatron.bridge.training.config import CheckpointConfig, ConfigContainer, LoggerConfig
from megatron.bridge.training.model_load_save import temporary_distributed_context
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.core.optimizer import OptimizerConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct-write HF INT4 -> Megatron checkpoint conversion",
    )
    parser.add_argument("--hf-model-path", required=True, help="Path to HF INT4 model directory")
    parser.add_argument("--megatron-path", required=True, help="Output Megatron checkpoint directory")
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=(
            "INT4 group size used when re-quantising weights. If omitted, read "
            "from HF config.quantization_config.config_groups.*.weights.group_size; "
            "falls back to 32 for legacy checkpoints that do not expose a "
            "quantization_config block (e.g. Kimi-K2.5)."
        ),
    )
    return parser.parse_args(argv)


@contextmanager
def keep_meta_model_unmaterialized():
    """Keep direct-conversion template models on meta instead of CUDA.

    The generic Bridge distributed-model builder materializes meta tensors onto
    CUDA after construction. Direct checkpoint conversion only needs module
    names, shapes, and sharded-state metadata, so CUDA materialization defeats
    the low-memory path for very large models.
    """

    import megatron.bridge.models.common.unimodal as unimodal_common
    import megatron.bridge.models.model_provider as model_provider

    original_unimodal_to_empty = unimodal_common.to_empty_if_meta_device
    original_provider_to_empty = getattr(model_provider, "to_empty_if_meta_device", None)

    def _identity_to_empty(module, *, device, recurse=True):
        return module

    unimodal_common.to_empty_if_meta_device = _identity_to_empty
    if original_provider_to_empty is not None:
        model_provider.to_empty_if_meta_device = _identity_to_empty

    try:
        yield
    finally:
        unimodal_common.to_empty_if_meta_device = original_unimodal_to_empty
        if original_provider_to_empty is not None:
            model_provider.to_empty_if_meta_device = original_provider_to_empty


def _bridge_name(architecture: Any) -> str:
    if isinstance(architecture, str):
        return architecture
    return architecture.__name__


def _infer_group_size(auto_bridge: AutoBridge, override: int | None) -> int:
    if override is not None:
        return override
    hf_config = getattr(auto_bridge.hf_pretrained, "config", None)
    qcfg = getattr(hf_config, "quantization_config", None)
    if isinstance(qcfg, dict):
        groups = qcfg.get("config_groups") or {}
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            weights = group.get("weights") or {}
            gs = weights.get("group_size")
            if isinstance(gs, int) and gs > 0:
                print(f"Inferred group_size={gs} from HF quantization_config")
                return gs
        top_gs = qcfg.get("group_size")
        if isinstance(top_gs, int) and top_gs > 0:
            print(f"Inferred group_size={top_gs} from HF quantization_config")
            return top_gs
    print("No group_size found in HF config; falling back to 32 (legacy default)")
    return 32


def _select_int4_bridge(auto_bridge: AutoBridge) -> Any:
    architecture_name = _bridge_name(auto_bridge._causal_lm_architecture)
    if architecture_name == "KimiK25ForConditionalGeneration":
        return KimiK25VLBridge()
    if architecture_name in {"DeepseekV3ForCausalLM", "DeepSeekV3ForCausalLM"}:
        return DeepSeekV3INT4Bridge()
    if architecture_name == "LlamaForCausalLM":
        return LlamaINT4Bridge()
    if architecture_name == "Qwen3ForCausalLM":
        return Qwen3INT4Bridge()
    if architecture_name == "Qwen3MoeForCausalLM":
        return Qwen3MoEINT4Bridge()
    raise ValueError(f"Unsupported INT4 direct-write architecture: {architecture_name}")


def _save_direct_checkpoint(
    provider: Any,
    path: str,
    model_state: dict[str, Any],
    *,
    pg_collection: Any,
    hf_tokenizer_path: str | None,
    hf_tokenizer_kwargs: dict[str, Any] | None,
) -> None:
    tokenizer_config = None
    if hf_tokenizer_path is not None:
        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=str(hf_tokenizer_path),
            hf_tokenizer_kwargs=hf_tokenizer_kwargs or {},
        )

    state = GlobalState()
    if hasattr(state, "train_state") and hasattr(state.train_state, "step"):
        state.train_state.step = 0

    state.cfg = ConfigContainer(
        model=provider,
        train=None,
        optimizer=OptimizerConfig(use_distributed_optimizer=False),
        ddp=None,
        scheduler=None,
        dataset=None,
        logger=LoggerConfig(),
        tokenizer=tokenizer_config,
        checkpoint=CheckpointConfig(
            async_save=False,
            # Force MCore's async strategy. The default "nvrx" path imports
            # nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async
            # .get_write_results_queue, which newer versions of the package
            # renamed to _get_write_results_queue, crashing even with
            # async_save=False because save_checkpoint still resolves the
            # strategy.
            async_strategy="mcore",
            save=str(path),
            save_optim=False,
            save_rng=False,
            ckpt_format="torch_dist",
            dist_ckpt_optim_fully_reshardable=True,
            fully_parallel_save=False,
            storage_writers_per_rank=16,
        ),
        dist=None,
    )

    prebuilt_state_dict = {
        "checkpoint_version": 3.0,
        "iteration": 0,
        "model": model_state,
    }

    t0 = time.monotonic()
    print("Saving checkpoint...", flush=True)
    from megatron.core.dist_checkpointing.strategies import torch as torch_dist_strategy

    original_have_nvrx = torch_dist_strategy.HAVE_NVRX
    torch_dist_strategy.HAVE_NVRX = False
    try:
        save_checkpoint(
            state=state,
            model=[],
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
            prebuilt_state_dict=prebuilt_state_dict,
            pg_collection=pg_collection,
        )
    finally:
        torch_dist_strategy.HAVE_NVRX = original_have_nvrx
    print(f"Checkpoint saved in {time.strftime('%H:%M:%S', time.gmtime(time.monotonic() - t0))}", flush=True)

    if tokenizer_config is not None:
        tokenizer = build_tokenizer(tokenizer_config)
        checkpoint_name = get_checkpoint_name(str(path), 0, release=False)
        save_tokenizer_assets(tokenizer, tokenizer_config, checkpoint_name)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"Converting INT4 checkpoint directly: {args.hf_model_path} -> {args.megatron_path}")
    print("Using single-process meta-model conversion (TP=PP=EP=1 checkpoint write)")

    auto_bridge, provider = build_single_rank_meta_provider(
        args.hf_model_path,
        trust_remote_code=True,
    )
    int4_bridge = _select_int4_bridge(auto_bridge)
    group_size = _infer_group_size(auto_bridge, args.group_size)

    if hasattr(provider, "finalize"):
        provider.finalize()

    trust_remote_code = getattr(auto_bridge.hf_pretrained, "trust_remote_code", False)
    tokenizer_kwargs = {"trust_remote_code": True} if trust_remote_code else None

    patch_meta_init_for_te_modules()

    with temporary_distributed_context(backend="gloo"):
        with keep_meta_model_unmaterialized():
            meta_model = provider.provide_distributed_model(
                wrap_with_ddp=False,
                use_cpu_initialization=True,
                init_model_with_meta_device=True,
                mixed_precision_wrapper=None,
            )

        pg_collection = get_pg_collection(meta_model)
        model_template = meta_model[0].sharded_state_dict(metadata={"dp_cp_group": pg_collection.dp_cp})
        model_state = build_int4_direct_model_state_dict(
            int4_bridge,
            auto_bridge.hf_pretrained,
            meta_model,
            model_template,
            group_size=group_size,
        )

        del meta_model
        del model_template

        _save_direct_checkpoint(
            provider,
            args.megatron_path,
            model_state,
            pg_collection=pg_collection,
            hf_tokenizer_path=args.hf_model_path,
            hf_tokenizer_kwargs=tokenizer_kwargs,
        )

    print(f"Done. Direct INT4 Megatron checkpoint saved to: {args.megatron_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
