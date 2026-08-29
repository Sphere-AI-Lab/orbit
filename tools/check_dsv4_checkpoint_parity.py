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

"""Byte-equality parity check for DSV4-Flash HF -> Megatron conversion.

Two legs (select with --mode):

  * ``in_memory`` -- Build a Megatron model on CUDA from the HF checkpoint
    via ``DeepSeekV4Bridge.load_weights_hf_to_megatron``, then call
    ``stream_weights_megatron_to_hf`` and byte-compare every emitted tensor
    against the source ``.safetensors``. Mirrors
    ``Megatron-Bridge/tests/integration_tests/test_dsv4_bridge_quant_scale_roundtrip.py``
    but exhaustively over the full key set instead of an 8-key sample.

  * ``disk`` -- Build an empty DSV4 model via
    ``auto_bridge.to_megatron_provider(load_weights=False)`` +
    ``provider.provide()`` (same code path as orbit's ``model_provider.py``
    at training boot, so ``_swap_decoder_to_dsv4`` fires), fill weights
    from the torch_dist at ``--megatron-path`` via orbit's
    ``load_dist_checkpoint(model, path)``, then stream back to HF and
    byte-compare. This is the orbit ``_load_checkpoint_dist`` load path.
    ``AutoBridge.load_megatron_model`` is intentionally NOT used: its
    ``build_and_load_model`` reconstructs the model from saved config
    without running the DSV4 decoder swap, so it builds vanilla LayerNorm
    with bias keys that DSV4 doesn't have.

  * ``both`` -- Run ``in_memory`` then ``disk``.

Byte-equality compares ``tensor.view(torch.uint8) + torch.equal``;
``torch.equal`` is unimplemented for the sub-byte-packed
``Float4_e2m1fn_x2`` dtype that holds DSV4's routed-expert weights.
Storage is one byte per element for ``Float4_e2m1fn_x2`` (two packed FP4
nibbles), ``Float8_e4m3fn``, and ``Float8_e8m0fnu``, so the uint8 view is
a zero-copy reinterpret.

Each key is classified as:
  * ``EQ``  -- bytes match exactly (expected for bf16 / fp32 / fp8_e4m3 /
    fp8_e8m0 / packed FP4 / int8 / int32 tensors).
  * ``VEQ`` -- bytes differ but values match after a known dtype
    promotion. DSV4 accepts int64->int32 for ``.tid2eid`` and source
    fp16/bf16->fp32 for the numerically sensitive norm, gate,
    compressor/indexer, and lm-head tensors, but only after exact downcast
    equality is verified.
  * ``MISMATCH`` -- dtype/shape regression or value drift.

Exit code is 0 iff every key is ``EQ`` or ``VEQ``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from glob import glob
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch


# Known dtype promotions where bytes differ but values are equal.
# (source_dtype, emitted_dtype, key_suffix) -> reason
_VALUE_EQ_PROMOTIONS: Dict[Tuple[torch.dtype, torch.dtype, str], str] = {
    (torch.int64, torch.int32, ".tid2eid"): "Megatron stores tid2eid as int32 (values in [0, 255], no overflow risk)",
}

_FP32_VALUE_EQ_PROMOTION_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"head\.weight"), "fp32 lm head"),
    (re.compile(r"norm\.weight"), "fp32 final RMSNorm"),
    (re.compile(r"layers\.\d+\.attn_norm\.weight"), "fp32 attention RMSNorm"),
    (re.compile(r"layers\.\d+\.ffn_norm\.weight"), "fp32 ffn RMSNorm"),
    (re.compile(r"layers\.\d+\.attn\.q_norm\.weight"), "fp32 q RMSNorm"),
    (re.compile(r"layers\.\d+\.attn\.kv_norm\.weight"), "fp32 kv RMSNorm"),
    (re.compile(r"layers\.\d+\.ffn\.gate\.weight"), "fp32 MoE gate"),
    (
        re.compile(r"layers\.\d+\.attn\.compressor\.(norm|wkv|wgate)\.weight"),
        "fp32 attention compressor",
    ),
    (
        re.compile(r"layers\.\d+\.attn\.indexer\.compressor\.(norm|wkv|wgate)\.weight"),
        "fp32 indexer compressor",
    ),
)


# Bridge bug workaround: MegatronParamMapping.gather_from_ep_ranks
# (Megatron-Bridge/src/megatron/bridge/models/conversion/param_mapping.py:782)
# calls torch.distributed.all_gather(group=ep_group) unconditionally,
# including when ep_size==1. NCCL refuses sub-byte / non-IEEE FP dtypes
# (Float8_e8m0fnu, Float4_e2m1fn_x2, Float8_e4m3fn). View as uint8 for the
# gather (same byte storage, NCCL-supported dtype), then view back.
_NCCL_REJECTED_DTYPES = (
    *(
        dt
        for dt in (
            getattr(torch, "float8_e8m0fnu", None),
            getattr(torch, "float4_e2m1fn_x2", None),
            getattr(torch, "float8_e4m3fn", None),
        )
        if dt is not None
    ),
)


def _install_ep_gather_nccl_workaround() -> None:
    """Monkey-patch MegatronParamMapping.gather_from_ep_ranks to uint8-view FP4/FP8 tensors."""
    import megatron.bridge.models.conversion.param_mapping as _pm

    if getattr(_pm.MegatronParamMapping.gather_from_ep_ranks, "_uint8_view_patch", False):
        return

    _orig = _pm.MegatronParamMapping.gather_from_ep_ranks

    def _patched_gather_from_ep_ranks(self, megatron_weights, megatron_module, hf_param_name):
        if (
            megatron_weights is None
            or megatron_weights.dtype not in _NCCL_REJECTED_DTYPES
            or self.ep_size == 1
        ):
            # ep_size==1: gather is a no-op identity. Skip NCCL entirely (also
            # works around the dtype block).
            if megatron_weights is not None and self.ep_size == 1:
                return {str(hf_param_name): megatron_weights}
            return _orig(self, megatron_weights, megatron_module, hf_param_name)

        # ep_size > 1 with FP4/FP8 dtype: gather byte storage as uint8, restore dtype.
        import re

        src_u8 = megatron_weights.view(torch.uint8)
        gathered_u8 = [torch.empty_like(src_u8) for _ in range(self.ep_size)]
        torch.distributed.all_gather(gathered_u8, src_u8, group=self.ep_group)
        gathered = [g.view(megatron_weights.dtype) for g in gathered_u8]

        if megatron_module is None:
            num_experts_per_rank = self.broadcast_obj_from_pp_rank(None, "num_experts_per_rank")
        else:
            num_experts = self._get_config(megatron_module).num_moe_experts
            num_experts_per_rank = self.broadcast_obj_from_pp_rank(
                num_experts // self.ep_size, "num_experts_per_rank"
            )
        from megatron.bridge.models.conversion.param_mapping import extract_expert_number_from_param

        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        local_expert_number = global_expert_number % num_experts_per_rank
        names = [
            re.sub(
                r"experts\.(\d+)",
                f"experts.{int(local_expert_number) + num_experts_per_rank * i}",
                str(hf_param_name),
            )
            for i in range(self.ep_size)
        ]
        out: Dict[str, torch.Tensor] = {}
        for name, t in zip(names, gathered):
            if name in out:
                out[name] = torch.cat([out[name], t.unsqueeze(0)], dim=0)
            else:
                out[name] = t.unsqueeze(0)
        for name in out:
            out[name] = out[name].squeeze()
        return out

    _patched_gather_from_ep_ranks._uint8_view_patch = True  # type: ignore[attr-defined]
    _pm.MegatronParamMapping.gather_from_ep_ranks = _patched_gather_from_ep_ranks


def _bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare two tensors by their raw storage bytes.

    Needed because torch.equal is unimplemented for sub-byte-packed dtypes like
    Float4_e2m1fn_x2. We compare via a uint8-view + tensor equality, which
    runs the vectorized C++ kernel (millions of bytes/sec). The previous
    ``bytes(untyped_storage())`` path iterates the storage in Python and is
    ~6 orders of magnitude slower on a 9 MB FP4 expert tensor.
    """
    a_c = a.contiguous().cpu()
    b_c = b.contiguous().cpu()
    if a_c.untyped_storage().nbytes() != b_c.untyped_storage().nbytes():
        return False
    # view(torch.uint8) flattens to a byte array regardless of the source
    # dtype (FP4 packed-x2, FP8 e4m3 / e8m0, bf16, int32, etc).
    a_b = a_c.view(torch.uint8).reshape(-1)
    b_b = b_c.view(torch.uint8).reshape(-1)
    return bool(torch.equal(a_b, b_b))


def _fp32_value_eq_promotion_reason(key: str) -> str | None:
    for pattern, reason in _FP32_VALUE_EQ_PROMOTION_PATTERNS:
        if pattern.fullmatch(key):
            return reason
    return None


def _build_source_index(hf_path: Path) -> Dict[str, str]:
    """Map every HF tensor key to the shard file containing it."""
    from safetensors import safe_open

    index: Dict[str, str] = {}
    shards = sorted(glob(str(hf_path / "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors shards in {hf_path}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                index[k] = shard
    return index


def _classify(
    key: str, emitted: torch.Tensor, source: torch.Tensor
) -> Tuple[str, str]:
    """Return (status, detail). Status in {EQ, VEQ, MISMATCH}."""
    if tuple(emitted.shape) != tuple(source.shape):
        return "MISMATCH", f"shape src={tuple(source.shape)} em={tuple(emitted.shape)}"

    if emitted.dtype == source.dtype:
        if _bytes_equal(emitted, source):
            return "EQ", ""
        return "MISMATCH", "bytes diverge (same dtype/shape)"

    # Dtype mismatch -- accept only documented value-preserving promotions.
    for (src_dt, em_dt, suffix), reason in _VALUE_EQ_PROMOTIONS.items():
        if source.dtype == src_dt and emitted.dtype == em_dt and key.endswith(suffix):
            promoted = emitted.cpu().to(source.dtype)
            if torch.equal(promoted, source.cpu()):
                return "VEQ", f"{source.dtype} -> {emitted.dtype} ({reason})"
            return "MISMATCH", f"value drift after {source.dtype} -> {emitted.dtype} promotion"
    fp32_reason = _fp32_value_eq_promotion_reason(key)
    if (
        fp32_reason is not None
        and source.dtype in (torch.bfloat16, torch.float16)
        and emitted.dtype == torch.float32
    ):
        downcast = emitted.cpu().to(source.dtype)
        if _bytes_equal(downcast, source):
            return "VEQ", f"{source.dtype} -> {emitted.dtype} ({fp32_reason})"
        return "MISMATCH", f"value drift after {source.dtype} -> {emitted.dtype} promotion"
    return "MISMATCH", f"dtype regression src={source.dtype} em={emitted.dtype}"


def _compare_emitted_vs_source(
    emitted: Dict[str, torch.Tensor], hf_path: Path
) -> int:
    """Compare an emitted HF-keyed dict against the on-disk source safetensors.

    Returns the process exit code (0 iff every key is EQ or VEQ).
    """
    src_index = _build_source_index(hf_path)

    common = sorted(set(emitted) & set(src_index))
    missing_in_emitted = sorted(set(src_index) - set(emitted))
    missing_in_source = sorted(set(emitted) - set(src_index))

    # Stream source tensors in shard-batches to keep peak RAM bounded.
    print(f"Comparing {len(common)} shared keys ...", flush=True)
    t0 = time.monotonic()
    log_interval = max(1, len(common) // 50)

    counters = defaultdict(int)
    mismatches: List[Tuple[str, str]] = []
    veq_examples: List[Tuple[str, str]] = []

    # Pull source tensors lazily, one at a time, to bound RAM (large FP4 experts).
    # Cache open shards through an ExitStack so each shard is opened once and
    # closed deterministically.
    from contextlib import ExitStack

    from safetensors import safe_open

    with ExitStack() as stack:
        open_shards: Dict[str, object] = {}
        for i, k in enumerate(common):
            shard = src_index[k]
            if shard not in open_shards:
                open_shards[shard] = stack.enter_context(
                    safe_open(shard, framework="pt", device="cpu")
                )
            src = open_shards[shard].get_tensor(k)
            em = emitted[k]
            status, detail = _classify(k, em, src)
            counters[status] += 1
            if status == "MISMATCH":
                mismatches.append((k, detail))
            elif status == "VEQ" and len(veq_examples) < 8:
                veq_examples.append((k, detail))
            if (i + 1) % log_interval == 0 or (i + 1) == len(common):
                elapsed = time.monotonic() - t0
                print(
                    f"  [{i+1}/{len(common)}] EQ={counters['EQ']} VEQ={counters['VEQ']} "
                    f"MISMATCH={counters['MISMATCH']} ({elapsed:.1f}s)",
                    flush=True,
                )

    print("")
    print("Verification summary")
    print(f"  shared keys:       {len(common)}")
    print(f"  EQ (byte-equal):   {counters['EQ']}")
    print(f"  VEQ (promoted):    {counters['VEQ']}")
    print(f"  MISMATCH:          {counters['MISMATCH']}")
    print(f"  in source only:    {len(missing_in_emitted)}")
    print(f"  in emitted only:   {len(missing_in_source)}")

    if veq_examples:
        print("")
        print("Value-equal (promoted) examples:")
        for k, detail in veq_examples:
            print(f"  {k}  --  {detail}")

    if mismatches:
        print("")
        print(f"MISMATCH details ({min(len(mismatches), 32)} of {len(mismatches)} shown):")
        for k, detail in mismatches[:32]:
            print(f"  {k}  --  {detail}")

    if missing_in_emitted:
        print("")
        print(f"Source keys not emitted by Megatron->HF stream ({min(20, len(missing_in_emitted))} of {len(missing_in_emitted)} shown):")
        for k in missing_in_emitted[:20]:
            print(f"  {k}")

    if missing_in_source:
        print("")
        print(f"Emitted keys not in source safetensors ({min(20, len(missing_in_source))} of {len(missing_in_source)} shown):")
        for k in missing_in_source[:20]:
            print(f"  {k}")

    ok = counters["MISMATCH"] == 0 and not missing_in_emitted and not missing_in_source
    print("")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _init_distributed_single_rank() -> None:
    """Set up a single-rank gloo PG so the bridge's distributed primitives work."""
    from megatron.core import parallel_state

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12377")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, rank=0, world_size=1)
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
    from megatron.core.tensor_parallel.random import (
        get_cuda_rng_tracker,
        model_parallel_cuda_manual_seed,
    )
    if "model-parallel-rng" not in get_cuda_rng_tracker().get_states():
        model_parallel_cuda_manual_seed(0)


def _set_official_dsv4_module_globals(ckpt_dir: Path) -> None:
    """Match the global config that DSV4 inference's ``model.py`` expects.

    Mirrors the prelude in test_dsv4_bridge_quant_scale_roundtrip.py so the
    bridge can build the experimental DSV4 attention/expert blocks.
    """
    inference_dir = os.environ.get("DSV4_INFERENCE_DIR", str(ckpt_dir / "inference"))
    if Path(inference_dir).exists() and inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    try:
        import model as official_model  # noqa: F401
    except Exception as exc:
        print(f"WARNING: could not import DSV4 inference model module: {exc}", flush=True)
        return

    official_model.world_size = 1
    official_model.rank = 0
    official_model.default_dtype = torch.float8_e4m3fn
    official_model.scale_fmt = "ue8m0"
    official_model.scale_dtype = torch.float8_e8m0fnu
    official_model.block_size = 128
    official_model.fp4_block_size = 32


def _native_config_to_hf(native_cfg: dict, ckpt_dir: Path) -> SimpleNamespace:
    """Translate DSV4 native config.json into the HF-style SimpleNamespace.

    Same translation as test_dsv4_bridge_quant_scale_roundtrip.py:
    DeepseekV4Bridge.provider_bridge reads HF-style fields, but the native
    config.json uses the DSV4 inference repo schema.
    """
    cfg = SimpleNamespace()
    cfg.architectures = ["DeepseekV4ForCausalLM"]
    cfg.model_type = "deepseek_v4"
    cfg.num_hidden_layers = native_cfg["n_layers"]
    cfg.hidden_size = native_cfg["dim"]
    cfg.num_attention_heads = native_cfg["n_heads"]
    cfg.num_key_value_heads = native_cfg["n_heads"]
    cfg.vocab_size = native_cfg["vocab_size"]
    cfg.max_position_embeddings = native_cfg.get("original_seq_len", 65536)
    cfg.rms_norm_eps = 1e-6
    cfg.initializer_range = 0.02
    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0
    cfg.tie_word_embeddings = False
    cfg.attention_bias = False
    cfg.mlp_bias = False
    cfg.use_qk_norm = True
    cfg.hidden_act = "silu"
    cfg.torch_dtype = "bfloat16"
    cfg.head_dim = native_cfg["head_dim"]
    cfg.qk_rope_head_dim = native_cfg["rope_head_dim"]
    cfg.q_lora_rank = native_cfg["q_lora_rank"]
    cfg.kv_lora_rank = native_cfg["head_dim"]
    cfg.qk_nope_head_dim = native_cfg["head_dim"]
    cfg.v_head_dim = native_cfg["head_dim"]
    cfg.first_k_dense_replace = 0
    cfg.moe_intermediate_size = native_cfg["moe_inter_dim"]
    cfg.n_routed_experts = native_cfg["n_routed_experts"]
    cfg.n_shared_experts = native_cfg["n_shared_experts"]
    cfg.num_experts_per_tok = native_cfg["n_activated_experts"]
    cfg.scoring_func = native_cfg["score_func"]
    cfg.routed_scaling_factor = native_cfg["route_scale"]
    cfg.aux_loss_alpha = 0.0
    cfg.swiglu_limit = native_cfg.get("swiglu_limit", 0.0)
    cfg.o_groups = native_cfg["o_groups"]
    cfg.o_lora_rank = native_cfg["o_lora_rank"]
    cfg.sliding_window = native_cfg["window_size"]
    cfg.compress_ratios = native_cfg.get("compress_ratios")
    cfg.compress_rope_theta = native_cfg.get("compress_rope_theta", 160000)
    cfg.hc_mult = native_cfg["hc_mult"]
    cfg.hc_sinkhorn_iters = native_cfg["hc_sinkhorn_iters"]
    cfg.hc_eps = native_cfg["hc_eps"]
    cfg.num_hash_layers = native_cfg.get("n_hash_layers", 0)
    cfg.num_nextn_predict_layers = native_cfg.get("n_mtp_layers", 0)
    cfg.index_n_heads = native_cfg["index_n_heads"]
    cfg.index_head_dim = native_cfg["index_head_dim"]
    cfg.index_topk = native_cfg["index_topk"]
    cfg.rope_theta = native_cfg.get("rope_theta", 10000)
    cfg.rope_scaling = {
        "type": "yarn",
        "factor": native_cfg.get("rope_factor", 1),
        "original_max_position_embeddings": native_cfg.get("original_seq_len", 65536),
        "beta_fast": native_cfg.get("beta_fast", 32),
        "beta_slow": native_cfg.get("beta_slow", 1),
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
    cfg.name_or_path = str(ckpt_dir)
    return cfg


def _make_dsv4_state_source(ckpt_dir: Path):
    """Per-tensor safe_open StateSource (handles non-IEEE FP4/FP8 dtypes).

    Lifted from test_dsv4_bridge_quant_scale_roundtrip.py.
    """
    from megatron.bridge.models.hf_pretrained.state import StateSource

    class _DSV4SafeTensorsStateSource(StateSource):  # type: ignore[misc]
        def __init__(self, path):
            self.path = Path(path)
            self._key_to_file: Dict[str, str] = {}
            self._scan()

        def _scan(self) -> None:
            from safetensors import safe_open

            for shard_path in sorted(glob(str(self.path / "*.safetensors"))):
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        self._key_to_file[k] = shard_path

        def get_all_keys(self) -> List[str]:
            return sorted(self._key_to_file.keys())

        def load_tensors(self, keys):
            from safetensors import safe_open

            by_file: Dict[str, List[str]] = defaultdict(list)
            for k in keys:
                shard = self._key_to_file.get(k)
                if shard is not None:
                    by_file[shard].append(k)
            out: Dict[str, torch.Tensor] = {}
            for shard_path, ks in by_file.items():
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for k in ks:
                        out[k] = f.get_tensor(k)
            return out

        def has_glob(self, pattern: str) -> bool:
            import fnmatch

            for key in self._key_to_file:
                if fnmatch.fnmatch(key, pattern):
                    return True
            return False

    return _DSV4SafeTensorsStateSource(ckpt_dir)


def _build_bridge_for_dsv4(hf_path: Path):
    """Return (bridge, pretrained) for DSV4 with the per-tensor StateSource.

    The default ``AutoBridge.from_hf_pretrained`` builds a ``PreTrainedCausalLM``
    whose state accessor and config parser do not handle DSV4's native
    ``config.json`` schema or non-IEEE FP4/FP8 dtypes. We mirror the test
    fixture: load native config, translate to HF schema, override the state
    accessor, then drive ``DeepSeekV4Bridge`` directly.
    """
    from megatron.bridge.orbit.model_bridges.deepseek_v4_bridge import DeepSeekV4Bridge
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
    from megatron.bridge.models.hf_pretrained.state import StateDict

    native_cfg = json.loads((hf_path / "config.json").read_text())
    hf_cfg = _native_config_to_hf(native_cfg, hf_path)

    pretrained = PreTrainedCausalLM(model_name_or_path=str(hf_path))
    pretrained._config = hf_cfg
    pretrained._state_dict_accessor = StateDict(_make_dsv4_state_source(hf_path))

    bridge = DeepSeekV4Bridge()
    return bridge, pretrained, native_cfg


def verify_in_memory(hf_path: Path) -> int:
    """In-memory leg: HF -> bridge.load_weights -> Megatron model -> stream -> compare."""
    if not torch.cuda.is_available():
        print("ERROR: in_memory leg requires CUDA (DSV4 attention kernels are CUDA-only)", flush=True)
        return 2

    os.environ.setdefault("MEGATRON_USE_KV_QAT", "1")
    _set_official_dsv4_module_globals(hf_path)

    torch.manual_seed(33377335)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)

    _init_distributed_single_rank()
    _install_ep_gather_nccl_workaround()

    from megatron.core.process_groups_config import ProcessGroupCollection

    bridge, pretrained, native_cfg = _build_bridge_for_dsv4(hf_path)

    print("Building Megatron model on CUDA (this may take ~90s for DSV4-debug) ...", flush=True)
    t_build = time.monotonic()
    provider = bridge.provider_bridge(pretrained)
    provider.vocab_size = native_cfg["vocab_size"]
    provider.params_dtype = torch.bfloat16
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    provider.finalize()
    model = provider.provide(pre_process=True, post_process=True).cuda()
    print(f"  model built in {time.monotonic() - t_build:.1f}s", flush=True)

    print("Loading HF weights into Megatron model ...", flush=True)
    t_load = time.monotonic()
    bridge.load_weights_hf_to_megatron(pretrained, model)
    model.eval()
    print(f"  weights loaded in {time.monotonic() - t_load:.1f}s", flush=True)

    print("Streaming Megatron -> HF ...", flush=True)
    t_stream = time.monotonic()
    emitted: Dict[str, torch.Tensor] = {}
    for name, t in bridge.stream_weights_megatron_to_hf(
        [model], pretrained, cpu=True, show_progress=False
    ):
        emitted[name] = t
    print(f"  streamed {len(emitted)} tensors in {time.monotonic() - t_stream:.1f}s", flush=True)

    rc = _compare_emitted_vs_source(emitted, hf_path)

    from megatron.core import parallel_state
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
    return rc


def verify_disk(hf_path: Path, megatron_path: Path) -> int:
    """Disk leg: build model via the orbit path, fill weights via
    ``load_dist_checkpoint``, then stream back to HF + byte-compare.

    This mirrors the orbit training boot:

      1. ``bridge.to_megatron_provider(load_weights=False)`` -> provider WITHOUT
         the HF-load pre-wrap hook. The DSV4 swap pre-wrap hook still fires
         when provider.provide() is called (registered separately by
         DeepSeekV4Bridge), so the model is built correctly with
         ``DeepSeekV4TransformerBlock`` + RMSNorm-no-bias.
      2. ``load_dist_checkpoint(model, megatron_path)`` fills the model's
         parameters from the saved torch_dist via
         ``dist_checkpointing.load(sharded_state_dict, resolved_dist_path)``.
         This is the same code path
         ``orbit/backends/megatron_utils/checkpoint.py:_load_checkpoint_dist``
         takes when ``LOAD_CKPT`` points at a torch_dist dir.
      3. Stream Megatron -> HF and byte-compare with the source safetensors.

    Note: ``AutoBridge.load_megatron_model(megatron_path)`` does NOT work here
    because its ``build_and_load_model`` reconstructs the model from the saved
    config without running the DSV4 decoder swap, so it builds the vanilla
    LayerNorm-with-bias structure and the dist-ckpt load misses a slot.
    """
    if not torch.cuda.is_available():
        print("ERROR: disk leg requires CUDA", flush=True)
        return 2

    os.environ.setdefault("MEGATRON_USE_KV_QAT", "1")
    _set_official_dsv4_module_globals(hf_path)

    torch.manual_seed(33377335)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)

    _init_distributed_single_rank()
    _install_ep_gather_nccl_workaround()

    from megatron.bridge import AutoBridge

    print(f"AutoBridge.from_hf_pretrained({hf_path}) ...", flush=True)
    t0 = time.monotonic()
    auto_bridge = AutoBridge.from_hf_pretrained(str(hf_path), trust_remote_code=True)
    print(f"  done in {time.monotonic() - t0:.1f}s", flush=True)

    print("bridge.to_megatron_provider(load_weights=False) + provider.provide(...) "
          "(orbit's model_provider.py path; runs DSV4 decoder swap, NO HF weight load) ...", flush=True)
    t0 = time.monotonic()
    provider = auto_bridge.to_megatron_provider(load_weights=False)
    if hasattr(provider, "finalize"):
        provider.finalize()
    model = provider.provide(pre_process=True, post_process=True).cuda()
    model.eval()
    print(f"  empty DSV4 model built in {time.monotonic() - t0:.1f}s", flush=True)

    print(f"load_dist_checkpoint(model, {megatron_path}) ...", flush=True)
    t0 = time.monotonic()
    # Same loader orbit's _load_checkpoint_dist calls. ORBIT_ROOT must be on
    # PYTHONPATH; the shell wrapper handles that via scripts/lib/tool_env.sh.
    from orbit.peft.megatron.low_precision_bootstrap import load_dist_checkpoint

    load_dist_checkpoint(model, str(megatron_path))
    print(f"  loaded in {time.monotonic() - t0:.1f}s", flush=True)

    print("Streaming Megatron -> HF ...", flush=True)
    t_stream = time.monotonic()
    emitted: Dict[str, torch.Tensor] = {}
    bridge = auto_bridge._model_bridge
    for tup in bridge.stream_weights_megatron_to_hf(
        [model], auto_bridge.hf_pretrained, cpu=True, show_progress=False
    ):
        emitted[tup.param_name if hasattr(tup, "param_name") else tup[0]] = (
            tup.weight if hasattr(tup, "weight") else tup[1]
        )
    print(f"  streamed {len(emitted)} tensors in {time.monotonic() - t_stream:.1f}s", flush=True)

    rc = _compare_emitted_vs_source(emitted, hf_path)

    from megatron.core import parallel_state
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
    return rc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-model-path", required=True, type=Path, help="HF DSV4-Flash checkpoint directory")
    parser.add_argument(
        "--megatron-path",
        type=Path,
        default=None,
        help="Megatron torch_dist checkpoint directory (required for --mode disk / both)",
    )
    parser.add_argument(
        "--mode",
        choices=["in_memory", "disk", "both"],
        default="both",
        help="Verification leg(s) to run (default: both)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.hf_model_path.exists():
        print(f"ERROR: HF path does not exist: {args.hf_model_path}", file=sys.stderr)
        return 2
    if args.mode in ("disk", "both"):
        if args.megatron_path is None:
            print("ERROR: --megatron-path required for --mode disk/both", file=sys.stderr)
            return 2
        if not args.megatron_path.exists():
            print(f"ERROR: Megatron path does not exist: {args.megatron_path}", file=sys.stderr)
            return 2

    rc = 0
    if args.mode in ("in_memory", "both"):
        print("=" * 70)
        print("LEG 1: in-memory HF -> Megatron -> HF byte-equality")
        print("=" * 70)
        rc_mem = verify_in_memory(args.hf_model_path)
        print(f"in_memory leg exit: {rc_mem}")
        rc = rc or rc_mem

    if args.mode in ("disk", "both"):
        print("")
        print("=" * 70)
        print("LEG 2: disk torch_dist -> Megatron -> HF byte-equality")
        print("=" * 70)
        rc_disk = verify_disk(args.hf_model_path, args.megatron_path)
        print(f"disk leg exit: {rc_disk}")
        rc = rc or rc_disk

    print("")
    print("OVERALL:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
