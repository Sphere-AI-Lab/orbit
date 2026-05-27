#!/usr/bin/env python3
"""Cross-repo DSV4 DeepGEMM FP8xFP4 parity check.

This is intentionally narrower than the runtime step-0 logprob parity tools:
it imports the local SGLang and Megatron-LM checkouts in one process, feeds
byte-identical quantized weights/layouts into both DSV4 DeepGEMM wrappers, and
checks bitwise parity plus batch invariance for common routed rows.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
SGLANG_PYTHON = PROJECT_ROOT / "sglang" / "python"
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"

os.environ.setdefault("DSV4_FP4_GEMM_BACKEND", "deepgemm")

for path in (str(MEGATRON_ROOT), str(SGLANG_PYTHON)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from megatron.core.transformer.experimental_attention_variant import (  # noqa: E402
    dsv4_kernels as meg_dsv4_kernels,
)
from megatron.core.transformer.experimental_attention_variant import (  # noqa: E402
    dsv4_grouped_fp4 as meg_grouped_fp4,
)
from sglang.srt.models import dsv4_kernels as sgl_dsv4_kernels  # noqa: E402
from sglang.srt.models import dsv4_moe_kernels as sgl_grouped_fp4  # noqa: E402


@dataclass(frozen=True)
class ShapeConfig:
    experts: int
    n: int
    k: int


_DEBUG_CKPT = Path(
    os.environ.get("DSV4_DEBUG_CKPT", "checkpoints/dsv4_bridge_emitted_ckpt")
)


def _parse_batch_list(value: str) -> list[int]:
    batches = [int(item) for item in value.split(",") if item.strip()]
    if not batches or any(batch <= 0 for batch in batches):
        raise argparse.ArgumentTypeError("batch list must contain positive integers")
    return sorted(set(batches))


def _align_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _shape_from_args(args: argparse.Namespace) -> ShapeConfig:
    if args.shape == "w1":
        return ShapeConfig(args.experts, args.inter_dim, args.hidden_dim)
    if args.shape == "w2":
        return ShapeConfig(args.experts, args.hidden_dim, args.inter_dim)
    raise ValueError(f"unsupported shape: {args.shape}")


def _counts_for_active_rows(active_rows: int, experts: int) -> list[int]:
    base = active_rows // experts
    rem = active_rows % experts
    return [base + (1 if expert < rem else 0) for expert in range(experts)]


def _make_routed_input(
    x_by_expert: torch.Tensor,
    counts: Iterable[int],
    *,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    chunks: list[torch.Tensor] = []
    pos_chunks: list[torch.Tensor] = []
    active_rows: list[int] = []
    keys: list[tuple[int, int]] = []
    row_offset = 0
    device = x_by_expert.device
    hidden = x_by_expert.shape[-1]

    for expert, count in enumerate(counts):
        if count == 0:
            continue
        aligned = _align_to(count, alignment)
        chunk = torch.zeros(
            aligned,
            hidden,
            device=device,
            dtype=x_by_expert.dtype,
        )
        chunk[:count].copy_(x_by_expert[expert, :count])
        chunks.append(chunk)

        pos = torch.full((aligned,), -1, device=device, dtype=torch.int32)
        pos[:count] = expert
        pos_chunks.append(pos)

        active_rows.extend(range(row_offset, row_offset + count))
        keys.extend((expert, local_row) for local_row in range(count))
        row_offset += aligned

    if not chunks:
        raise ValueError("at least one routed row is required")
    x = torch.cat(chunks, dim=0).contiguous()
    pos_to_expert = torch.cat(pos_chunks, dim=0).contiguous()
    active = torch.tensor(active_rows, device=device, dtype=torch.long)
    return x, pos_to_expert, active, keys


def _summarize_diff(name: str, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    equal = torch.equal(actual, expected)
    diff = (actual.float() - expected.float()).abs()
    nonzero = int((actual != expected).sum().item())
    print(
        f"{name}: equal={equal} mean_abs={diff.mean().item():.10g} "
        f"max_abs={diff.max().item():.10g} nonzero={nonzero}/{actual.numel()}",
        flush=True,
    )
    return equal


def _summarize_float_diff(name: str, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    equal = torch.equal(actual, expected)
    diff = (actual.float() - expected.float()).abs()
    nonzero = int((actual != expected).sum().item())
    print(
        f"{name}: equal={equal} mean_abs={diff.mean().item():.10g} "
        f"max_abs={diff.max().item():.10g} nonzero={nonzero}/{actual.numel()}",
        flush=True,
    )
    return equal


def _filter_expected_leftovers(leftover: Iterable[str]) -> list[str]:
    return [
        key
        for key in leftover
        if not key.startswith("mtp.") and not key.endswith(".kv_cache")
    ]


def _load_debug_config(
    ckpt_dir: Path,
    *,
    batch_size: int,
    seq_len: int,
) -> SimpleNamespace:
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    cfg.setdefault("norm_eps", cfg.get("rms_norm_eps", 1.0e-6))
    cfg.setdefault("score_func", cfg.get("scoring_func", "sqrtsoftplus"))
    cfg.setdefault("route_scale", cfg.get("routed_scaling_factor", 1.5))
    cfg.setdefault("n_hash_layers", cfg.get("num_hash_layers", 0))
    cfg.setdefault("n_layers", cfg.get("num_hidden_layers", 6))
    cfg.setdefault("n_routed_experts", cfg.get("num_experts", 256))
    cfg.setdefault("n_activated_experts", cfg.get("num_experts_per_tok", 6))
    cfg.setdefault("moe_inter_dim", cfg.get("moe_intermediate_size", 2048))
    cfg["max_batch_size"] = max(int(cfg.get("max_batch_size", 1)), batch_size)
    cfg["max_seq_len"] = max(int(cfg.get("max_seq_len", 0)), seq_len)
    return SimpleNamespace(**cfg)


def _make_debug_input_ids(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str,
) -> torch.Tensor:
    base = torch.arange(seq_len, device=device, dtype=torch.long)
    rows = []
    for batch in range(batch_size):
        rows.append(((base * 1543 + 17 + batch * 7919) % vocab_size).long())
    return torch.stack(rows, dim=0).contiguous()


def _enable_identity_expert_oft(model: torch.nn.Module, *, block_size: int) -> None:
    for layer in model.model.layers:
        mlp = layer.mlp
        for proj in ("w1", "w2", "w3"):
            mlp.ensure_dsv4_expert_oft_r(
                proj,
                block_size=block_size,
                dtype=torch.bfloat16,
            )


def _set_full_model_debug_env() -> None:
    defaults = {
        "SGLANG_DSV4_DET_FP8_GEMM": "1",
        "MEGATRON_DSV4_DET_FP8_GEMM": "1",
        "SGLANG_DSV4_USE_TILEKERNELS": "1",
        "MEGATRON_DSV4_USE_TILEKERNELS": "1",
        "MEGATRON_DSV4_ATTN_IMPL": "tilelang",
        "MEGATRON_USE_KV_QAT": "1",
        "MEGATRON_DSV4_CKPT_VERSION": "0415",
        "SGLANG_DSV4_GATE_LINEAR_IMPL": "persistent",
        "MEGATRON_DSV4_GATE_LINEAR_IMPL": "persistent",
        "SGLANG_DSV4_CANONICAL_RMSNORM": "1",
        "MEGATRON_DSV4_CANONICAL_RMSNORM": "1",
        "SGLANG_DSV4_CANONICAL_RMSNORM_IMPL": "persistent",
        "MEGATRON_DSV4_CANONICAL_RMSNORM_IMPL": "persistent",
        "SGLANG_DSV4_CANONICAL_COMPRESSOR_LINEAR": "1",
        "MEGATRON_DSV4_CANONICAL_COMPRESSOR_LINEAR": "1",
        "SGLANG_DSV4_CANONICAL_COMPRESSOR_LINEAR_IMPL": "fixed_order",
        "MEGATRON_DSV4_CANONICAL_COMPRESSOR_LINEAR_IMPL": "fixed_order",
        "SGLANG_DSV4_CANONICAL_COMPRESS_TOPK_ORDER": "1",
        "MEGATRON_DSV4_CANONICAL_COMPRESS_TOPK_ORDER": "1",
        "SGLANG_DSV4_PREFILL_FIXED_WINDOW_LAYOUT": "1",
        "MEGATRON_DSV4_PREFILL_FIXED_WINDOW_LAYOUT": "1",
        "SGLANG_DSV4_CANONICAL_LM_HEAD": "1",
        "MEGATRON_DSV4_CANONICAL_LM_HEAD": "1",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


def _run_sglang_full_debug_model(
    args: argparse.Namespace,
    cfg: SimpleNamespace,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    from sglang.srt.models.deepseek_v4_loader import load_dsv4_native_checkpoint
    from sglang.srt.model_loader.utils import set_default_torch_dtype

    model = DeepseekV4ForCausalLM(cfg)
    leftover = _filter_expected_leftovers(
        load_dsv4_native_checkpoint(model, str(args.ckpt_dir))
    )
    if leftover:
        raise RuntimeError(f"SGLang leftover checkpoint keys: {leftover[:20]}")

    model = model.cuda().eval()
    if args.force_oft:
        _enable_identity_expert_oft(model, block_size=args.oft_block_size)

    traces: dict[str, torch.Tensor] = {}
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(
            layer.self_attn.register_forward_hook(
                lambda _module, _inputs, output, layer_idx=layer_idx: traces.__setitem__(
                    f"layer_{layer_idx}_attn", output.detach().cpu()
                )
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                lambda _module, _inputs, output, layer_idx=layer_idx: traces.__setitem__(
                    f"layer_{layer_idx}_mlp", output.detach().cpu()
                )
            )
        )
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, layer_idx=layer_idx: traces.__setitem__(
                    f"layer_{layer_idx}", output.detach().cpu()
                )
            )
        )

    import sglang.srt.model_executor.cuda_graph_runner as cuda_graph_runner

    old_capture = cuda_graph_runner.get_is_capture_mode
    old_graph_env = os.environ.get("SGLANG_DSV4_MOE_CUDA_GRAPH")
    try:
        if args.sglang_moe_path == "grouped":
            os.environ["SGLANG_DSV4_MOE_CUDA_GRAPH"] = "1"
            cuda_graph_runner.get_is_capture_mode = lambda: True
        elif args.sglang_moe_path == "eager":
            os.environ["SGLANG_DSV4_MOE_CUDA_GRAPH"] = "0"

        with torch.inference_mode():
            with set_default_torch_dtype(torch.bfloat16):
                h_sbd = model.model.forward_stateful(input_ids, start_pos=0, req_idx=0)
        torch.cuda.synchronize()
        traces["final"] = h_sbd.detach().cpu()
        return traces["final"], traces
    finally:
        for hook in hooks:
            hook.remove()
        cuda_graph_runner.get_is_capture_mode = old_capture
        if old_graph_env is None:
            os.environ.pop("SGLANG_DSV4_MOE_CUDA_GRAPH", None)
        else:
            os.environ["SGLANG_DSV4_MOE_CUDA_GRAPH"] = old_graph_env
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _run_megatron_full_debug_model(
    args: argparse.Namespace,
    cfg: SimpleNamespace,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4 import (
        DeepSeekV4TransformerBlock,
    )
    from megatron.core.transformer.experimental_attention_variant.dsv4_loader import (
        _make_dsv4_config_from_official,
        load_dsv4_native_safetensors,
    )

    meg_cfg = _make_dsv4_config_from_official(cfg)
    block = DeepSeekV4TransformerBlock(config=meg_cfg).cuda().eval()
    embed_w = torch.empty(
        cfg.vocab_size,
        cfg.dim,
        device="cuda",
        dtype=torch.float32,
    )
    head_w = torch.empty(
        cfg.vocab_size,
        cfg.dim,
        device="cuda",
        dtype=torch.float32,
    )
    leftover = _filter_expected_leftovers(
        load_dsv4_native_safetensors(
            block=block,
            embed_weight=embed_w,
            head_weight=head_w,
            ckpt_dir=str(args.ckpt_dir),
            mp_rank=0,
            mp_world=1,
        )
    )
    if leftover:
        raise RuntimeError(f"Megatron leftover checkpoint keys: {leftover[:20]}")

    if args.force_oft:
        for layer in block.layers:
            for proj in ("w1", "w2", "w3"):
                layer.mlp.ensure_dsv4_expert_oft_r(
                    proj,
                    block_size=args.oft_block_size,
                    dtype=torch.bfloat16,
                )

    traces: dict[str, torch.Tensor] = {}
    hooks = []
    for layer_idx, layer in enumerate(block.layers):
        hooks.append(
            layer.self_attention.register_forward_hook(
                lambda _module, _inputs, output, layer_idx=layer_idx: traces.__setitem__(
                    f"layer_{layer_idx}_attn", output.detach().cpu()
                )
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                lambda _module, _inputs, output, layer_idx=layer_idx: traces.__setitem__(
                    f"layer_{layer_idx}_mlp", output.detach().cpu()
                )
            )
        )
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, layer_idx=layer_idx: traces.__setitem__(
                    f"layer_{layer_idx}", output.detach().cpu()
                )
            )
        )

    input_ids = input_ids.to("cuda")
    try:
        with torch.inference_mode():
            emb = F.embedding(input_ids, embed_w).to(torch.bfloat16)
            h_sbd = emb.permute(1, 0, 2).contiguous()
            h_sbd = block(h_sbd, input_ids=input_ids)
        torch.cuda.synchronize()
        traces["final"] = h_sbd.detach().cpu()
        return traces["final"], traces
    finally:
        for hook in hooks:
            hook.remove()
        del block, embed_w, head_w
        gc.collect()
        torch.cuda.empty_cache()


def _make_debug_hidden(
    *,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    seed: int,
    device: str,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.device("cpu"):
        hidden = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            dtype=torch.float32,
            generator=generator,
        ).to(torch.bfloat16)
    return hidden.to(device)


def _run_sglang_moe_debug(
    args: argparse.Namespace,
    cfg: SimpleNamespace,
    input_ids: torch.Tensor,
    hidden: torch.Tensor,
) -> dict[str, torch.Tensor]:
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    from sglang.srt.models.deepseek_v4_loader import load_dsv4_native_checkpoint
    from sglang.srt.model_loader.utils import set_default_torch_dtype

    model = DeepseekV4ForCausalLM(cfg).cuda().eval()
    leftover = _filter_expected_leftovers(
        load_dsv4_native_checkpoint(model, str(args.ckpt_dir))
    )
    if leftover:
        raise RuntimeError(f"SGLang leftover checkpoint keys: {leftover[:20]}")
    if args.force_oft:
        _enable_identity_expert_oft(model, block_size=args.oft_block_size)

    import sglang.srt.model_executor.cuda_graph_runner as cuda_graph_runner

    old_capture = cuda_graph_runner.get_is_capture_mode
    old_graph_env = os.environ.get("SGLANG_DSV4_MOE_CUDA_GRAPH")
    traces: dict[str, torch.Tensor] = {}
    try:
        if args.sglang_moe_path == "grouped":
            os.environ["SGLANG_DSV4_MOE_CUDA_GRAPH"] = "1"
            cuda_graph_runner.get_is_capture_mode = lambda: True
        elif args.sglang_moe_path == "eager":
            os.environ["SGLANG_DSV4_MOE_CUDA_GRAPH"] = "0"

        with torch.inference_mode(), set_default_torch_dtype(torch.bfloat16):
            for layer_idx, layer in enumerate(model.model.layers):
                out = layer.mlp(hidden, input_ids)
                traces[f"layer_{layer_idx}_mlp"] = out.detach().cpu()
        torch.cuda.synchronize()
        return traces
    finally:
        cuda_graph_runner.get_is_capture_mode = old_capture
        if old_graph_env is None:
            os.environ.pop("SGLANG_DSV4_MOE_CUDA_GRAPH", None)
        else:
            os.environ["SGLANG_DSV4_MOE_CUDA_GRAPH"] = old_graph_env
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _run_megatron_moe_debug(
    args: argparse.Namespace,
    cfg: SimpleNamespace,
    input_ids: torch.Tensor,
    hidden: torch.Tensor,
) -> dict[str, torch.Tensor]:
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4 import (
        DeepSeekV4TransformerBlock,
    )
    from megatron.core.transformer.experimental_attention_variant.dsv4_loader import (
        _make_dsv4_config_from_official,
        load_dsv4_native_safetensors,
    )

    meg_cfg = _make_dsv4_config_from_official(cfg)
    block = DeepSeekV4TransformerBlock(config=meg_cfg).cuda().eval()
    embed_w = torch.empty(
        cfg.vocab_size,
        cfg.dim,
        device="cuda",
        dtype=torch.float32,
    )
    head_w = torch.empty(
        cfg.vocab_size,
        cfg.dim,
        device="cuda",
        dtype=torch.float32,
    )
    leftover = _filter_expected_leftovers(
        load_dsv4_native_safetensors(
            block=block,
            embed_weight=embed_w,
            head_weight=head_w,
            ckpt_dir=str(args.ckpt_dir),
            mp_rank=0,
            mp_world=1,
        )
    )
    if leftover:
        raise RuntimeError(f"Megatron leftover checkpoint keys: {leftover[:20]}")
    if args.force_oft:
        for layer in block.layers:
            for proj in ("w1", "w2", "w3"):
                layer.mlp.ensure_dsv4_expert_oft_r(
                    proj,
                    block_size=args.oft_block_size,
                    dtype=torch.bfloat16,
                )

    traces: dict[str, torch.Tensor] = {}
    try:
        with torch.inference_mode():
            for layer_idx, layer in enumerate(block.layers):
                out = layer.mlp(hidden, input_ids)
                traces[f"layer_{layer_idx}_mlp"] = out.detach().cpu()
        torch.cuda.synchronize()
        return traces
    finally:
        del block, embed_w, head_w
        gc.collect()
        torch.cuda.empty_cache()


def _target_logprob(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float(), dim=-1).gather(
        1,
        target.reshape(-1, 1),
    ).squeeze(1)


def _run_sglang_decode_debug(
    args: argparse.Namespace,
    cfg: SimpleNamespace,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    from sglang.srt.models.deepseek_v4_loader import load_dsv4_native_checkpoint
    from sglang.srt.model_loader.utils import set_default_torch_dtype

    model = DeepseekV4ForCausalLM(cfg).cuda().eval()
    leftover = _filter_expected_leftovers(
        load_dsv4_native_checkpoint(model, str(args.ckpt_dir))
    )
    if leftover:
        raise RuntimeError(f"SGLang leftover checkpoint keys: {leftover[:20]}")
    if args.force_oft:
        _enable_identity_expert_oft(model, block_size=args.oft_block_size)

    prompt_len = args.prompt_len
    response_len = args.response_len
    logits_steps: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    import sglang.srt.model_executor.cuda_graph_runner as cuda_graph_runner

    old_capture = cuda_graph_runner.get_is_capture_mode
    old_graph_env = os.environ.get("SGLANG_DSV4_MOE_CUDA_GRAPH")
    try:
        if args.sglang_moe_path == "grouped":
            os.environ["SGLANG_DSV4_MOE_CUDA_GRAPH"] = "1"
            cuda_graph_runner.get_is_capture_mode = lambda: True
        elif args.sglang_moe_path == "eager":
            os.environ["SGLANG_DSV4_MOE_CUDA_GRAPH"] = "0"

        with torch.inference_mode(), set_default_torch_dtype(torch.bfloat16):
            prompt = token_ids[:, :prompt_len].contiguous()
            h_sbd = model.model.forward_stateful(prompt, start_pos=0, req_idx=0)
            logits_steps.append(F.linear(h_sbd[-1].float(), model.lm_head.weight))
            targets.append(token_ids[:, prompt_len])

            for step in range(1, response_len):
                start_pos = prompt_len + step - 1
                cur = token_ids[:, start_pos : start_pos + 1].contiguous()
                h_sbd = model.model.forward_stateful(
                    cur,
                    start_pos=start_pos,
                    req_idx=0,
                )
                logits_steps.append(F.linear(h_sbd[-1].float(), model.lm_head.weight))
                targets.append(token_ids[:, prompt_len + step])
        torch.cuda.synchronize()
        return (
            torch.stack([x.detach().cpu() for x in logits_steps], dim=1),
            torch.stack([x.detach().cpu() for x in targets], dim=1),
        )
    finally:
        cuda_graph_runner.get_is_capture_mode = old_capture
        if old_graph_env is None:
            os.environ.pop("SGLANG_DSV4_MOE_CUDA_GRAPH", None)
        else:
            os.environ["SGLANG_DSV4_MOE_CUDA_GRAPH"] = old_graph_env
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _run_megatron_training_debug(
    args: argparse.Namespace,
    cfg: SimpleNamespace,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4 import (
        DeepSeekV4TransformerBlock,
    )
    from megatron.core.transformer.experimental_attention_variant.dsv4_loader import (
        _make_dsv4_config_from_official,
        load_dsv4_native_safetensors,
    )

    meg_cfg = _make_dsv4_config_from_official(cfg)
    block = DeepSeekV4TransformerBlock(config=meg_cfg).cuda().eval()
    embed_w = torch.empty(
        cfg.vocab_size,
        cfg.dim,
        device="cuda",
        dtype=torch.float32,
    )
    head_w = torch.empty(
        cfg.vocab_size,
        cfg.dim,
        device="cuda",
        dtype=torch.float32,
    )
    leftover = _filter_expected_leftovers(
        load_dsv4_native_safetensors(
            block=block,
            embed_weight=embed_w,
            head_weight=head_w,
            ckpt_dir=str(args.ckpt_dir),
            mp_rank=0,
            mp_world=1,
        )
    )
    if leftover:
        raise RuntimeError(f"Megatron leftover checkpoint keys: {leftover[:20]}")
    if args.force_oft:
        for layer in block.layers:
            for proj in ("w1", "w2", "w3"):
                layer.mlp.ensure_dsv4_expert_oft_r(
                    proj,
                    block_size=args.oft_block_size,
                    dtype=torch.bfloat16,
                )

    prompt_len = args.prompt_len
    response_len = args.response_len
    try:
        with torch.inference_mode():
            emb = F.embedding(token_ids, embed_w).to(torch.bfloat16)
            h_sbd = emb.permute(1, 0, 2).contiguous()
            h_sbd = block(h_sbd, input_ids=token_ids)
            h_bsd = h_sbd.permute(1, 0, 2).contiguous()
            positions = torch.arange(
                prompt_len - 1,
                prompt_len + response_len - 1,
                device=h_bsd.device,
                dtype=torch.long,
            )
            logits = F.linear(h_bsd[:, positions].float(), head_w)
            targets = token_ids[:, prompt_len : prompt_len + response_len]
        torch.cuda.synchronize()
        return logits.detach().cpu(), targets.detach().cpu()
    finally:
        del block, embed_w, head_w
        gc.collect()
        torch.cuda.empty_cache()


def _make_fp4_weights(
    cfg: ShapeConfig,
    *,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    weights: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for expert in range(cfg.experts):
        dense = (
            torch.randn(
                cfg.n,
                cfg.k,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            * 0.02
        ).to(torch.bfloat16)
        weight, scale = sgl_dsv4_kernels.fp4_act_quant(dense, 32)
        weights.append(weight)
        scales.append(scale)
        if cfg.experts >= 64 and (expert + 1) % 64 == 0:
            print(f"quantized experts {expert + 1}/{cfg.experts}", flush=True)
    return torch.stack(weights).contiguous(), torch.stack(scales).contiguous()


def _sglang_deepgemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if sgl_grouped_fp4.has_deep_gemm_official_fp8_fp4():
        x_q, x_s = sgl_grouped_fp4.dsv4_deep_gemm_act_quant(x.contiguous(), 128)
        packed_scale = sgl_grouped_fp4.pack_fp8_e8m0_scale_for_deep_gemm(scale)
    else:
        x_q, x_s = sgl_dsv4_kernels.act_quant(
            x.contiguous(),
            128,
            "ue8m0",
            torch.float8_e8m0fnu,
        )
        packed_scale = scale.contiguous()
    out = sgl_grouped_fp4.grouped_fp4_gemm(
        x_q,
        x_s,
        weight,
        packed_scale,
        pos_to_expert.contiguous(),
        torch.float8_e8m0fnu,
    )
    return out, x_q, x_s, packed_scale


def _megatron_deepgemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if meg_grouped_fp4.has_deep_gemm_official_fp8_fp4():
        x_q, x_s = meg_grouped_fp4.dsv4_deep_gemm_act_quant(x.contiguous(), 128)
        packed_scale = meg_grouped_fp4.pack_fp8_e8m0_scale_for_deep_gemm(scale)
    else:
        x_q, x_s = meg_dsv4_kernels.act_quant(
            x.contiguous(),
            128,
            "ue8m0",
            torch.float8_e8m0fnu,
        )
        packed_scale = scale.contiguous()
    out = meg_grouped_fp4.grouped_fp4_gemm(
        x_q,
        x_s,
        weight,
        packed_scale,
        pos_to_expert.contiguous(),
        torch.float8_e8m0fnu,
    )
    return out, x_q, x_s, packed_scale


def _assert_deepgemm_available() -> None:
    if os.environ.get("DSV4_FP4_GEMM_BACKEND", "deepgemm").strip().lower() == "tilelang":
        return
    if not sgl_grouped_fp4.has_deep_gemm_official_fp8_fp4():
        raise RuntimeError("SGLang did not enable deep_gemm_official FP8xFP4")
    if not meg_grouped_fp4.has_deep_gemm_official_fp8_fp4():
        raise RuntimeError("Megatron did not enable deep_gemm_official FP8xFP4")


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    _assert_deepgemm_available()

    cfg = _shape_from_args(args)
    max_active_rows = max(args.batches) * args.topk
    max_counts = _counts_for_active_rows(max_active_rows, cfg.experts)
    max_rows_per_expert = max(max_counts)
    print(
        "config: "
        f"shape={args.shape} experts={cfg.experts} n={cfg.n} k={cfg.k} "
        f"topk={args.topk} batches={args.batches} max_rows_per_expert={max_rows_per_expert}",
        flush=True,
    )

    weight, scale = _make_fp4_weights(cfg, device=device, seed=args.seed + 1)
    x_gen = torch.Generator(device=device).manual_seed(args.seed + 2)
    x_by_expert = (
        torch.randn(
            cfg.experts,
            max_rows_per_expert,
            cfg.k,
            device=device,
            dtype=torch.float32,
            generator=x_gen,
        )
        * args.activation_scale
    ).to(torch.bfloat16)

    baseline_sglang: dict[tuple[int, int], torch.Tensor] = {}
    baseline_megatron: dict[tuple[int, int], torch.Tensor] = {}
    ok = True

    baseline_batch = max(args.batches)
    for batch in sorted(args.batches, reverse=True):
        active_rows = batch * args.topk
        counts = _counts_for_active_rows(active_rows, cfg.experts)
        x, pos_to_expert, active, keys = _make_routed_input(
            x_by_expert,
            counts,
            alignment=args.alignment,
        )
        print(
            f"\nbatch={batch} active_rows={active_rows} padded_rows={x.shape[0]} "
            f"active_experts={sum(count > 0 for count in counts)}",
            flush=True,
        )

        sgl_out, sgl_x_q, sgl_x_s, sgl_scale = _sglang_deepgemm(
            x, weight, scale, pos_to_expert
        )
        meg_out, meg_x_q, meg_x_s, meg_scale = _megatron_deepgemm(
            x, weight, scale, pos_to_expert
        )
        torch.cuda.synchronize()

        ok &= _summarize_diff(
            "activation_q_sglang_vs_megatron",
            sgl_x_q.view(torch.uint8),
            meg_x_q.view(torch.uint8),
        )
        ok &= _summarize_diff(
            "activation_scale_sglang_vs_megatron",
            sgl_x_s.view(torch.int32),
            meg_x_s.view(torch.int32),
        )
        ok &= _summarize_diff(
            "weight_scale_layout_sglang_vs_megatron",
            sgl_scale.view(torch.int32),
            meg_scale.view(torch.int32),
        )
        active_sgl = sgl_out[active]
        active_meg = meg_out[active]
        ok &= _summarize_diff(
            "deepgemm_output_sglang_vs_megatron",
            active_sgl,
            active_meg,
        )

        if batch == baseline_batch:
            baseline_sglang = {
                key: active_sgl[row].detach().clone() for row, key in enumerate(keys)
            }
            baseline_megatron = {
                key: active_meg[row].detach().clone() for row, key in enumerate(keys)
            }
            continue

        sgl_ref = torch.stack([baseline_sglang[key] for key in keys])
        meg_ref = torch.stack([baseline_megatron[key] for key in keys])
        ok &= _summarize_diff(
            f"sglang_batch_invariance_vs_batch_{baseline_batch}",
            active_sgl,
            sgl_ref,
        )
        ok &= _summarize_diff(
            f"megatron_batch_invariance_vs_batch_{baseline_batch}",
            active_meg,
            meg_ref,
        )

    return 0 if ok else 1


def run_full_debug_model(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.ckpt_dir = Path(args.ckpt_dir)
    if not (args.ckpt_dir / "config.json").exists():
        raise FileNotFoundError(f"missing debug config: {args.ckpt_dir / 'config.json'}")
    if not (args.ckpt_dir / "model0-mp1.safetensors").exists():
        raise FileNotFoundError(
            f"missing debug safetensors: {args.ckpt_dir / 'model0-mp1.safetensors'}"
        )

    _set_full_model_debug_env()
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(f"cuda:{args.device}")

    cfg = _load_debug_config(
        args.ckpt_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    input_ids = _make_debug_input_ids(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=cfg.vocab_size,
        device=f"cuda:{args.device}",
    )
    print(
        "full_debug_model_config: "
        f"ckpt={args.ckpt_dir} layers={cfg.n_layers} hidden={cfg.dim} "
        f"experts={cfg.n_routed_experts} topk={cfg.n_activated_experts} "
        f"moe_inter={cfg.moe_inter_dim} n_hash_layers={cfg.n_hash_layers} "
        f"batch={args.batch_size} seq_len={args.seq_len} "
        f"force_oft={args.force_oft} sglang_moe_path={args.sglang_moe_path}",
        flush=True,
    )
    print(
        "kernel_env: "
        f"DSV4_FP4_GEMM_BACKEND={os.environ.get('DSV4_FP4_GEMM_BACKEND')} "
        f"SGLANG_DSV4_USE_TILEKERNELS={os.environ.get('SGLANG_DSV4_USE_TILEKERNELS')} "
        f"MEGATRON_DSV4_ATTN_IMPL={os.environ.get('MEGATRON_DSV4_ATTN_IMPL')}",
        flush=True,
    )

    prev_dtype = torch.get_default_dtype()
    try:
        sgl_hidden, sgl_traces = _run_sglang_full_debug_model(args, cfg, input_ids)
        meg_hidden, meg_traces = _run_megatron_full_debug_model(args, cfg, input_ids)
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_default_device("cpu")

    ok = True
    for name in sorted(set(sgl_traces) & set(meg_traces)):
        ok &= _summarize_float_diff(
            f"full_debug_{name}_sglang_vs_megatron",
            sgl_traces[name],
            meg_traces[name],
        )
    ok &= _summarize_float_diff(
        "full_debug_decoder_hidden_sglang_vs_megatron",
        sgl_hidden,
        meg_hidden,
    )
    return 0 if ok else 1


def run_moe_debug(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.ckpt_dir = Path(args.ckpt_dir)
    if not (args.ckpt_dir / "config.json").exists():
        raise FileNotFoundError(f"missing debug config: {args.ckpt_dir / 'config.json'}")
    if not (args.ckpt_dir / "model0-mp1.safetensors").exists():
        raise FileNotFoundError(
            f"missing debug safetensors: {args.ckpt_dir / 'model0-mp1.safetensors'}"
        )

    _set_full_model_debug_env()
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(f"cuda:{args.device}")

    cfg = _load_debug_config(
        args.ckpt_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    input_ids = _make_debug_input_ids(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=cfg.vocab_size,
        device=f"cuda:{args.device}",
    )
    hidden = _make_debug_hidden(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=cfg.dim,
        seed=args.seed + 3,
        device=f"cuda:{args.device}",
    )
    print(
        "moe_debug_config: "
        f"ckpt={args.ckpt_dir} layers={cfg.n_layers} hidden={cfg.dim} "
        f"experts={cfg.n_routed_experts} topk={cfg.n_activated_experts} "
        f"moe_inter={cfg.moe_inter_dim} n_hash_layers={cfg.n_hash_layers} "
        f"batch={args.batch_size} seq_len={args.seq_len} "
        f"force_oft={args.force_oft} sglang_moe_path={args.sglang_moe_path}",
        flush=True,
    )
    print(
        "kernel_env: "
        f"DSV4_FP4_GEMM_BACKEND={os.environ.get('DSV4_FP4_GEMM_BACKEND')} "
        f"SGLANG_DSV4_USE_TILEKERNELS={os.environ.get('SGLANG_DSV4_USE_TILEKERNELS')} "
        f"MEGATRON_DSV4_ATTN_IMPL={os.environ.get('MEGATRON_DSV4_ATTN_IMPL')}",
        flush=True,
    )

    prev_dtype = torch.get_default_dtype()
    try:
        sgl_traces = _run_sglang_moe_debug(args, cfg, input_ids, hidden)
        meg_traces = _run_megatron_moe_debug(args, cfg, input_ids, hidden)
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_default_device("cpu")

    ok = True
    for name in sorted(set(sgl_traces) & set(meg_traces)):
        ok &= _summarize_float_diff(
            f"moe_debug_{name}_sglang_vs_megatron",
            sgl_traces[name],
            meg_traces[name],
        )
    return 0 if ok else 1


def run_decode_training_debug(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.ckpt_dir = Path(args.ckpt_dir)
    if args.prompt_len <= 0 or args.response_len <= 0:
        raise ValueError("--prompt-len and --response-len must be positive")
    total_len = args.prompt_len + args.response_len
    if not (args.ckpt_dir / "config.json").exists():
        raise FileNotFoundError(f"missing debug config: {args.ckpt_dir / 'config.json'}")
    if not (args.ckpt_dir / "model0-mp1.safetensors").exists():
        raise FileNotFoundError(
            f"missing debug safetensors: {args.ckpt_dir / 'model0-mp1.safetensors'}"
        )

    _set_full_model_debug_env()
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(f"cuda:{args.device}")

    cfg = _load_debug_config(
        args.ckpt_dir,
        batch_size=args.batch_size,
        seq_len=total_len,
    )
    token_ids = _make_debug_input_ids(
        batch_size=args.batch_size,
        seq_len=total_len,
        vocab_size=cfg.vocab_size,
        device=f"cuda:{args.device}",
    )
    print(
        "decode_training_debug_config: "
        f"ckpt={args.ckpt_dir} layers={cfg.n_layers} hidden={cfg.dim} "
        f"experts={cfg.n_routed_experts} topk={cfg.n_activated_experts} "
        f"prompt_len={args.prompt_len} response_len={args.response_len} "
        f"total_len={total_len} batch={args.batch_size} force_oft={args.force_oft} "
        f"sglang_moe_path={args.sglang_moe_path}",
        flush=True,
    )
    print(
        "kernel_env: "
        f"DSV4_FP4_GEMM_BACKEND={os.environ.get('DSV4_FP4_GEMM_BACKEND')} "
        f"SGLANG_DSV4_USE_TILEKERNELS={os.environ.get('SGLANG_DSV4_USE_TILEKERNELS')} "
        f"MEGATRON_DSV4_ATTN_IMPL={os.environ.get('MEGATRON_DSV4_ATTN_IMPL')}",
        flush=True,
    )

    prev_dtype = torch.get_default_dtype()
    try:
        sgl_logits, sgl_targets = _run_sglang_decode_debug(args, cfg, token_ids)
        meg_logits, meg_targets = _run_megatron_training_debug(args, cfg, token_ids)
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_default_device("cpu")

    ok = _summarize_diff(
        "decode_training_targets_sglang_vs_megatron",
        sgl_targets,
        meg_targets,
    )
    for step in range(args.response_len):
        sgl_step = sgl_logits[:, step]
        meg_step = meg_logits[:, step]
        target = sgl_targets[:, step]
        logit_diff = (sgl_step.float() - meg_step.float()).abs()
        sgl_lp = _target_logprob(sgl_step, target)
        meg_lp = _target_logprob(meg_step, target)
        lp_diff = (sgl_lp - meg_lp).abs()
        equal = torch.equal(sgl_step, meg_step) and torch.equal(sgl_lp, meg_lp)
        print(
            f"decode_training_step_{step}: equal={equal} "
            f"logit_mean_abs={logit_diff.mean().item():.10g} "
            f"logit_max_abs={logit_diff.max().item():.10g} "
            f"target_lp_sglang={sgl_lp.tolist()} "
            f"target_lp_megatron={meg_lp.tolist()} "
            f"target_lp_abs_diff={lp_diff.tolist()}",
            flush=True,
        )
        ok &= equal
    return 0 if ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "grouped-gemm",
            "full-debug-model",
            "moe-debug",
            "decode-training-debug",
        ),
        default="grouped-gemm",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shape", choices=("w1", "w2"), default="w1")
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=3072)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--alignment", type=int, default=128)
    parser.add_argument("--batches", type=_parse_batch_list, default="1,2,4,128,512,1024,4096")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--activation-scale", type=float, default=0.7)
    parser.add_argument("--ckpt-dir", type=Path, default=_DEBUG_CKPT)
    parser.add_argument("--seq-len", type=int, default=82)
    parser.add_argument("--prompt-len", type=int, default=82)
    parser.add_argument("--response-len", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--sglang-moe-path",
        choices=("eager", "grouped"),
        default="grouped",
        help="For full-debug-model, force SGLang's eager per-expert or grouped CUDA-graph MoE path.",
    )
    parser.add_argument(
        "--force-oft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach identity routed-expert OFT buffers so Megatron and SGLang use OFT MoE paths.",
    )
    parser.add_argument("--oft-block-size", type=int, default=128)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller shape for fast smoke testing.",
    )
    args = parser.parse_args()
    if isinstance(args.batches, str):
        args.batches = _parse_batch_list(args.batches)
    if args.quick:
        if args.mode == "grouped-gemm":
            args.experts = 8
            args.hidden_dim = 1024
            args.inter_dim = 512
            args.batches = [1, 2, 4, 16]
        else:
            args.seq_len = min(args.seq_len, 4)
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "full-debug-model":
        raise SystemExit(run_full_debug_model(args))
    if args.mode == "moe-debug":
        raise SystemExit(run_moe_debug(args))
    if args.mode == "decode-training-debug":
        raise SystemExit(run_decode_training_debug(args))
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
