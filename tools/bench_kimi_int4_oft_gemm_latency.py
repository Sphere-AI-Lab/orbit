#!/usr/bin/env python3
"""Profile Kimi-K2.5 Megatron INT4/OFT dense-dequant GEMM paths.

This isolated profiler creates synthetic Kimi-shaped Megatron INT4 expert
weights and measures these paths:

* existing_w4a16: Megatron-Bridge INT4 dequantize to BF16, then F.linear per
  active expert.
* triton_dequant_bf16_linear: Triton INT4 dequantize to a dense BF16/FP16
  weight, then F.linear per active expert.
* te_grouped_chunked: Megatron-style chunked path that dequantizes active
  experts to dense BF16/FP16 weights, then uses TransformerEngine GroupedLinear
  to reduce per-expert GEMM launch overhead.
* existing_w4a16_bwd / fp8_qdq_bf16_linear_bwd: forward plus backward-input
  timings with frozen packed weights.
* fp8_qdq_bf16_linear: current Megatron fp8 reference behavior: activation
  FP8 QDQ, INT4 weight dequant plus FP8 QDQ, then BF16/FP16 F.linear.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_DEFAULT_MEGATRON_BRIDGE = Path(
    os.environ.get("MEGATRON_BRIDGE_SRC")
    or os.environ.get("MEGATRON_BRIDGE_ROOT")
    or "../Megatron-Bridge"
).expanduser()
DEFAULT_MEGATRON_BRIDGE_SRC = str(
    _DEFAULT_MEGATRON_BRIDGE
    if _DEFAULT_MEGATRON_BRIDGE.name == "src"
    else _DEFAULT_MEGATRON_BRIDGE / "src"
)


@dataclass
class BenchResult:
    path: str
    layer: str
    m: int
    local_routed_tokens: int
    active_experts: int
    input_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    dtype: str
    backend: str
    warmup: int
    iters: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float


@triton.jit
def _int4_dequant_kernel(
    w_packed_ptr,
    w_scale_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wkpack: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_on: tl.constexpr,
    stride_ok: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    out_type: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    packed = tl.load(
        w_packed_ptr
        + offs_n[:, None] * stride_wn
        + (offs_k[None, :] // 8) * stride_wkpack,
        mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
        other=0,
    )
    scale = tl.load(
        w_scale_ptr
        + offs_n[:, None] * stride_sn
        + (offs_k[None, :] // GROUP_SIZE) * stride_sg,
        mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    shift = (offs_k[None, :] % 8) * 4
    weight = (((packed >> shift) & 0xF).to(tl.float32) - 8.0) * scale
    out = weight.to(out_type)
    out_ptrs = out_ptr + offs_n[:, None] * stride_on + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, out, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K))


def _insert_path(path: str | Path) -> None:
    path_obj = Path(path).expanduser()
    path_str = str(path_obj)
    if path_obj.is_dir() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _megatron_bridge_paths(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).expanduser()
    root = candidate.parent if candidate.name == "src" else candidate
    return root / "3rdparty" / "Megatron-LM", root / "src"


def _insert_megatron_bridge_paths(path: str | Path) -> Path:
    megatron_lm_path, bridge_src_path = _megatron_bridge_paths(path)
    _insert_path(megatron_lm_path)
    _insert_path(bridge_src_path)
    return bridge_src_path


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_choices(value: str, *, allowed: set[str], label: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown {label}(s) {unknown}; allowed: {','.join(sorted(allowed))}"
        )
    return items


def _parse_layers(value: str) -> list[str]:
    return _parse_choices(value, allowed={"fc1", "fc2"}, label="layer")


def _parse_paths(value: str) -> list[str]:
    return _parse_choices(
        value,
        allowed={
            "existing_w4a16",
            "existing_w4a16_bwd",
            "triton_dequant_bf16_linear",
            "te_grouped_chunked",
            "fp8_qdq_bf16_linear",
            "fp8_qdq_bf16_linear_bwd",
        },
        label="path",
    )


def _dtype_from_name(name: str) -> torch.dtype:
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _measure_cuda_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    enable_grad: bool = False,
) -> tuple[dict[str, float], torch.Tensor]:
    last = None
    for _ in range(warmup):
        if enable_grad:
            last = fn()
        else:
            with torch.no_grad():
                last = fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if enable_grad:
            last = fn()
        else:
            with torch.no_grad():
                last = fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    torch.cuda.synchronize()

    ordered = sorted(times)
    return (
        {
            "mean_ms": statistics.fmean(times),
            "p50_ms": _percentile(ordered, 0.50),
            "p90_ms": _percentile(ordered, 0.90),
            "min_ms": ordered[0],
            "max_ms": ordered[-1],
        },
        last,
    )


def _make_megatron_int4_packed(
    out_features: int,
    in_features: int,
    *,
    group_size: int,
    device: torch.device,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if in_features % 8 != 0:
        raise ValueError(f"in_features must be divisible by 8, got {in_features}")
    if in_features % group_size != 0:
        raise ValueError(f"in_features must be divisible by group_size, got {in_features}")

    q = torch.randint(0, 16, (out_features, in_features), device=device, dtype=torch.int32)
    packed = torch.empty((out_features, in_features // 8), device=device, dtype=torch.int32)
    packed.zero_()
    for lane in range(8):
        packed |= q[:, lane::8] << (4 * lane)
    scale = (
        torch.rand(
            (out_features, in_features // group_size),
            device=device,
            dtype=torch.float32,
        )
        * 0.02
        + 0.001
    ).to(scale_dtype)
    shape = torch.tensor([out_features, in_features], device=device, dtype=torch.int64)
    return packed, scale, shape


def _make_grouped_megatron_int4_packed(
    experts: int,
    out_features: int,
    in_features: int,
    *,
    group_size: int,
    device: torch.device,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed_parts = []
    scale_parts = []
    shape_parts = []
    for _ in range(experts):
        packed, scale, shape = _make_megatron_int4_packed(
            out_features,
            in_features,
            group_size=group_size,
            device=device,
            scale_dtype=scale_dtype,
        )
        packed_parts.append(packed)
        scale_parts.append(scale)
        shape_parts.append(shape)
    return torch.stack(packed_parts), torch.stack(scale_parts), torch.stack(shape_parts)


def _dequantize_int4_local(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    out_features = int(weight_shape.reshape(-1)[0].item())
    in_features = int(weight_shape.reshape(-1)[1].item())
    shifts = torch.arange(8, device=weight_packed.device, dtype=torch.int32) * 4
    unpacked = ((weight_packed.unsqueeze(-1) >> shifts) & 0xF).float()
    unpacked = unpacked.reshape(out_features, -1)[:, :in_features] - 8.0
    scale = weight_scale.float().repeat_interleave(group_size, dim=1)[:, :in_features]
    return (unpacked * scale).to(torch.bfloat16)


def _triton_dequantize_int4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    group_size: int,
    out_dtype: torch.dtype,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    out_features = int(weight_shape.reshape(-1)[0].item())
    in_features = int(weight_shape.reshape(-1)[1].item())
    out = torch.empty((out_features, in_features), device=weight_packed.device, dtype=out_dtype)
    out_type = tl.bfloat16 if out_dtype is torch.bfloat16 else tl.float16
    grid = (triton.cdiv(out_features, block_n), triton.cdiv(in_features, block_k))
    _int4_dequant_kernel[grid](
        weight_packed,
        weight_scale,
        out,
        out_features,
        in_features,
        weight_packed.stride(0),
        weight_packed.stride(1),
        weight_scale.stride(0),
        weight_scale.stride(1),
        out.stride(0),
        out.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        out_type=out_type,
        num_warps=4,
        num_stages=3,
    )
    return out


def _fp8_qdq_per_group(
    x: torch.Tensor,
    *,
    group_size: int,
    quant_dim: int = -1,
    ste: bool = False,
) -> torch.Tensor:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if x.shape[quant_dim] % group_size != 0:
        raise ValueError(
            f"FP8 QDQ group size {group_size} must divide dim {quant_dim}; shape={tuple(x.shape)}"
        )
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    original_shape = x.shape
    if quant_dim < 0:
        quant_dim += x.ndim
    moved = x.movedim(quant_dim, -1).contiguous()
    flat = moved.view(-1, moved.shape[-1])
    grouped = flat.float().reshape(flat.shape[0], flat.shape[1] // group_size, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / fp8_info.max
    q = torch.clamp(grouped / scale, fp8_info.min, fp8_info.max).to(torch.float8_e4m3fn)
    qdq = (q.float() * scale).to(x.dtype).reshape_as(flat).view_as(moved)
    qdq = qdq.movedim(-1, quant_dim).reshape(original_shape)
    if ste:
        return x + (qdq - x).detach()
    return qdq


def _try_megatron_dequantize_int4(args: argparse.Namespace) -> tuple[Callable[..., torch.Tensor], str]:
    bridge_src_path = _insert_megatron_bridge_paths(args.megatron_bridge_src)
    try:
        from megatron.bridge.models.conversion.low_precision.int4 import dequantize_int4

        return dequantize_int4, f"Megatron-Bridge:dequantize_int4 ({bridge_src_path}) + F.linear"
    except Exception as exc:
        print(
            f"[bench] Megatron-Bridge dequantize_int4 import failed; using local equivalent "
            f"({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )

        def local_dequantize(
            weight_packed: torch.Tensor,
            weight_scale: torch.Tensor,
            weight_shape: torch.Tensor,
            group_size: int = 32,
            device: str | torch.device | None = None,
        ) -> torch.Tensor:
            if device is not None:
                target = torch.device(device)
                weight_packed = weight_packed.to(target)
                weight_scale = weight_scale.to(target)
                weight_shape = weight_shape.to(target)
            return _dequantize_int4_local(
                weight_packed,
                weight_scale,
                weight_shape,
                group_size=group_size,
            )

        return local_dequantize, "local dequantize_int4 clone + F.linear"


def _layer_shape(args: argparse.Namespace, layer: str) -> tuple[int, int]:
    if layer == "fc1":
        return 2 * args.moe_intermediate, args.hidden
    if layer == "fc2":
        return args.hidden, args.moe_intermediate
    raise ValueError(f"Unknown layer {layer!r}")


def _balanced_token_splits(total_tokens: int, experts: int) -> list[int]:
    base = total_tokens // experts
    rem = total_tokens % experts
    return [base + (1 if expert < rem else 0) for expert in range(experts)]


def _sparse_token_splits(total_tokens: int, experts: int, active_experts: int) -> list[int]:
    active = max(1, min(int(active_experts), experts, total_tokens))
    active_splits = _balanced_token_splits(total_tokens, active)
    return active_splits + [0] * (experts - active)


def _token_splits_for_args(total_tokens: int, experts: int, args: argparse.Namespace) -> list[int]:
    if args.active_experts is None:
        return _balanced_token_splits(total_tokens, experts)
    return _sparse_token_splits(total_tokens, experts, args.active_experts)


def _local_routed_tokens(m: int, *, topk: int, ep_size: int) -> int:
    return max(1, (m * topk + ep_size - 1) // ep_size)


def _existing_w4a16_grouped(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    shape: torch.Tensor,
    token_splits: list[int],
    *,
    dequantize_int4: Callable[..., torch.Tensor],
    group_size: int,
) -> torch.Tensor:
    out_features = int(shape[0].reshape(-1)[0].item())
    input_chunks = list(torch.split(x, token_splits, dim=0))
    output_chunks = [x.new_empty((split, out_features), dtype=x.dtype) for split in token_splits]
    for expert, split in enumerate(token_splits):
        if split == 0:
            continue
        weight = dequantize_int4(
            packed[expert],
            scale[expert],
            shape[expert],
            group_size=group_size,
            device=packed.device,
        ).to(x.dtype)
        output_chunks[expert] = F.linear(input_chunks[expert], weight)
    return torch.cat(output_chunks, dim=0)


def _triton_dequant_bf16_grouped(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    shape: torch.Tensor,
    token_splits: list[int],
    *,
    group_size: int,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    out_features = int(shape[0].reshape(-1)[0].item())
    input_chunks = list(torch.split(x, token_splits, dim=0))
    output_chunks = [x.new_empty((split, out_features), dtype=x.dtype) for split in token_splits]
    for expert, split in enumerate(token_splits):
        if split == 0:
            continue
        weight = _triton_dequantize_int4(
            packed[expert],
            scale[expert],
            shape[expert],
            group_size=group_size,
            out_dtype=x.dtype,
            block_n=block_n,
            block_k=block_k,
        )
        output_chunks[expert] = F.linear(input_chunks[expert], weight)
    return torch.cat(output_chunks, dim=0)


def _try_make_te_grouped_linear(
    *,
    num_gemms: int,
    in_features: int,
    out_features: int,
    device: torch.device,
    dtype: torch.dtype,
):
    try:
        from transformer_engine import pytorch as te
    except Exception as exc:
        raise RuntimeError(
            "te_grouped_chunked requires transformer_engine.pytorch; run after loading CUDA modules"
        ) from exc

    module = te.GroupedLinear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        return_bias=False,
        params_dtype=dtype,
        device=device,
        init_method=lambda weight: None,
    )
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    if hasattr(module, "disable_parameter_transpose_cache"):
        module.disable_parameter_transpose_cache = True
    if hasattr(module, "is_first_microbatch"):
        module.is_first_microbatch = False
    return module


def _te_grouped_chunked(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    shape: torch.Tensor,
    token_splits: list[int],
    *,
    dequantize_int4: Callable[..., torch.Tensor],
    group_size: int,
    chunk_size: int,
    module_cache: dict[tuple[int, int, int, str], torch.nn.Module],
) -> torch.Tensor:
    out_features = int(shape[0].reshape(-1)[0].item())
    in_features = x.shape[-1]
    input_chunks = list(torch.split(x, token_splits, dim=0))
    output_chunks = [x.new_empty((split, out_features), dtype=x.dtype) for split in token_splits]
    active_experts = [expert for expert, split in enumerate(token_splits) if split > 0]

    for chunk_start in range(0, len(active_experts), chunk_size):
        chunk_experts = active_experts[chunk_start : chunk_start + chunk_size]
        cache_key = (len(chunk_experts), in_features, out_features, str(x.dtype))
        module = module_cache.get(cache_key)
        if module is None:
            module = _try_make_te_grouped_linear(
                num_gemms=len(chunk_experts),
                in_features=in_features,
                out_features=out_features,
                device=x.device,
                dtype=x.dtype,
            )
            module_cache[cache_key] = module

        for local_idx, expert in enumerate(chunk_experts):
            weight = dequantize_int4(
                packed[expert],
                scale[expert],
                shape[expert],
                group_size=group_size,
                device=packed.device,
            ).to(x.dtype)
            getattr(module, f"weight{local_idx}").data = weight

        chunk_input = torch.cat([input_chunks[expert] for expert in chunk_experts], dim=0)
        chunk_splits = [token_splits[expert] for expert in chunk_experts]
        chunk_output = module(chunk_input, chunk_splits)
        if isinstance(chunk_output, tuple):
            chunk_output = chunk_output[0]
        for expert, expert_output in zip(chunk_experts, torch.split(chunk_output, chunk_splits, dim=0)):
            output_chunks[expert] = expert_output

    return torch.cat(output_chunks, dim=0)


def _fp8_qdq_bf16_grouped(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    shape: torch.Tensor,
    token_splits: list[int],
    *,
    dequantize_int4: Callable[..., torch.Tensor],
    int4_group_size: int,
    fp8_group_size: int,
) -> torch.Tensor:
    out_features = int(shape[0].reshape(-1)[0].item())
    input_chunks = list(torch.split(x, token_splits, dim=0))
    output_chunks = [x.new_empty((split, out_features), dtype=x.dtype) for split in token_splits]
    for expert, split in enumerate(token_splits):
        if split == 0:
            continue
        x_qdq = _fp8_qdq_per_group(input_chunks[expert], group_size=fp8_group_size, ste=True)
        weight = dequantize_int4(
            packed[expert],
            scale[expert],
            shape[expert],
            group_size=int4_group_size,
            device=packed.device,
        ).to(x.dtype)
        weight_qdq = _fp8_qdq_per_group(weight, group_size=fp8_group_size, ste=False)
        output_chunks[expert] = F.linear(x_qdq, weight_qdq)
    return torch.cat(output_chunks, dim=0)


def _forward_backward_input(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    x.grad = None
    out = forward_fn(x)
    out.backward(grad_output)
    if x.grad is None:
        raise RuntimeError("Backward did not produce x.grad")
    return x.grad


def _bench(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> list[BenchResult]:
    dequantize_int4, existing_backend = _try_megatron_dequantize_int4(args)
    local_experts = args.num_experts // args.megatron_ep_size
    results: list[BenchResult] = []

    def record(
        *,
        path: str,
        layer: str,
        m: int,
        local_tokens: int,
        active_experts: int,
        input_shape: tuple[int, ...],
        weight_shape: tuple[int, ...],
        backend: str,
        stats: dict[str, float],
    ) -> None:
        results.append(
            BenchResult(
                path=path,
                layer=layer,
                m=m,
                local_routed_tokens=local_tokens,
                active_experts=active_experts,
                input_shape=input_shape,
                weight_shape=weight_shape,
                dtype=_dtype_name(dtype),
                backend=backend,
                warmup=args.warmup,
                iters=args.iters,
                **stats,
            )
        )

    for layer in args.layers:
        out_features, in_features = _layer_shape(args, layer)
        packed, scale, shape = _make_grouped_megatron_int4_packed(
            local_experts,
            out_features,
            in_features,
            group_size=args.int4_group_size,
            device=device,
            scale_dtype=dtype,
        )
        te_module_cache: dict[tuple[int, int, int, str], torch.nn.Module] = {}
        for m in args.m_values:
            local_tokens = _local_routed_tokens(m, topk=args.topk, ep_size=args.megatron_ep_size)
            token_splits = _token_splits_for_args(local_tokens, local_experts, args)
            active_experts = sum(1 for split in token_splits if split > 0)
            x = torch.randn((local_tokens, in_features), device=device, dtype=dtype)
            weight_shape = (local_experts, out_features, in_features)
            if "existing_w4a16" in args.paths:

                def existing_fn() -> torch.Tensor:
                    return _existing_w4a16_grouped(
                        x,
                        packed,
                        scale,
                        shape,
                        token_splits,
                        dequantize_int4=dequantize_int4,
                        group_size=args.int4_group_size,
                    )

                stats, out = _measure_cuda_ms(existing_fn, warmup=args.warmup, iters=args.iters)
                record(
                    path="existing_w4a16",
                    layer=layer,
                    m=m,
                    local_tokens=local_tokens,
                    active_experts=active_experts,
                    input_shape=tuple(x.shape),
                    weight_shape=weight_shape,
                    backend=existing_backend,
                    stats=stats,
                )
                del out
            if "triton_dequant_bf16_linear" in args.paths:

                def triton_dequant_fn() -> torch.Tensor:
                    return _triton_dequant_bf16_grouped(
                        x,
                        packed,
                        scale,
                        shape,
                        token_splits,
                        group_size=args.int4_group_size,
                        block_n=args.dequant_block_n,
                        block_k=args.dequant_block_k,
                    )

                stats, out = _measure_cuda_ms(triton_dequant_fn, warmup=args.warmup, iters=args.iters)
                record(
                    path="triton_dequant_bf16_linear",
                    layer=layer,
                    m=m,
                    local_tokens=local_tokens,
                    active_experts=active_experts,
                    input_shape=tuple(x.shape),
                    weight_shape=weight_shape,
                    backend="Triton INT4->dense BF16/FP16 dequant + F.linear",
                    stats=stats,
                )
                del out
            if "te_grouped_chunked" in args.paths:

                def te_grouped_fn() -> torch.Tensor:
                    return _te_grouped_chunked(
                        x,
                        packed,
                        scale,
                        shape,
                        token_splits,
                        dequantize_int4=dequantize_int4,
                        group_size=args.int4_group_size,
                        chunk_size=args.te_chunk_size,
                        module_cache=te_module_cache,
                    )

                stats, out = _measure_cuda_ms(te_grouped_fn, warmup=args.warmup, iters=args.iters)
                record(
                    path="te_grouped_chunked",
                    layer=layer,
                    m=m,
                    local_tokens=local_tokens,
                    active_experts=active_experts,
                    input_shape=tuple(x.shape),
                    weight_shape=weight_shape,
                    backend=f"dense dequant + TE GroupedLinear chunks of {args.te_chunk_size}",
                    stats=stats,
                )
                del out
            if "fp8_qdq_bf16_linear" in args.paths:

                def fp8_qdq_fn() -> torch.Tensor:
                    return _fp8_qdq_bf16_grouped(
                        x,
                        packed,
                        scale,
                        shape,
                        token_splits,
                        dequantize_int4=dequantize_int4,
                        int4_group_size=args.int4_group_size,
                        fp8_group_size=args.fp8_group_size,
                    )

                stats, out = _measure_cuda_ms(fp8_qdq_fn, warmup=args.warmup, iters=args.iters)
                record(
                    path="fp8_qdq_bf16_linear",
                    layer=layer,
                    m=m,
                    local_tokens=local_tokens,
                    active_experts=active_experts,
                    input_shape=tuple(x.shape),
                    weight_shape=weight_shape,
                    backend="FP8 QDQ activation + dequant/QDQ weight + F.linear",
                    stats=stats,
                )
                del out
            backward_paths = {
                "existing_w4a16_bwd",
                "fp8_qdq_bf16_linear_bwd",
            } & set(args.paths)
            if backward_paths:
                x_bwd = torch.randn((local_tokens, in_features), device=device, dtype=dtype, requires_grad=True)
                grad_output = torch.randn((local_tokens, out_features), device=device, dtype=dtype)
                if "existing_w4a16_bwd" in backward_paths:

                    def existing_bwd_fn() -> torch.Tensor:
                        return _forward_backward_input(
                            x_bwd,
                            grad_output,
                            lambda inp: _existing_w4a16_grouped(
                                inp,
                                packed,
                                scale,
                                shape,
                                token_splits,
                                dequantize_int4=dequantize_int4,
                                group_size=args.int4_group_size,
                            ),
                        )

                    stats, grad = _measure_cuda_ms(
                        existing_bwd_fn,
                        warmup=args.warmup,
                        iters=args.iters,
                        enable_grad=True,
                    )
                    record(
                        path="existing_w4a16_bwd",
                        layer=layer,
                        m=m,
                        local_tokens=local_tokens,
                        active_experts=active_experts,
                        input_shape=tuple(x_bwd.shape),
                        weight_shape=weight_shape,
                        backend=f"{existing_backend}; forward + dX backward",
                        stats=stats,
                    )
                    del grad
                if "fp8_qdq_bf16_linear_bwd" in backward_paths:

                    def fp8_bwd_fn() -> torch.Tensor:
                        return _forward_backward_input(
                            x_bwd,
                            grad_output,
                            lambda inp: _fp8_qdq_bf16_grouped(
                                inp,
                                packed,
                                scale,
                                shape,
                                token_splits,
                                dequantize_int4=dequantize_int4,
                                int4_group_size=args.int4_group_size,
                                fp8_group_size=args.fp8_group_size,
                            ),
                        )

                    stats, grad = _measure_cuda_ms(
                        fp8_bwd_fn,
                        warmup=args.warmup,
                        iters=args.iters,
                        enable_grad=True,
                    )
                    record(
                        path="fp8_qdq_bf16_linear_bwd",
                        layer=layer,
                        m=m,
                        local_tokens=local_tokens,
                        active_experts=active_experts,
                        input_shape=tuple(x_bwd.shape),
                        weight_shape=weight_shape,
                        backend="FP8 QDQ path; forward + dX backward",
                        stats=stats,
                    )
                    del grad
                del x_bwd, grad_output
            del x
            torch.cuda.empty_cache()
        del packed, scale, shape, te_module_cache
        torch.cuda.empty_cache()
    return results


def _print_table(results: Iterable[BenchResult]) -> None:
    rows = list(results)
    headers = [
        "path",
        "layer",
        "m",
        "local_tokens",
        "active_experts",
        "input",
        "weight",
        "p50_ms",
        "p90_ms",
        "mean_ms",
        "backend",
    ]
    table_rows = [
        [
            result.path,
            result.layer,
            str(result.m),
            str(result.local_routed_tokens),
            str(result.active_experts),
            "x".join(map(str, result.input_shape)),
            "x".join(map(str, result.weight_shape)),
            f"{result.p50_ms:.3f}",
            f"{result.p90_ms:.3f}",
            f"{result.mean_ms:.3f}",
            result.backend,
        ]
        for result in rows
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in table_rows)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in table_rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def _check_correctness(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> None:
    group_size = args.int4_group_size
    experts = 3
    out_features = 64
    in_features = 128
    local_tokens = 7
    token_splits = [2, 3, 2]
    packed, scale, shape = _make_grouped_megatron_int4_packed(
        experts,
        out_features,
        in_features,
        group_size=group_size,
        device=device,
        scale_dtype=dtype,
    )
    x = torch.randn((local_tokens, in_features), device=device, dtype=dtype)
    dequantize_int4 = (
        lambda weight_packed, weight_scale, weight_shape, group_size=32, device=None: _dequantize_int4_local(
            weight_packed,
            weight_scale,
            weight_shape,
            group_size=group_size,
        ).to(device or weight_packed.device)
    )
    reference = _existing_w4a16_grouped(
        x,
        packed,
        scale,
        shape,
        token_splits,
        dequantize_int4=dequantize_int4,
        group_size=group_size,
    )
    dense_triton = _triton_dequant_bf16_grouped(
        x,
        packed,
        scale,
        shape,
        token_splits,
        group_size=group_size,
        block_n=args.dequant_block_n,
        block_k=args.dequant_block_k,
    )
    torch.testing.assert_close(dense_triton, reference, atol=0.12, rtol=0.08)

    grad_output = torch.randn_like(reference)
    x_ref = x.detach().clone().requires_grad_(True)
    ref_out = _existing_w4a16_grouped(
        x_ref,
        packed,
        scale,
        shape,
        token_splits,
        dequantize_int4=dequantize_int4,
        group_size=group_size,
    )
    ref_out.backward(grad_output)
    if x_ref.grad is None:
        raise AssertionError("Correctness check did not produce x gradient")
    print("[check] Triton dense dequant matches reference and existing dX backward runs.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        type=_parse_paths,
        default=[
            "existing_w4a16",
            "triton_dequant_bf16_linear",
            "fp8_qdq_bf16_linear",
        ],
    )
    parser.add_argument("--layers", type=_parse_layers, default=["fc1", "fc2"])
    parser.add_argument("--m-values", type=_parse_csv_ints, default=[1, 8, 16, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["bf16", "bfloat16", "fp16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--moe-intermediate", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--megatron-ep-size", type=int, default=8)
    parser.add_argument(
        "--active-experts",
        type=int,
        default=None,
        help=(
            "If set, use a sparse local routing pattern with at most this many "
            "nonzero local experts instead of balancing tokens over all local experts."
        ),
    )
    parser.add_argument("--int4-group-size", type=int, default=32)
    parser.add_argument("--fp8-group-size", type=int, default=128)
    parser.add_argument("--dequant-block-n", type=int, default=16)
    parser.add_argument("--dequant-block-k", type=int, default=128)
    parser.add_argument("--te-chunk-size", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--megatron-bridge-src",
        default=DEFAULT_MEGATRON_BRIDGE_SRC,
        help=(
            "Megatron-Bridge repo root or src directory. Defaults to MEGATRON_BRIDGE_SRC, "
            "MEGATRON_BRIDGE_ROOT/src, or ../Megatron-Bridge/src."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.iters <= 0 or args.warmup < 0:
        raise SystemExit("--iters must be > 0 and --warmup must be >= 0")
    if args.num_experts % args.megatron_ep_size != 0:
        raise SystemExit("--num-experts must be divisible by --megatron-ep-size")
    if args.active_experts is not None and args.active_experts <= 0:
        raise SystemExit("--active-experts must be > 0 when set")
    if args.te_chunk_size <= 0:
        raise SystemExit("--te-chunk-size must be > 0")
    if args.dequant_block_k % 8 != 0:
        raise SystemExit("--dequant-block-k must be divisible by 8")
    if args.hidden % args.int4_group_size != 0 or args.moe_intermediate % args.int4_group_size != 0:
        raise SystemExit("--hidden and --moe-intermediate must be divisible by --int4-group-size")
    if args.hidden % args.fp8_group_size != 0 or args.moe_intermediate % args.fp8_group_size != 0:
        raise SystemExit("--hidden and --moe-intermediate must be divisible by --fp8-group-size")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this profiler.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        torch.cuda.set_device(device)

    dtype = _dtype_from_name(args.dtype)
    torch.manual_seed(1234)
    torch.cuda.empty_cache()
    print(
        "[bench] Kimi Megatron INT4/OFT: "
        f"hidden={args.hidden}, moe_intermediate={args.moe_intermediate}, "
        f"global_experts={args.num_experts}, ep={args.megatron_ep_size}, "
        f"local_experts={args.num_experts // args.megatron_ep_size}, topk={args.topk}, "
        f"dtype={_dtype_name(dtype)}"
    )
    print(f"[bench] Megatron-Bridge source: {_insert_megatron_bridge_paths(args.megatron_bridge_src)}")
    if args.active_experts is not None:
        print(f"[bench] sparse local routing: active_experts<=min(local_tokens,{args.active_experts}).")
    print("[bench] local_routed_tokens = ceil(m * topk / ep), matching expected per-EP Megatron MoE load.")
    if args.check:
        _check_correctness(args, dtype, device)
    results = _bench(args, dtype, device)
    _print_table(results)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n")
        print(f"Wrote JSON results to {args.json}")


if __name__ == "__main__":
    main()
