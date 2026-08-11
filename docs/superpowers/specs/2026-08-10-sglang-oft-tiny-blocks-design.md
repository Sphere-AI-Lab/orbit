# SGLang OFT Tiny Block Sizes Design

## Goal

Support power-of-two OFT block sizes beginning at 4 across every public OFT
execution path in Orbit's pinned Sphere-AI-Lab SGLang fork. In particular, the
E4 `b8` arms must start and run instead of aborting in the fused rotate-project
kernel with `Triton tl.dot requires BS >= 16`.

The implementation must also measure the cost of the new `BS=4` and `BS=8`
paths against `BS=16`, where Triton's normal matrix-multiply path is available.

## Context

Orbit currently pins `Sphere-AI-Lab/sglang` at
`89ea43812ec6fb161fe29902a6c6f1fbefb524dd`. At that revision:

- `gemm_oft_r_fwd`, the uniform single-adapter rotation kernel, already uses an
  elementwise fallback below block size 16.
- `sgemm_oft_r_fwd`, the segmented multi-adapter rotation kernel, uses the same
  fallback.
- `test_gemm_oft_r_tiled.py` exercises those fallbacks at block sizes 4 and 8.
- `fused_rotate_project.py` rejects block sizes below 16 before launch, even
  though the unfused operation is valid there.
- the two grouped-MoE rotate/project implementations and the gradient-R
  helpers still call `tl.dot` with a 4- or 8-wide matrix dimension.
- the shared `apply_oft_rotation_triton` helper in the legacy fused-MoE layer
  also contracts the OFT block with `tl.dot`. It is used by the legacy Triton
  MoE path, Marlin MoE, and DeepSeek-V4 expert rotation, so it is part of the
  supported OFT surface even though it lives outside `python/sglang/srt/oft`.
- the ordinary adapter-loading path already avoids the Triton Cayley kernel
  below 16, but the directly exported Triton Cayley API does not.

This is a pure-Python SGLang change. Nothing under `sgl-kernel/` changes, so
Orbit's separately compiled `sgl-kernel` revision remains pinned.

## Supported Domain

Configured OFT block sizes are powers of two greater than or equal to 4:

```text
4, 8, 16, 32, 64, 128, ...
```

The block size must continue to divide the relevant hidden dimension. Runtime
value zero is not a configured block size: it remains SGLang's sentinel for the
base/identity adapter during CUDA-graph capture and reference-model forwards.

Block sizes 1 and 2 and non-power-of-two block sizes fail in Python validation
with a message that states the supported domain. They must not reach Triton
compilation and fail there opaquely.

## Chosen Approach

Use a compile-time elementwise fallback for block sizes 4 and 8, following the
implementation that already ships in `gemm_oft_r.py` and `sgemm_oft_r.py`.
Keep every block-size-16-and-larger `tl.dot` path unchanged.

"Elementwise" does not mean CPU execution or one-token-at-a-time execution.
The Triton program remains vectorized over its token and output-feature tiles.
Only the 4- or 8-element contraction dimension is expressed as unrolled
multiply-adds.

For a rotation, the tiny path computes:

```python
for output_col in range(BS):
    rotated_col = 0.0
    for reduction_col in range(BS):
        rotated_col += x[:, reduction_col] * R[reduction_col, output_col]
```

For fused rotate-and-project, each complete rotated column is cast at the same
point as the existing kernel and accumulated into an output tile:

```python
rotated_col = rotated_col.to(tl.bfloat16)
output_acc += rotated_col[:, None] * W[:, output_col][None, :]
```

The key branch is a `tl.constexpr` condition:

```python
if BS >= 16:
    # Existing tl.dot implementation.
else:
    # BS=4/8 elementwise implementation.
```

Triton removes the unused branch at compilation. Consequently the generated
path for block sizes 16 and above does not acquire tiny-block masks, extra
runtime branches, or a different accumulation order.

## Alternatives Rejected

### Pad every tiny block to 16

Padding would retain `tl.dot`, but a block-size-4 rotation would perform up to
16 times as much matrix work and every load would need masks to prevent reads
from the next OFT block. The masking and address complexity would repeat in
dense, grouped-MoE, identity, and backward kernels. This is a higher-risk change
than the established elementwise fallback.

### Dispatch tiny blocks to PyTorch or unfused operations

This would be a small patch for dense inference, but it would change allocation
and launch behavior under CUDA graphs, weaken the fused fast path, and still
leave grouped-MoE and direct Triton APIs inconsistent. It does not satisfy the
requirement that every OFT execution path support the same domain.

## Component Design

### Configuration validation

`python/sglang/srt/oft/utils.py` provides one shared
`validate_oft_block_size(block_size: int, *, allow_zero: bool = False) -> int`
helper. `python/sglang/srt/oft/oft_config.py` uses it when reading
`oft_block_size`, and the OFT manager uses it for an explicitly supplied
`max_oft_block_size`. Public launchers that derive a block size directly from
tensor shapes also call it before launching, so direct API calls receive the
same error behavior as adapter-backed calls.

Validation preserves zero only where it is already a runtime device-tensor
sentinel. A saved adapter configuration may not use zero.

### Dense fused forward paths

`python/sglang/srt/oft/triton_ops/fused_rotate_project.py` owns three public
entry points:

- `fused_rotate_project_qkv`
- `fused_rotate_project_gate_up`
- `fused_rotate_gate_up_inputs`

For block sizes 4 and 8, QKV and gate/up projection rotate each small block with
an unrolled reduction, cast the completed rotated column to BF16, and accumulate
its outer product with each active weight tile into FP32 output accumulators.
The `bsv == 0` identity branch skips the rotation and directly accumulates each
input column against its weight column.

The gate/up-input helper performs the corresponding unrolled rotations for its
two output buffers and stores the completed BF16 columns. Its existing
block-size-16-and-larger tiled-dot path remains unchanged.

### Unfused and segmented forward paths

`gemm_oft_r_fwd` and `sgemm_oft_r_fwd` already implement and test the chosen
fallback. Their kernel implementations do not change. Their tiny-block tests
remain part of the aggregate verification suite, and the new cross-path tests
ensure the fused, unfused, and segmented results agree.

Embedding, LM-head, row-parallel, and ordinary non-fused linear rotations flow
through these generic kernels and inherit their existing block-size-4/8 support.

### Grouped-MoE forward paths

`python/sglang/srt/oft/triton_ops/grouped_moe_rotate_project.py` has two public
implementations:

- `fused_split_w13_oft_grouped_moe`, which rotates and projects directly.
- `packed_bmm_split_w13_oft_grouped_moe`, which rotates routed inputs into a
  packed buffer before `torch.bmm` projection.

Both use the elementwise rotation for block sizes 4 and 8. The direct variant
then accumulates each completed rotated column against its expert weight column.
The packed variant stores the completed rotated columns and leaves its existing
batched projection unchanged. Routing, expert padding, non-local-expert zeroing,
and output scattering do not change.

The shared legacy helper
`python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py::apply_oft_rotation_triton`
gets the same compile-time split. For block sizes 4 and 8, its Triton program
loads one input column and one R row per unrolled reduction step and accumulates
the `(BLOCK_M, BS)` result in FP32. For block size 16 and above, its existing
tiled `tl.dot` loop is unchanged. This one helper covers the legacy Triton MoE,
Marlin MoE, and DeepSeek-V4 callers; those callers do not need separate kernel
changes.

Grouped-MoE down projections that use the generic rotation kernels already
inherit the existing tiny-block fallback.

### Backward helpers

The gradient-input sides of `gemm_oft_r_backward.py` and the segmented rotation
already reuse or implement elementwise tiny-block rotation. The remaining gap is
the gradient-R contraction:

```text
grad_R[k, c] = sum_t x[t, k] * grad_y[t, c]
```

At block sizes 4 and 8, the gradient-R kernels compute these small output blocks
with compile-time loops and Triton reductions over the token tile. Block size 16
and above retains the current `tl.dot(x.T, grad_y)` implementation.

Tests cover `gemm_oft_r_bwd_grad_x`, `gemm_oft_r_bwd_grad_R`, the combined
`gemm_oft_r_bwd`, and `sgemm_oft_r_grad_R` at block sizes 4, 8, and 16.

### Cayley-Neumann preparation

The normal `precompute_oft_r` loader already chooses the Torch Neumann
implementation below block size 16. The three directly exported APIs
`cayley_neumann_fwd`, `cayley_neumann_bwd`, and `cayley_neumann` are aligned
with that behavior. At block sizes 4 and 8, forward uses the Torch Neumann
calculation and backward uses the same matrix recurrence as the existing
Triton backward kernel expressed with Torch matrix multiplies. They do not
enter a Triton kernel whose matrix dimensions violate `tl.dot` requirements.

Forward values and backward gradients are compared with the Torch reference at
block sizes 4, 8, and 16.

## Numerical Contract

Rotation accumulation remains FP32. Dense and grouped fused paths cast each
fully reduced rotated column to BF16 before projection, matching the existing
kernel's cast point. Projection accumulation remains FP32, and final outputs are
stored as BF16.

The correctness tolerance remains `max_abs <= 2e-3`; no tolerance is relaxed to
make the new path pass. Identity rotations, nontrivial rotations, and the
runtime identity sentinel are tested separately.

## GPU Test Design

The SGLang GPU suite covers:

1. Dense QKV at block sizes 4, 8, and 16 for decode and larger token tiles.
2. Dense gate/up fused projection at the same sizes.
3. Fused gate/up input rotation at the same sizes.
4. Uniform single-adapter and segmented multi-adapter rotation parity.
5. Direct, packed-BMM, and shared legacy/Marlin grouped-MoE rotation parity,
   including multiple routed experts, padded token rows, and BS16 boundary
   control.
6. Gradient-input and gradient-R parity.
7. Cayley-Neumann forward and autograd parity.
8. `bsv == 0` identity behavior under the 4- and 8-sized compiled buffers.
9. CUDA-graph capture and replay for dense block sizes 4 and 8, including an
   adapter-slot update between replays.
10. Explicit validation errors for block sizes 1, 2, and a non-power-of-two
    divisor.

Every block-size-16 case is a boundary control: it must exercise the existing
`tl.dot` path rather than the new fallback.

## Benchmark Design

### Dense microbenchmark

Extend `test/srt/oft/bench_fused_rotate_project_blocks.py` rather than creating
a separate timing convention.

- block sizes: 4, 8, 16, 32, and 64 for the focused report; retain the existing
  larger-block coverage for regression checking.
- token counts: 1, 8, 32, 64, 256, and 1024.
- primary dense shapes: Llama-3.1-8B as sharded by the E4 rollout engine's TP=2
  configuration: QKV input 4096 with output slices `[2048, 512, 512]`, and
  gate/up input 4096 with output slices `[7168, 7168]`.
- secondary dense shape: the benchmark's existing Qwen2.5-7B case.
- modes: active rotation (`bsv=BS`) and base/identity (`bsv=0`).

Timing uses CUDA events, 20 warmup launches, 100 measured iterations, and five
repeats. The fastest repeat is recorded, matching the existing harness's method
for excluding host and shared-GPU interference from sub-millisecond kernels.
Compilation time is reported separately and excluded from steady-state latency.

For every `(shape, tokens, mode)` row, the report records absolute latency and
the `BS4 / BS16` and `BS8 / BS16` ratios. It also reports correctness error.
Raw latency is the primary comparison: for a fixed hidden width, rotation work
scales approximately with block size while projection work is nearly constant,
so normalizing all rows to equal rotation FLOPs would obscure the cost users
actually pay for each OFT configuration.

### Grouped-MoE microbenchmark

Extend the existing grouped-MoE benchmark using its representative synthetic
shape: hidden size 2048, half-intermediate size 384, 128 experts, and top-k 8.
Sweep block sizes 4, 8, and 16 across its decode-to-prefill token range, and
report the direct variant, packed-BMM variant, and legacy baseline separately.
The legacy row is also the performance exercise for
`apply_oft_rotation_triton`, so it must report a real BS4/8 timing after the
patch rather than an unsupported/error row.

### Acceptance and reporting

- correctness is a hard gate at every new block size.
- block-size-16-and-larger rows are compared before and after the patch with the
  existing 10% regression tolerance.
- block sizes 4 and 8 are report-only for performance on the first
  implementation. Their latency relative to 16 is recorded without inventing a
  threshold before measurement.
- a result that is correct but pathologically slower than both block size 16 and
  the existing unfused fallback is reported and investigated before Orbit pins
  the commit.

## End-to-End Verification

After SGLang unit tests and microbenchmarks pass, Orbit pins the exact verified
SGLang commit and refreshes `uv.lock`. The compiled `sgl-kernel` dependency does
not move.

Orbit then runs its fast OFT, launcher, and E4 campaign tests. On the cluster,
three short E4-style probes run at block sizes 4, 8, and 16 using the same
Llama-3.1-8B rollout topology as the campaign. Each probe must demonstrate:

- SGLang server startup;
- OFT adapter allocation and streamed loading;
- CUDA-graph capture and replay;
- generation with the active adapter;
- at least one adapter update followed by another generation cycle; and
- clean shutdown with no Triton assertion.

The probe records per-rollout timing so the microbenchmark comparison can be
related to observed system-level cost. The full seven-column (`lr0` through
`lr6`) E4 OFT sweeps remain paused until the block-size-8 probe passes multiple
rollout/update cycles.

## Repository and Delivery Flow

Implementation uses one isolated task branch and project-local worktree in each
repository:

1. Branch the Sphere-AI-Lab SGLang fork from
   `89ea43812ec6fb161fe29902a6c6f1fbefb524dd`.
2. Implement, test, and benchmark the SGLang change on the cluster GPU runtime.
3. Commit the verified SGLang change and retain its exact SHA as the handoff
   identity.
4. In an isolated Orbit worktree, update `pyproject.toml`, `uv.lock`, and the
   pin's explanatory comment to that SHA.
5. Run Orbit CPU tests, synchronize the committed Orbit revision to its
   dedicated remote execution worktree, and run the three probes.
6. Report correctness, dense and grouped-MoE benchmark tables, end-to-end probe
   timing, exact commits, and any unrun hardware coverage before resuming the
   campaign.

The reserved `lr0` allocation is not modified during planning. Cluster work
begins only during implementation under the project's Condor and durable-run
provenance rules.

## Non-Goals

- Changing the mathematical definition or parameterization of OFT.
- Changing the E4 capacity ladder; block size 4 is a diagnostic and supported
  configuration, while the existing campaign still begins at block size 8.
- Retuning block-size-16-and-larger kernels.
- Changing adapter wire formats, memory-pool layouts, or the runtime identity
  sentinel.
- Moving the compiled `sgl-kernel` dependency.
- Establishing a performance threshold for BS4/8 before measuring them.
