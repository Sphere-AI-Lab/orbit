# SGLang OFT Tiny Block Sizes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every public SGLang OFT path accept power-of-two block sizes beginning at 4, preserve the existing `tl.dot` paths at BS16+, benchmark BS4/8 against BS16, and pin the verified SGLang commit in Orbit.

**Architecture:** Add Python-level block-size validation and compile-time Triton fallbacks for BS4/8. The fallback keeps token/output tiles vectorized while expressing only the tiny rotation or gradient contraction as unrolled multiply-adds. SGLang is implemented and GPU-verified first; Orbit then pins that exact pure-Python commit and runs short BS4/8/16 E4-style probes.

**Tech Stack:** Python 3.12, PyTorch 2.11, Triton, SGLang's OFT backend, pytest, uv, HTCondor, H100 CUDA 13.2 runtime.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-08-10-sglang-oft-tiny-blocks-design.md` at Orbit commit `a8df8229a2eccc81e965d1bb2bf563a8c2fdc0cd`.
- SGLang implementation base is exactly `89ea43812ec6fb161fe29902a6c6f1fbefb524dd` on `Sphere-AI-Lab/sglang` branch `orbit-sgl-v0.5.9`.
- Supported configured sizes are powers of two at least 4. Runtime zero remains the identity-adapter sentinel only.
- BS16 and larger keep their existing `tl.dot` code paths and their `max_abs <= 2e-3` correctness contract.
- Existing BS16+ benchmark rows must remain within the harness's 10% regression tolerance.
- BS4/8 performance is measured and reported against BS16; it is not assigned a threshold before measurement.
- Do not edit installed `site-packages` as a deliverable.
- Do not change `sgl-kernel`, adapter wire formats, memory-pool layouts, or the E4 capacity ladder.
- Use one task branch and project-local `.worktrees/oft-bs4` worktree per repository.
- SGLang working branch: `codex/oft-bs4`; Orbit working branch: `codex/oft-bs4`.
- Local SGLang repository: `/Users/zqiu/Documents/GitHub/sglang-spherelab`; local SGLang worktree: `/Users/zqiu/Documents/GitHub/sglang-spherelab/.worktrees/oft-bs4`.
- Local Orbit worktree: `/Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.worktrees/oft-bs4`.
- Before creating either worktree, use `superpowers:using-git-worktrees`, verify `.worktrees/` is ignored, and report the base branch, task branch, and path.
- Before remote synchronization, follow `develop-on-remote-clusters` and verify that `/fast/zqiu/software/proj/spherelab/sglang-spherelab` and `/lustre/fast/fast/zqiu/orbit-iclr/orbit` are not dirty, shared by another task, or supporting an active job. If either fails, stop instead of reusing its Git metadata.
- The local task worktrees are authoritative. Remote task worktrees are execution-only and must match the pushed task branch and exact commit.
- Every GPU command runs inside the existing reserved Condor allocation or a newly approved allocation, never on a login node.
- Every remote run writes stdout, stderr, provenance, and completion status beneath the canonical `remote-cluster-runs` root before it starts.
- Commit messages are short conventional-commit lines with no AI attribution.

---

### Task 1: Isolated repositories and benchmark baseline

**Files:**
- Modify: `test/srt/oft/bench_fused_rotate_project_blocks.py` (SGLang)
- Modify: `python/sglang/srt/oft/triton_ops/grouped_moe_rotate_project.py` (benchmark section only in this task)
- Create: `test/srt/oft/test_tiny_block_benchmark_report.py` (SGLang)

**Interfaces:**
- Consumes: SGLang base commit `89ea43812ec6fb161fe29902a6c6f1fbefb524dd`.
- Produces: `relative_to_bs16(rows: list[dict]) -> list[dict]`, dense benchmark rows keyed by `(shape, mode, M, BS)`, and grouped-MoE tables covering BS4/8/16.

- [ ] **Step 1: Create or verify the two local authoritative worktrees**

Use `superpowers:using-git-worktrees`. Clone the Sphere fork only if the local repository does not exist, verify its origin, and create the worktrees:

```bash
git clone --branch orbit-sgl-v0.5.9 https://github.com/Sphere-AI-Lab/sglang.git /Users/zqiu/Documents/GitHub/sglang-spherelab
git -C /Users/zqiu/Documents/GitHub/sglang-spherelab worktree add \
  /Users/zqiu/Documents/GitHub/sglang-spherelab/.worktrees/oft-bs4 \
  -b codex/oft-bs4 89ea43812ec6fb161fe29902a6c6f1fbefb524dd
git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit worktree add \
  /Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.worktrees/oft-bs4 \
  -b codex/oft-bs4 HEAD
```

If the SGLang clone already exists, do not clone over it. Verify:

```bash
git -C /Users/zqiu/Documents/GitHub/sglang-spherelab remote get-url origin
git -C /Users/zqiu/Documents/GitHub/sglang-spherelab rev-parse 89ea43812ec6fb161fe29902a6c6f1fbefb524dd
git -C /Users/zqiu/Documents/GitHub/sglang-spherelab status --short
```

Expected: Sphere-AI-Lab origin, the base commit resolves, and the repository is clean. Stop on a conflicting branch/worktree or dirty shared state.

- [ ] **Step 2: Write the failing benchmark-report unit test**

Create `test/srt/oft/test_tiny_block_benchmark_report.py`:

```python
from test.srt.oft.bench_fused_rotate_project_blocks import relative_to_bs16


def test_relative_to_bs16_is_partitioned_by_shape_mode_and_batch():
    rows = [
        {"shape": "llama31-8b-tp2-qkv", "mode": "rotate", "M": 8, "BS": 4, "ms": 1.5},
        {"shape": "llama31-8b-tp2-qkv", "mode": "rotate", "M": 8, "BS": 8, "ms": 1.2},
        {"shape": "llama31-8b-tp2-qkv", "mode": "rotate", "M": 8, "BS": 16, "ms": 1.0},
        {"shape": "llama31-8b-tp2-qkv", "mode": "identity", "M": 8, "BS": 4, "ms": 0.8},
        {"shape": "llama31-8b-tp2-qkv", "mode": "identity", "M": 8, "BS": 16, "ms": 1.0},
    ]
    enriched = relative_to_bs16(rows)
    by_key = {(row["mode"], row["BS"]): row for row in enriched}
    assert by_key[("rotate", 4)]["vs_bs16"] == 1.5
    assert by_key[("rotate", 8)]["vs_bs16"] == 1.2
    assert by_key[("rotate", 16)]["vs_bs16"] == 1.0
    assert by_key[("identity", 4)]["vs_bs16"] == 0.8
```

- [ ] **Step 3: Run the report test to verify it fails**

Run from the SGLang worktree:

```bash
pytest -q test/srt/oft/test_tiny_block_benchmark_report.py
```

Expected: FAIL because `relative_to_bs16` does not exist.

- [ ] **Step 4: Extend the dense benchmark**

Add exact focused shapes and sizes:

```python
SHAPES = [
    ("llama31-8b-tp2-qkv", 4096, [2048, 512, 512]),
    ("qwen25-7b-qkv", 3584, [3584, 512, 512]),
]
BATCHES = [1, 8, 32, 64, 256, 1024]
BLOCK_SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
MODES = ["rotate", "identity"]


def relative_to_bs16(rows: list[dict]) -> list[dict]:
    baselines = {
        (row["shape"], row["mode"], row["M"]): row["ms"]
        for row in rows
        if row["BS"] == 16 and row.get("ms") is not None
    }
    enriched = []
    for row in rows:
        copy = dict(row)
        base = baselines.get((row["shape"], row["mode"], row["M"]))
        copy["vs_bs16"] = None if base is None or row.get("ms") is None else row["ms"] / base
        enriched.append(copy)
    return enriched
```

Use 4D `R`, persistent `slot_idx_t`, and `bsv_t`. Set `bsv_t` to `BS` for `rotate` and zero for `identity`. Keep compilation outside `_time_ms`, record a separate first-call `compile_ms`, and print `vs_bs16` after enriching the complete result list.

- [ ] **Step 5: Extend the grouped-MoE benchmark loop**

Keep its existing shape (`hidden=2048`, `half=384`, `experts=128`, `top_k=8`). Replace the single `block_size = 64` assignment with:

```python
block_sizes = [4, 8, 16]
ms = [1, 8, 32, 64, 256, 1024]
```

Add an outer `for block_size in block_sizes:` immediately before the existing
`for m in ms:` loop and indent that loop's complete body once. Add
`"BS": block_size` to the direct, packed-BMM, and legacy result dictionaries
and print `block_size` in every table row. Add this wrapper so a pre-change
BS4/8 compilation failure is recorded without preventing the BS16 baseline:

```python
def _measure_or_error(fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        return out, _bench_cuda(fn), None
    except Exception as exc:  # benchmark must retain unsupported baseline rows
        return None, None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
```

Use it independently for the direct and packed-BMM calls. When `out is None`,
store `max_abs=None`, `us=None`, and the returned error. Add an `--json` argparse
option that writes the complete row list; preserve the current printed table.

- [ ] **Step 6: Run the report test and static checks**

```bash
pytest -q test/srt/oft/test_tiny_block_benchmark_report.py
python -m compileall -q test/srt/oft/bench_fused_rotate_project_blocks.py \
  python/sglang/srt/oft/triton_ops/grouped_moe_rotate_project.py
```

Expected: PASS.

- [ ] **Step 7: Commit the benchmark harness before changing kernels**

```bash
git add test/srt/oft/bench_fused_rotate_project_blocks.py \
  test/srt/oft/test_tiny_block_benchmark_report.py \
  python/sglang/srt/oft/triton_ops/grouped_moe_rotate_project.py
git commit -m "bench(oft): compare tiny blocks with bs16"
```

- [ ] **Step 8: Synchronize this committed revision and record the pre-change baseline**

Push the SGLang task branch, create the dedicated remote execution worktree only after verifying the remote repository is safe, set `PYTHONPATH` to that worktree's `python/`, and run:

```bash
python test/srt/oft/bench_fused_rotate_project_blocks.py --json "$RUN_DIR/dense-before.json"
python python/sglang/srt/oft/triton_ops/grouped_moe_rotate_project.py \
  --json "$RUN_DIR/grouped-before.json"
```

Expected: BS16+ rows have timings; fused/grouped BS4/8 rows fail with the current `tl.dot`/assertion behavior. The authoritative JSON and streams are written under the pre-created run directory, not the repository.

---

### Task 2: Shared block-size validation

**Files:**
- Modify: `python/sglang/srt/oft/utils.py`
- Modify: `python/sglang/srt/oft/oft_config.py`
- Modify: `python/sglang/srt/oft/oft_manager.py`
- Create: `test/srt/oft/test_tiny_block_validation.py`

**Interfaces:**
- Produces: `validate_oft_block_size(block_size: int, *, allow_zero: bool = False) -> int`.
- Consumed by: adapter config, manager maximum, and direct launchers in Tasks 3-5.

- [ ] **Step 1: Write the failing validation tests**

```python
import pytest

from sglang.srt.oft.oft_config import OFTConfig
from sglang.srt.oft.utils import validate_oft_block_size


@pytest.mark.parametrize("block_size", [4, 8, 16, 1024])
def test_power_of_two_block_sizes_are_valid(block_size):
    assert validate_oft_block_size(block_size) == block_size


@pytest.mark.parametrize("block_size", [1, 2, 3, 6, 12])
def test_unsupported_block_sizes_fail_before_triton(block_size):
    with pytest.raises(ValueError, match="power of two.*at least 4"):
        validate_oft_block_size(block_size)


def test_zero_is_only_allowed_as_an_explicit_runtime_sentinel():
    with pytest.raises(ValueError, match="power of two.*at least 4"):
        validate_oft_block_size(0)
    assert validate_oft_block_size(0, allow_zero=True) == 0


def test_adapter_config_accepts_bs4_and_rejects_bs2():
    base = {"target_modules": ["q_proj"], "oft_block_size": 4}
    assert OFTConfig.from_dict(base).block_size == 4
    with pytest.raises(ValueError, match="power of two.*at least 4"):
        OFTConfig.from_dict({**base, "oft_block_size": 2})
```

- [ ] **Step 2: Run the tests to verify failure**

```bash
pytest -q test/srt/oft/test_tiny_block_validation.py
```

Expected: import failure for `validate_oft_block_size`.

- [ ] **Step 3: Implement the shared helper**

Add to `python/sglang/srt/oft/utils.py`:

```python
def validate_oft_block_size(block_size: int, *, allow_zero: bool = False) -> int:
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError(f"OFT block size must be an integer, got {block_size!r}")
    if allow_zero and block_size == 0:
        return 0
    if block_size < 4 or block_size & (block_size - 1):
        raise ValueError(
            f"OFT block size must be a power of two and at least 4; got {block_size}"
        )
    return block_size
```

In `OFTConfig.__init__`, replace the raw assignment with:

```python
from sglang.srt.oft.utils import validate_oft_block_size

self.block_size = validate_oft_block_size(int(self.hf_config["oft_block_size"]))
```

In `OFTManager`, validate a non-`None` explicit `max_oft_block_size` before storing it:

```python
self.max_oft_block_size = validate_oft_block_size(int(max_oft_block_size))
```

- [ ] **Step 4: Run validation and existing config tests**

```bash
pytest -q test/srt/oft/test_tiny_block_validation.py
python -m compileall -q python/sglang/srt/oft
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/oft/utils.py python/sglang/srt/oft/oft_config.py \
  python/sglang/srt/oft/oft_manager.py test/srt/oft/test_tiny_block_validation.py
git commit -m "feat(oft): validate block sizes from four"
```

---

### Task 3: Dense fused BS4/8 forward kernels

**Files:**
- Modify: `python/sglang/srt/oft/triton_ops/fused_rotate_project.py`
- Modify: `test/srt/oft/test_fused_rotate_project_tiled.py`

**Interfaces:**
- Consumes: `validate_oft_block_size` from Task 2.
- Produces: BS4/8 support in `fused_rotate_project_qkv`, `fused_rotate_project_gate_up`, and `fused_rotate_gate_up_inputs`.

- [ ] **Step 1: Extend the failing dense tests to BS4/8**

Add BS4/8 to QKV, gate/up projection, gate/up-input, and identity cases. Add a 4D-buffer runtime-sentinel test:

```python
@pytest.mark.parametrize("BS", [4, 8])
def test_tiny_qkv_runtime_identity_slot(BS):
    x, R, W = _inputs(32, BS, rotate=True)
    R4 = torch.stack([R, R], dim=0).contiguous()
    slot = torch.tensor(1, device="cuda", dtype=torch.int32)
    bsv = torch.tensor(0, device="cuda", dtype=torch.int32)
    out = fused_rotate_project_qkv(
        x, R4, W, OUT, slot_idx_t=slot, bsv_t=bsv
    )
    torch.cuda.synchronize()
    err = (out.float() - (x.float() @ W.float().T)).abs().max().item()
    assert err <= TOL
```

Update tile-picker tests so BS4/8 are valid but do not require a `TILE_K >= 16` dot tile.

- [ ] **Step 2: Run the focused tests to verify the current assertion**

```bash
pytest -q test/srt/oft/test_fused_rotate_project_tiled.py -k 'tiny or small or gate_up'
```

Expected: FAIL with `Triton tl.dot requires BS >= 16` for BS4/8.

- [ ] **Step 3: Add the tiny gate/up-input rotation**

In `_fused_rotate_gate_up_inputs_kernel`, retain the current loop under `if BS >= 16`. For `else`, compute each output column as GPU vectors:

```python
for j in range(BS):
    gate_j = tl.zeros((BLOCK_M,), dtype=tl.float32)
    up_j = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(BS):
        x_k = tl.load(
            x_ptr + offs_m * K + block_idx * BS + k,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        gate_r = tl.load(R_ptr + gate_R_base_t + k * BS + j).to(tl.float32)
        up_r = tl.load(R_ptr + up_R_base_t + k * BS + j).to(tl.float32)
        gate_j += x_k * gate_r
        up_j += x_k * up_r
    out_col = block_idx * BS + j
    tl.store(out_gate_ptr + offs_m * K + out_col, gate_j.to(tl.bfloat16), mask=m_mask)
    tl.store(out_up_ptr + offs_m * K + out_col, up_j.to(tl.bfloat16), mask=m_mask)
```

For `bsv == 0`, copy each scalar column to both outputs. The branch remains runtime because CUDA-graph replay changes `bsv_t`.

- [ ] **Step 4: Add the tiny fused rotate/project contraction**

In `_fused_rotate_project_inner`, guard the current tiled-dot rotate/project and identity paths with `if BS >= 16`. The BS4/8 rotation path uses:

```python
for block_idx in range(0, blocks_per_slice):
    k_block_start = block_idx * BS
    R_block_base = slot_R_offset + (rotation_block_start + block_idx) * BS * BS
    for j in range(BS):
        rotated_j = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k in range(BS):
            x_k = tl.load(
                x_ptr + offs_m * K + k_block_start + k,
                mask=m_mask,
                other=0.0,
            ).to(tl.float32)
            r_kj = tl.load(R_ptr + R_block_base + k * BS + j).to(tl.float32)
            rotated_j += x_k * r_kj
        projected_j = rotated_j.to(tl.bfloat16)
        w0 = tl.load(
            W_ptr + (slice_offset + offs_n0) * K + k_block_start + j,
            mask=n_mask0,
            other=0.0,
        )
        acc0 += projected_j[:, None] * w0[None, :]
```

Apply the same explicit outer product to `acc1` through `acc7` inside their existing `GROUP_N` constexpr guards. In the tiny identity branch, use the input column in place of `projected_j` and do not read `R`.

- [ ] **Step 5: Replace the hard minimum and validate launchers**

Call `validate_oft_block_size(BS)` in `_validate_inputs` and `fused_rotate_gate_up_inputs`; remove both `BS >= 16` assertions. Keep hidden-dimension divisibility and dtype/contiguity checks unchanged.

- [ ] **Step 6: Run dense tests**

```bash
pytest -q test/srt/oft/test_fused_rotate_project_tiled.py
```

Expected: all BS4/8/16/large parity and identity cases PASS at `max_abs <= 2e-3`.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/oft/triton_ops/fused_rotate_project.py \
  test/srt/oft/test_fused_rotate_project_tiled.py
git commit -m "feat(oft): support tiny fused rotation blocks"
```

---

### Task 4: Grouped-MoE BS4/8 forward kernels

**Files:**
- Modify: `python/sglang/srt/oft/triton_ops/grouped_moe_rotate_project.py`
- Modify: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`
- Create: `test/srt/oft/test_tiny_block_grouped_moe.py`

**Interfaces:**
- Produces: BS4/8 support in `fused_split_w13_oft_grouped_moe`, `packed_bmm_split_w13_oft_grouped_moe`, and the shared `apply_oft_rotation_triton` path used by legacy Triton MoE, Marlin MoE, and DeepSeek-V4.

- [ ] **Step 1: Write grouped-MoE parity tests**

Build a deterministic CUDA fixture with `M=8`, `hidden=32`, `half=16`, `experts=2`, `top_k=1`, and BS4/8/16. Use `moe_align_block_size`, call both public implementations, and compare them with a Torch reference that selects each token's expert, applies blockwise `x @ R`, then projects through the gate and up halves of `w13`.

The assertions are:

```python
assert direct.shape == (M, TOP_K, 2 * HALF)
assert packed.shape == direct.shape
assert (direct.float() - reference.float()).abs().max().item() <= 2e-3
assert (packed.float() - reference.float()).abs().max().item() <= 2e-3
```

Include a padded-route case where `num_tokens_post_padded > M * TOP_K` and verify unused output rows remain zero.

In the same test module, call `apply_oft_rotation_triton` directly with the
same routing metadata for BS4/8/16. Build the reference output in sorted routed
order, leave padded/non-local rows out of the comparison, and assert:

```python
assert rotated.shape == (M * TOP_K, HIDDEN)
assert (rotated[valid_rows].float() - reference.float()).abs().max().item() <= 2e-3
```

This direct helper test is the hardware-independent coverage point for all its
legacy Triton, Marlin, and DeepSeek-V4 callers; caller-specific tests remain in
their existing suites.

- [ ] **Step 2: Run the tests to verify failure**

```bash
pytest -q test/srt/oft/test_tiny_block_grouped_moe.py
```

Expected: BS4/8 fail during Triton compilation at `tl.dot` in all three kernel
families; BS16 passes.

- [ ] **Step 3: Implement the direct tiny path**

In `_split_w13_oft_grouped_moe_kernel`, retain the existing two dots for `BLOCK_SIZE >= 16`. For BS4/8, rotate one column at a time and immediately accumulate its expert-weight outer product:

```python
for block_idx in range(0, BLOCKS):
    r_base = expert * BLOCKS * BLOCK_SIZE * BLOCK_SIZE + block_idx * BLOCK_SIZE * BLOCK_SIZE
    for j in range(BLOCK_SIZE):
        x_rot_j = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k in range(BLOCK_SIZE):
            x_k = tl.load(
                hidden_states_ptr + token_idx * K + block_idx * BLOCK_SIZE + k,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            if half_id == 0:
                r_kj = tl.load(w1_oft_r_ptr + r_base + k * BLOCK_SIZE + j)
            else:
                r_kj = tl.load(w3_oft_r_ptr + r_base + k * BLOCK_SIZE + j)
            x_rot_j += x_k * r_kj.to(tl.float32)
        w_j = tl.load(
            w13_ptr + expert * N * K + (half_offset + offs_n) * K
            + block_idx * BLOCK_SIZE + j,
            mask=n_mask,
            other=0.0,
        )
        acc += x_rot_j.to(tl.bfloat16)[:, None] * w_j[None, :]
```

- [ ] **Step 4: Implement the packed tiny path**

In `_pack_split_oft_grouped_bmm_inputs_kernel`, retain the dot for BS16+. For BS4/8, compute and store each rotated scalar column:

```python
for j in range(BLOCK_SIZE):
    x_rot_j = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(BLOCK_SIZE):
        x_k = tl.load(
            hidden_states_ptr + token_idx * K + block_idx * BLOCK_SIZE + k,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        if half_id == 0:
            r_kj = tl.load(
                w1_oft_r_ptr + r_base + k * BLOCK_SIZE + j
            ).to(tl.float32)
        else:
            r_kj = tl.load(
                w3_oft_r_ptr + r_base + k * BLOCK_SIZE + j
            ).to(tl.float32)
        x_rot_j += x_k * r_kj
    packed_col = block_idx * BLOCK_SIZE + j
    tl.store(
        packed_inputs_ptr
        + (packed_batch * MAX_PADDED_TOKENS_PER_EXPERT + rank_offsets) * K
        + packed_col,
        x_rot_j.to(tl.bfloat16),
        mask=token_mask,
    )
```

Validate the tensor-derived block size in both public launchers.

- [ ] **Step 5: Implement the shared legacy/Marlin tiny rotation**

In `_oft_block_rotate_kernel`, retain the existing tiled loop for
`OFT_BLOCK_SIZE >= 16`. For BS4/8, express only the reduction dimension as
compile-time scalar loads while preserving the vectorized token and output
dimensions:

```python
if OFT_BLOCK_SIZE >= 16:
    for k_off in range(0, OFT_BLOCK_SIZE, TILE_K):
        # Keep the existing a_tile/r_sub loads and tl.dot unchanged.
        rot_accum += tl.dot(a_tile, r_sub, input_precision="ieee")
else:
    out_cols = tl.arange(0, OFT_BLOCK_SIZE).to(tl.int64)
    for k in range(OFT_BLOCK_SIZE):
        a_col = tl.load(
            A_ptr + orig_ids * stride_am + (k_base + k) * stride_ak,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        r_row = tl.load(
            R_ptr + expert * stride_re + pid_blk * stride_rb
            + k * stride_ri + out_cols * stride_rj
        ).to(tl.float32)
        rot_accum += a_col[:, None] * r_row[None, :]
```

Call `validate_oft_block_size(bs)` in `apply_oft_rotation_triton` before launch
and verify `oft_r.shape[-2:] == (bs, bs)` and `K % bs == 0`. Do not change any
legacy Triton, Marlin, or DeepSeek-V4 caller: they inherit support through the
shared helper.

- [ ] **Step 6: Run grouped tests and the BS16 boundary**

```bash
pytest -q test/srt/oft/test_tiny_block_grouped_moe.py
```

Expected: PASS for direct, packed, and shared legacy/Marlin rotation at
BS4/8/16.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/oft/triton_ops/grouped_moe_rotate_project.py \
  python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py \
  test/srt/oft/test_tiny_block_grouped_moe.py
git commit -m "feat(oft): support tiny grouped moe blocks"
```

---

### Task 5: Tiny-block backward and Cayley APIs

**Files:**
- Modify: `python/sglang/srt/oft/triton_ops/gemm_oft_r_backward.py`
- Modify: `python/sglang/srt/oft/triton_ops/sgemm_oft_r_bwd.py`
- Modify: `python/sglang/srt/oft/triton_ops/cayley_neumann.py`
- Create: `test/srt/oft/test_tiny_block_backward_cayley.py`

**Interfaces:**
- Produces: BS4/8 support for gradient-R helpers and direct Cayley forward/backward APIs.

- [ ] **Step 1: Write backward and Cayley parity tests**

Add this Torch reference and use it for each BS in `[4, 8, 16]`:

```python
def _reference_backward(x, weights, grad_y, num_slices=1):
    _, _, block_size, _ = weights.shape
    total_tokens, input_dim = x.shape
    num_blocks = input_dim // block_size
    grad_x = torch.zeros_like(x, dtype=torch.float32)
    grad_R = torch.zeros_like(weights, dtype=torch.float32)
    for slice_id in range(num_slices):
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            stop = start + block_size
            weight_idx = slice_id * num_blocks + block_idx
            x_block = x[:, start:stop].float()
            gy_block = grad_y[
                :, slice_id * input_dim + start : slice_id * input_dim + stop
            ].float()
            R_block = weights[0, weight_idx].float()
            grad_x[:, start:stop] += gy_block @ R_block.transpose(0, 1)
            grad_R[0, weight_idx] = x_block.transpose(0, 1) @ gy_block
    return grad_x, grad_R


grad_x, grad_R = gemm_oft_r_bwd(x, weights, grad_y, slot, bsv)
expected_grad_x, expected_grad_R = _reference_backward(x, weights, grad_y)
torch.testing.assert_close(grad_x.float(), expected_grad_x.float(), atol=2e-3, rtol=0)
torch.testing.assert_close(grad_R.float(), expected_grad_R.float(), atol=2e-3, rtol=0)
```

Also call `sgemm_oft_r_grad_R`. For Cayley, compare `cayley_neumann_fwd` and `cayley_neumann_bwd` with the Torch recurrence, then run `torch.autograd.gradcheck` on the public `cayley_neumann` wrapper in float64 on CPU and a BF16/FP32 CUDA tolerance check at BS4/8/16.

- [ ] **Step 2: Run to verify BS4/8 fail**

```bash
pytest -q test/srt/oft/test_tiny_block_backward_cayley.py
```

Expected: gradient-R and direct Cayley BS4/8 cases fail at `tl.dot`.

- [ ] **Step 3: Add tiny gradient-R reductions**

In both gradient-R kernels, keep the existing matrix dot under `if BLOCK_SIZE >= 16`. For BS4/8, compute each output scalar with a token-vector reduction:

```python
for k in range(BLOCK_SIZE):
    for c in range(BLOCK_SIZE):
        value = 0.0
        for t_start in range(0, total_tokens, TILE_T):
            t_offsets = t_start + tl.arange(0, TILE_T)
            t_mask = t_offsets < total_tokens
            x_col = tl.load(
                x_ptr + t_offsets * x_stride_0 + col_base_x + k,
                mask=t_mask,
                other=0.0,
            ).to(tl.float32)
            gy_col = tl.load(
                grad_y_ptr + t_offsets * grad_y_stride_0 + col_base_gy + c,
                mask=t_mask,
                other=0.0,
            ).to(tl.float32)
            value += tl.sum(x_col * gy_col, axis=0)
        tl.store(out_base + k * grad_R_stride_2 + c * grad_R_stride_3, value)
```

Use each kernel's existing strides and slice offset names. Do not change the BS16+ matrix accumulator.

- [ ] **Step 4: Route all direct tiny Cayley APIs through Torch**

Add a Torch backward recurrence:

```python
def _torch_cayley_neumann_bwd(grad_R: torch.Tensor, Q_skew: torch.Tensor) -> torch.Tensor:
    q_t = Q_skew.transpose(-1, -2)
    g_prev = grad_R
    acc = grad_R
    for _ in range(3):
        g_k = (2.0 * grad_R + g_prev @ q_t).to(grad_R.dtype)
        g_prev = g_k
        acc = (g_k + q_t @ acc).to(grad_R.dtype)
    return acc
```

In `cayley_neumann_fwd` and `cayley_neumann_bwd`, return the Torch forward/backward implementations when `block_size < 16`. In `cayley_neumann`, call the differentiable Torch forward directly below 16; preserve existing maximum-size fallback behavior.

- [ ] **Step 5: Run backward/Cayley tests**

```bash
pytest -q test/srt/oft/test_tiny_block_backward_cayley.py
```

Expected: PASS at BS4/8/16 without relaxing tolerance.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/oft/triton_ops/gemm_oft_r_backward.py \
  python/sglang/srt/oft/triton_ops/sgemm_oft_r_bwd.py \
  python/sglang/srt/oft/triton_ops/cayley_neumann.py \
  test/srt/oft/test_tiny_block_backward_cayley.py
git commit -m "feat(oft): support tiny backward and cayley paths"
```

---

### Task 6: SGLang aggregate GPU verification and benchmark report

**Files:**
- Modify if measurements require a correctness fix: only files from Tasks 1-5.
- Produce outside Git: dense/grouped before-and-after JSON, test logs, provenance, and completion status.

**Interfaces:**
- Produces: one exact verified SGLang commit SHA suitable for Orbit's dependency pin.

- [ ] **Step 1: Run static checks**

```bash
python -m compileall -q python/sglang/srt/oft \
  python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py \
  test/srt/oft
git diff --check 89ea43812ec6fb161fe29902a6c6f1fbefb524dd...HEAD
```

Expected: exit 0.

- [ ] **Step 2: Run the complete focused GPU suite in the allocation**

```bash
pytest -q \
  test/srt/oft/test_tiny_block_validation.py \
  test/srt/oft/test_gemm_oft_r_tiled.py \
  test/srt/oft/test_fused_rotate_project_tiled.py \
  test/srt/oft/test_tiny_block_grouped_moe.py \
  test/srt/oft/test_tiny_block_backward_cayley.py \
  test/srt/oft/test_streamed_chunk_limit.py
```

Expected: PASS. Record GPU name, CUDA/Triton/PyTorch/SGLang versions, command, SHA, and clean/dirty state in provenance.

- [ ] **Step 3: Capture CUDA-graph evidence**

Run the dense CUDA-graph tests for BS4 and BS8 with a slot/bsv update between replays. Expected: both replays match eager references and allocate no new adapter tensors inside capture.

- [ ] **Step 4: Run after benchmarks with the exact same harness**

```bash
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --json "$RUN_DIR/dense-after.json" \
  --compare "$RUN_DIR/dense-before.json"
python python/sglang/srt/oft/triton_ops/grouped_moe_rotate_project.py \
  --json "$RUN_DIR/grouped-after.json"
```

Expected: BS4/8 rows now have timings and errors within `2e-3`; BS16+ comparison exits 0 under the existing 10% ceiling.
The grouped legacy rows must contain real BS4/8 timings, proving the shared
legacy/Marlin rotation helper compiled and ran rather than being skipped.

- [ ] **Step 5: Inspect BS4/8 vs BS16 and unfused ratios**

Report absolute microseconds, `BS4/BS16`, `BS8/BS16`, and fused/unfused ratios for every shape/mode/batch. If a correct tiny fused path is slower than both BS16 and its same-BS unfused fallback across representative decode batches, stop and profile before pinning.

- [ ] **Step 6: Commit any measurement-driven correctness fixes, rerun, and push**

Do not tune BS16+. Any fix commit must name the specific path, for example:

```bash
git commit -m "fix(oft): correct tiny block identity replay"
git push -u origin codex/oft-bs4
```

Record the final clean SHA as `SGLANG_TINY_BS_SHA` in run provenance.

---

### Task 7: Pin the verified SGLang commit in Orbit

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `tests/fast/utils/test_lora_regret_arms_coverage.py`

**Interfaces:**
- Consumes: exact `SGLANG_TINY_BS_SHA` from Task 6.
- Produces: Orbit dependency metadata that resolves that SHA while keeping `sgl-kernel` at `9c83ae8be07cbb1eb6898ce608ae244e3be375b4`.

- [ ] **Step 1: Write the failing pin/contract test**

Extend `TestOftBlockCeilingUnderRl` with:

```python
def test_sglang_runtime_supports_power_of_two_blocks_from_four(self):
    from tools.lora_regret.arms import OFT_MAX_BLOCK_SGLANG

    supported = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    assert supported[0] == 4
    assert all(block & (block - 1) == 0 for block in supported)
    assert supported[-1] == OFT_MAX_BLOCK_SGLANG
```

Add a TOML assertion that the SGLang source revision equals `SGLANG_TINY_BS_SHA` and the `sgl-kernel` revision remains unchanged.

- [ ] **Step 2: Run the focused Orbit test before the pin**

```bash
pytest -q tests/fast/utils/test_lora_regret_arms_coverage.py -k sglang_runtime
```

Expected: the revision assertion fails against `89ea43812`.

- [ ] **Step 3: Update the two SGLang revision fields and dependency comment**

In `pyproject.toml`, replace the Python-package `sglang` `rev` and `[tool.orbit.release.backend-pins.sglang].tested-ref` with the full `SGLANG_TINY_BS_SHA`. Extend the comment to state:

```text
The pinned Python revision also adds elementwise BS4/8 fallbacks to every OFT
rotation path and keeps the BS16+ tl.dot path unchanged. sgl-kernel does not move.
```

- [ ] **Step 4: Refresh the lockfile**

```bash
uv lock --upgrade-package sglang
```

Verify every `sglang` git source in `uv.lock` contains the exact new SHA and every `sgl-kernel` source still contains `9c83ae8be07cbb1eb6898ce608ae244e3be375b4`.

- [ ] **Step 5: Run Orbit's fast contract suite**

```bash
pytest -q \
  tests/fast/utils/test_lora_regret_arms_coverage.py \
  tests/fast/utils/test_lora_regret_lr_columns.py \
  tests/fast/utils/test_lora_regret_preflight.py \
  tests/fast/utils/test_peft_param_match.py \
  tests/fast/utils/test_lora_arguments.py
```

Expected: PASS.

- [ ] **Step 6: Commit and push Orbit**

```bash
git add pyproject.toml uv.lock tests/fast/utils/test_lora_regret_arms_coverage.py
git commit -m "chore(deps): pin sglang tiny oft blocks"
git push -u origin codex/oft-bs4
```

---

### Task 8: BS4/8/16 E4-style probes

**Files:**
- Modify only if needed for a reusable probe selector: `scripts/lora_regret/coverage_probe.sh`
- Test if modified: `tests/fast/utils/test_lora_regret_preflight.py`
- Produce outside Git: three run-label directories with logs, provenance, completion status, and timing artifacts.

**Interfaces:**
- Consumes: clean matching local/remote Orbit SHA and installed SGLang `SGLANG_TINY_BS_SHA`.
- Produces: end-to-end evidence for BS4, BS8, and BS16 under the E4 Llama-3.1-8B TP=2 rollout topology.

- [ ] **Step 1: Verify remote worktrees and runtime identity**

Before launch, verify both remote worktrees are on `codex/oft-bs4`, clean, and match the local full SHAs. Set `PYTHONPATH` or install only the verified SGLang task revision through the repository-scoped environment; do not patch `site-packages` manually.

Run:

```bash
python - <<'PY'
import inspect
import sglang
from sglang.srt.oft.triton_ops import fused_rotate_project_qkv

print(sglang.__version__)
print(inspect.getsourcefile(fused_rotate_project_qkv))
PY
```

Expected: source resolves to the dedicated remote SGLang worktree or the exact installed git revision, never the old `89ea43812` package.

- [ ] **Step 2: Pre-create three durable run directories and provenance files**

Resolve one execution ID and create run labels `e4-bs4-probe`, `e4-bs8-probe`, and `e4-bs16-probe` beneath the canonical remote run root. Report the remote/local snapshot paths, stdout, stderr, provenance, and completion status paths before launching.

- [ ] **Step 3: Run the BS4 probe**

Use the E4 launcher protocol with:

```bash
PEFT_METHOD=oft OFT_BLOCK_SIZE=4 NUM_ROLLOUT=3 \
EVAL_INTERVAL=2 SAVE_INTERVAL=100000 WANDB_MODE=offline \
bash examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh
```

Bind streams and status to the BS4 run directory. Expected: startup, adapter load, CUDA-graph replay, three rollouts, at least one weight update followed by generation, and exit 0.

- [ ] **Step 4: Run the BS8 probe**

Run the same command with `OFT_BLOCK_SIZE=8`. This is the campaign-unblocking gate. Expected: the previous `BS >= 16` assertion is absent and multiple rollout/update cycles finish.

- [ ] **Step 5: Run the BS16 control**

Run the same command with `OFT_BLOCK_SIZE=16`. Expected: exit 0 through the unchanged dot path.

- [ ] **Step 6: Snapshot and compare evidence**

Take one bounded local snapshot of the three run directories. Report startup time, adapter-load/update time, rollout times, peak memory if available, final scheduler state, and BS4/BS8 ratios against BS16. Do not infer success from queue disappearance; inspect completion status and final logs.

- [ ] **Step 7: Resume-gate decision**

The E4 OFT columns may resume only if BS8 has clean completion evidence. BS4 is diagnostic support; BS16 is the control. If BS8 fails, leave the campaign paused and report the exact failing phase and log path.

---

### Task 9: Final verification and handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-08-10-sglang-oft-tiny-blocks.md` only to check completed boxes and append exact measured artifact paths; do not paste large logs.

**Interfaces:**
- Produces: reproducible SGLang and Orbit commits, benchmark summary, probe summary, and a safe campaign-resume decision.

- [ ] **Step 1: Run final repository checks**

In both worktrees:

```bash
git status --short --branch
git log -5 --oneline
git diff --check HEAD^..HEAD
```

Expected: clean worktrees and focused commits.

- [ ] **Step 2: Verify the dependency identity one final time**

Confirm `pyproject.toml`, `uv.lock`, the installed SGLang package, and the remote SGLang worktree all identify the same `SGLANG_TINY_BS_SHA`. Confirm `sgl-kernel` remains at `9c83ae8be07cbb1eb6898ce608ae244e3be375b4`.

- [ ] **Step 3: Summarize evidence**

Report:

- SGLang branch/SHA and Orbit branch/SHA.
- Exact GPU and software versions.
- Focused pytest counts and commands.
- Dense and grouped BS4/8/16 latency tables.
- BS16+ regression result.
- BS4/8/16 E4 probe timing and terminal status.
- Durable remote run directories and local snapshot directories.
- Whether the seven E4 OFT LR columns are safe to resume.
- Any hardware or path not tested.

- [ ] **Step 4: Use the branch-finishing workflow**

Invoke `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch`. Do not merge, delete worktrees, or resume campaign jobs without the user's selected integration/resume action.
