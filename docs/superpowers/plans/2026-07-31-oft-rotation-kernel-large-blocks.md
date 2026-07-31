# OFT Rotation Kernel — Large Block Sizes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SGLang's fused OFT rotate-project kernels launch at block sizes above 128 — where they currently cannot start at all — without losing any speed at the block sizes that work today.

**Architecture:** The kernels stage the entire `BS x BS` rotation block in shared memory, so their footprint is `6·BS·(BS+128)` bytes and blows the 232,448 B per-SM limit above `BS=128`. This plan adds a *tiled* inner loop that walks the rotation block in `BK x BK` sub-tiles, making shared memory O(BK²) and independent of `BS`, at identical FLOPs. The existing single-shot path is kept and selected by a `tl.constexpr` switch so `BS <= 128` compiles to exactly the code it does today — the new path is additive, never a replacement.

**Tech Stack:** Triton (bundled with torch 2.11 / CUDA 13), PyTorch, SGLang (Neckarium fork), pytest.

## Global Constraints

- **Repository:** the kernels live in the **SGLang fork**, not in orbit. Clone/worktree of `https://github.com/Neckarium/sglang.git`; the local clone is `/lustre/fast/fast/zqiu/NeckariumAI/clthegoat/env_for_cc/sglang`.
- **The installed build is commit `9c83ae8be`**, which that clone does **not** yet contain — `git fetch` before branching. Byte-identical source is cached at `/lustre/home/zqiu/.cache/uv_cu13_orbit/git-v0/checkouts/6d53cf772bb1c77f/9c83ae8be/python/build/lib/sglang/srt/oft/triton_ops/fused_rotate_project.py`; use it as the reference if the fetch cannot reach the remote.
- **File under change:** `python/sglang/srt/oft/triton_ops/fused_rotate_project.py`. Three public entry points: `fused_rotate_project_qkv`, `fused_rotate_project_gate_up`, `fused_rotate_gate_up_inputs`.
- **Hardware limit is 232,448 B** of shared memory per block on H100 (sm_90). Measured, not assumed — see Task 2.
- **Numerical tolerance is `max_abs <= 2e-3`**, the tolerance the file's own parity harness already uses. Do not loosen it.
- **No regression at `BS <= 128`.** Any change on that path must be justified by a measured speedup from the file's own benchmark, not by inspection.
- **Never edit `site-packages` as a deliverable.** Iterate there only for throwaway experiments, and copy findings back into the repo.
- **GPU work:** every task needs one H100. Verify with `nvidia-smi` before running; these are single-GPU microbenchmarks, not reservations.
- **Environment:** `source /fast/zqiu/orbit-iclr/orbit_env/bin/activate`, then `export CUDA_HOME=/is/software/nvidia/cuda-13.2 && source env.sh` from the orbit repo. `CUDA_HOME` is also exported globally from `~/.bashrc`.
- **Commit style:** one short conventional-commit line, no AI attribution trailer.

## Why this is not a tuning exercise

The footprint formula is exact, verified against four measurements:

```
shared = 3 stages · 2 bytes · (BLOCK_M·BS + BS·BS + BLOCK_N·BS)
       = 6 · BS · (BS + 128)        with BLOCK_M = BLOCK_N = 64

  BS=128 →   196,608 B   (85% of the 232,448 B budget — fits)
  BS=256 →   589,824 B   (measured 589,824)
  BS=512 → 1,966,080 B   (measured 1,966,080)
  BS=1024→ 7,077,888 B   (measured 7,077,888)
```

The `BS·BS` term dominates and grows quadratically while the budget is fixed, so no choice of `BLOCK_M`/`BLOCK_N`/`num_stages` reaches 256. `BS=128` only fits because `_pick_qkv_tiles` already spends the last of the budget on it by forcing `GROUP_N=1`. The rotation block itself has to stop being resident.

## Prior art — read before starting

The fork already contains one attempt at this and a revert:

```
4476ead90  feat(oft): re-use sgemm kernel to accelerate QKV OFT
d9da13a81  roll back to golden QKV baseline      (also deleted tests/bench_oft_shared_r.py)
```

Read `git show 4476ead90` and `git show d9da13a81` first. If the rollback was for a correctness or perf reason this plan would repeat, stop and report before writing code.

## File Structure

| File | Responsibility |
|---|---|
| `python/sglang/srt/oft/triton_ops/fused_rotate_project.py` | Both kernel paths, tile pickers, the three public entry points. Modified throughout. |
| `test/srt/oft/test_fused_rotate_project_tiled.py` | New. Parity of tiled vs untiled vs torch reference, and the shared-memory ceiling regression. |
| `test/srt/oft/bench_fused_rotate_project_blocks.py` | New. Per-block-size throughput table; the gate for "no regression at ≤128". |
| `tools/lora_regret/arms.py` (orbit repo) | `OFT_MAX_BLOCK_SGLANG` is raised once the kernel ships. Task 8 only. |

---

### Task 1: Working copy and a reproduction

**Files:**
- Create: nothing yet — this task establishes the workspace.

**Interfaces:**
- Consumes: nothing.
- Produces: a branch in the SGLang fork whose `fused_rotate_project.py` is byte-identical to the installed build, and a one-command reproduction of the failure.

- [ ] **Step 1: Get the source at the installed commit**

```bash
cd /lustre/fast/fast/zqiu/NeckariumAI/clthegoat/env_for_cc/sglang
git fetch origin
git cat-file -t 9c83ae8be   # must print "commit"
git worktree add /lustre/home/zqiu/.config/superpowers/worktrees/sglang/oft-large-blocks -b zqiu/oft-large-blocks 9c83ae8be
```

If `git cat-file` still fails after the fetch, the commit is not on the remote. Fall back to branching from `HEAD` and overwriting the one file from the uv cache:

```bash
git worktree add /lustre/home/zqiu/.config/superpowers/worktrees/sglang/oft-large-blocks -b zqiu/oft-large-blocks
cp /lustre/home/zqiu/.cache/uv_cu13_orbit/git-v0/checkouts/6d53cf772bb1c77f/9c83ae8be/python/build/lib/sglang/srt/oft/triton_ops/fused_rotate_project.py \
   /lustre/home/zqiu/.config/superpowers/worktrees/sglang/oft-large-blocks/python/sglang/srt/oft/triton_ops/fused_rotate_project.py
```

- [ ] **Step 2: Prove the working copy matches what is installed**

```bash
diff -q /lustre/home/zqiu/.config/superpowers/worktrees/sglang/oft-large-blocks/python/sglang/srt/oft/triton_ops/fused_rotate_project.py \
        /fast/zqiu/orbit-iclr/orbit_env/lib/python3.12/site-packages/sglang/srt/oft/triton_ops/fused_rotate_project.py
```

Expected: no output. If they differ, every measurement below is against the wrong code — stop and resolve.

- [ ] **Step 3: Read the prior attempt**

```bash
cd /lustre/fast/fast/zqiu/NeckariumAI/clthegoat/env_for_cc/sglang
git show 4476ead90 --stat
git show d9da13a81
```

Write two sentences in the commit message of Step 5 saying what the rollback undid and why this plan does not repeat it. If it *does* repeat it, stop and report.

- [ ] **Step 4: Reproduce the failure standalone**

Create `test/srt/oft/repro_shared_memory_ceiling.py` in the worktree:

```python
"""Minimal reproduction: the fused QKV kernel cannot launch above BS=128.

Run: python test/srt/oft/repro_shared_memory_ceiling.py
"""

import torch

from sglang.srt.oft.triton_ops.fused_rotate_project import fused_rotate_project_qkv

# Llama-3.1-8B fused QKV: hidden 4096 in, (32 + 8 + 8) * 128 = 6144 out.
K, OUT, M = 4096, [4096, 1024, 1024], 64
dev, dt = "cuda", torch.bfloat16

x = (torch.randn(M, K, device=dev, dtype=dt) * 0.01).contiguous()
W = (torch.randn(sum(OUT), K, device=dev, dtype=dt) * 0.02).contiguous()

for BS in (16, 32, 64, 128, 256, 512, 1024):
    blocks = 3 * (K // BS)
    # Identity rotation: the fused result must equal a plain projection, which
    # makes any mismatch a kernel bug rather than a bad reference.
    R = torch.eye(BS, device=dev, dtype=dt).expand(blocks, BS, BS).contiguous()
    try:
        out = fused_rotate_project_qkv(x, R, W, OUT)
        torch.cuda.synchronize()
        err = (out.float() - (x @ W.T).float()).abs().max().item()
        print(f"BS={BS:>5}  OK    max|out-ref|={err:.2e}")
    except Exception as exc:  # noqa: BLE001 -- report whatever Triton raises
        print(f"BS={BS:>5}  FAIL  {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
    del R
    torch.cuda.empty_cache()
```

- [ ] **Step 5: Run it and commit the reproduction**

```bash
python test/srt/oft/repro_shared_memory_ceiling.py
```

Expected, exactly:

```
BS=   16  OK    max|out-ref|=0.00e+00
BS=   32  OK    max|out-ref|=0.00e+00
BS=   64  OK    max|out-ref|=0.00e+00
BS=  128  OK    max|out-ref|=0.00e+00
BS=  256  FAIL  OutOfResources: out of resource: shared memory, Required: 589824, ...
BS=  512  FAIL  ... Required: 1966080 ...
BS= 1024  FAIL  ... Required: 7077888 ...
```

If the `Required:` numbers differ, the GPU is not sm_90 — record the actual limit and adjust Task 2's expectations rather than the code.

```bash
git add test/srt/oft/repro_shared_memory_ceiling.py
git commit -m "test(oft): reproduce the fused rotate-project shared-memory ceiling"
```

---

### Task 2: Baseline numbers, before touching the kernel

**Files:**
- Create: `test/srt/oft/bench_fused_rotate_project_blocks.py`

**Interfaces:**
- Consumes: `fused_rotate_project_qkv` from Task 1's working copy.
- Produces: `bench_block_sizes(shapes, block_sizes) -> list[dict]` and a CLI printing a throughput table. Every later task compares against the numbers this produces.

- [ ] **Step 1: Write the benchmark**

Create `test/srt/oft/bench_fused_rotate_project_blocks.py`:

```python
"""Throughput of the fused OFT rotate-project kernel, per block size.

This is the gate for "no regression at BS <= 128". The file's own __main__
benchmark compares fused/legacy/merged at ONE block size; this one sweeps the
block size, which is the axis this work changes.

Run: python test/srt/oft/bench_fused_rotate_project_blocks.py
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from sglang.srt.oft.triton_ops.fused_rotate_project import fused_rotate_project_qkv

# (name, K, output_sizes). Llama-3.1-8B and Qwen2.5-7B fused QKV.
SHAPES = [
    ("llama31-8b-qkv", 4096, [4096, 1024, 1024]),
    ("qwen25-7b-qkv", 3584, [3584, 512, 512]),
]
# Decode (1, 8, 64), CUDA-graph capture (256), prefill (1024).
BATCHES = [1, 8, 64, 256, 1024]
BLOCK_SIZES = [16, 32, 64, 128, 256, 512, 1024]


def _time_ms(fn, warmup=5, iters=20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def bench_block_sizes(shapes=SHAPES, block_sizes=BLOCK_SIZES, batches=BATCHES) -> list[dict]:
    """One row per (shape, M, BS). `ms` is None when the kernel cannot launch."""
    dev, dt = "cuda", torch.bfloat16
    rows: list[dict] = []
    for name, K, out_sizes in shapes:
        W = (torch.randn(sum(out_sizes), K, device=dev, dtype=dt) * 0.02).contiguous()
        for M in batches:
            x = (torch.randn(M, K, device=dev, dtype=dt) * 0.01).contiguous()
            for BS in block_sizes:
                if K % BS:
                    continue
                blocks = 3 * (K // BS)
                R = torch.eye(BS, device=dev, dtype=dt).expand(blocks, BS, BS).contiguous()
                row = {"shape": name, "M": M, "BS": BS, "ms": None, "err": None}
                try:
                    out = fused_rotate_project_qkv(x, R, W, out_sizes)
                    torch.cuda.synchronize()
                    row["err"] = (out.float() - (x @ W.T).float()).abs().max().item()
                    row["ms"] = _time_ms(lambda: fused_rotate_project_qkv(x, R, W, out_sizes))
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
                rows.append(row)
                del R
                torch.cuda.empty_cache()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default=None, help="also write rows here")
    args = parser.parse_args()

    rows = bench_block_sizes()
    print(f"{'shape':16} {'M':>5} {'BS':>5} {'ms':>9} {'max_err':>9}  note")
    for r in rows:
        ms = f"{r['ms']:9.4f}" if r["ms"] is not None else f"{'--':>9}"
        err = f"{r['err']:9.1e}" if r["err"] is not None else f"{'--':>9}"
        print(f"{r['shape']:16} {r['M']:>5} {r['BS']:>5} {ms} {err}  {r.get('error', '')}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Record the baseline**

```bash
python test/srt/oft/bench_fused_rotate_project_blocks.py --json /tmp/oft_baseline.json
cp /tmp/oft_baseline.json test/srt/oft/baseline_9c83ae8be.json
```

Expected: rows for `BS <= 128` carry a time and `max_err` of `0.0e+00`; rows for 256/512/1024 carry `--` and an `OutOfResources` note.

**This JSON is the contract for the rest of the plan.** Task 5 and Task 7 compare against it.

- [ ] **Step 3: Commit**

```bash
git add test/srt/oft/bench_fused_rotate_project_blocks.py test/srt/oft/baseline_9c83ae8be.json
git commit -m "bench(oft): sweep fused rotate-project throughput by block size"
```

---

### Task 3: The failing parity test for the tiled path

**Files:**
- Create: `test/srt/oft/test_fused_rotate_project_tiled.py`

**Interfaces:**
- Consumes: `fused_rotate_project_qkv`.
- Produces: the test suite every later task must keep green. Introduces the name `OFT_TILE_K` (the sub-tile width) which Task 4 defines.

- [ ] **Step 1: Write the test**

Create `test/srt/oft/test_fused_rotate_project_tiled.py`:

```python
"""Large OFT block sizes must launch, and must agree with the small ones.

The kernel stages the BS x BS rotation block in shared memory, so its footprint
is 6*BS*(BS+128) bytes against a 232,448 B limit -- exact, verified at
BS=256/512/1024. Above BS=128 it cannot launch at all. The tiled path walks the
rotation block in OFT_TILE_K sub-tiles so the footprint stops depending on BS.

Correctness is anchored on an identity rotation: with R = I the fused result
must equal a plain projection, so any mismatch is the kernel's, not a drifting
reference implementation's.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.oft.triton_ops.fused_rotate_project import (
    OFT_TILE_K,
    fused_rotate_project_qkv,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# Llama-3.1-8B fused QKV.
K, OUT = 4096, [4096, 1024, 1024]
TOL = 2e-3  # the tolerance this file's own parity harness already uses


def _inputs(M, BS, device="cuda", dtype=torch.bfloat16, rotate=True, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(M, K, device=device, dtype=dtype, generator=g) * 0.01).contiguous()
    W = (torch.randn(sum(OUT), K, device=device, dtype=dtype, generator=g) * 0.02).contiguous()
    blocks = 3 * (K // BS)
    eye = torch.eye(BS, device=device, dtype=dtype)
    if not rotate:
        R = eye.expand(blocks, BS, BS).contiguous()
        return x, R, W
    # A real orthogonal-ish rotation per block: identity plus a small skew, which
    # exercises every element of R rather than only its diagonal.
    noise = torch.randn(blocks, BS, BS, device=device, dtype=torch.float32, generator=g) * 0.02
    skew = noise - noise.transpose(-1, -2)
    R = (eye.float().unsqueeze(0) + skew).to(dtype).contiguous()
    return x, R, W


def _reference(x, R, W, out_sizes):
    """Rotate each block of x, then project. fp32 throughout."""
    BS = R.shape[-1]
    blocks_per_slice = R.shape[0] // len(out_sizes)
    outs = []
    offset = 0
    for s, width in enumerate(out_sizes):
        Ws = W[offset:offset + width].float()
        offset += width
        acc = torch.zeros(x.shape[0], width, device=x.device, dtype=torch.float32)
        for b in range(blocks_per_slice):
            k0 = b * BS
            xb = x[:, k0:k0 + BS].float()
            Rb = R[s * blocks_per_slice + b].float()
            acc += (xb @ Rb) @ Ws[:, k0:k0 + BS].T
        outs.append(acc)
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize("BS", [256, 512, 1024])
@pytest.mark.parametrize("M", [1, 64, 256])
def test_large_blocks_launch_and_are_correct(BS, M):
    """The whole point: these three block sizes cannot launch today."""
    x, R, W = _inputs(M, BS)
    out = fused_rotate_project_qkv(x, R, W, OUT)
    torch.cuda.synchronize()
    err = (out.float() - _reference(x, R, W, OUT)).abs().max().item()
    assert err <= TOL, f"BS={BS} M={M} max_abs={err:.2e}"


@pytest.mark.parametrize("BS", [16, 32, 64, 128])
@pytest.mark.parametrize("M", [1, 64, 256])
def test_small_blocks_still_correct(BS, M):
    """The untiled path must be untouched. If this breaks, the constexpr switch
    is selecting the tiled path where it should not."""
    x, R, W = _inputs(M, BS)
    out = fused_rotate_project_qkv(x, R, W, OUT)
    torch.cuda.synchronize()
    err = (out.float() - _reference(x, R, W, OUT)).abs().max().item()
    assert err <= TOL, f"BS={BS} M={M} max_abs={err:.2e}"


@pytest.mark.parametrize("BS", [128, 256, 1024])
def test_identity_rotation_is_exactly_a_projection(BS):
    """With R = I the fused kernel must reproduce x @ W.T bit-for-bit in the
    untiled path and within tolerance in the tiled one. A nonzero error here
    means the rotation matmul is wrong, independent of any reference."""
    x, R, W = _inputs(64, BS, rotate=False)
    out = fused_rotate_project_qkv(x, R, W, OUT)
    torch.cuda.synchronize()
    err = (out.float() - (x.float() @ W.float().T)).abs().max().item()
    assert err <= TOL, f"BS={BS} max_abs={err:.2e}"


def test_the_two_paths_agree_at_the_boundary():
    """BS=128 is the largest untiled size. Forcing the tiled path at the same
    BS must give the same answer -- this is what proves the tiled path is a
    reimplementation and not a different operator."""
    from sglang.srt.oft.triton_ops.fused_rotate_project import fused_rotate_project_qkv as f

    x, R, W = _inputs(64, 128)
    untiled = f(x, R, W, OUT)
    tiled = f(x, R, W, OUT, force_tiled=True)
    torch.cuda.synchronize()
    err = (untiled.float() - tiled.float()).abs().max().item()
    assert err <= TOL, f"paths disagree by {err:.2e}"


def test_the_tile_width_divides_every_supported_block():
    """OFT_TILE_K must divide each BS the kernel accepts, or the inner loop
    reads past the rotation block."""
    for BS in (16, 32, 64, 128, 256, 512, 1024):
        assert BS % min(OFT_TILE_K, BS) == 0, BS


def test_shared_memory_no_longer_scales_with_block_size():
    """The regression guard. If someone reinstates a full-BS load, BS=1024
    stops launching again and this fails with the OutOfResources message."""
    x, R, W = _inputs(8, 1024)
    out = fused_rotate_project_qkv(x, R, W, OUT)   # must not raise
    torch.cuda.synchronize()
    assert out.shape == (8, sum(OUT))
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest test/srt/oft/test_fused_rotate_project_tiled.py -x -q
```

Expected: collection error — `ImportError: cannot import name 'OFT_TILE_K'`.

- [ ] **Step 3: Commit the test**

```bash
git add test/srt/oft/test_fused_rotate_project_tiled.py
git commit -m "test(oft): pin large-block parity for the fused rotate-project kernel"
```

---

### Task 4: The tiled QKV kernel

**Files:**
- Modify: `python/sglang/srt/oft/triton_ops/fused_rotate_project.py`

**Interfaces:**
- Consumes: the failing test from Task 3.
- Produces: module constant `OFT_TILE_K: int = 64`; `_fused_rotate_project_inner` gains a `TILED: tl.constexpr` parameter; `fused_rotate_project_qkv` gains a keyword-only `force_tiled: bool = False`.

- [ ] **Step 1: Add the tile constant and the launch-side switch**

Near the top of the module, next to the other constants:

```python
# Sub-tile width for the tiled rotation path, in elements.
#
# The untiled path stages the whole BS x BS rotation block in shared memory, so
# its footprint is 6*BS*(BS+128) bytes and exceeds the 232,448 B per-SM limit
# above BS=128 (measured: 589,824 at 256, 1,966,080 at 512, 7,077,888 at 1024).
# The tiled path walks that block in OFT_TILE_K x OFT_TILE_K sub-tiles, so the
# footprint is O(OFT_TILE_K**2) and does not depend on BS at all.
#
# 64 rather than 32 or 128: tl.dot wants at least 16 per dimension, 64 keeps the
# three staged tiles at 3*2*(64*64)*3 = 73,728 B (a third of the budget) and
# matches the BLOCK_M/BLOCK_N the QKV path already uses, so the tiled loop reuses
# the same tile shapes the untiled one was tuned for.
OFT_TILE_K = 64

# Largest block the untiled path fits in shared memory on sm_90. Above this the
# tiled path is selected automatically; see OFT_TILE_K.
OFT_UNTILED_MAX_BS = 128
```

- [ ] **Step 2: Add the tiled branch to the inner routine**

In `_fused_rotate_project_inner`, add `TILED: tl.constexpr` to the signature and wrap the existing rotation with a branch. The existing body becomes the `else`:

```python
        if TILED:
            # Walk the rotation block in TILE_K-wide column tiles. x_rot_j is a
            # complete BS-reduction before it is cast, exactly as in the untiled
            # path -- only the summation order inside the reduction differs, so
            # the cast point and therefore the numerics are unchanged in kind.
            for j in range(0, BS, TILE_K):
                offs_j = j + tl.arange(0, TILE_K)
                x_rot_j = tl.zeros((BLOCK_M, TILE_K), dtype=tl.float32)
                for i in range(0, BS, TILE_K):
                    offs_i = i + tl.arange(0, TILE_K)
                    x_i = tl.load(
                        x_ptr + offs_m[:, None] * K + (k_block_start + offs_i)[None, :],
                        mask=m_mask[:, None],
                        other=0.0,
                    )
                    R_ij = tl.load(
                        R_ptr + R_block_base + offs_i[:, None] * BS + offs_j[None, :]
                    )
                    x_rot_j += tl.dot(x_i, R_ij, out_dtype=tl.float32)
                x_for_proj_j = x_rot_j.to(tl.bfloat16)

                W_block0_j = tl.load(
                    W_ptr + w_rows0[:, None] * K + (k_block_start + offs_j)[None, :],
                    mask=n_mask0[:, None],
                    other=0.0,
                )
                acc0 += tl.dot(
                    x_for_proj_j, tl.trans(W_block0_j), out_dtype=tl.float32, allow_tf32=False
                )
                if GROUP_N >= 2:
                    W_block1_j = tl.load(
                        W_ptr + w_rows1[:, None] * K + (k_block_start + offs_j)[None, :],
                        mask=n_mask1[:, None],
                        other=0.0,
                    )
                    acc1 += tl.dot(
                        x_for_proj_j, tl.trans(W_block1_j), out_dtype=tl.float32,
                        allow_tf32=False,
                    )
        else:
            <the existing x_block / R_block / x_rot body, unchanged>
```

`w_rows0`/`w_rows1`/`n_mask0`/`n_mask1` are already computed above the branch in the current code; hoist them out of the `else` if they are not.

> **Note for the implementer:** the FLOP count is unchanged. The `i` loop covers the same `BS` reduction the single `tl.dot` did, and the `j` loop covers the same `BS` output columns. The only added cost is re-reading `x` once per `j` tile — `BS/TILE_K` times. That is why `TILED` is a compile-time constant and not a runtime flag: `BS <= 128` must not pay it.

- [ ] **Step 3: Thread the switch through the QKV entry point**

In `fused_rotate_project_qkv`, add the parameter and pass it down:

```python
def fused_rotate_project_qkv(
    x: torch.Tensor,
    R: torch.Tensor,
    W: torch.Tensor,
    output_sizes: List[int],
    bias: Optional[torch.Tensor] = None,
    *,
    slot_idx_t: Optional[torch.Tensor] = None,
    bsv_t: Optional[torch.Tensor] = None,
    force_tiled: bool = False,
) -> torch.Tensor:
```

and, after `BLOCK_M, BLOCK_N, GROUP_N = _pick_qkv_tiles(M, max_slice_width, BS)`:

```python
    # Tiled only where the untiled path cannot launch, so every block size that
    # works today compiles to exactly the code it does today. `force_tiled` is
    # for the boundary test that proves the two paths agree at BS=128.
    tiled = force_tiled or BS > OFT_UNTILED_MAX_BS
    tile_k = min(OFT_TILE_K, BS)
```

Pass `TILED=tiled, TILE_K=tile_k` in the kernel launch, and add both to `_fused_rotate_project_qkv_kernel`'s signature as `tl.constexpr`, forwarding them to `_fused_rotate_project_inner`.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest test/srt/oft/test_fused_rotate_project_tiled.py -q
```

Expected: all pass, including `BS in {256, 512, 1024}`.

If `test_the_two_paths_agree_at_the_boundary` fails while the others pass, the tiled path has an indexing bug that the identity rotation hides — check the `offs_i[:, None] * BS + offs_j[None, :]` term against `R`'s row-major `(BS, BS)` layout.

- [ ] **Step 5: Re-run the reproduction**

```bash
python test/srt/oft/repro_shared_memory_ceiling.py
```

Expected: all seven block sizes now print `OK`, with `max|out-ref|` at most `2e-3` for the tiled ones.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/oft/triton_ops/fused_rotate_project.py
git commit -m "feat(oft): tile the rotation block so large OFT block sizes launch"
```

---

### Task 5: Prove no regression at BS <= 128

**Files:**
- Modify: `test/srt/oft/bench_fused_rotate_project_blocks.py` (add the comparison mode)

**Interfaces:**
- Consumes: `baseline_9c83ae8be.json` from Task 2, the kernel from Task 4.
- Produces: `compare(baseline_rows, current_rows, tolerance) -> list[dict]` and a `--compare` CLI flag that exits non-zero on a regression.

- [ ] **Step 1: Add the comparison**

Append to `bench_fused_rotate_project_blocks.py`:

```python
# A kernel that was not changed should time within noise. 5% absorbs run-to-run
# variance on an idle H100 without hiding a real slowdown; measured spread over
# 20 iterations after 5 warmups is well under 2%.
REGRESSION_TOLERANCE = 0.05


def compare(baseline_rows: list[dict], current_rows: list[dict],
            tolerance: float = REGRESSION_TOLERANCE) -> list[dict]:
    """Rows that got slower, or that stopped working.

    Only compares rows the baseline could actually run: the whole point of this
    change is that 256/512/1024 have no baseline time, so a new number there is
    a gain, not a regression.
    """
    key = lambda r: (r["shape"], r["M"], r["BS"])  # noqa: E731
    current = {key(r): r for r in current_rows}
    problems = []
    for base in baseline_rows:
        if base.get("ms") is None:
            continue
        now = current.get(key(base))
        if now is None or now.get("ms") is None:
            problems.append({**base, "reason": "no longer runs"})
            continue
        ratio = now["ms"] / base["ms"]
        if ratio > 1.0 + tolerance:
            problems.append({**base, "now_ms": now["ms"], "ratio": ratio,
                             "reason": f"{(ratio - 1) * 100:.1f}% slower"})
    return problems
```

and in `main`, before `return 0`:

```python
    if args.compare:
        with open(args.compare, encoding="utf-8") as fh:
            baseline = json.load(fh)
        problems = compare(baseline, rows)
        if problems:
            print("\nREGRESSIONS against the baseline:")
            for p in problems:
                print(f"  {p['shape']} M={p['M']} BS={p['BS']}: {p['reason']}")
            return 1
        print("\nno regression at any block size the baseline could run")
    return 0
```

with `parser.add_argument("--compare", type=str, default=None, help="baseline JSON to check against")`.

- [ ] **Step 2: Run the comparison**

```bash
python test/srt/oft/bench_fused_rotate_project_blocks.py --compare test/srt/oft/baseline_9c83ae8be.json
```

Expected: exit 0, `no regression at any block size the baseline could run`, and timings present for 256/512/1024 where the baseline had none.

If a `BS <= 128` row regressed, the `TILED` constexpr is not doing its job — confirm with `TRITON_PRINT_AUTOTUNING=1` that the untiled path is still being compiled for those sizes.

- [ ] **Step 3: Record what the large blocks cost**

Write the new table into the commit message. The expected shape is that 256/512/1024 are *slower per call* than 128 — they do more rotation work — but they run at all, which they did not before. A large-block time worse than `BS/128` times the 128 time by more than ~2x suggests the `x` re-read is dominating; note it for Task 7 rather than fixing it here.

- [ ] **Step 4: Commit**

```bash
git add test/srt/oft/bench_fused_rotate_project_blocks.py
git commit -m "bench(oft): gate the tiled kernel against the pre-change baseline"
```

---

### Task 6: The gate-up paths

**Files:**
- Modify: `python/sglang/srt/oft/triton_ops/fused_rotate_project.py`
- Modify: `test/srt/oft/test_fused_rotate_project_tiled.py`

**Interfaces:**
- Consumes: `OFT_TILE_K`, `OFT_UNTILED_MAX_BS`, the `TILED` branch from Task 4.
- Produces: `force_tiled` on `fused_rotate_project_gate_up` and `fused_rotate_gate_up_inputs`.

**Why separate from Task 4:** QKV is what fails in production and is worth shipping alone. The gate-up path has a different tile picker (`_pick_tiles`, `GROUP_N` up to 4) and `fused_rotate_gate_up_inputs` has no projection at all, so a reviewer can accept QKV while rejecting these.

- [ ] **Step 1: Extend the test to the other two entry points**

Append to `test/srt/oft/test_fused_rotate_project_tiled.py`:

```python
from sglang.srt.oft.triton_ops.fused_rotate_project import (  # noqa: E402
    fused_rotate_gate_up_inputs,
    fused_rotate_project_gate_up,
)

# Llama-3.1-8B FC1: hidden 4096 in, gate and up of 14336 each.
FC1_K, FC1_OUT = 4096, [14336, 14336]


def _fc1_inputs(M, BS, device="cuda", dtype=torch.bfloat16, seed=1):
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(M, FC1_K, device=device, dtype=dtype, generator=g) * 0.01).contiguous()
    W = (torch.randn(sum(FC1_OUT), FC1_K, device=device, dtype=dtype, generator=g) * 0.02).contiguous()
    blocks = 2 * (FC1_K // BS)
    eye = torch.eye(BS, device=device, dtype=dtype)
    noise = torch.randn(blocks, BS, BS, device=device, dtype=torch.float32, generator=g) * 0.02
    R = (eye.float().unsqueeze(0) + (noise - noise.transpose(-1, -2))).to(dtype).contiguous()
    return x, R, W


@pytest.mark.parametrize("BS", [128, 256, 512, 1024])
def test_gate_up_projection_at_large_blocks(BS):
    x, R, W = _fc1_inputs(64, BS)
    out = fused_rotate_project_gate_up(x, R, W, FC1_OUT)
    torch.cuda.synchronize()
    err = (out.float() - _reference(x, R, W, FC1_OUT)).abs().max().item()
    assert err <= TOL, f"BS={BS} max_abs={err:.2e}"


@pytest.mark.parametrize("BS", [128, 256, 512, 1024])
def test_gate_up_inputs_at_large_blocks(BS):
    """No projection here -- it returns the two rotated inputs, so the reference
    is the rotation alone."""
    x, R, _ = _fc1_inputs(64, BS)
    x_gate, x_up = fused_rotate_gate_up_inputs(x, R)
    torch.cuda.synchronize()
    blocks_per_slice = R.shape[0] // 2
    for idx, got in enumerate((x_gate, x_up)):
        expect = torch.empty_like(got, dtype=torch.float32)
        for b in range(blocks_per_slice):
            k0 = b * BS
            expect[:, k0:k0 + BS] = x[:, k0:k0 + BS].float() @ R[idx * blocks_per_slice + b].float()
        err = (got.float() - expect).abs().max().item()
        assert err <= TOL, f"BS={BS} slice={idx} max_abs={err:.2e}"
```

- [ ] **Step 2: Run to verify the new tests fail**

```bash
python -m pytest test/srt/oft/test_fused_rotate_project_tiled.py -q -k "gate_up"
```

Expected: `OutOfResources` for BS 256/512/1024; BS=128 passes.

- [ ] **Step 3: Apply the same branch**

`fused_rotate_project_gate_up` shares `_fused_rotate_project_inner`, so it needs only the launch-side switch — the same four lines added to `fused_rotate_project_qkv` in Task 4 Step 3, using `_pick_tiles` instead of `_pick_qkv_tiles`, plus `TILED`/`TILE_K` on `_fused_rotate_project_gate_up_kernel`.

`_fused_rotate_gate_up_inputs_kernel` is separate — it does two `tl.dot`s against `R_gate` and `R_up` (around line 192) and no projection. Tile it the same way, accumulating one `TILE_K` column tile of each output at a time:

```python
        if TILED:
            for j in range(0, BS, TILE_K):
                offs_j = j + tl.arange(0, TILE_K)
                gate_j = tl.zeros((BLOCK_M, TILE_K), dtype=tl.float32)
                up_j = tl.zeros((BLOCK_M, TILE_K), dtype=tl.float32)
                for i in range(0, BS, TILE_K):
                    offs_i = i + tl.arange(0, TILE_K)
                    x_i = tl.load(
                        x_ptr + offs_m[:, None] * K + (k_block_start + offs_i)[None, :],
                        mask=m_mask[:, None], other=0.0,
                    )
                    gate_j += tl.dot(
                        x_i, tl.load(R_gate_ptr + R_gate_base + offs_i[:, None] * BS + offs_j[None, :]),
                        out_dtype=tl.float32,
                    )
                    up_j += tl.dot(
                        x_i, tl.load(R_up_ptr + R_up_base + offs_i[:, None] * BS + offs_j[None, :]),
                        out_dtype=tl.float32,
                    )
                tl.store(gate_out_ptr + offs_m[:, None] * K + (k_block_start + offs_j)[None, :],
                         gate_j.to(tl.bfloat16), mask=m_mask[:, None])
                tl.store(up_out_ptr + offs_m[:, None] * K + (k_block_start + offs_j)[None, :],
                         up_j.to(tl.bfloat16), mask=m_mask[:, None])
        else:
            <the existing two-dot body, unchanged>
```

Match the existing pointer and offset names in the file rather than the placeholders above.

- [ ] **Step 4: Run the whole test file and the benchmark gate**

```bash
python -m pytest test/srt/oft/test_fused_rotate_project_tiled.py -q
python test/srt/oft/bench_fused_rotate_project_blocks.py --compare test/srt/oft/baseline_9c83ae8be.json
```

Expected: all tests pass; the comparison still exits 0.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/oft/triton_ops/fused_rotate_project.py test/srt/oft/test_fused_rotate_project_tiled.py
git commit -m "feat(oft): tile the gate-up rotation paths for large block sizes"
```

---

### Task 7: Make BS <= 128 faster, or prove it cannot be

**Files:**
- Modify: `python/sglang/srt/oft/triton_ops/fused_rotate_project.py`

**Interfaces:**
- Consumes: the benchmark and baseline from Tasks 2 and 5.
- Produces: either a measured improvement in `_pick_qkv_tiles` / `_pick_tiles`, or a comment recording that the current tiles are already the best of those tried.

**This is the "or even better" half of the goal, and it is measurement-first.** No change ships here without a benchmark row proving it.

- [ ] **Step 1: Sweep the tile parameters at BS <= 128**

Add a throwaway script `/tmp/sweep_tiles.py` (do not commit):

```python
import itertools, json, torch
from sglang.srt.oft.triton_ops import fused_rotate_project as F

results = []
for BLOCK_M, BLOCK_N, GROUP_N in itertools.product((32, 64, 128), (32, 64, 128), (1, 2)):
    shared = 3 * 2 * (BLOCK_M * 128 + 128 * 128 + GROUP_N * BLOCK_N * 128)
    if shared > 232448:
        results.append({"tiles": (BLOCK_M, BLOCK_N, GROUP_N), "skip": f"{shared} B > limit"})
        continue
    F._pick_qkv_tiles = lambda M, w, bs, t=(BLOCK_M, BLOCK_N, GROUP_N): t
    from test.srt.oft.bench_fused_rotate_project_blocks import bench_block_sizes
    rows = bench_block_sizes(block_sizes=[64, 128], batches=[1, 64, 256])
    results.append({"tiles": (BLOCK_M, BLOCK_N, GROUP_N),
                    "ms": {f"{r['M']}/{r['BS']}": r["ms"] for r in rows}})
print(json.dumps(results, indent=2))
```

```bash
python /tmp/sweep_tiles.py
```

- [ ] **Step 2: Decide from the numbers**

If some configuration beats the current `(64, 64, 1)` at `BS=128` by more than the 5% tolerance **on every M**, adopt it in `_pick_qkv_tiles`. If the win is only at some batch sizes, make the picker branch on `M` — it already does for the `BS < 128` path.

If nothing wins, add this comment above `_pick_qkv_tiles` and change no code:

```python
# Tile choice swept over BLOCK_M/BLOCK_N/GROUP_N in {32,64,128}x{32,64,128}x{1,2}
# at BS in {64,128} and M in {1,64,256} on 2026-XX-XX; (64,64,1) was the fastest
# configuration that fits the 232,448 B budget at BS=128. Do not re-tune without
# re-running test/srt/oft/bench_fused_rotate_project_blocks.py.
```

- [ ] **Step 3: Re-run the gate**

```bash
python -m pytest test/srt/oft/test_fused_rotate_project_tiled.py -q
python test/srt/oft/bench_fused_rotate_project_blocks.py --compare test/srt/oft/baseline_9c83ae8be.json
```

Expected: tests pass; comparison exits 0. If a tile change made something slower, revert it — the baseline is the contract.

- [ ] **Step 4: Commit**

```bash
git add python/sglang/srt/oft/triton_ops/fused_rotate_project.py
git commit -m "perf(oft): record the tile sweep for the untiled rotation path"
```

---

### Task 8: Ship it, and let orbit use the larger blocks

**Files:**
- Modify: `tools/lora_regret/arms.py` (orbit repo)
- Modify: `tests/fast/utils/test_lora_regret_arms_coverage.py` (orbit repo)
- Modify: `docs/superpowers/plans/2026-07-30-lora-without-regret-runbook.md` (orbit repo)

**Interfaces:**
- Consumes: a published SGLang build containing Tasks 4 and 6.
- Produces: `OFT_MAX_BLOCK_SGLANG` raised from 128 to the new measured ceiling.

- [ ] **Step 1: Rebuild the environment against the new kernel**

Push the branch and rebuild the orbit env against it, or install the worktree editable into a scratch venv. Then re-run the reproduction from *inside orbit's* environment:

```bash
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
export CUDA_HOME=/is/software/nvidia/cuda-13.2 && source env.sh
python /path/to/worktree/test/srt/oft/repro_shared_memory_ceiling.py
```

Expected: all seven block sizes `OK`. If not, orbit is still resolving the old pinned build — check `python -c "import sglang; print(sglang.__version__, sglang.__file__)"`.

- [ ] **Step 2: Raise the cap**

In `tools/lora_regret/arms.py`, replace the value and rewrite the comment to record the new measurement:

```python
# The largest OFT block SGLang's fused rotate-project kernel can launch.
#
# Was 128: the kernel staged the whole BS x BS rotation block in shared memory,
# costing 6*BS*(BS+128) bytes against a 232,448 B limit. Since <SGLang commit>
# the rotation is walked in OFT_TILE_K sub-tiles and the footprint no longer
# depends on BS, so this is now bounded by <what Task 5's benchmark showed>
# rather than by shared memory.
OFT_MAX_BLOCK_SGLANG = <new value>
```

- [ ] **Step 3: Update the tests that pin the old ceiling**

In `tests/fast/utils/test_lora_regret_arms_coverage.py`, `TestOftBlockCeilingUnderRl`:

- `test_the_ceiling_is_the_measured_one` — assert the new value.
- `test_the_ceiling_matches_every_working_example_launcher` — keep as is; it asserts `max(example blocks) <= OFT_MAX_BLOCK_SGLANG`, which stays true when the ceiling rises.
- `test_the_rl_placement_cells_are_not_capacity_matched` — **this should now fail**, because the capacity match becomes reachable. Delete it and remove the `pytest.skip("capacity match unreachable under the SGLang block cap")` from `test_the_oft_capacity_is_in_the_neighbourhood_of_a_lora_arm_it_sits_beside`, so `e4place` is checked like every other matrix again.

- [ ] **Step 4: Verify the RL OFT arms are matched again**

```bash
python -m pytest tests/fast/utils/test_lora_regret_arms_coverage.py -q
python -c "
from tools.lora_regret.arms import MATRICES
for m in ('e4','e4place'):
    for a in MATRICES[m](4096,14336,0,None,None):
        if a.method=='oft': print(m, a.name, f'ratio={a.matched_ratio:.3f}')
" | sort -u
```

Expected: `e4place`'s OFT blocks now match its r256 / r92 LoRA arms, and the neighbourhood test passes without a skip.

- [ ] **Step 5: Run orbit's full suite**

```bash
pytest tests -q 2>&1 | tail -3
```

Expected: **0 failed**.

- [ ] **Step 6: Note it in the runbook**

In `docs/superpowers/plans/2026-07-30-lora-without-regret-runbook.md` §20, replace the sentence about OFT being capped at 128 under RL with the new ceiling and a pointer to this plan.

- [ ] **Step 7: Commit**

```bash
git add tools/lora_regret/arms.py tests/fast/utils/test_lora_regret_arms_coverage.py \
        docs/superpowers/plans/2026-07-30-lora-without-regret-runbook.md
git commit -m "feat(lora_regret): raise the OFT block ceiling to the new kernel limit"
```

---

## Verification

From a clean shell, in the SGLang worktree:

```bash
python test/srt/oft/repro_shared_memory_ceiling.py            # 7/7 OK
python -m pytest test/srt/oft/test_fused_rotate_project_tiled.py -q   # all pass
python test/srt/oft/bench_fused_rotate_project_blocks.py \
    --compare test/srt/oft/baseline_9c83ae8be.json            # exit 0
```

and in orbit:

```bash
pytest tests -q 2>&1 | tail -3                                # 0 failed
```

## What this plan does not do

**It does not make large blocks as fast as small ones.** A `BS=1024` rotation is 64x the FLOPs of a `BS=128` one per block, though there are 8x fewer blocks — so expect roughly 8x the rotation cost per token, plus the `x` re-read. The goal is that they *run*, and that the sizes that already ran do not get slower.

**It does not touch the backward pass.** `sgemm_oft_r_bwd.py` and the training-side rotation are unaffected; Megatron never hits this kernel, which is why the campaign's SFT OFT arms work at `BS=1024` today.

**It does not change the OFT numerics.** The tiled path reorders the summation inside one reduction and casts at the same point, so results move by bf16 rounding only — bounded by the existing `2e-3` parity tolerance and asserted by `test_the_two_paths_agree_at_the_boundary`.
