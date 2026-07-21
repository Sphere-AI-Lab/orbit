# sglang v0.5.13 on cu129 bare-metal — env test (2026-07-02)

Goal: build a **fresh** env (`miles_v0513_test`) running sglang **v0.5.13** source +
**torch 2.11.0+cu129** + the existing `cu129-x86_64-v0.5.12` wheels bundle, WITHOUT touching
the working `miles` env (torch 2.9.1). Fixes the `/begin_weight_update` 404 (routes exist at
v0.5.13: `http_server.py:1281/1294`) and executes the deferred sglang-sync's env half.

History: the June attempt (`miles-sync-2026-06-02/install-findings.md`) died on **Issue 8**:
no public sgl-kernel wheel was BOTH cu12 AND torch-2.11 (PyPI kernels were cu13-only;
host driver 570.195.03 = CUDA 12.8 max). Fallback then = old torch-2.9.1 binaries +
`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK`.

## Phase-0 probe results (2026-07-02, readelf-verified)

**The June wall is gone — sgl-project now publishes cu12 kernel twins:**
- `sglang_kernel-0.4.3+cu129` (github.com/sgl-project/whl releases, indexed at
  `docs.sglang.ai/whl/cu129/sglang-kernel/`): every `.so` links `libcudart.so.12`/
  `libnvrtc.so.12`/`libcublas.so.12`. The **PyPI default `0.4.3` is still 100% cu13** —
  the `+cu129` local version must be pinned explicitly.
- `sgl_deep_gemm-0.1.2+cu129` likewise cu12 (PyPI default cu13). June's uninstall hack unneeded.
- **NEW trap:** `torchao==0.17.0` PyPI wheel links `libcudart.so.13` → must use
  `0.17.0+cu129` from `download.pytorch.org/whl/cu129`.
- `flash-attn-4` (>=b9): pure-Python CuTe-DSL JIT, no ELF lock; b20/quack-0.5.1+ pin
  `cutlass-dsl==4.6.0.dev0` (conflicts sglang's `==4.5.2`) → pin `b19` + `quack-kernels==0.5.0`.
- `cuda-tile[tileiras]` (via flashinfer): 1.4.0 wants cuda-toolkit 13.2-13.4, conflicts torch
  cu129's `cuda-toolkit==12.9.1` → constrain `<1.4`.
- `cuda-python>=13.0` still in v0.5.13 pyproject (June Issue 1) — metadata-impossible next to
  torch cu129 (`cuda-bindings<13`); runtime-safe to lower (srt use is lazy w/ bindings-12 fallback).
- flashinfer 0.6.12: `[cu12]` extra = plain cutlass-dsl; `flashinfer_cubin` arch-agnostic.
- transformers 5.8.1 + kernels: June Issue 6 moot (sglang itself caps `kernels<0.15`;
  transformers 5.8.1 has no unconditional kernels dep).
- sglang-router 0.3.2 on PyPI = manylinux_2_17 (OK on GLIBC 2.35); the miles-wheels release
  wheel is still manylinux_2_39 → June GLIBC source-build fallback still needed.
- Engine floors at v0.5.13: sglang-kernel>=0.4.3, flashinfer>=0.6.12 (`engine.py:1330/1338`),
  still behind `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK` — but with matching versions installed,
  no skip needed.
- mrope patch (`c74db48da`) NOT subsumed at v0.5.13; 3-way merge clean.

## What was set up

1. **Worktree** `~/workspace/sglang-v0513`, branch `sync-v0.5.13-20260702` =
   `6ee17b436` (sgl-project@sglang-miles tip, v0.5.13-35) + `4f3aaf47a` (mrope re-apply)
   + `fe3f5fced` "[sglang-miles cu129] bare-metal cu12 dep flavors" (pyproject: cuda-python
   `>=12.9.4,<13`, flashinfer `[cu12]`, plain cutlass-dsl, `sglang-kernel==0.4.3+cu129`,
   `sgl-deep-gemm==0.1.2+cu129`, `torchao==0.17.0+cu129`). The `thirdparty/sglang` submodule
   tree stays at `c74db48da` — the working `miles` env's editable install is untouched.
2. **June hardening restored** onto `scripts/slurm/setup/` (from the recovered 2026-06-02
   blobs; they were never committed): cudnn derived-from-torch + 3× reassert + LD prepend,
   TE decomposed source-build + onnx/onnxscript, cuda-python override + torch `+cu129`
   constraint through the sglang resolution, router GLIBC source-build fallback, KERNELS_SPEC.
3. **New knobs** in `install_env.sh`: `SGLANG_SRC` (external worktree), 
   `MILES_ALLOW_WHEELS_SGLANG_LAG=1` (bundle wheels are torch-ABI-bound, not sglang-version-
   bound; upstream pairs the v0.5.13 image with v0.5.12 wheels), `SGL_WHL_INDEX_URL`,
   `SGLANG_EXTRA_CONSTRAINT`. `verify_env.py` honors `SGLANG_SRC` for the editable check.
   `extract_pins.py`: `WHEELS_TAG(?:_X86)?` regex for the split upstream Dockerfile ARGs;
   CUDNN pin row removed (derived at install time now). `pins.env`: CUDNN line removed
   (escape hatch stays as env var, e.g. 9.16.0.29 for torch-2.9.x rebuilds per #168167).

## Install run

Jobs on slinky-5 (preflighted: weka read 2s, nvcc 12.8, GLIBC 2.35):
`install-v0513-test.sbatch` → env `miles_v0513_test`, `MILES_WHEELS_TAG=cu129-x86_64-v0.5.12`,
`TORCH_VERSION=2.11.0`, `MOONCAKE_VERSION=0.3.11.post1` (v0.5.13 Dockerfile pin),
constraints `flash-attn-4==4.0.0b19, quack-kernels==0.5.0, cuda-tile<1.4`.

**Run 1 (job 22055): FAILED at the router source build — everything before it PASSED.**
The whole cu129 strategy resolved as designed (from the log):
`sglang-kernel==0.4.3+cu129`, `sgl-deep-gemm==0.1.2+cu129`, `torchao==0.17.0+cu129`,
`cuda-python==12.9.0`/`cuda-bindings==12.9.4`, `flash-attn-4==4.0.0b19`,
`quack-kernels==0.5.0`, `cuda-tile==1.3.0`, `flashinfer 0.6.12`, `transformers==5.8.1`,
torch held at `2.11.0+cu129`; TE 2.10.0 torch-ext source-compiled; FA2/FA3/apex installed;
router GLIBC guard correctly fell back to source build. The build then died on
**maturin 1.14.1**: `project.readme path '../../README.md' resolves outside allowed
metadata root .../bindings/python` — sgl-router-for-miles declares its readme outside the
package dir (`bindings/python/pyproject.toml:16`) and maturin >=1.14 refuses that (the June
build predated the restriction). Fix: install_env.sh router fallback now inlines the README
and rewrites the reference before `maturin build`.

**Run 2 (job 22102): router built (fix worked: README inlined → maturin OK →
`sglang_router-0.3.2 manylinux_2_34` installed), reached verify → 31/37.** The 6 fails were
ONE real bug + artifacts: sglang's unpinned `scipy` resolved to **1.18.0 whose runtime
requires numpy>=2**, while the late `numpy<2` step (upstream-Dockerfile parity, line 142)
left numpy at 1.26.4 → `np.long` AttributeError broke `import sglang`/`megatron.bridge`/
`mbridge`/fp8_kernel. The `torch==2.9.1`/`mooncake==0.3.9` fails were verify_env comparing
against pins.env FILE defaults, ignoring the run's env overrides.

**Fixes:** install_env.sh now does `$UV 'numpy<2' 'scipy<1.16'` (scipy 1.15.x runs on numpy
1.26); verify_env.py `load_pins()` now overlays exported env vars over file defaults (same
`${KEY:-default}` contract as sourcing pins.env).

**Surgical re-verify on slinky-5 (scipy 1.18.0→1.15.3): `=== 37/37 pass, 0 fail ===`** —
including the decisive `sgl_kernel symbol: sgl_per_token_quant_fp8` (the +cu129 kernel .so
LOADED on the CUDA-12.8-driver host — June Issue 8 cleared at binary level),
`sglang engine import: srt.layers.quantization.fp8_kernel`, torch 2.11.0+cu129 (build cu129),
runtime cuDNN 9.17.1, TE 2.10.0, mooncake 0.3.11.post1, FA3 symbols, apex C exts, and
`sglang editable @ ~/workspace/sglang-v0513/python/` (worktree — production tree untouched).
`import sglang` reports `0.5.14.dev37+gfe3f5fced` (the worktree branch tip). Only noise: the
known benign Qwen3-ASR `cache_position` docstring ERRORs from Megatron-Bridge.

NOTE: the clean-slate end-to-end proof of install_env.sh with ALL fixes in one run is still
pending (the scipy fix landed as a surgical env edit after run 2; June followed the same
pattern — hand-validated env first, fresh-run proof after).

## Smoke run (geo3k multi-turn colocate 1-node)

Job **22113**, run dir `runs/geo3k-mt-smoke-v0513/260702_194517`, env `miles_v0513_test`,
June sizing (`NUM_ROLLOUT=2, ROLLOUT_BATCH_SIZE=8, N_SAMPLES=2`), `TMPDIR=/tmp`,
**no `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK`** (kernel 0.4.3+cu129 == the v0.5.13 floor;
the assert passing is part of the validation). Also first end-to-end exercise of
`POST /begin_weight_update`/`end_weight_update` (colocate weight sync calls them — the
routes the 21656 404 was missing). **RESULT: ✅ SUCCEEDED end-to-end (2026-07-02 20:41 UTC).**

- ray bring-up: attempt 1 collapsed fast (`ray_client_server` node-info timeout — the known
  cold-cache/slow-GCS class from 21571/21582; first-ever ray start on a fresh env), the NEW
  fail-fast+retry logic caught it instantly and **attempt 2 succeeded** — the launcher fix
  paying off in production.
- engines: weights loaded ×8 (974 s, weka-slow but clean), **cuda-graph capture 68 s ×8** —
  the `+cu129` kernel stack (sgl-kernel 0.4.3, FA3, flashinfer sampling) functionally proven;
  `fired up and ready to roll` ×8. The v0.5.13 `sglang-kernel>=0.4.3` startup assert passed
  with NO skip flag.
- **`POST /begin_weight_update` → 200 and `/end_weight_update` → 200, 3 cycles × 8 engines,
  zero 404s** — the job-21656 sync regression is fixed end-to-end.
- training: rollout ×2 (multi-turn geo3k + tool calls, cuda graphs active, ~430 tok/s/req),
  step 0 `grad_norm 0.383`, step 1 `grad_norm 0.658`;
  `train_rollout_logprob_abs_diff ≈ 0.015`, `train_rollout_kl ≈ 0.0007` — trainer-vs-engine
  numerics consistent (June torch-2.9.1 reference: same order).
- teardown: `ray job logs observed terminal state SUCCEEDED; not reconnecting`
  (ray_lifecycle fix working); only the known benign wandb `teardown_atexit` noise.
- torchcodec FFmpeg probe tracebacks at engine start are NON-fatal noise (no FFmpeg libs in
  the env; video decode unused by geo3k). Parity TODO: `conda install -n miles_v0513_test
  -c conda-forge ffmpeg` (upstream docker base ships ffmpeg via apt).

## Clean-slate proof (2026-07-03)

Job **22160** (slinky-5, 1:10:31): `install_env.sh` from an EMPTY env
(`miles_v0513_fresh_20260703`) with all fixes in-script → **INSTALL_EXIT=0, verify 37/37,
zero manual intervention**. Both previously-surgical fixes engaged automatically (router
README inline → maturin OK; scipy 1.18.0→1.15.3 alongside numpy<2). The script is proven
end-to-end for the v0.5.13/cu129 line. (`miles_v0513_test` remains the smoke-validated env;
`miles_v0513_fresh_20260703` is the script-proof env.)

## 3-node async validation (2026-07-03) — found & fixed a SECOND dropped patch

Recipe `async/geo3k-vlm-multi-turn-fully-async-prefetch2-3node` (Qwen3-VL-8B, 24 GPUs),
env `miles_v0513_test`, excluded slinky-2/50.

- **Attempt 1 (job 22261):** healthcheck false-failed slinky-44/13 — tier2's cold
  `import torch` vs the 60 s probe timeout (the known class). Launcher default now **300 s**
  (`launch_miles.sbatch`); with it, all three nodes pass and tier-nccl reads 347 GB/s.
- **Attempt 2 (job 22263):** everything green through engines-up + **`begin_weight_update`
  → 200 ×16 on the async path** (the 21656 site, fixed) + step 0 (`grad_norm 0.375`,
  `logprob_diff 0.0117`) — then **FAILED at the post-step flush**:
  `GET /flush_cache → 400 ×951`, `Cache not flushed because there are pending requests.
  #queue-req: 16, #running-req: 0` → miles' 60-try poll → `TimeoutError` → run dies.
  **Root cause: the v0.5.13 rebase of sglang-miles dropped the pause-aware flush**
  (`is_fully_idle() OR (_engine_paused and running_batch empty)`) introduced by
  "[sglang-miles] Fix pause-aware weight update deadlocks" (#22754/#22623). The rest of
  that patch series survived (`_engine_paused` exists; #29675 locking present) — only the
  flush_cache disjunct was lost in the refactor, and the **live upstream tip (f8cfad35b)
  still lacks it**. Colocate never hits this (nothing queued during update); fully-async
  always will (prefetched rollout requests sit in waiting_queue at weight-update time,
  and queued requests hold no KV state, so the pause-aware flush is safe by design).
  **Fix: worktree commit `723ed7d19` restores the disjunct + the paused empty_cache guard.
  Must go upstream to sgl-project@sglang-miles with the /sglang-sync landing.**
- **Attempt 3 (job 22272):** the flush restore WORKED (`Cache flushed successfully!` ×16, zero
  400s) — then a NEW failure ~2.5 min after the first update: all 8 engines on slinky-44
  crashed with `PermissionError: /tmp/triton/<key>` from stock triton's JIT cache
  (`mem_cache/common.py write_cache_indices` → triton jit → cache makedirs). slinky-44 is
  multi-tenant (zichen/evanz jobs); a shared `/tmp/triton` is collision-prone. No setter of
  `TRITON_CACHE_DIR=/tmp/triton` was found in miles/sglang/triton/tokenspeed/torch/ray code
  (all defaults are home-based or user-suffixed) — unresolved WHO, but defended regardless:
  **miles now pins `TRITON_CACHE_DIR` per-user + per-rank** (`/tmp/triton_$USER/<type>_rank_N`)
  in `server_group.py` engine env (mirroring the `SGLANG_DG_CACHE_DIR` pattern) and per-user in
  `actor_group.py` trainer env.
- **Attempt 4 (job 22338): ✅ SUSTAINED — validation complete.** Past every prior failure
  point; 150+ steps at ~40 s/step, `logprob_abs_diff` tightening to 0.0065,
  **outlived the pre-sync baseline** (21100 crashed at step ~137 on the same recipe).
  wandb: https://wandb.ai/M3TRL/async_envpack/runs/22338

**RL-metrics regression check (vs pre-sync run 21100, same recipe, matched step windows,
`rollout/raw_reward`):** 0-26: 0.587→0.576 · 27-68: 0.665→0.650 · 69-110: 0.712→0.700 ·
111-136: 0.730→0.712 · full: 0.676 vs 0.662. Identical learning slope; the ~0.014 offset is
constant from step 0 (pre-training-divergence), ~1.4 SE (per-step σ≈0.08, n=137), and smaller
than the spread between the two pre-sync baselines themselves. **Verdict: no regression.**

Long-window confirmation vs pre-sync 21021 (`geo3k-async-mt-8b-mf07`, 375 steps —
the longest pre-sync async-mt-8b run): 0-99: 0.651→0.642 · 100-199: 0.745→0.738 ·
200-299: 0.801→0.791 · 300-375: 0.820→0.817 — the offset SHRINKS with training
(−0.008 → −0.003); trajectories converge at reward ≈0.82. 22338 passed step 400+,
outliving both baselines (which crashed at 137 and 375).

## Remaining to land this (not yet done)
1. ~~Commit the setup-script changes~~ → committed (see branch).
2. ~~Clean-slate end-to-end install_env.sh proof~~ → done (job 22160, above).
3. The real `/sglang-sync` (approval-gated): mirror-publish the worktree branch
   (= tip + mrope + cu129-flavors as the 2nd local patch), bump the gitlink, realign
   pins.env ACTIVE / WHEELS_STACK; teach the skill the validated "bundle may lag source
   when torch matches" rule.
4. Scale-up validation: the 3-node async geo3k (the original 21656 recipe) + RL-metrics
   regression check vs a known-good baseline.

## Watch items at install/verify/smoke

- resolver backtracking (cuda-tile/flash-attn-4), tokenspeed_mla → tokenspeed-triton fork
  (may fight torch 2.11's triton), torchcodec needs FFmpeg shared libs (conda ffmpeg if
  import fails), smg-grpc-servicer needs grpcio>=1.81.1, flashinfer-cubin ships
  cu13-toolchain cubins (sm90 SASS should load on r570; PTX-JIT fallback would not — verify
  at smoke), rust ext `sglang.srt.grpc._core` builds via cargo at editable-install time.
- verify_env 35/35 is INSTALL proof, not engine proof (June Issue 9) — the June verify_env
  adds `from sgl_kernel import ...` engine-path imports; the real gate is the geo3k smoke
  (which also exercises `begin_weight_update` end-to-end: colocate weight sync calls it).
