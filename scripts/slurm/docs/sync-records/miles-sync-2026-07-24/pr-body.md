## Summary

Sync with upstream `radixark/miles` — **187 upstream commits** merged (merge-base `5e97c865`, 2026-06-29 → upstream tip `c94c2fa9`, 2026-07-23), **plus the combined sglang-sync**: `thirdparty/sglang` advances to the **v0.5.15 sglang-miles line** (`38d4bbef5` = `sgl-project/sglang@sglang-miles` `v0.5.15-31` + 4 re-applied local patches). The ACTIVE wheels bundle stays `cu129-x86_64-v0.5.12` — torch is unchanged at 2.11.0, so this is a textbook bundle-may-lag sync (upstream's own Dockerfile pairs its v0.5.15 image the same way).

Major upstream content: the fault-tolerance megaseries (~50 commits: witness ids, independent-DP cells, in-memory checkpoints, heartbeat/control-server, event logging), multi-LoRA (7 parts), session-server scale-out (N instances on a port range), metric-history CI gate, miles dashboard, disk-delta weight sync, dual-clip PPO, GLM-5 LoRA RL, entropy observation, pass@k relocation, and two sglang bumps (v0.5.14, v0.5.15).

## Commits on top of the merge

| commit | what |
|---|---|
| `f3e2c8d3` | pins/install refresh + sglang v0.5.15 advance (single bundle commit; see below) |

## Conflicts resolved (20 files — all reviewed per-file)

The largest conflict surface of any sync so far: upstream's FT megaseries and our OPD series (#23–#40, freshly merged to main) touch the same core files. Every file was resolved with per-item review; the recurring pattern was **take upstream's structural evolution + graft our feature back in**:

- **`miles/ray/actor_group.py`** — upstream extracted actor construction into `miles/ray/train/actor_factory.py` (#1416); took the factory, **ported our per-user `TRITON_CACHE_DIR` pin into it** (cross-user `/tmp/triton` EACCES, job 22272).
- **`miles/backends/megatron_utils/model_provider.py`** — upstream's `_apply_bridge_runtime_config()` (#1483) is a superset of our manual propagation block; kept **our freeze-vision block** (no upstream equivalent exists; frozenvit recipes depend on it).
- **`miles/backends/megatron_utils/model.py`** — kept our `mm_token_type_ids` filtering (`**mm_inputs`) — upstream's raw `**batch["multimodal_train_inputs"]` passthrough would crash Qwen3-VL — plus their witness kwargs and multi-LoRA adapter block.
- **`miles/utils/data.py`** — upstream #1767 fixed the same `filter_long_prompt` bug class as our #34 with a text/multimodal split; took their structure, kept **our stored-`multimodal_inputs` reuse** (their loop still re-runs `process_vision_info` on templated strings → crash), kept our `call_processor` (kimi-vl medias support). Restored the `call_processor` import the auto-merge dropped.
- **`miles/backends/training_utils/data.py`** — our OPD DAgger device-move helper + their `witness_info` signature/materialization (#1754's CP alignment sits in the auto-merged region; OPD fast tests green).
- **`examples/fully_async/fully_async_rollout.py`** — our prefetch design **layered on** upstream #1677: explicit `--async-max-concurrent-samples` overrides absolutely; otherwise the window is `rollout_batch_size × --fully-async-prefetch-batches` (replaces upstream's legacy one-batch fallback; degenerates to it at prefetch=1). John's `_active_tasks` fail-closed plumbing kept.
- **`miles/ray/rollout/router_manager.py`** — took upstream's N-instance session-server scale-out (#1659/#1602) with readiness `timeout=300` (upstream's 30s re-introduces the bare-metal cold-import false-kill we fixed in 84f65a8d; 600→300 per review).
- **`train.py` / `train_async.py`** — kept our watchdog sentinel (`set_progress`/`write_train_status`; consumed by `launch_miles.sbatch` outside ray — upstream's new FT heartbeat lives inside ray, different layer), took their `rollout_id == args.start_rollout_id` eval fix (#1579) and tracking package paths.
- **`.github/workflows/pr-test.yml`** — upstream's `resolve-ci-policy` refactor taken verbatim; our `run-ci-h200-gpu`/`run-ci-h100-gpu` hardware-label gates re-ANDed onto all 4 GPU stages (fork has no self-hosted GPU runners).
- **`miles/utils/arguments.py`, `tracking_utils/base.py`, `log_utils.py`, `metrics.py`, `rollout_manager.py`, `train_data_conversion.py`, `misc.py`, `broadcast.py`, 3 test files** — unions (both sides' args/validators/metrics/attrs/imports kept; redundant `is_lora` line dropped in favor of upstream's `_init_lora`).

Three test-double updates were needed (not regressions): fakes lacked upstream's new interface (`async_max_concurrent_samples` attr, `compute_policy_loss` 5th param `eps_clip_c` ×2, `session_id` field in an sglang test).

## sglang: v0.5.15 line, patch-by-patch

`thirdparty/sglang` → `38d4bbef5` = `v0.5.15-31` (`94949da73`) + re-applied local patches (verified by CONTENT on the new base, not titles):

1. **mrope text-only gate** (`get_rl_on_policy_target()` disjunct in forward_batch) — absent on v0.5.15, re-applied clean.
2. **cu12 dep flavors** — re-ported onto the v0.5.15 dep set: `flashinfer_python[cu12]==0.6.12`, `sglang-kernel==0.4.4+cu129`, `sgl-deep-gemm==0.1.4+cu129`, `torchao==0.17.0+cu129`, `cuda-python<13`, plain `nvidia-cutlass-dsl` — **all +cu129 wheels verified to exist** on their indexes before committing.
3. **exact multimodal scoring suffix** (impossible-inc/sglang #3) — absent on v0.5.15, re-applied; 4 conflict files were tail-of-parameter-list unions with upstream's new `session_id` plumbing.
4. **qwen-vl pretokenized-IDs fix** (impossible-inc/sglang #4) — absent on v0.5.15, re-applied clean.
5. **pause-aware `flush_cache` restore — DROPPED**: upstream fixed it officially (#31962, `is_fully_idle(ignore_waiting=self._engine_paused)`), semantics fully cover our patch. One less mirror patch to carry.

Patch-shipped unit tests on the new pin: **54 passed + 13 subtests** (scoring suffix ×3 files, pretokenized IDs, io_struct).

Mirror publish plan (waits for approval, ordered): push `sync-v0.5.15-20260724` branch → open review PR (per impossible-inc/sglang #1/#2 convention) → archive `sglang-miles-v0.5.13-final` + date tag → lease-guarded force-advance `sglang-miles` → date-tag new tip.

## Pin / install changes

- **`pins.env`**: `UPSTREAM_*` → v0.5.15 / `cu130-x86_64` (upstream's new **torch-ABI-only wheels-tag naming** — no sglang-version suffix; releases republished only on torch bumps). ACTIVE untouched.
- **`TMS_COMMIT` becomes a hand-owned pin at `6d5bce48`** — upstream unpinned torch_memory_saver (#1773); we keep determinism for bare-metal rebuilds. `extract_pins.py` gains `read_preserved()`; the dead Dockerfile pattern is gone.
- **`extract_pins.py` pending comparator**: version-less upstream tags no longer trip `[sglang-sync pending]` (nothing to version-compare; the torch-ABI check in `abi_errors()` is the real guard). `--check` is clean at exit 0 with **no pending**.
- **`install_env.sh`**: `TMS_CUDA_MAJOR` derived from the env's torch for the newer TMS source build (#1774); **`numpy<2` + `scipy<1.16` cap removed** — upstream dropped the numpy 1.x force at the v0.5.14 bump (#1587); the merged `requirements.txt` (installed directly) brings `transformers<5.13`, blake3/xxhash/zstandard, nvidia-resiliency-ext, psycopg, polars==1.42.1.
- **Deliberately NOT mirrored** (model-family-specific, unused here): Emerging-Optimizers (Muon — import is try/except-guarded), tilelang/tile_kernels/FlashQLA (DSV4/Qwen-GDN), causal-conv1d/mamba-ssm (nemotron-h).

## ⚠️ Attention items

- **Env rebuild required before the next run**: sglang tree is now v0.5.15 (editable install picks it up immediately) while the env's binary deps are still the v0.5.13-era set — rebuild with `install_env.sh` and expect `transformers` to move under the new `<5.13` cap and **numpy to major-bump to 2.x** (validation owns this risk per the test plan).
- **FT megaseries / multi-LoRA / dashboard are flag-gated** — nothing enables them by default; our recipes are unaffected until opted in.
- **`fully-async` knobs**: `--async-max-concurrent-samples` (upstream, absolute) vs `--fully-async-prefetch-batches` (ours, pipeline depth) both live; explicit absolute setting wins. Unification is a recorded follow-up.
- **Launcher watchdog sentinel retirement** is a recorded follow-up for when the launcher reads upstream's FT event stream instead.
- **Upstream `sglang-miles` tip == our pin base** (v0.5.15-31) at sync time; no deferred sglang delta.

## Divergence from upstream after sync

258 files, +40,575/−249 vs the merged upstream snapshot — `scripts/slurm/` launcher infra, OPD recipes/presentation, `miles_plugins/envpack_adapter/`, `.claude/skills/`, `examples/geo3k_vlm_multi_turn/` + fully-async hardening, in-place `miles/*` modifications listed above, and the `thirdparty/*` submodules. Full stat in [divergence.stat](divergence.stat).

## Validation

Done locally (pre-push):
- [x] Fast tests on merged tree: `test_data_filter_long_prompt` 4/4, `test_on_policy_distillation` **68/68**, `test_true_on_policy_loss_metrics` 8/8.
- [x] sglang patch unit tests on the new pin: **54 passed + 13 subtests**.
- [x] `extract_pins.py --check` exit 0, no pending; black/isort clean on all hand-edited files.

Remaining (post-push, fresh GPU salloc):
- [ ] `bash scripts/slurm/setup/install_env.sh` clean-slate rebuild (numpy 2.x + transformers <5.13 + v0.5.15 kernels land here).
- [ ] `python scripts/slurm/setup/verify_env.py` all checks pass.
- [ ] Sanity-launch: geo3k multimodal multiturn OPD smoke (06a) to completion + a fully-async 3-node smoke to first eval.

⚠️ **Merge mode**: this PR MUST be merged via **"Create a merge commit"**. Squash or rebase will break future `merge-base` detection (upstream SHAs must survive).
