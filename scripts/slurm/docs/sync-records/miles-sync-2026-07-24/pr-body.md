## Summary

Sync with upstream `radixark/miles` — **187 upstream commits** merged (merge-base `5e97c865`, 2026-06-29 → upstream tip `c94c2fa9`, 2026-07-23), **plus the combined sglang-sync**: `thirdparty/sglang` advances to the **v0.5.15 sglang-miles line** (`38d4bbef5` = `sgl-project/sglang@sglang-miles` `v0.5.15-31` + 4 re-applied local patches). The ACTIVE wheels bundle follows upstream #1784 onto rolling `cu129-x86_64`; its Apex/FA2/FA3 files are SHA256-identical to the torch 2.11.0+cu129 set validated by clean rebuild job 28782 and both baseline runs.

Major upstream content: the fault-tolerance megaseries (~50 commits: witness ids, independent-DP cells, in-memory checkpoints, heartbeat/control-server, event logging), multi-LoRA (7 parts), session-server scale-out (N instances on a port range), metric-history CI gate, miles dashboard, disk-delta weight sync, dual-clip PPO, GLM-5 LoRA RL, entropy observation, pass@k relocation, and two sglang bumps (v0.5.14, v0.5.15).

## Fork-side changes on top of the merge

| commit | what |
|---|---|
| `f3e2c8d3` | pins/install refresh + sglang v0.5.15 advance (single bundle commit; see below) |
| `dc1fac05` | make the OPD baseline W&B project configurable without changing the default |
| `8d5726f8` | skip the unused `nvidia-resiliency-ext` package on cluster glibc older than 2.39 |
| `1a134f6d` | put the environment cuDNN directory first for batch-shell and envpack-server scopes |
| `3d743d4a` | apply the repository pre-commit import cleanup to the merged tree |
| `02ff7524` | repair import and test-double regressions exposed by the merged upstream APIs |
| `80cc75e8` | add SGLang source-pin diagnostics and stabilize fork CPU checks |
| `c8da137a` | close post-sync API/CI gaps and migrate the retired wheels release to the byte-verified rolling tag |
| `34335c5e` | initialize the rollout-dispose test double with the synced `data_source` lifecycle field |
| `2d529d88` | distinguish dynamic-sampling live diagnostics from the final all-samples hook in its integration test |
| sync-record commits | capture the upstream analysis, SGLang patch classification, cluster validation, performance comparison, and OPD spike retrospective |

The initial one-code-commit preparation rule was not preserved after the PR was
published: CI and review found real follow-up defects. Public history was left
intact; each repair is recorded here under the review/CI repair protocol in the
sync skill.

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
- **`.github/workflows/pr-test.yml`** — upstream's `resolve-ci-policy` refactor taken verbatim; our `run-ci-h200-gpu`/`run-ci-h100-gpu` hardware-label gates re-ANDed onto all 5 GPU stages (fork has no self-hosted GPU runners).
- **`miles/utils/arguments.py`, `tracking_utils/base.py`, `log_utils.py`, `metrics.py`, `rollout_manager.py`, `train_data_conversion.py`, `misc.py`, `broadcast.py`, 2 test files** — unions (both sides' args/validators/metrics/attrs/imports kept; redundant `is_lora` line dropped in favor of upstream's `_init_lora`).

Three test-double updates were needed (not regressions): fakes lacked upstream's new interface (`async_max_concurrent_samples` attr, `compute_policy_loss` 5th param `eps_clip_c` ×2, `session_id` field in an sglang test).

Post-publication review found four additional compatibility gaps, fixed in
`c8da137a`: OPD's device transfer now accepts the CPU test double returned by
`torch.cuda.current_device()`; the rollout metric test patches the new
`tracking` import; the Envpack adapter follows `Sample.routing_key` (including
manual/consistent-hashing headers); and the new 8-GPU H200 CI stage has the
fork's hardware-label gate. The same commit also keeps `check_run.sh` on the
actual eval score rather than newly added top-level diagnostics.

A later CPU CI run exposed one more stale test-double contract, fixed in
`34335c5e`: `TestRolloutManagerDispose` constructs the actor with
`object.__new__()` to isolate fork-owned cleanup, so it must provide the
`data_source` field that the synced production constructor always initializes.
The repair is test-only and leaves `RolloutManager.dispose()` unchanged. The
other red CPU shard was a Ray worker OOM at the hosted runner's 95% memory
threshold, not an assertion or product-code failure.

That rerun reached 846 passing tests before exposing a second stale assertion,
fixed in `2d529d88`. Dynamic sampling intentionally invokes the all-samples hook
for live diagnostics while refill is in progress and once more for the final
batch. The integration test now selects and validates the unique non-live final
call instead of requiring the hook's total call count to be one. Runtime
behavior is unchanged.

## sglang: v0.5.15 line, patch-by-patch

`thirdparty/sglang` → `38d4bbef5` = `v0.5.15-31` (`94949da73`) + re-applied local patches (verified by CONTENT on the new base, not titles):

1. **mrope text-only gate** (`get_rl_on_policy_target()` disjunct in forward_batch) — absent on v0.5.15, re-applied clean.
2. **cu12 dep flavors** — re-ported onto the v0.5.15 dep set: `flashinfer_python[cu12]==0.6.12`, `sglang-kernel==0.4.4+cu129`, `sgl-deep-gemm==0.1.4+cu129`, `torchao==0.17.0+cu129`, `cuda-python<13`, plain `nvidia-cutlass-dsl` — **all +cu129 wheels verified to exist** on their indexes before committing.
3. **exact multimodal scoring suffix** (impossible-inc/sglang #3) — absent on v0.5.15, re-applied; 4 conflict files were tail-of-parameter-list unions with upstream's new `session_id` plumbing.
4. **qwen-vl pretokenized-IDs fix** (impossible-inc/sglang #4) — absent on v0.5.15, re-applied clean.
5. **pause-aware `flush_cache` restore — DROPPED**: upstream fixed it officially (#31962, `is_fully_idle(ignore_waiting=self._engine_paused)`), semantics fully cover our patch. One less mirror patch to carry.

Patch-shipped unit tests on the new pin: **54 passed + 13 subtests** (scoring suffix ×3 files, pretokenized IDs, io_struct).

Mirror publication status: `sync-v0.5.15-20260724` is already published and
resolves to the exact Miles gitlink `38d4bbef5`. The review PR to
`sglang-miles` is [impossible-inc/sglang#5](https://github.com/impossible-inc/sglang/pull/5).
It is intentionally non-fast-forward because the upstream line was rebased.
After review, landing requires archiving `sglang-miles-v0.5.13-final`, then a
lease-guarded force-advance of the rebased line and a date tag. **This Miles PR
is blocked by #5 and must not merge before that mirror landing completes.**

## Pin / install changes

- **`pins.env`**: `UPSTREAM_*` → v0.5.15 / `cu130-x86_64`; ACTIVE cu12 moves from the retired versioned release to rolling `cu129-x86_64`, following upstream #1784. Server SHA256 comparison confirmed that Apex, FA2, and FA3 are byte-identical to the job-28782 torch 2.11.0+cu129 cache.
- **SGLang source and wheel identity remain separate pins**:
  `MILES_SGLANG_SOURCE_VERSION=v0.5.15` tracks the checked-out source line,
  while `MILES_WHEELS_SGLANG_VERSION=v0.5.15` records the source line against
  which the current rolling assets were validated. The values happen to match
  in this sync, but neither is inferred from the version-less release tag.
- **`TMS_COMMIT` becomes a hand-owned pin at `6d5bce48`** — upstream unpinned torch_memory_saver (#1773); we keep determinism for bare-metal rebuilds. `extract_pins.py` gains `read_preserved()`; the dead Dockerfile pattern is gone.
- **`extract_pins.py` pending comparator**: compares the ACTIVE source line
  directly with `UPSTREAM_SGLANG_IMAGE_TAG`, so version-less wheel tags cannot
  hide a future source lag. Wheel/source label lag remains legal when the
  independent torch-ABI check passes. `--check` is clean at exit 0 with **no
  pending**.
- **`install_env.sh`**: `TMS_CUDA_MAJOR` derived from the env's torch for the newer TMS source build (#1774); **`numpy<2` + `scipy<1.16` cap removed** — upstream dropped the numpy 1.x force at the v0.5.14 bump (#1587); the merged `requirements.txt` (installed directly) brings `transformers<5.13`, blake3/xxhash/zstandard, nvidia-resiliency-ext, psycopg, polars==1.42.1.
- **Deliberately NOT mirrored** (model-family-specific, unused here): Emerging-Optimizers (Muon — import is try/except-guarded), tilelang/tile_kernels/FlashQLA (DSV4/Qwen-GDN), causal-conv1d/mamba-ssm (nemotron-h).

## ⚠️ Attention items

- **The validated environment changed materially**: the clean rebuild installed
  numpy 2.3.5, transformers 5.12.1, TMS `6d5bce48`, and the cu129 SGLang
  kernels. Future deployments must use `install_env.sh`; reusing the old
  v0.5.13-era environment is outside the validated configuration.
- **FT megaseries / multi-LoRA / dashboard are flag-gated** — nothing enables them by default; our recipes are unaffected until opted in.
- **Claude skill rename**: upstream removed `/doc-first-principle` and replaced
  it with the broader `/doc-dev` workflow. The old documentation-first behavior
  is still available as `/doc-dev --docfirst`, but the command name and default
  mode changed.
- **`fully-async` knobs**: `--async-max-concurrent-samples` (upstream, absolute) vs `--fully-async-prefetch-batches` (ours, pipeline depth) both live; explicit absolute setting wins. Unification is a recorded follow-up.
- **Launcher watchdog sentinel retirement** is a recorded follow-up for when the launcher reads upstream's FT event stream instead.
- **Upstream `sglang-miles` tip == our pin base** (v0.5.15-31) at sync time; no deferred sglang delta.

## Divergence from upstream after sync

266 files, +40,770/−267 vs the merged upstream snapshot — `scripts/slurm/` launcher infra, OPD recipes/presentation, `miles_plugins/envpack_adapter/`, `.claude/skills/`, `examples/geo3k_vlm_multi_turn/` + fully-async hardening, in-place `miles/*` modifications listed above, and the `thirdparty/*` submodules. Full stat in [divergence.stat](divergence.stat).

## Validation

Static and fast validation:
- [x] Fast tests on merged tree: `test_data_filter_long_prompt` 4/4, `test_on_policy_distillation` **68/68**, `test_true_on_policy_loss_metrics` 8/8.
- [x] sglang patch unit tests on the new pin: **54 passed + 13 subtests**.
- [x] `extract_pins.py --check` exit 0 with no pending.
- [x] `tests/fast/utils/test_extract_pins.py` 4/4.
- [x] Repository pre-commit hooks pass across the complete PR diff.
- [x] Post-publication test repairs `34335c5e` and `2d529d88` pass all
  repository pre-commit hooks. Their targeted pytest coverage is delegated to
  CPU CI because the local macOS test environment does not contain PyTorch.
- [x] Rolling `cu129-x86_64` Apex/FA2/FA3 SHA256 values match the previously validated job-28782 cache exactly; no replacement binaries were introduced.

Fresh cluster environment and end-to-end validation:
- [x] Clean `install_env.sh` rebuild completed in job 28782; `verify_env.py`
  passed with numpy 2.3.5, transformers 5.12.1, TMS `6d5bce48`, and the
  v0.5.15/cu129 SGLang dependency line.
- [x] Fully async Geo3K baseline run 28789 reached 2034 optimizer steps over
  22h43m and was stopped intentionally. Across matched windows, rollout time
  improved by 9–14%, effective token throughput improved by 15–26%, and reward
  stayed within 0.006 of the pre-sync baseline.
- [x] Multimodal multiturn OPD baseline run 28790 completed 200/200 steps.
  Stable-window sampled reverse-KL reproduced within 0.0004 of the pre-sync
  baseline; the higher trainer time is explained by the bridge path now
  honoring full activation recomputation.

Detailed curves, metric definitions, and validation boundaries are recorded in
[`baseline-validation.html`](baseline-validation.html).

⚠️ **Merge mode**: this PR MUST be merged via **"Create a merge commit"**. Squash or rebase will break future `merge-base` detection (upstream SHAs must survive).
