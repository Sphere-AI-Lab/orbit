## Summary

Sync with upstream `radixark/miles` — **68 upstream commits** merged (merge-base `1e1679706`, 2026-06-01 → upstream tip `5e97c8650`, 2026-06-29), **plus the previously-deferred sglang-sync**: `thirdparty/sglang` advances to the **v0.5.13 sglang-miles line**, and the whole stack is **validated end-to-end on cu129 bare-metal** (clean-slate install → geo3k colocate smoke → 500-step 3-node fully-async run with RL metrics matching the pre-sync baseline).

Major upstream content: DeepSeek V4 (+ V3.2/TITO), GLM-5.2 744B-A40B, Gemma-4, on-policy-distillation (OPD) refactor + Megatron teacher mode, LoRA disaggregated training, Qwen3-VL THD mRoPE fix, FlashQLA / Qwen-GDN backend, a Ray-based Megatron→HF converter, ROCm/AMD CI, a docs migration into `docs/`, and a PPO-ratio NaN-logit clamp.

This sync's validation surfaced and fixed **two real post-sync regressions** (the `/begin_weight_update` 404 and a fully-async weight-update deadlock — both traced to sglang-side patches missing from the v0.5.13 line) plus three infra hardening fixes discovered en route.

## Commits on top of the merge

| commit | what |
|---|---|
| `84f65a8d9` | rollout: spawned router/session-server readiness timeout 30→600 s (#1367 fork→spawn cold-imports torch+sglang ~4 min on bare metal) |
| `889415a66` | slurm: fail-fast + retry ray bring-up; terminal MANIFEST on exhaustion |
| `7e52893c6` | slurm: WekaFS-liveness healthcheck tier (`tier-weka`) |
| `c86aee34b` | setup: cu129 bare-metal install for the torch-2.11/sglang-v0.5.13 line (June-2026-06-02 hardening ported + scipy cap, router maturin fix, external-SGLANG_SRC support, engine-path verify checks) |
| `6213f30cb` | slurm: healthcheck probe timeout default 60→300 s (cold `import torch` false-failed healthy nodes) |
| `d477a5c67` | ray: per-user node-local triton JIT cache dirs (shared `/tmp/triton` collides across users on shared nodes) |
| `090c8045c` | **[sglang] sync sglang-miles to the v0.5.13 line; ACTIVE → cu129-x86_64-v0.5.12** |
| `3559df006` | skills: sglang-sync lessons (3 mirror patches, content-verify rule, bundle-may-lag rule, review-PR-first) |

## Conflicts resolved (3, all keep-both)

1. **`.github/workflows/pr-test.yml`** — accepted upstream's CI overhaul (CPU-stage partition matrices, `resolve-ci-image`, `bypass-fastfail`, nightly `schedule` cron, `--match-all-labels`/`--continue-on-error`). Kept upstream's GPU-stage `if:` blocks **verbatim**, with exactly one change: the final `(github.event.pull_request)` conjunct is AND-gated with a **GPU-hardware label** — `run-ci-h200-gpu` for the H200 stages and `run-ci-h100-gpu` for `stage-c-8-gpu-h100`. This fork has no self-hosted GPU runners; without the gate a normal PR would queue those jobs forever. These are workflow-gate labels (like `bypass-fastfail`), **not** domain labels — test selection still works the upstream way.
2. **`miles/backends/megatron_utils/update_weight/update_weight_from_distributed/broadcast.py`** — unioned our Megatron-Bridge/AutoBridge VLM export path with upstream's LoRA-disaggregate support (#988). Both `update_weights` (bridge) and `_update_lora_weight_implementation` (LoRA) kept; bridge+LoRA still fails closed via the existing `NotImplementedError`.
3. **`miles/utils/tracking_utils/base.py`** — kept upstream's `step_key` threading (#1234) AND our no-backend drop-and-warn guard + monotonic W&B row-step.

> #1361 `hf_weight_iterator_bridge.py` was flagged by pre-analysis as a hidden import break, then **verified to be a false positive** (symbol relocated into `common.py` by #1281 in the same merge).

## sglang: v0.5.13 line + why two patches had to be restored

`thirdparty/sglang` now pins `723ed7d19` = **sgl-project/sglang@sglang-miles `v0.5.13-35`** plus three local mirror patches (mirror advance tracked in **impossible-inc/sglang#2**; old tip archived as `sglang-miles-v0.5.12-final` + date tags):

1. **mrope text-only gate re-apply** (`rl_on_policy_target` gating in forward_batch) — our geo3k VLM fix; upstream's rebase does not carry it.
2. **cu12 dep flavors** (mirror-only by design): cuda-python `<13`, flashinfer `[cu12]`, plain cutlass-dsl, and `+cu129` local-version pins for `sglang-kernel`/`sgl-deep-gemm`/`torchao` — **the PyPI default wheels of those three are cu13-linked** and cannot load on this cluster's CUDA-12.8 driver. The cu12 twins come from `docs.sglang.ai/whl/cu129` (new derived `SGL_WHL_INDEX_URL`).
3. **pause-aware `flush_cache` restore** — the v0.5.13 rebase of the sglang-miles patch stack carried "Fix pause-aware weight update deadlocks" (#22754/#22623) **with its key hunk silently dropped** (the rebased twin kept a one-line fragment). Without it, every fully-async weight update deadlocks: `pause_generation → flush_cache` refuses while the async rollout's queued requests sit in `waiting_queue` (400 × ∞ → client timeout). Colocate flows never hit this (nothing queued during an update). Upstream candidate — to be reported to sgl-project.

The 404 this sync originally shipped (`POST /begin_weight_update → 404`, first weight update, job 21656) was the same class: upstream miles #1329's client half arrived via the merge while the sglang server half (#28001/#28710 routes) lives only on the v0.5.13 sglang line. Advancing the submodule fixes it; validated at 200 across every run.

## Pin / install changes

- **ACTIVE bundle: `cu129-x86_64-v0.5.12`** (torch **2.11.0**, router 0.3.2) with the sglang **source** at v0.5.13 — the bundle wheels (FA2/FA3/apex/router/gateway) are **torch-ABI-bound, not sglang-version-bound**; upstream's own Dockerfile pairs its v0.5.13 image with the v0.5.12 bundle. The pin model now encodes this: the install guard **warns** on bundle-sglang lag and stays **fatal on torch mismatch**; `[sglang-sync pending]` compares the version component of wheels tags (upstream docker moved to cu130; bare-metal is driver-bound to cu129). Practical upshot: **miles-wheels releases are now only needed on torch bumps**, not per sglang version.
- **`install_env.sh`** gains the previously-unlanded 2026-06-02 hardening (torch-derived cuDNN with reasserts, TE torch-extension source build, torch `+cu129` constraint through the sglang resolution, router GLIBC source-build fallback) plus: `scipy<1.16` capped with `numpy<2` (sglang's unpinned scipy resolves to 1.18 which requires numpy≥2), the router README inline for maturin ≥1.14, external `SGLANG_SRC` checkout support, and the `+cuNNN` kernel index.
- **`extract_pins.py`** parses upstream's split `WHEELS_TAG_X86/_ARM64` (fixes the parser fatal this merge introduced); cuDNN pin rows removed (derived from torch metadata at install time); `SGL_WHL_INDEX_URL` derived alongside the other index URLs.
- `pins.env` regenerated: torch 2.11.0, mooncake 0.3.11.post1, UPSTREAM_* = v0.5.13 / cu130-x86_64-v0.5.12.

## ⚠️ Attention items

- **Old torch-2.9.1 envs no longer pair with this branch.** The submodule tree is v0.5.13; envs built for the pre-sync line should be rebuilt with `install_env.sh` (validated: fresh env `verify_env.py` 37/37 with no manual intervention). The old pairing remains reachable via the mirror's `sglang-miles-v0.5.12-final` archive branch.
- **Upstream sglang-miles tip moved 2 commits past our validated pin** (`f8cfad35b`: LoRA/DSA fixes) — deliberate: we pin what we validated; a routine future bump catches up.
- **Dead RadixTreeMiddleware guard** in `miles_plugins/envpack_adapter/config.py` (~L177-180): #1290 removed the arg; our `getattr(...)` degrades to `[]` (dead branch, no crash). Cleanup follow-up.
- **OPD estimator value dropped** (#993): upstream removed `on_policy_distillation` from `--advantage-estimator` (now `--use-opd`). No miles-imp recipe used it.
- **WekaFS `/data` read-wedge** seen during validation (slinky-2/50, kernel D-state) is cluster infra, not this PR — ops ticket exists; the new `tier-weka` healthcheck excludes wedged nodes at launch.

## Divergence from upstream after sync

153 files, +31,198/−79 vs the merged upstream snapshot — our local additions (`scripts/slurm/` launcher infra, `examples/vagen/`, `miles_plugins/envpack_adapter/`, recipes, `.claude/skills/`, the `thirdparty/*` submodules) plus in-place `miles/*` modifications (router timeout, triton cache pins) and the setup-script line. Full stat/patch in the local sync-event folder (`scripts/slurm/docs/debug-notes/miles-sync-2026-06-30/`, gitignored).

## Validation (done)

- [x] **Clean-slate `install_env.sh`** → fresh env, `verify_env.py` **37/37**, `INSTALL_EXIT=0`, zero manual steps (slurm job 22160).
- [x] **geo3k multi-turn colocate smoke** → engines up (cuda-graph 68 s ×8 on the `+cu129` kernels), `begin/end_weight_update` **200** ×3 cycles, 2 train steps, MANIFEST SUCCEEDED (job 22113).
- [x] **3-node fully-async geo3k (Qwen3-VL-8B, 24 GPUs)** → **500 steps** at ~40 s/step, weight sync clean under async load throughout, outlived both pre-sync baselines (crashed at 137/375 on the old stack) (job 22338, [wandb](https://wandb.ai/M3TRL/async_envpack/runs/22338)).
- [x] **RL-metrics regression check** vs pre-sync baselines (same/closest recipes, matched step windows, `rollout/raw_reward`): 0-26: 0.587→0.576 · 111-136: 0.730→0.712 · 300-375: 0.820→**0.817** — identical learning slope, offset within run-to-run noise and shrinking with training. `train_rollout_logprob_abs_diff` ≈ 0.006 (tighter than any recorded pre-sync run).

⚠️ **Merge mode**: this PR MUST be merged via **"Create a merge commit"**. Squash or rebase will break future `merge-base` detection (upstream SHAs must survive).

<details><summary><b>Full upstream PR list (68)</b> — watchlist / flagged / clean</summary>

## ⚠️ Watchlist hits (pin sources)

The cumulative `docker/Dockerfile` + `requirements.txt` deltas (merge-base → upstream tip):

**`docker/Dockerfile`**
- `SGLANG_IMAGE_TAG`: `v0.5.12-cu129` → **`v0.5.13`** (#1301 dropped the `-cu129` suffix; #1352 bumped `v0.5.12`→`v0.5.13`).
- `ENABLE_CUDA_13`: `0` → **`1`** (#1301 — x86 default now CUDA 13.0).
- `WHEELS_TAG` (single ARG) **removed**, replaced by `TARGETARCH` + `WHEELS_TAG_X86=cu130-x86_64-v0.5.12` + `WHEELS_TAG_ARM64=cu130-aarch64-v0.5.12` (#1301 moved cu129→cu130; #1312 split per-arch). Net x86 target: **`cu130-x86_64-v0.5.12`**.
- New pinned pip RUN lines (#1181 DeepSeek-V4, #1318 FlashQLA): `tilelang` → `tilelang==0.1.8`; **+** `tile_kernels==1.0.0` (`--no-deps`); **+** `git+.../FlashQLA.git` (unpinned, `--no-build-isolation`); **+** `fast-hadamard-transform@e7706faf…` (`--no-build-isolation`).
- New trailing `ENV PATH="/root/.cargo/bin:${PATH}"` (#1301 — cu130 base ships rust off-PATH).
- Removed a commented-out Megatron-Bridge line (no action).

**`requirements.txt`** (#1339, #1371, #1312)
- `+ backports.strenum; python_version < "3.11"` (ROCm py3.10).
- `hypothesis` → `hypothesis>=5.40`.
- `torchft-nightly==2026.4.3` marker tightened to `… and platform_machine == "x86_64"` (skipped on arm64).
- `+ tqdm`.

> ⚠️ **`extract_pins.py` parser risk (#1312)**: the single `WHEELS_TAG=` ARG the extractor
> greps for is **gone**, replaced by `WHEELS_TAG_X86`/`WHEELS_TAG_ARM64`. After the merge,
> verify `extract_pins.py` still resolves `UPSTREAM_WHEELS_TAG` from the new per-arch keys —
> if it silently misses them, `UPSTREAM_*` won't advance. This is independent of the rc=2
> ABI hold below.

Per-PR watchlist detail:

| PR | Title | Pin impact |
|----|-------|-----------|
| #1181 | deepseek v4 model support | `tilelang==0.1.8`, `+tile_kernels==1.0.0`, `+fast-hadamard-transform@e7706faf…` (Dockerfile RUN lines). No ARG/version-ARG change. Also edits `arguments.py` (auto-merges). |
| #1301 | Default x86 image to CUDA 13 (cu130) | `SGLANG_IMAGE_TAG v0.5.12-cu129→v0.5.12`; `ENABLE_CUDA_13 0→1`; `WHEELS_TAG cu129…→cu130-x86_64-v0.5.12`; `+ENV PATH cargo`. |
| #1318 | FlashQLA backend for Qwen GDN | `+RUN pip install git+…/FlashQLA.git` (unpinned, `--no-build-isolation`). Adds `--linear-attention-backend` arg (auto-merges in `arguments.py`). |
| #1312 | doc-driven CI + docker flow refactor | `WHEELS_TAG`→`WHEELS_TAG_X86`/`WHEELS_TAG_ARM64` (x86 net unchanged); `requirements.txt` torchft→x86-only. Also edits `pr-test.yml` (part of conflict #1). |
| #1339 | AMD CI: Python 3.10 on ROCm base | `requirements.txt`: `+backports.strenum; py<3.11`, `hypothesis>=5.40`. `docker/Dockerfile` (CUDA) untouched. |
| #1352 | Bump sglang to v0.5.13 | `SGLANG_IMAGE_TAG v0.5.12→v0.5.13` (only ARG change). |
| #1371 | Converter: Megatron→HF on Ray | `requirements.txt`: `+tqdm`. `docker/Dockerfile` untouched. |

---

## PRs that touch files we've modified (flagged)

All overlaps below **auto-merge** except the 3 textual + 1 hidden-semantic listed in the
conflict section. The `arguments.py` hotspot (touched by 11 upstream PRs) auto-merges in
every case — upstream's additions land in different argument groups / validation regions
than ours.

| PR | Title | Overlap files | Merge note |
|----|-------|--------------|-----------|
| #1230 | partitioned stage-a-cpu CI | `pr-test.yml` | **CONFLICT #1** (GPU-stage `if:`). |
| #1304 | bypass-fastfail label | `pr-test.yml` | **CONFLICT #1**. |
| #1312 | doc-driven CI + docker refactor | `pr-test.yml` (+watchlist) | **CONFLICT #1** (cosmetic part); pins above. |
| #1491 | nightly full-suite CI | `pr-test.yml` | **CONFLICT #1**; ⚠️ don't let schedule/nightly enable GPU stages. |
| #988 | lora disaggregate training | `broadcast.py`, `arguments.py` | **CONFLICT #2** (broadcast.py keep-both); arguments.py auto-merges. |
| #1234 | scope step-key strip | `tracking_utils/base.py` | **CONFLICT #3** (keep-both; retain `**kwargs`). |
| #1361 | pair q_a_proj/kv_a_proj | `hf_weight_iterator_bridge.py` | Auto-merges; symbol relocated by #1281 (same merge) — **no fix needed** (flagged then dismissed). |
| #993 | OPD [1/N] decouple + Megatron teacher | `model.py`, `arguments.py` | Auto-merges. ⚠️ **drops `on_policy_distillation` as `--advantage-estimator` value** — migrate any recipe using it to `--use-opd …`. |
| #994 | OPD [2/N] move to miles/rollout | `sglang_rollout.py`, `arguments.py` | Auto-merges (disjoint regions). |
| #1290 | remove router middleware_hub | `generate_endpoint_utils.py`, `sglang_rollout.py`, `arguments.py` | Auto-merges; deletes lines we reference — no dangling refs (our other refs deleted too). Grep `RadixTreeMiddleware\|middleware_hub` post-merge to confirm. |
| #1280 | Kimi-K2.5 CI | `hf_weight_iterator_bridge.py`, `arguments.py` | Auto-merges (disjoint). |
| #1197 | Megatron Muon optimizer | `model.py` | Auto-merges (different functions). |
| #1232 | Gemma-4 model support | `model_provider.py`, `hf_weight_iterator_bridge.py`, `arguments.py` | Auto-merges (complementary regions). |
| #1272 | Qwen3-VL THD packed mRoPE | `model_provider.py` | Auto-merges (non-adjacent to our freeze-vision block). |
| #1329 | refactor weight-update post-process | `arguments.py`, `train.py`, `train_async.py` | Auto-merges (ours = train-status/heartbeat). |
| #1058 | configurable Miles DSA top-k backend | `arguments.py` | Auto-merges (adds `--miles-dsa-topk-backend`). |
| #1353 | step0 eval for train_async.py | `train_async.py` | Auto-merges (3-line block). |
| #1503 | remove trailing comma in help | `arguments.py` | Auto-merges; fixes a bug our HEAD still carries. |
| #1370 | ban huggingface-cli in pre-commit | `.pre-commit-config.yaml` | Auto-merges (our isort `rev` bump untouched). |

---

## Other PRs (no overlap detected) — 43

Sorted by PR number ascending.

| PR | Theme | One-line |
|----|-------|----------|
| #838 | chore | Support `jdopensource/JoyAI-LLM-Flash` (DeepSeek-V3-style small model). |
| #1065 | deepseek-v4 | Add DeepSeek V4 TITO model support. |
| #1131 | deepseek-v32 | DeepSeek V3.2 TITO tokenizer family + thinking_mode alias. |
| #1233 | rocm/amd | Honor `HIP_VISIBLE_DEVICES` in Ray GPU helpers / NOSET list. |
| #1258 | chore | Remove the `eval/terminal_bench` example. |
| #1269 | ci | Fix + re-enable `test_run_megatron_worker_main`. |
| #1271 | ci | Re-enable `test_qwen3_4B_ckpt.py`. |
| #1276 | ci | Make sglang EP explicit in Megatron DeepEP e2e; broadcast test pp=3. |
| #1281 | lora | q-LoRA atomic update groups (drop conversion cache). |
| #1282 | ci | Disable flaky `TestStopEnginesRealKill`. |
| #1283 | bugfix | Force `enable_metrics=True` on every SGLang engine (Prometheus). |
| #1291 | docs | Rewrite agentic chat-template guide for session-server TITO. |
| #1294 | rocm/amd | Merge MI300/MI350-5 AMD Dockerfiles. |
| #1309 | deepseek-v4 | Per-rank `SGLANG_DG_CACHE_DIR` (DeepGEMM cache race). |
| #1310 | deepseek-v4 | DSV4 disaggregated launcher. |
| #1311 | chore | Drop deepseek-v4/v32/glm model-specific docker tags. |
| #1315 | docs | Fix deepseek-v4 launcher env-var / path-default guidance. |
| #1316 | deepseek-v4 | Fix deepseek-v4 Blackwell path. |
| #1321 | docs | Migrate documentation into `docs/` with history. |
| #1323 | rollout | Stop folding agentic turns at first non-COMPLETED turn. |
| #1330 | deepseek-v32 | Set `override_hf_native=True` for deepseek-v32. |
| #1337 | deepseek-v4 | Fix DeepSeek V4 blockwise FP8 on Blackwell. |
| #1342 | deepseek-v4 | Optimize DeepSeek V4 SGLang config. |
| #1346 | ci | Fix kimi-2.5 CI OOM. |
| #1347 | deepseek-v32 | Use SGLang router in DeepSeek V3.2 script. |
| #1349 | sglang/perf | Migrate to renamed `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK` env var. |
| #1363 | bugfix | Clamp PPO ratio exponentials (fix NaN logits). |
| #1367 | rollout | Spawn (not fork) router/session-server subprocesses. |
| #1369 | qwen | Tune qwen3.5-35B-A3B MTP cp2/ep8 perf args. |
| #1374 | bugfix | `NVSHMEM_DISABLE_NCCL` to avoid CUDA graph hang. |
| #1375 | sglang/perf | `flashinfer_trtllm` for mxfp8 GEMM (sm103). |
| #1376 | glm | Support GLM-5.2 744B-A40B. |
| #1379 | rocm/amd | ROCm CI for MI350X e2e (2/N). |
| #1382 | deepseek-v4 | Dynamically size DeepSeek-V4 rope `freqs_cis` (high CP). |
| #1470 | bugfix | Don't force `use_distributed_optimizer=True` for sgd. |
| #1475 | rocm/amd | Point ROCm Megatron-Bridge at the `bridge` branch. |
| #1477 | glm | Fix GLM 5 & 5.2 on Blackwell. |
| #1478 | docs | Minor doc fixes. |
| #1479 | docs | Drop deprecated `--use-miles-router` from recipe docs. |
| #1480 | chore | Remove `--use-miles-router` flag from launch scripts. |
| #1484 | ci | Pre-commit: route `load_hf_config` through miles helper. |
| #1505 | deepseek-v4 | Keep `linear_weights_proj` bf16 under DSV4 blockwise FP8. |
| #1509 | ci | Move `tests/ci` harness unit tests into `tests/ci/test/`. |

---


</details>
