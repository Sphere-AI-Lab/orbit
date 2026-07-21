# miles upstream PRs report

**Period**: since `1e1679706` (`2026-06-01` — "Support atomic weight update groups (#1264)")
**Synced upstream snapshot**: through `5e97c8650` (`2026-06-29`) "Clamp PPO ratio exponentials to fix NaN logits (#1363)"
**Total commits**: 68 (one PR each — upstream squash-merges)
**Flagged (touch files we've modified)**: 21 PRs (18 here + 3 that are also watchlist hits)
**Watchlist hits (touch pin-source files)**: 7 (`docker/Dockerfile` ×5, `requirements.txt` ×3; one PR hits both)
**Predicted merge conflicts**: **3 textual** — see the conflict section. (A 4th candidate — #1361's `hf_weight_iterator_bridge.py` — was flagged by the per-PR pass, then **investigated and dismissed**: it auto-merges cleanly and is correct. Details at the end of the conflict section.)

> Built from a full per-PR fan-out (`gh pr view` + `git show` for every PR) **plus an
> authoritative real-merge dry run** (`git merge upstream/main` in a throwaway worktree,
> aborted + removed). The conflict set below is what the actual merge will produce, not a
> heuristic. Pre-merge `extract_pins.py --check` is **rc=2** — see the Pin / ABI section;
> this is the known held-divergence, not new.

---

## ⚠️ Predicted merge conflicts (the part that matters)

A real merge of `upstream/main` into `main` produces **exactly 3 textual conflicts**. Per the
skill's HARD RULE #1, the sync will **STOP at the merge and surface these — nothing
auto-resolved.** (A 4th file — #1361's `hf_weight_iterator_bridge.py` — was flagged by the
per-PR pass, then **dismissed on verification**; see the end of this section.)

### 1. `.github/workflows/pr-test.yml` — TEXTUAL (4 hunks)
**Upstream PRs**: #1230, #1304, #1312, #1491 (cumulative CI rewrite).
- **Ours**: GPU stages gated by fork-specific labels — `stage-b-2-gpu-h200` requires `run-ci-fast-gpu`; `stage-c-{8-gpu-h100,4-gpu-h200,2-gpu-h200}` require `run-ci-full-gpu` (this fork has **no self-hosted GPU runners**, so ungated GPU jobs queue forever).
- **Upstream**: rewrote those same GPU-stage `if:` blocks — dropped `!failure()`, added `needs.resolve-ci-image.result == 'success'`, a `stage-a-cpu success || (failure && bypass-fastfail)` group, `--continue-on-error`, and a `schedule`/`nightly` full-suite path.
- **Resolution — keep-both by merging conjuncts**: take upstream's cross-stage gating AND re-apply our trigger-side label gate on each GPU job. **Critical (per #1491):** a bare `schedule` event carries no PR labels, so do **NOT** let `schedule`/`nightly` alone enable GPU stages — `AND` the schedule condition with our runner-label gate, never `OR` it, until GPU runners are attached. The CPU-stage and `schedule:` header hunks from upstream auto-merged; only the GPU `if:` blocks collide.

### 2. `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/broadcast.py` — TEXTUAL (3 regions)
**Upstream PR**: #988 (LoRA on the disaggregated/distributed weight-update path).
- **Ours**: Megatron-Bridge / AutoBridge VLM export (`_bridge_mode`, `_hf_weight_iterator`, rewritten `_is_source`/`_update_weight_implementation`, new `update_weights`).
- **Upstream**: LoRA support (`_init_lora` in `__init__`, `_is_lora_source` property, new `_update_lora_weight_implementation`, two new imports).
- Both edit the **same** insertion points (imports, `__init__`, methods).
- **Resolution — keep-both (independent features)**: union the imports; keep both the bridge-mode `__init__` block and upstream's `_init_lora(...)`; keep our rewritten `_is_source` and add upstream's `_is_lora_source`; keep our `_update_weight_implementation`/`update_weights` and add upstream's `_update_lora_weight_implementation`. They're non-contradictory — our bridge path already raises `NotImplementedError` on bridge+LoRA. (`_init_lora`/`_check_weight_sync_results` arrive via upstream's clean-merging `mixin.py`/`common.py`.)

### 3. `miles/utils/tracking_utils/base.py` — TEXTUAL (3 regions)
**Upstream PR**: #1234 (scope step-key strip to the caller's exact key).
- **Ours**: `import time`; `WandbBackend` monotonic row-step (`_next_row_step`); `init()` returns `bool|None`; `TrackingManager` no-backend drop-and-warn guard + skip-failed-backend.
- **Upstream**: threads `step_key` through — adds `**kwargs` to every `backend.log` override, gives `TrackingManager.log` a `step_key` param, `TensorboardBackend` strips `k != step_key`.
- **Resolution — keep-both (take-upstream-then-reapply-ours)**: in `WandbBackend.log`, take upstream's `**kwargs` on the signature AND keep our monotonic-row-step body; in `TrackingManager.log`, add upstream's `step_key` param, keep our early-return guard, then pass through `backend.log(metrics, step=step, step_key=step_key)`. **Critical**: our `WandbBackend.log` override **must** keep `**kwargs`, or `TrackingManager` passing `step_key=` raises `TypeError`.

### (dismissed) `miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py` — #1361 — NOT a conflict
**Status**: the per-PR fan-out flagged this as a "clean merge that breaks at import." **A
post-merge verification (real merge in a throwaway worktree) refuted it.** No action needed —
the file auto-merges and is correct. **Do NOT apply the import rewrite the per-PR pass
suggested** (`from ..megatron_to_hf import …`): post-merge that points at a deleted location
and would *create* the ImportError.

- **What the per-PR agent saw**: #1361 adds `from .common import get_atomic_update_groups` to `bridge.py`. In our **pre-merge** tree that symbol lives in `megatron_to_hf/__init__.py`, not `update_weight/common.py`, so `from .common` looks broken.
- **What it missed**: **#1281 `1f74c412a` "Use q lora atomic update groups"** — in this *same* sync window (it's in the "clean / no-overlap" bucket) — **relocates** `AtomicUpdateGroup` + `get_atomic_update_groups` from `megatron_to_hf/__init__.py` *into* `update_weight/common.py` and repoints every existing caller (`direct.py`, `mixin.py`) to `from …common`. #1281 and #1361 are one coordinated refactor split across two PRs.
- **Post-merge truth (verified)**: we never touched any of these 4 files, so the merge takes upstream's `common.py` + `megatron_to_hf/__init__.py` wholesale. After merge: `common.py` **defines** `get_atomic_update_groups`/`AtomicUpdateGroup`; the old home no longer does; `bridge.py`'s `from .common import …` resolves; grep finds **zero** surviving imports from the old location. No `ImportError`.
- **Lesson**: per-PR isolation can't see a refactor that spans two PRs in the same merge (mover in one bucket, consumer in another). The cumulative real-merge dry run is authoritative — trust it over single-PR import reasoning.

---

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

## Pin / ABI state (the sglang gate)

**Pre-merge `extract_pins.py --check` is rc=2 (ABI MISMATCH)** — and this is the *current
tree, before any merge*:

```
FATAL: ABI MISMATCH: MILES_WHEELS_TAG=cu129-x86_64 ships torch-2.9.1 wheels,
       but the sglang submodule pins torch==2.11.0.
```

This is the **known, intentional held-divergence** the 2026-05-29 sync recorded: torch-2.11
was not bare-metal-viable on this cu129 / CUDA-12.8 host, so **ACTIVE stays on the
torch-2.9.1 bundle** (`MILES_WHEELS_TAG=cu129-x86_64`, sglang v0.5.10, router 0.3.2) while
the `thirdparty/sglang` source submodule sits ahead at v0.5.12 / torch-2.11. The
install-time fail-closed ABI guard keeps a fresh build safe meanwhile.

**What this sync changes**: upstream's `docker/Dockerfile` now targets **sglang v0.5.13 /
CUDA 13 (cu130) / `cu130-x86_64-v0.5.12` wheels** — i.e. `UPSTREAM_TARGET` jumps *further*
ahead (cu129→cu130, +CUDA-13). The merge does **not** move `thirdparty/sglang` (upstream
doesn't track it) or `pins.env`.

**Consequence for Step 5b/5d**:
- Post-merge `extract_pins.py --check` will still be **rc=2** (the pre-existing ACTIVE-vs-submodule hold persists). Per the skill: **rc=2 ⇒ do NOT `--write`** (it refuses anyway). Surface, don't regenerate.
- The sglang gate is a **DEFER**, not sync-together: ACTIVE is held at cu129/torch-2.9.1 by an explicit host-viability decision, and upstream now wants an even bigger jump (cu130/CUDA-13/v0.5.13). Keep ACTIVE put; record the deferral in the PR body; the install-time guard backstops it.
- See [`upstream-sync-design.md`](../upstream-sync-design.md) for the ACTIVE-vs-UPSTREAM_TARGET model.

---

## Recommendation

Proceed with the **miles-code merge** (it's a large, valuable sync — DeepSeek V4 / V3.2, GLM-5.2, Gemma-4, OPD, LoRA-disaggregate, Qwen-VL fixes, NaN-logit clamp), resolving the **3 textual conflicts** with the user (all keep-both). **Defer the sglang/pin bump** — leave ACTIVE held; do not `--write`; document the deferral. Verify the `extract_pins.py` `WHEELS_TAG_X86` parser change post-merge.
