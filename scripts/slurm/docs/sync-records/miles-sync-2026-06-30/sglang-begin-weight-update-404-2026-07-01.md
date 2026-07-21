# Post-sync SGLang `/begin_weight_update` 404 — 2026-07-01 (job 21656)

**Status:** confirmed failure; **root cause fully identified** (verified against upstream git
history). A genuine **sync coordination gap** on `sync-upstream-20260630`, distinct from the
WekaFS wedge (that was job 21623). Fix is scoped but not yet applied.

## What happened
geo3k run **21656** (nodes slinky-19,20,36) cleared healthcheck + tier-nccl + ray, **loaded
weights, and brought all SGLang engines up healthy** (`/health` 200 continuously 02:16–02:30).
At the **first weight update** it failed:

- `init_weights_update_group` → 200, `pause_generation` → 200, `flush_cache` → 200, then
  **`POST /begin_weight_update` → 404 Not Found** (`run.log:2472`, repeated 15× across engines).
- Traceback: `update_weight_from_tensor.py:196 → common.py:388 ray.get([engine.begin_weight_update.remote()…]) → sglang_engine.py:560 _make_request("begin_weight_update", {}) → requests.exceptions.HTTPError: 404 … http://…:15015/begin_weight_update` (`run.log:2528-2545`).
- MANIFEST: `state=FAILED job_rc=1`.

Not the WekaFS issue — `grep -c 'wekafs|commit_blocking|O_DIRECT' run.log` = 0, no
`weka_probe.*.log`. The init was merely *slow*, then completed.

## Root cause (verified)
This is a **coordinated client+server protocol change** whose two halves landed on opposite
sides of the sync. It is **not** an sgl-project *mainline* API change, and it is **not** fixed by
a routine submodule bump.

**miles side (arrived via the sync):** radixark/miles PR **#1329** — "refactor weight update
post-process and weight-checker" (`dc2ced1d5`, Yueming Yuan, merged 2026-06-25 00:27:06Z) — added
a **weight-update session**: the client now brackets the per-tensor update with
`begin_weight_update` (restore/unpack packed weights so they can be loaded) and
`end_weight_update` (post-load + re-quant on the full model). Confirmed *new* to this sync:
`begin_weight_update` is present in `upstream/main` but **absent in our pre-sync side**
(`82d59e680^1`). Call sites: `sglang_engine.py:558-564`, `common.py:386-393`,
`update_weight_from_tensor.py:196/240`, `update_weight_from_distributed/mixin.py:273/278`.

**sglang side (the matching routes — NOT in our fork):** the server half is two `[sglang-miles]`
PRs on sgl-project/sglang, merged in the same ~30-second train as #1329:
- **#28710** (merged 2026-06-19, `d5243e82f`) — scheduler/model-runner **session machinery**:
  `begin_weight_update()` opens a session recording the runner selector; `update_weights_from_*`
  now *assert* an open session; `end_weight_update(run_post_load=…)` finalizes it.
- **#28001** (merged 2026-06-25 00:26:57Z, `64b0059aa`) — the **HTTP routes**:
  `@app.post("/begin_weight_update")` + `@app.post("/end_weight_update")` in `http_server.py`,
  `BeginWeightUpdateReqInput` in `io_struct.py`, and the `tokenizer_manager` methods.
- (sibling) **#28082** (merged 2026-06-25 00:26:37Z) — the `allow_quant_error` weight-checker
  half of #1329; this is the one #1329's body pins as `ci-sglang-pr: #28082`.
- (later) **#29675** (merged 2026-06-29) — `[sglang-miles]` cherry-pick "pause-aware post-process
  weight locking" (follow-up hardening).

**Why the running server 404s:** the miles-imp `thirdparty/sglang` submodule tracks
`impossible-inc/sglang` branch **`sglang-miles`**, pinned at `c74db48da` =
`sglang-miles-v0.5.12-20260604` (2026-06-04). None of #28710 / #28001 / #28082 / #29675 have been
cherry-picked into that fork — its newest RL cherry-picks are in the **#26xxx** range (e.g.
#26736, #26430, #26287), all older than the late-June route PRs. Verified: full-tree
`git grep begin_weight_update` = **0** on `sglang-miles`, `sync-v0.5.12-20260603`, and
`sglang-miles-v0.5.10-final`. The routes are also **not** on sgl-project *main*
(`gh search code` = 0) because they are `[sglang-miles]`-tagged branch work, not merged to main.

**Corrections to the earlier version of this note:** the routes were *never a miles patch dropped
by the sync*, and this is *not* fixed by "bump the pin to the `sglang-miles` tip" — the tip
(13bcf73c4, only 6 commits ahead of the pin) still lacks them (grep = 0). The server half simply
was never brought into the impossible-inc fork.

## Repro / verification
```bash
# client half is new to this sync:
git -C miles-imp grep -n "def begin_weight_update" upstream/main -- miles/backends/sglang_utils/sglang_engine.py   # present
git -C miles-imp grep -n "def begin_weight_update" 82d59e680^1  -- miles/backends/sglang_utils/sglang_engine.py   # absent (pre-sync)

# server half is absent from every fork branch:
for b in sglang-miles sync-v0.5.12-20260603 sglang-miles-v0.5.10-final; do
  git -C miles-imp/thirdparty/sglang grep -c "begin_weight_update" origin/$b || echo "$b: 0"; done

# the upstream sglang PRs that add the routes:
gh pr view 28001 -R sgl-project/sglang   # http routes + tokenizer_manager
gh pr view 28710 -R sgl-project/sglang   # scheduler/runner session machinery
```
(A full end-to-end repro needs a healthy `/data` so the run reaches the first weight update.)

## Fix — run `/sglang-sync` (advance the submodule; do NOT hand cherry-pick)
The routes already live on the miles sglang line — **`sgl-project/sglang@sglang-miles`** (the
official repo hosts the branch; `impossible-inc/sglang` is only our mirror, advanced by
`/sglang-sync`). #28001 (routes) + #28710 (session machinery) landed there. So the fix is the
**deferred `/sglang-sync`** (`.claude/skills/sglang-sync/SKILL.md`): advance `thirdparty/sglang`
to the sgl-project@sglang-miles tip, re-apply our one local patch (`[sglang-miles] forward_batch
mrope gate`), bump the gitlink, realign the ACTIVE pin bundle.

Verified target state (2026-07-01): PIN `sglang-miles-v0.5.12-20260604` → TARGET
`v0.5.13-35-g6ee17b436` (route-adding commit `64b0059aa`/#28001 is on a **v0.5.13** base; the
v0.5.12 snapshots have **zero** route hits, so there is no v0.5.12 shortcut). It's a
v0.5.12→v0.5.13 version bump but **torch is unchanged (2.11.0)** — no ABI jump.

**No new wheels bundle is needed.** upstream `radixark/miles@main` `docker/Dockerfile` pairs
`SGLANG_IMAGE_TAG=v0.5.13` (`FROM lmsysorg/sglang:v0.5.13`) with `WHEELS_TAG_X86=cu130-x86_64-v0.5.12`
— i.e. it runs sglang **v0.5.13 on the v0.5.12 wheels bundle**. The bundle wheels
(flash-attn/apex/TE/router/gateway) are **torch-ABI-bound, not sglang-version-bound**, and torch is
2.11.0 for both; the sglang-version-specific kernels (`sglang-kernel`, `flashinfer`) come from the
base image / PyPI, not the bundle. So the existing `cu129-x86_64-v0.5.12` bundle is the correct
overlay for v0.5.13. (Earlier claim that a `cu129-x86_64-v0.5.13` release must be published/built
was wrong.)

**Actual blockers are our tooling being stricter than upstream:**
1. **`install_env.sh:248` guard** fail-closes when submodule sglang (`v0.5.13`) ≠ bundle sglang
   (`v0.5.12`) — but upstream deliberately makes that pairing. Fix: key the bundle on **torch**, not
   the sglang tag (allow wheels-sglang to lag submodule-sglang).
2. **`extract_pins.py` broke on the same Dockerfile change** —
   `FATAL: pattern '^ARG\s+WHEELS_TAG=' matched nothing`. Upstream split `WHEELS_TAG` →
   `WHEELS_TAG_X86`/`_ARM64` and added `SGLANG_IMAGE_TAG`. The extractor needs the new layout AND to
   treat a lagging wheels-sglang as valid, not drift.
3. **cu129 vs cu130 — RESOLVED: stay cu129.** The cluster driver is `570.195.03` = **CUDA 12.8**, so
   cu130 (CUDA 13) is impossible without a root-level driver upgrade (no major forward-compat;
   `install_env.sh:173` refuses it). conda can't help — the driver is the host kernel module.
   **cu129 is fully supported for v0.5.13:** upstream ships `lmsysorg/sglang:v0.5.13-cu129`
   (and `v0.5.13.post1-cu129`); `torch-2.11.0+cu129` exists; `flashinfer-python 0.6.12` has a
   first-class `[cu12]` extra (`provides_extra: [cu12, cu13, nvep]`); `flashinfer-cubin 0.6.12` is
   arch-agnostic; `sglang-kernel 0.4.3` has no cuda-major lock (same shape as the 0.4.1 running today
   on the 12.8 driver). Our `TORCH_INDEX_URL`/`FLASHINFER_INDEX_URL`/wheels are already cu129.
   **Only adaptation:** v0.5.13's pyproject hardcodes `flashinfer_python[cu13]` /
   `nvidia-cutlass-dsl[cu13]` — override to `[cu12]` at install (flashinfer's `[cu12]` extra pulls
   plain cutlass). Not an incompatibility. (Verify `sglang-kernel 0.4.3` imports on cu129 at build —
   metadata/precedent strong but not import-proven at 0.4.3.)
   Note: the installed `miles` env is torch 2.9.1 (stale); the canonical path builds a **fresh** env
   (keep `miles` intact) at torch 2.11.0 + cu129-v0.5.12 bundle + v0.5.13 sglang.

Once the guard + extractor are fixed and the cu129/cu130 call is made, advance the submodule to
sgl-project@sglang-miles (re-applying the mrope patch), pin the bundle to `cu129-x86_64-v0.5.12`
(torch 2.11.0), rebuild the env, then re-validate the sequence `init_weights_update_group →
pause_generation → flush_cache → begin_weight_update → update_weights_from_tensor* →
end_weight_update → continue_generation` (all 200).

Alternative (not recommended): revert #1329 on the miles side to the pre-session flow — diverges
from upstream and loses the packed/quant post-process the refactor added.

## Context
Caught validating `sync-upstream-20260630`. Everything up to the first weight update works
(router 600 s fix, ray bring-up, tier-nccl, weight load, engine health). This 404 is the first
*application-level* regression the validation surfaced — a real submodule-sync coordination gap:
the miles client was synced past the vendored sglang fork. It blocks training on the synced branch
independent of the WekaFS storage issue.
