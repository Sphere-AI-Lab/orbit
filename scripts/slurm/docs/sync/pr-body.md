## Summary

Sync with upstream `radixark/miles` — **100 upstream commits** merged (merge-base
`7a6cf48e6` → `1e1679706`). Brings in the **sglang v0.5.10 → v0.5.12** source bump
(upstream's Dockerfile moved the sglang line and the merged miles code targets it).

> **Scope note:** upstream pairs v0.5.12 with torch 2.11, which is **not bare-metal-viable**
> on this cu129 / CUDA-12.8 host (its `sgl-kernel` / `deep_gemm` are published cu13-only). So
> this PR moves only the **sglang source** (submodule) and keeps the **torch-2.9.1 wheel
> stack** — the v0.5.12 source runs on the existing `miles` env with a version-check skip
> (see Attention items). Install/pin scripts stay at `main`; the launcher fix and the
> sglang-mirror tooling ship as separate PRs.

## Upstream changes

~99 PRs. Highlights (full list in [prs.md](prs.md)):

- **Rollout refactor** (#899–#947, #1070–#1084): `miles/ray/rollout.py` decomposed into
  the `miles/ray/rollout/` package + async rollout management + extensive tests.
- **sglang v0.5.12** (#1164): Dockerfile pins → `SGLANG_IMAGE_TAG=v0.5.12-cu129`,
  `WHEELS_TAG=cu129-x86_64-v0.5.12`.
- **Loss refactor** (#753/#1121/#1132), **model support** (DeepSeek v3.2/v4, Kimi 2.5,
  GLM-5, qwen3.5, npu qwen3-4b), **indexer/replay** (#1248–#1257), **atomic weight update
  groups** (#1264), **CI restructure** (#1149/#1227).

## Pin / install changes

**`pins.env` / `install_env.sh` / `extract_pins.py` / `verify_env.py`: untouched — left exactly
as `main` (the #12 pin bundle).** This PR does NOT advance the wheel bundle: `pins.env` still
pins `TORCH_VERSION=2.9.1` / `MILES_WHEELS_TAG=cu129-x86_64` (sglang v0.5.10), which correctly
describes the env we run. torch 2.11 is not bare-metal-viable here, so we keep the torch-2.9.1
stack; the torch-2.11 install hardening is parked locally for a future cu13/docker path.

Because the submodule moves *ahead* of that wheel bundle, the pin-model's guards trip **by
design**: `extract_pins.py --check` and `install_env.sh` fail closed with an ABI mismatch
(`wheels torch 2.9.1 != submodule pyproject torch 2.11`). That is expected for this sync — the
source is deliberately ahead of the wheels. **Don't** `extract_pins.py --write` and **don't**
fresh-install; use the prebuilt `miles` env. (The guards are dev tooling, not CI-gated, so this
does not fail the PR.) Full rationale: `docs/launcher.md` → "Running the v0.5.12 sync".

**`thirdparty/sglang` submodule** is the only thing that moves: `4d795356c` (v0.5.10-40) →
`c74db48da` (v0.5.12-24, + the re-applied mrope patch).

## ⚠️ Attention items

- **Runs on the existing torch-2.9.1 env + a version-check skip.** The v0.5.12 engine asserts
  `sglang-kernel >= 0.4.2.post2` (and `flashinfer >= 0.6.11.post1` for the flashinfer backend)
  at launch; the `miles` env has 0.4.1 / 0.6.7.post2 — below those floors but functionally fine
  (the assert is a guard, not a real ABI gate). Set **`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true`**
  when submitting; it chains via `--export=ALL` to each engine subprocess. Full rationale +
  rebuild caveat: `docs/launcher.md` → "Running the v0.5.12 sync".
- **Local sglang patch re-applied.** The geo3k VLM mrope fix (`forward_batch`: gate text-only
  path on `rl_on_policy_target`) was NOT carried forward by upstream's v0.5.12 rebase and is
  NOT subsumed (v0.5.12 keys `is_true_on_policy_enabled()` solely off `true_on_policy_contract`).
  Re-applied on top of the v0.5.12-23 target as `c74db48da`. The mirror advance is **already
  done** (separate PR, merged): `impossible-inc/sglang@sglang-miles` was advanced via the
  force-push + archive flow — the old v0.5.10 tip is archived as tag `sglang-miles-v0.5.10-20260604`
  (`4d795356c`), and the gitlink `c74db48da` we ship here is anchored by tag
  `sglang-miles-v0.5.12-20260604`, so it stays reachable independent of the branch tip.
- **Merge conflicts resolved** (5): `.gitignore` (union), `model.py` (walrus + our
  `mm_token_type_ids` drop), `sglang_rollout.py` (our richer all-samples soft-call),
  `pr-test.yml` (took upstream's `stage-b-2-gpu-h200`, kept our `run-ci-fast-gpu` gate),
  `miles/ray/rollout.py` modify/delete (our `_compute_zero_std_metrics` tweak ported to the
  new `miles/ray/rollout/metrics.py`).
- **CI follow-up:** upstream's new `stage-c-{8gpu-h100,4gpu-h200,2gpu-h200}` jobs are
  **ungated** (`if: pull_request`) and will queue on this runner-less fork. Only
  `stage-b-2-gpu-h200` was gated behind `run-ci-fast-gpu` in this PR; the stage-c jobs were
  left upstream-exact. Gate or attach runners before relying on PR CI.

## Divergence from upstream after sync

See [divergence.stat](divergence.stat) — snapshot at merge time (72 files, +9293/−39, mostly
additive slurm/skills/examples content; substantive code diffs are `metrics.py`, `model.py`,
`sglang_rollout.py`, `broadcast.py`, `pr-test.yml`). The `setup/*` scripts in the snapshot are
the `main` (#12) pin-model, unchanged by this PR.

## Validation (done — all passing)

Validated the v0.5.12 source on the existing **torch-2.9.1 `miles` env** with
`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true` (the supported runtime):

- **Install / env.** The supported runtime is the existing torch-2.9.1 `miles` env (built
  previously from `main`'s `install_env.sh`); on it `verify_env.py --imports-only` is green —
  note it only import-checks (the engine version-assert fires at *launch*), so the runtime
  smokes below are the real gate. A *fresh* `install_env.sh` is not part of this path: against
  the v0.5.12 submodule it fail-closes on the ABI guard (by design — see "Pin / install
  changes"). A separate torch-2.11 fresh build was tried and *completes* (`verify_env` green),
  but the sglang **engine does not run** bare-metal (cu13 kernels) — which is why we keep
  torch 2.9.1.
- **geo3k VLM multi-turn (Qwen3-VL-2B), smoke** — engine init incl. cuda-graph capture →
  rollout → train step ✅ (grad_norm 0.50).
- **VAGEN FrozenLake (Qwen3-VL-2B)** — smoke ✅; 30-step e2e ✅ stable (no OOM/crash).
- **VAGEN Sokoban (Qwen2.5-VL-3B), 160-step e2e** — ✅ full GRPO loop, and the model **learned**
  (`eval/sokoban_val` 0.478 → 0.699, `traj_success` 0.156 → 0.281). Confirms a second model
  arch + a second env.

No outstanding test TODOs.

⚠️ **Merge mode**: MUST be merged via "Create a merge commit". Squash or rebase breaks
future `merge-base` detection.
