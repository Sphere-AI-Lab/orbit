## Summary

Sync with upstream `radixark/miles` — **100 upstream commits** merged (merge-base
`7a6cf48e6` → `1e1679706`). Bundled with the **sglang v0.5.10 → v0.5.12** upgrade,
since upstream's Dockerfile bumped the sglang line and the merged miles code targets it.

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

`pins.env` — sglang bundle advanced (ACTIVE == UPSTREAM, `extract_pins.py --check` clean):
- `MILES_WHEELS_TAG`: `cu129-x86_64` → `cu129-x86_64-v0.5.12`
- `TORCH_VERSION`: **2.9.1 → 2.11.0**
- sglang: `v0.5.10` → `v0.5.12`; `MOONCAKE_VERSION`: `0.3.9 → 0.3.10.post2`

`thirdparty/sglang`: `4d795356c` (v0.5.10-40) → `c74db48da` (v0.5.12-24).
`install_env.sh`: no changes (upstream Dockerfile moved only the ARG pins — no new RUN/ENV).

## ⚠️ Attention items

- **torch 2.9.1 → 2.11.0 (heavyweight).** Forces a full rebuild of Megatron-LM / TE / apex /
  flash-attn against the new ABI. The wheels release `cu129-x86_64-v0.5.12` ships matched
  prebuilt wheels (torch 2.11.0, TE 2.10.0, router 0.3.2); `install_env.sh` fails closed on
  any ABI mismatch.
- **Local sglang patch re-applied.** The geo3k VLM mrope fix (`forward_batch`: gate text-only
  path on `rl_on_policy_target`) was NOT carried forward by upstream's v0.5.12 rebase and is
  NOT subsumed (v0.5.12 keys `is_true_on_policy_enabled()` solely off `true_on_policy_contract`).
  Re-applied on top of the v0.5.12-23 target as `c74db48da`. **The `impossible-inc/sglang`
  mirror's `sglang-miles` branch must be force-updated to `c74db48da` BEFORE this PR is
  fetched, or the gitlink won't resolve.**
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

See [divergence.stat](divergence.stat) (72 files, +9293/−39 — mostly our additive
slurm/skills/examples content; substantive code diffs are `metrics.py`, `model.py`,
`sglang_rollout.py`, `broadcast.py`, `pr-test.yml`).

## Test plan

- [ ] `bash scripts/slurm/setup/install_env.sh` in a fresh GPU salloc (full rebuild — torch changed).
- [ ] `python scripts/slurm/setup/verify_env.py` passes.
- [ ] Sanity-launch a geo3k VLM recipe to first eval (exercises the re-applied mrope patch).

⚠️ **Merge mode**: MUST be merged via "Create a merge commit". Squash or rebase breaks
future `merge-base` detection.
