## Summary

Sync with upstream `radixark/miles` — **274 upstream commits** merged (merge-base `c94c2fa9`, 2026-07-23 → upstream tip `fc04f666`, 2026-08-18), **plus the combined sglang-sync**: `thirdparty/sglang` advances to the **v0.5.16 sglang-miles line** (`36982fef0` = `sgl-project/sglang@sglang-miles` `cb05a44f3` + 7 re-applied local patches), **plus the Megatron-Bridge advance** to `7f0fb345` (TE 2.17 grouped-linear contract; clean fast-forward of 21 commits). Same `cu129-x86_64` torch-2.11 wheels bundle — no torch ABI jump.

Major upstream content: fully-async rewrite onto the class-based rollout API (#1716/#1717/#2522, sample-granularity submission #1673, unified async data buffer #2030, fully-async eval #1740), session server v2 (trajectory trees #2126/#2128, server-side sample assembly #1758/#1759), TE 2.17 + cuDNN 9.22 + FA3 3.0.0 docker chain, sglang v0.5.16 (#1795), OPD per-position sparse top-k scoring + multi-teacher routing (#1298), NVFP4/Nemotron/Inkling model enablement, variable global batch size (#1927), training log-prob reuse (#2219), PPO fixes (CP terminal rewards, GAE over trainable tokens, critic-save derivation), dashboard series, CI restructure (#2221/#2529/#2530/#2548), scripts/models converted to python model_args, release workflow.

Commit shape: upstream SHAs preserved under merge commit `4c93a4b4`; fork adjustments in `2a080f79`; the sync record in `720d1a01`. Five further commits then landed on this still-unpublished branch (`14d126a8`, `7c4e8040`, `6bb4af5b`, `a91c2a29`, `b4b13326`), so the sync skill's single-commit shape (HARD RULE 4) does **not** hold here: the review-repair protocol was applied deliberately *before* publication rather than after it. Each follow-up is a focused change carrying its own validation; no history was rewritten, and this body plus `divergence.stat` are kept current so every follow-up is represented.

## Fork-side changes on top of the merge (`2a080f79` + follow-ups)

| area | what |
|---|---|
| pins/install | TE 2.10→2.17, TMS re-extracted @`74d68c5e`, FA3-interface pin now hand-owned, ACTIVE sglang v0.5.16 (same wheels bundle); install_env.sh mirrors mathdx + pydantic/nvdlfw-inspect |
| fully-async port | fail-closed staleness buffer (`examples/fully_async/fail_closed_data_buffer.py`), 18 recipes → `--fully-async`, OPD scoring-session close re-wired into `dispose_rollout_function` |
| import tax | `megatron_utils/__init__` side effects → `runtime_hooks.install_runtime_hooks()` (from `initialize.init()`): `import miles.utils.logging_utils` 75s → 0.6s per process on WekaFS (upstream candidate) |
| scripts/models | five .sh shims delegating to `load_model_args` keep 26 recipes working; **new recipes use `load_model_args` directly** |
| review repairs (post-publication) | PR-review findings, all verified against the tree before fixing: `--fully-async-prefetch-batches` was inert after the upstream rewrite (nothing read it) so migrated recipes ran at depth 1 — now derives `--async-max-concurrent-samples`, with the prefetch-vs-staleness sanity warning restored; `FailClosedDataBuffer` was never selected by any recipe, leaving upstream's fail-open staleness admission live — now the `--fully-async` default (no-op unless `--max-weight-staleness` is set, explicit `--custom-async-data-buffer-path` still wins); `--opd-topk-per-position` was a silent no-op — now rejected in validation with matching help text, and the orphaned comment removed; `scripts/models/qwen3-30B-A3B.sh` now also forwards `MODEL_ARGS_NUM_LAYERS` (its `.py` reads it); `--fully-async-max-completed-queue-groups` warns that it is superseded by `--async-data-buffer-capacity-factor` |
| follow-ups (pre-publication) | `14d126a8` `verify_env.py` sglang deep-import probe repointed for the v0.5.16 module layout (found by the fresh-env validation); `7c4e8040` pre-merge regression-gate wrappers under `scripts/experiments/baseline/`; `6bb4af5b` `rollout_train_kl/{k1,k2,k3}` train-vs-rollout mismatch estimators (no reference model needed) plus the upstream-R3-vs-fork-R3 analysis appended to `resolution-notes.md`; `a91c2a29` OPD baseline teacher repointed at the 2026-08-10 shared-model reorg; `b4b13326` model shims forward unexported recipe knobs to the python child |
| merge completions | `actor.py` missing import; `actor_factory` assert retired (#1785); geo3k dir-rename sweep (61 refs); examples README/doc mirror; PYTHONUNBUFFERED on two fork launchers; sync-records excluded from hygiene scan; fork test doubles/suites updated to synced contracts |

## Conflicts resolved (32 files — all reviewed per-file)

Full log: [resolution-notes.md](resolution-notes.md). Highlights: fully-async modify/delete pair resolved by porting the fork deltas onto the class API; OPD kept the fork multimodal/persistent-transport scoring pipeline and wired upstream's multi-teacher routing into it (no-op without `--opd-teacher-urls`); `losses.py` took upstream's detach discipline + logprob-reuse while keeping fork mismatch diagnostics and DAgger; `log_utils` took upstream's reduction core, kept the fork `interaction/*` namespace; CI hardware-label gates re-ANDed onto upstream's restructured stages; `.gitmodules` kept fork submodules, dropped swe-agent.

## sglang: v0.5.16 line, patch-by-patch

`thirdparty/sglang` → `36982fef0` = `sgl-project/sglang@sglang-miles` `cb05a44f3` (rebased v0.5.16) + re-applied local patches (verified by content):

1. mrope text-only gate — re-applied (import drift only: `true_on_policy` became a package).
2. cu12 dep flavors — re-ported onto the v0.5.16 dep set: `flashinfer_python[cu12]==0.6.14`, `nvidia-cutlass-dsl==4.6.2` (upstream #2600 sm_103 FA4-hang fix), `sglang-kernel==0.4.5+cu129`, `sgl-deep-gemm==0.1.4.post1+cu129` — **all +cu129 wheels verified to exist**; the old `flash-attn-4==4.0.0b15` pin retired in favor of the base `>=4.0.0b18` (universal wheel).
3. exact multimodal scoring suffix (#3) — re-applied (`allow_auto_truncate` moved to an instance attribute; test double updated in the same pick).
4. qwen-vl pretokenized IDs (#4) — re-applied clean.
5. #3's CI registration fix — re-applied clean.
6. token_ids_logprobs host-list guard (#7) — re-applied clean.
7. local_checkpoint reseed — re-applied clean.

Patch battery on the rebased tip: **52 passed + 13 subtests**.

## ⚠️ Attention items

- **OPD top-k student-side strategies need `MILES_USE_LEGACY_ROLLOUT_V1=1`** until `opd_student_top_logprobs` is ported to the class-based rollout (upstream gate in arguments.py).
- **OPD per-position sparse scoring is dormant**: client payload/helper/flag are merged, but the server capability does not exist on the sglang-miles line. Port ruling recorded: keep the fork's `logprob_start_len = prompt_length - 1` materialization window, keep upstream's absolute-position array convention.
- **Fully-async prefetch knob**: recipes keep `--fully-async-prefetch-batches`, and the arguments-side derivation to `--async-max-concurrent-samples` now genuinely ships (it did not in the first published revision — the flag was inert and every migrated recipe silently ran at depth 1; caught by PR review). The two knobs are mutually exclusive, and depth 1 reproduces the upstream default exactly. Verify pacing on the first async run.
- Multi-turn wandb metrics stay under the fork `interaction/*` namespace (upstream renamed theirs `multi_turn`); dashboards need no change.
- **Wheels bundle stays on cu129 by design**: ACTIVE `MILES_WHEELS_TAG=cu129-x86_64` (torch 2.11) while `UPSTREAM_WHEELS_TAG` now targets `cu130-x86_64`. The sglang *source* is fully synced (ACTIVE `v0.5.16` == `UPSTREAM_SGLANG_IMAGE_TAG`), so this is a held-back torch-ABI/wheels jump, not a pending source sync — `extract_pins.py --check` stays consistent and `install_env.sh` fails closed on any ABI mismatch meanwhile. Run `/sglang-sync` for the cu130 bundle as its own torch-ABI-scoped change.
- TE 2.17 requires the Bridge pin in this PR (`7f0fb345`); fresh envs before this PR's install_env.sh will not build TE 2.17 correctly.
- The rolling `cu129-x86_64` wheels bundle still ships the interface-less FA3, so the guarded hopper fetch stays; delete it together with the hand-owned pin when the bundle moves to FA3 ≥3.0.0.

## Divergence from upstream after sync

319 files changed, +45,211 / −331 vs the merged upstream tip `fc04f666` — see [divergence.stat](divergence.stat) (slurm launcher/env tooling, skills, sync records, fork recipes, OPD/multimodal/fully-async fork features, envpack adapter).

## Test plan

- [ ] **Re-run the regression gate after the review repairs** — the first baseline attempt was invalid: the prefetch knob was inert and the staleness filter fail-open, so neither arm exercised what its wrapper claimed.
- [x] `pre-commit run --all-files`: clean. The first PR CI run caught it failing — the test plan had never claimed it, so it was never run branch-wide. The repair applied ruff/isort/black to five fork files (`initialize.py`, `on_policy_distillation.py`, and three test modules) and two upstream files (`miles_plugins/models/inkling/layers.py`, `tests/fast-gpu/test_nvfp4_quantizer.py`) whose imports do not satisfy this fork's isort line length — a recurring per-sync cost worth expecting.
- [x] `tests/fast` full suite: **5408 passed, 45 skipped, 0 failed**.
- [x] SGLang patch battery on the rebased v0.5.16 tip: 52 passed + 13 subtests.
- [x] `install_env.sh` fresh-env build (job 42842, env `miles_sync0818_test`): TE 2.17 source build + cutlass 4.6.2 + v0.5.16 editable + Bridge 7f0fb345 all installed.
- [x] `verify_env.py`: 38 checks OK, 0 FAIL (job 42862). Follow-up commit updated its sglang deep-import probe to `quantization.fp8` (v0.5.16 removed `fp8_kernel`).
- [ ] **Pre-merge regression gate** (`scripts/experiments/baseline/`, wandb `M3TRL/baseline`): `sync20260818-opd-geo3k-mm-mt-fullyasync-200step` + `sync20260818-rl-geo3k-mt-fullyasync-prefetch2-3node` — curves manually compared against prior baseline entries; **this PR merges only after both show no regression**. (r3moe slot skipped this round: the R3 route plane lives on `feature/moe_multimodal`.)

⚠️ **Merge mode**: this PR MUST be merged via "Create a merge commit". Squash or rebase will break future `merge-base` detection.
