# miles-sync 2026-08-18 — resolution log and design rulings

Merge: upstream `c94c2fa9..fc04f666` (274 PRs) into `sync-upstream-20260818`;
merge commit `4c93a4b4`. Combined with sglang-sync v0.5.15 → v0.5.16
(`36982fef0` = sgl-project/sglang@sglang-miles `cb05a44f3` + 7 re-applied local
patches) and Megatron-Bridge `c092daca` → `7f0fb345` (TE 2.17 grouped-linear
contract; fast-forward, 21 commits).

## Conflict resolutions (32 files, per-file user review)

- `.gitmodules`: keep fork submodules, drop swe-agent entries (upstream #1918).
- `examples/fully_async/*`: accept upstream deletion (#1716/#1717 rewrote
  fully-async as class-based `miles/rollout/fully_async_rollout.py`; #2522 made
  class-based the default and `--fully-async` asserts an empty
  `--rollout-function-path`). Fork port (plan "A'"):
  - prefetch expressed through `--async-max-concurrent-samples` derivation
    (recipes keep `--fully-async-prefetch-batches` env vars; wiring lands with
    the arguments.py fork block);
  - fail-closed staleness contract preserved as
    `examples/fully_async/fail_closed_data_buffer.py` (upstream's
    DefaultDataBuffer silently admits version-less groups — the exact failure
    mode that cost a 15-hour run pre-fork-fix);
  - `_CachedWeightVersion` router polling deleted (upstream #2244 passes the
    trainer weight version straight through);
  - 18 recipes migrated `--rollout-function-path examples.fully_async...` →
    `--fully-async`.
- `examples/geo3k_vlm_multi_turn` → `examples/geo3k_vlm/multi_turn` rename
  followed; 61 references swept.
- OPD `miles/rollout/on_policy_distillation.py` (7 hunks): fork scoring
  pipeline kept (multimodal exact-suffix payloads, persistent session
  transport, telemetry); upstream's multi-teacher routing wired INTO the fork
  transport (`_teacher_url_for_sample` at all three teacher call sites — no-op
  without `--opd-teacher-urls`); upstream's per-position sparse scoring kept
  dormant (see design ruling below).
- `losses.py`: upstream detach discipline + logprob-reuse (#2219/#1966)
  adopted; fork `_rollout_mismatch` diagnostics + DAgger reducer kept.
- `log_utils.py`: upstream reduction core ((sum,count) tuples + extrema #1968);
  fork `interaction/*` metric namespace kept (has consumers) + upstream's
  rounds/min added.
- `actor.py`: upstream `verify_megatron_parallel_state` + `_asleep` +
  rematerialize `main_cast_ctx` adopted; fork `resolve_start_rollout_id_after_load`
  (#54 bridge-resume) replaces upstream's naive `loaded_rollout_id + 1`
  (strict superset).
- CI `pr-test.yml`: upstream's simplified fastfail clauses + fork
  hardware-label gates re-ANDed (5 stages; fork has no self-hosted GPU runners).
- Full list and remaining unions: see the merge commit and pr-body.

## Post-merge validation findings (full fast suite: 5381 tests)

- REAL bug caught: `actor.py` used upstream's `verify_megatron_parallel_state`
  without its import (F821) — fixed.
- `test_openai_endpoint_utils.py` resolution REVISED (user-ratified): upstream
  moved fork's `compute_samples_from_openai_records` AND its test battery into
  `miles/rollout/session/samples/` — took upstream's test file, ported the two
  teacher-topk propagation tests upstream had not migrated into
  `tests/fast/rollout/session/test_samples.py`.
- Import-tax fix (user-ratified design A): `megatron_utils/__init__` side
  effects (deep_ep TMS patch + bridge plugin registration) moved to
  `runtime_hooks.install_runtime_hooks()`, called from `initialize.init()`.
  `import miles.utils.logging_utils` (pulled by every process via the audit
  event models) drops 75s → 0.6s on WekaFS. Upstream candidate.
- OPD transport-close regression (user-ratified design B): the legacy
  fully-async worker owned the #24 `close_scoring_transport()` hook; re-wired
  into `dispose_rollout_function` via a `sys.modules` probe (no-op unless OPD
  scored). Legacy fully-async tests rewritten against the new architecture;
  `_CachedWeightVersion` tests deleted with the class.
- PYTHONUNBUFFERED: upstream's new launcher policy test applied to two
  fork launchers (ray_lifecycle.sh runtime-env, muon 2-node runner).

## Design rulings (user-ratified)

1. **scripts/models convention**: upstream converted all model-args scripts to
   python (`model_args()` + `load_model_args`). Fork ships five thin .sh shims
   (qwen2.5-3B, qwen3-1.7B, qwen3-4B, qwen3-8B, qwen3-30B-A3B) that delegate to
   `load_model_args` so 26 existing recipes keep sourcing them with zero drift.
   **NEW recipes must not add shims — consume `load_model_args` directly.**
2. **OPD per-position alignment** (for the future sglang-miles server port —
   the capability does not exist on v0.5.16): keep the fork's materialization
   window `logprob_start_len = prompt_length - 1` (the expensive side: teacher
   never materializes prompt logprobs; upstream's start_len=0 was a limitation
   of a payload builder with no response-window concept, not a choice); keep
   upstream's positions-array convention (absolute per-input-position indexing,
   empty prompt slots — the cheap side; keeps client helper/tests/upstreaming
   compatible). Contract recorded in `_score_payload`.
3. **cutlass-dsl 4.6.2** (not base 4.6.0) in the sglang cu12 flavors patch:
   adopts upstream #2600's sm_103 FA4-backward hang fix; validated at
   env-build time by the test plan.
4. OPD top-k student-side strategies currently require
   `MILES_USE_LEGACY_ROLLOUT_V1=1` (upstream gate: the class-based rollout does
   not produce `opd_student_top_logprobs` yet) — port item for the OPD series.

## Validation round 2 (the nine stragglers outside the first six files)

- `log_utils` metric tests: upstream's extrema test rewritten to the fork
  contract (`interaction` namespace + locally built reduction map — ruling 
  above); fork's reduce test migrated from the deleted `_reduce_gathered_log_dicts`
  helper onto upstream's `reduce_gathered_log_dict` core; `rounds/min` added to
  the compact-section expectation. Verified the production path is safe:
  `_compute_rollout_kl_statistics` builds explicit extrema reductions and the
  call site still passes them through.
- `actor_factory`: RESOLUTION REVISED — the fork's `os.path.exists` assert on
  the torch_memory_saver preload path was retired; upstream #1785's
  package-based resolver owns existence semantics and upstream's test
  monkeypatches the resolver with fake paths.
- Shell hygiene (upstream's new policy test): frozen `sync-records/` scripts
  excluded from the hardcoded-checkout scan (records are history, not
  launchers — same exclusion as the divergence diff); PYTHONUNBUFFERED added
  to the two fork ray launchers (`ray_lifecycle.sh` runtime env, muon 2-node
  runner).
- Docs mirror (upstream #2481): fork example dirs (disk_delta_weight_sync,
  muon, vagen) added to `examples/README.md`; stale `../p2p_weight_transfer`
  links fixed for the infra_features move; mirror regenerated via
  `scripts/tools/sync_example_docs.py`.
- Test doubles: `ft_components` (v1 group lifecycle) and
  `eval_generate_rollout` (RolloutManager dispose) added to fork doubles that
  predate the synced constructor/dispose contracts.
