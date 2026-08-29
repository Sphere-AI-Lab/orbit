# Phase 3: heavy-file extraction — plan from measured recon

Companion to 2026-08-29-miles-isolation-target-layout.md (hook table) and the
per-file structural recon (subagent, 2026-08-29; summarized in the verdicts
below). Execution mode: one slice = one opus/sonnet subagent working from a
short per-slice spec derived from this doc; Fable reviews, re-runs gates,
stamps seams, regenerates the manifest, commits. Subagents never commit.

## Extraction patterns (named, so slice specs can just cite them)

- **P1 lift-out**: orbit-added module-level functions/classes move verbatim to
  a home module; the miles file keeps a stamped import seam. For code only
  orbit calls, the miles file keeps nothing.
- **P2 mixin**: orbit-added METHODS on a miles class move to a home mixin
  class; the miles class declaration gains one stamped base
  (`class X(OrbitXMixin, ...)`). Interleaved edits inside base methods are NOT
  solved by this — only whole added methods.
- **P3 hook re-expression**: an interleaved modification becomes a stamped
  one-line call-out to a home function (before/after hook), or routes through
  an existing miles `--custom-*-path` hook per the verified hook table.
- **P4 override**: a base method orbit rewrote wholesale is overridden in the
  home mixin/subclass; the base copy returns to pristine. Accepted drift risk:
  upstream changes to the shadowed method stop applying — use only where the
  rewrite is already total.
- **P5 stamp-and-accept**: the diff IS the seam (delegation shells, deletions);
  stamp every hunk, no extraction. Already the case for lora_utils.py.

## Per-file verdicts and slice grouping

Slice 3a (insertion-dominated, low risk — do first):
- `update_weight/common.py` (+96/−12): P1 — adapter-param enumeration +
  expert-TP detection → `orbit/megatron/` (or fold into existing modules);
  small gather tweaks stay as stamped seams.
- `miles/ray/rollout.py` (+237/−26): P1 — OPD teacher-pool block (91 lines) →
  `orbit/opd/teacher_servers.py`; math-alignment eval scoring call-out P3;
  the ~18 small modification hunks become stamped seams.
- `miles/rollout/sglang_rollout.py` (+394/−23): P1 — the 222-line phase-stats
  subsystem → `orbit/rollout/phase_stats.py`; OPD capture + prefill
  recompute + eval concurrency stay as stamped P3 call-outs.

Slice 3b: `checkpoint.py` (+864/−29): P1 — the 797-line
  marker/preflight/save_checkpoint_with_peft subsystem →
  `orbit/megatron/checkpointing.py`; `load_checkpoint`'s 13 dispatch hunks
  become one P3 delegate seam ("orbit checkpoint dispatch: try home loader
  first"), per the hook table's save/load delegate.

Slice 3c: `sglang_engine.py` (+506/−31): P2 mixin for the ~10 added methods
  (adapter tensor loading, distributed PEFT sync, double-buffer activation) →
  `orbit/sglang/engine_ext.py`; launch-env helpers P1 → `orbit/sglang/launch.py`;
  the small modified methods stay stamped seams. Factory seam per hook table.

Slice 3d: `loss.py` (+863/−69): P1 — the 567-line OPD loss block + helpers →
  `orbit/opd/losses.py`; the `loss_type` match arms route via
  `--custom-loss-function-path`-shaped dispatch ONLY if that keeps CLI compat —
  otherwise keep the match arms as thin stamped one-line calls into home.
  `get_log_probs_and_entropy` top-k branch and `compute_advantages_and_returns`
  whitening-group change are P3 seams.

Slice 3e: `ppo_utils.py` (+450/−64): P1 — OPD advantage shaping, explained-var,
  true-on-policy gather block → home (`orbit/opd/advantages.py`,
  `orbit/critic/value_stats.py`, `orbit/true_on_policy/full_logits.py`).
  The `get_advantages_and_returns_batch` masked-compression rework is a real
  algorithmic change to base: DECISION NEEDED — P4 override in home vs
  leave-in-place stamped. Default: leave in place, stamped, revisit at Phase 4.
- Note: ppo_utils.py imports nothing from orbit today; after 3e it will (loss.py
  already imports home modules, so direction stays miles → orbit).

Slice 3f: `model.py` (+280/−53) + `model_provider.py` (+25/−24): P1 for critic
  reinit/memory-snapshot/optimizer-builder helpers → `orbit/critic/` and
  `orbit/megatron/`; provider work routes through `--custom-model-provider`
  per the verified hook table (orbit's in-place critic extension moves INTO the
  custom provider); `initialize_model_and_optimizer`'s low-precision split and
  `save_hf_model` stay P3 seams.

Slice 3g (deepest, last): `actor.py` (+769/−95) + `update_weight_from_tensor.py`
  (+446/−115): P2 mixins for the added method blocks (eval-NLL, teacher
  state, adapter push/teacher push) + P1 for module helpers; the interleaved
  `init`/`train_actor`/`update_weights`/`connect_rollout_engines` control flow
  becomes named P3 hooks (lifecycle: post-init, pre/post-train-step already
  exist as miles hooks — use them where the verified table allows). Expect the
  largest residual seam count here; that is acceptable.

lora_utils.py (+59/−335): P5 — stamp every delegation hunk; no extraction.

## Gates per slice (all CPU; identical to Phases 1-2)

1. compileall + tools/check_import_integrity.py clean.
2. `python3 tools/miles_purity.py --write`: the slice's files' delta drops;
   NOTHING else newly budgeted; pristine count never decreases.
3. Fast-suite failure set == the 7 known pre-existing failures.
4. Every residual hunk stamped `# ORBIT-SEAM:`; manifest regenerated.
5. GPU bitwise requalification (frozen-batch + e2e smoke) is deferred and
   user-owned; each slice records "CPU-verified only" until it runs.

## Sequencing constraint

Slices are serial by default (shared manifest). A slice may run in parallel
with another ONLY if their file sets are disjoint and neither regenerates the
manifest itself (Fable regenerates at commit time).
