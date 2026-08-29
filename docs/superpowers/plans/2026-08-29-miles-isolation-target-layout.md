# Miles isolation: target directory layout and purity enforcement

Proposal. Applies the sglang (`srt/peft` + audited seams) and Megatron-LM
(`experimental_attention_variant/`) isolation pattern to orbit itself, so that
"orbit = pristine miles + an orbit home layer + a bounded seam list" becomes an
enforced property instead of archaeology. Companion evidence:
`docs/reports/2026-08-29-miles-fork-delta.html` (full delta inventory and the
transfer dry-run onto miles@2026-08-27).

## Measured starting point (orbit-main @ 2026-08-29, vs miles ef7481ae3)

After normalizing the mechanical rename (`miles`->`orbit`, `miles_plugins`->
`orbit_plugins`, `MILES_`->`ORBIT_`):

| Class | Files | Normalized delta | Disposition |
|:--|--:|--:|:--|
| Pristine (identical or rename-only) | 107 | 0 | Enforced pristine from day one |
| Seam (<=20 lines) | 57 | 389 lines | Keep; mark and budget each hook |
| Moderate (21–100 lines) | 26 | 1,286 lines | Extract into home layer where possible |
| Heavy (>100 lines) | 30 (~22 code) | 12,301 lines | The real campaign; see hook inventory |
| Orbit-only files | 789 | — | Already isolated; some need moving into the home |
| Dropped miles files | 406 | — | Keep-or-drop list; not purity-relevant |

## Target layout

```
orbit/                          # package keeps its name; most files pristine miles
  peft/                         # THE home layer (mirrors sglang's srt/peft)
    transport/                  #   <- backends/megatron_utils/peft_transport/*
    megatron/                   #   <- oft_utils, peft_offload, bridge_peft_helpers,
    sglang/                     #      lora extraction from lora_utils, engine-side
    audit/                      #      helpers, audit/peft_wrap, low_precision_bootstrap
    critic/                     #   <- adapter-critic logic extracted from
                                #      training_utils/loss.py and utils/ppo_utils.py
  ...everything else            # pristine miles (purity-enforced) + marked seams
orbit_plugins/                  # unchanged: mbridge patches, model_args, conversion
tools/ scripts/ examples/ docs/ # already orbit-only at the top level
```

Rules:

1. **Pristine layer.** Any file with a miles counterpart that is not on the seam
   or owned list must be byte-identical to the pinned miles base after the
   rename normalization. New orbit code never lands in these files.
2. **Home layer.** All orbit logic lives under `orbit/peft/`, `orbit_plugins/`,
   or the orbit-only top-level trees. New orbit features start here.
3. **Seam layer.** A miles file may carry a bounded hook, marked
   `# ORBIT-SEAM:` with one line of rationale, calling into the home layer.
   Budgets are recorded in the purity manifest; a seam growing past its budget
   fails CI until deliberately re-recorded.
4. **Owned list.** Files orbit owns outright and diffs freely: `README.md`,
   `pyproject.toml`, `train.py`, `examples/*/README.md`, CI config. Explicitly
   enumerated, so "owned" is never a loophole.

## The existing miles hook surface — use it first

Reference: **slime-agentic** (LMIS-ORG/slime-agentic on THUDM/slime) realizes
the end state this plan aims for — measured: 136/137 slime-layer blobs
byte-identical to upstream, exactly **one** modified core file
(`slime/ray/rollout.py`, +16 lines, insertions only), zero files added inside
`slime/`, all extensions in a top-level `agentic/` home consuming slime's
dotted-path plugin hooks. Two rules carry over: **hook-first re-expression**
(try the existing extension surface before inventing a seam) and
**additive-only seams** (insertions that call out, never in-place edits — they
merge cleanly across upstream bumps).

The decisive fact: miles at the fork base ships a *richer* version of that
hook family than slime, and orbit retains it — 14 `--custom-*-path` dotted
hooks: `config`, `convert-samples-to-train-data`, `eval-rollout-log-function`,
`generate-function`, `loss-function`, `megatron-before-log-prob-hook`,
`megatron-before-train-step-hook`, `megatron-init`, `model-provider`,
`pg-loss-reducer-function`, `reward-post-process`, `rm`,
`rollout-log-function`, `tis-function`. Part of the extension interface this
plan was going to design already exists.

Because these are single-slot hooks and orbit has many features wanting the
same points, the home layer registers **one orbit dispatcher per hook** that
fans out internally (a small piece of `orbit/peft/`, not a miles change).
Before committing any mapping, verify each hook actually fires at the point
orbit needs (cheap CPU trace per hook).

## Hook mapping for the heavy code files

**Verified 2026-08-29** (CPU trace of every call site; firing points and
signatures below are read from code, not inferred). Two additional pluggable
entry points surfaced beyond the 14 `custom-*` hooks:
`--rollout-function-path` / `--eval-function-path` make the *entire rollout
procedure* pluggable (orbit already swaps them for SFT mode), and
`--custom-agent-function-path` exists in the agentic generate hub. Proof the
pattern works in production: OPD teacher scoring **already rides
`--custom-rm-path`** (`default_async_rm` is deliberately exposed for
pass-through when the reward slot is hijacked).

Verified firing points: `custom-megatron-init(args)` fires at the end of
Megatron init, **before any model exists** — env/patch setup only, not a wrap
point. `custom-megatron-before-train-step(args, rollout_id, step_id, model,
optimizer, opt_param_scheduler)` fires per train step after zero-grad.
`custom-megatron-before-log-prob(args, model, store_prefix)` fires before
log-prob forwards. `custom-loss-function(args, batch, logits,
sum_of_sample_mean) -> (loss, log)` is a full per-microbatch loss replacement
selected by `--loss-type custom_loss` — and the `loss_type` match is exactly
where orbit's in-place additions (`value_loss`, `opd_jsd_loss`,
`opd_topk_loss`) sit today, so they migrate behind one dispatcher.
`custom-model-provider` *replaces* model construction per chunk (pre-DDP);
orbit's critic value-head wrapping is currently an in-place extension of this
very hook site and must move into the custom provider.
`custom-config-path` is a config-file loader (opens a file, overrides args) —
**not** usable for argument registration; the registration seam stands.

| Entangled code | Lines | Verdict | Still needs |
|:--|--:|:--|:--|
| Verified-reward backends + router | (rm_hub + args) | **OK, proven in prod** — `custom-rm-path` (async `(args, sample, **kw)`), `custom-reward-post-process-path` | nothing |
| OPD/MOPD + adapter-critic losses (`loss.py` 1,031) | 1,031 | **OK** — `custom-loss-function-path` + home dispatcher replaces the in-match loss types | teacher serving/pools stay home-layer |
| Advantage/return calc (`ppo_utils.py` 582) | 582 | **partial** — outside the loss hook | own extraction review; candidate: `convert-samples-to-train-data` + before-train-step |
| Actor lifecycle (`actor.py` 955) | 955 | **partial** — before-train-step/before-log-prob verified with model+optimizer in scope; init hook is pre-model (env setup only) | weight-sync/double-buffer points need 1–2 additive seams |
| Model wrapping (`model.py` 374, `model_provider.py`) | ~450 | **plausible** — provider replaces construction per chunk, pre-DDP (PEFT wrap timing fits); verify against the actual wrap call path | move orbit's in-place critic extension into the custom provider |
| Rollout drivers (`sglang_rollout.py` 449, fully-async) | 449 | **OK** — `rollout-function-path` (whole procedure) + `custom-generate-function-path` (incl. per-eval-dataset override) | engine construction factory seam |
| Arguments (`arguments.py` 2,027) | 2,027 | **hook ruled out** — `custom-config-path` is a config loader | one registration-loop seam (as planned) |
| Checkpoint adapter-state (`checkpoint.py` 1,028) | 1,028 | no hook | save/load delegate seam |
| `update_weight/*` + transport | 738 | no hook | finish the transport registry; double-buffer slots stay home-layer |
| Engine (`sglang_engine.py` 606) | 606 | no hook | subclass + factory seam in `orbit/peft/sglang/` |
| `lora_utils.py` (467), `ray/rollout.py` (289), `ray/actor_group.py` (125) | 881 | partial | case-by-case; several shrink once the hooks above are in use |

Where miles lacks the needed extension point, prefer contributing the generic
hook upstream (as with sglang) over widening a seam.

## Migration order

Isolate first, port second: refactoring on the current base gives every step a
bitwise reference (pure code movement must reproduce today's GPU-qualified
behavior bit-identically), and isolation collapses the port — git auto-resolves
any file whose our-side is byte-identical to the base, so the measured 45
both-modified conflicts against latest miles mostly evaporate once the
entangled files are pristine-plus-seams.

1. Agree this layout + the owned list with the team (the delta report is the
   negotiation document).
2. Hook verification pass: CPU-trace each of the 14 `--custom-*-path` hooks to
   confirm it fires where the mapping table needs it; correct the table.
3. Phase 1 (near-mechanical, bitwise-neutral): move orbit-only files that sit
   inside shared directories into the home; stamp the 57 surgical seams with
   `# ORBIT-SEAM:` marks.
4. Phase 2: the `arguments.py` registration refactor (~2k lines back to
   pristine, the loudest signal the approach works).
5. Phase 3: the heavy files, one at a time, hook-first per the mapping table;
   every extraction commit gated on the frozen-batch/e2e smoke reproducing
   bit-identically. Regenerate the purity manifest per step; budgeted counts
   fall monotonically.
6. Phase 4: merge `refs/miles/upstream` (now cheap), re-anchor the few seam
   lines upstream moved, regenerate the manifest against the new base.
7. Qualify with a numerical-equivalence campaign (pattern:
   `docs/reports/2026-08-27-restructure-numerical-equivalence.html`) before the
   result becomes orbit-main.

## Enforcement (live today)

`tools/miles_purity.py` + `tests/fast/test_miles_purity_ratchet.py` +
`tests/fast/miles_purity_manifest.json` are committed and green against the
current tree: 107 pristine files must stay pristine, and any edit to one of the
113 budgeted files fails CI until the manifest is regenerated — so from now on
the fork delta changes only deliberately. The manifest is self-contained
(hashes only); regeneration needs the miles base objects once:

```
git fetch https://github.com/radixark/miles.git ef7481ae3bfbcc641d031e7e6113b646bb764382:refs/miles/base
python3 tools/miles_purity.py --write
```
