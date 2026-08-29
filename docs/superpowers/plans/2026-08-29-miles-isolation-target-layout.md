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

## Hook inventory for the heavy code files

The ~22 heavy code files reduce to home-layer extractions behind these
interfaces (per-file verdicts belong in the campaign plan, not here):

- `utils/arguments.py` (2,027) — namespaced argument registration: the home
  layer contributes an argparse group; miles file gains one seam call.
- `training_utils/loss.py` (1,031) + `utils/ppo_utils.py` (582) — loss/advantage
  strategy object; adapter-critic implementation moves to `orbit/peft/critic/`.
- `megatron_utils/actor.py` (955) — train/rollout lifecycle callbacks
  (init, pre/post step, weight-sync points); orbit registers listeners.
- `megatron_utils/checkpoint.py` (1,028) — pluggable adapter-state
  save/load path beside the base checkpoint flow.
- `update_weight/*` (738 across files) — transport already plugin-shaped;
  finish the registry so PEFT transport is a registered backend, not a diff.
- `sglang_utils/sglang_engine.py` (606) + `rollout/sglang_rollout.py` (449) —
  engine wrapper subclass in `orbit/peft/sglang/`; seam = which class to build.
- `megatron_utils/lora_utils.py` (467, mostly deletions) + `model.py` (374),
  `ray/rollout.py` (289), `ray/actor_group.py` (125) — case-by-case; several
  shrink to seams once the callbacks above exist.

Where miles lacks the needed extension point, prefer contributing the generic
hook upstream (as with sglang) over widening a seam.

## Migration order

1. Agree this layout + the owned list with the team (the delta report is the
   negotiation document).
2. Execute as part of the port to latest miles (`miles-graft` +
   `git -c merge.renameLimit=20000 merge refs/miles/upstream`, 45 code
   conflicts measured): resolve each conflict *into* the target structure.
   Seams extracted against the frozen April base would land on moved upstream
   code (e.g. miles restructured `ray/rollout/`).
3. Regenerate the purity manifest against the new base; drive budgeted counts
   down file by file; pristine set should grow monotonically.
4. Qualify with a numerical-equivalence campaign (pattern:
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
