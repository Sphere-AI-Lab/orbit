# Phase 2: arguments.py registration refactor — implementation plan

Goal: `miles/utils/arguments.py` (4,003 lines, 2,015 delta lines vs the miles
base, 48 hunks) shrinks to pristine-plus-bounded-seams; all orbit argument
definitions, predicates, and validators move to a new home module
`orbit/arguments.py`. Loudest signal that the isolation approach scales.

## Measured structure of the delta (recon 2026-08-29)

Orbit-ADDED top-level functions (whole-function moves, no untangling needed):

- Predicates: `uses_rollout_engines`, `needs_opd_teacher`, `uses_separate_critic`,
  `uses_adapter_critic`, `uses_head_critic`, `uses_one_trunk_critic`,
  `_is_peft_enabled`, `_is_default_rollout_function_path`
- Arg groups: `add_on_policy_distillation_arguments` (lines 156-608)
- Validators/normalizers: `validate_async_off_policy_correction`,
  `validate_rollout_temperature`, `validate_opd_topk_reference_kl_args`,
  `validate_opd_topk_vocab_size`, `validate_opd_topk_loss_args`,
  `_validate_judge_args`, `_validate_reward_router_args`, `_validate_genrm_args`,
  `_validate_opd_args` (806-1179), `_validate_teacher_adapter_config`,
  `_normalize_peft_args` (1275-1372), `_normalize_and_validate_peft_args`,
  `_validate_dsv4_cp_args`, `_validate_ppo_args`, `_apply_training_mode_args`,
  `_apply_critic_args`, `_apply_custom_config_args`, `_finalize_train_offload_args`,
  `_common_orbit_validate_args` (3629-3940)

Base functions orbit MODIFIED (these become seams):

- `get_miles_extra_args_provider` — renamed to `get_orbit_extra_args_provider`
  and internally interleaved with orbit arg definitions (1402-3346; the bulk of
  the delta). Contains inner `add_*_arguments` defs; orbit args and orbit
  default-overrides are mixed into the base groups.
- `miles_validate_args` (renamed `miles_validate_args` kept) — calls orbit
  validation.
- `parse_args`, `parse_args_train_backend`, `_resolve_eval_datasets`,
  `_maybe_apply_dumper_overrides`, `hf_validate_args` — inspect per hunk; small.
- `reset_arg` (1259) is BASE code: miles' own mechanism for overriding an
  already-registered argument. Use it from the home side for every orbit
  default-change to a miles argument, so those base lines go back pristine.

## Target architecture

```
orbit/arguments.py                  # new home module
  add_orbit_arguments(parser)       # every orbit-added argument (groups kept),
                                    #   then reset_arg(...) overrides for the
                                    #   miles defaults orbit changes
  orbit_validate_args(args)         # orchestrates all orbit validators above
  predicates + validators           # moved verbatim (same names)

miles/utils/arguments.py            # pristine + seams:
  # ORBIT-SEAM re-export block: from orbit.arguments import uses_separate_critic,
  #   needs_opd_teacher, ... (so no OTHER miles file changes its imports)
  get_miles_extra_args_provider     # base name RESTORED; one seam line inside
                                    #   calls orbit.arguments.add_orbit_arguments
  get_orbit_extra_args_provider = get_miles_extra_args_provider   # seam alias
  miles_validate_args               # one seam line calls orbit_validate_args
  parse_args wrappers               # minimal seams for whatever remains
```

Rules: seams are additive-only where possible, each stamped
`# ORBIT-SEAM: <rationale>`; the re-export block keeps every existing
`from miles.utils.arguments import X` call site working unchanged (heavy files
are Phase 3's problem, do NOT touch their imports now). Import direction is
miles -> orbit at the seam only; orbit/arguments.py must not import back into
miles.utils.arguments at module level (circular import).

## Equivalence gates (all CPU)

1. **Argument-surface golden** (built BEFORE the refactor, committed):
   `tools/dump_args_surface.py` builds the parser exactly the way `parse_args`
   does (mimic the parser-construction pattern used by the existing
   tests/fast/utils/test_peft_arguments.py) and dumps every parser action —
   option_strings, dest, repr(default), type name, choices, nargs, required,
   help — sorted by dest, to JSON. `tests/fast/test_args_surface_golden.py`
   compares the live dump against the committed
   `tests/fast/args_surface_golden.json`. Order-insensitive (group/help
   ordering may shift), content-exact.
2. Import integrity (`tools/check_import_integrity.py`) and `compileall` clean.
3. `python3 tools/miles_purity.py --write`: arguments.py delta 2,015 -> target
   <= 60 lines; NOTHING else newly budgeted.
4. Fast suite failure set identical to the 7 known pre-existing failures
   (the arg-heavy tests — test_peft_arguments, test_lora_arguments,
   test_wandb_run_naming, test_eval_nll, test_llama3_loss_mask — all pass today
   and must keep passing).
5. Seam stamps on every remaining hunk; manifest regenerated; plan docs updated.

## Execution

Step 1 (subagent): build gate 1 and commit it green against the CURRENT tree.
Step 2 (subagent): the extraction, driven by this plan; no commits; gates 1-4
  run locally by the agent, re-verified independently before commit.
Step 3 (main): stamp seams, regenerate manifest, docs, commit.

Worktree: /lustre/fast/fast/zqiu/clthegoat-orbit/orbit-isolation (branch
miles-isolation). Env python:
/lustre/fast/fast/zqiu/clthegoat-orbit/uv_env_build/orbit-cu130-v1/bin/python
with PYTHONPATH set to the worktree. If any write fails with "Disk quota
exceeded", clear __pycache__ trees under the workspace repos and retry.
