# Static consistency guards for mechanical-refactor bugs

Every bug this campaign leaked was a NAME or LOCATION inconsistency introduced by
a mechanical transformation, not a logic error. Four of seven were statically
detectable but were caught by an 8-GPU smoke run (minutes) or a bespoke audit
instead of CI (seconds). This adds the guard family that closes that gap.

Convention to follow exactly (mirror `tools/check_import_integrity.py` +
`tests/fast/test_import_integrity.py`): a `tools/check_*.py` module exposing
`collect_errors() -> list[str]` and a `__main__` block that prints errors and
exits 1, plus a thin `tests/fast/test_*.py` that loads it by path with
`importlib.util.spec_from_file_location` and asserts `not errors`.

## G1 — argparse dest consistency  (tools/check_args_dest_consistency.py)

Bug class it closes: `verify_env.py` registered `--orbit-root` (dest
`orbit_root`) but `main()` read `args.miles_root` -> unconditional
AttributeError, invisible to CI. Same shape as the `--use-miles-router` vs
`use_orbit_router` bug the both-forms rename scan class-missed, because argparse
derives the dest at runtime and no source token spells it.

Rule A (per-file, standalone parsers): for any file that calls
`add_argument`, every BARE `args.<name>` / `<ns>.<name>` read of the parsed
namespace in that file must be a dest registered in that file, or assigned in
that file, or read via `getattr(..., default)`.

Rule B (repo-wide, production parser): the authoritative dest set is
`tests/fast/args_surface_golden.json` (2,198 records) UNION every
`args.<name> = ...` assignment and `setattr(args, "<name>", ...)` in the repo.
Every bare `args.<name>` read repo-wide must be in that set.

Derivation of a dest from an `add_argument` call: explicit `dest=` kwarg wins;
else the first long option string with leading dashes stripped and `-`->`_`;
else the positional name.

FALSE-POSITIVE CONTROL (this decides whether the guard is usable): a local
named `args` is often NOT the training namespace (`*args` tuples, a list of CLI
tokens, a dataclass). Tune empirically: run it, read EVERY hit, and either
(a) tighten the rule, or (b) allowlist with a one-line reason per entry.
Prefer tightening. A guard with noisy allowlists is worse than no guard.
Target: zero errors on the current tree with an allowlist under ~15 entries.

## G2 — call-signature agreement  (tools/check_call_signatures.py)

Bug class: orbit's one-trunk critic called `forward_only()` without upstream's
newly-added required `rollout_id` -> TypeError only reachable on GPU.

Scope to what is statically resolvable with high confidence:
- `self.<method>(...)` calls inside a class, resolved against that class and its
  in-repo base classes (including the orbit mixins, MRO order left-to-right).
- module-level `from x import f` / `import x` then `x.f(...)` where `x` resolves
  to an in-repo module and `f` to a module-level `def` in it.
Check: every required positional-or-keyword parameter without a default is
supplied (positionally or by keyword); no unexpected keyword is passed unless
the callee has `**kwargs`; `*args`/`**kwargs` on the callee relaxes the check.
SKIP: decorated functions (a decorator may change the signature), classmethods
resolved through metaclasses, anything whose callee cannot be resolved uniquely.
Under-reporting is fine; a false positive is not.

## G3 — path-anchor validity  (tools/check_path_anchors.py)

Bug class: `Path(__file__).resolve().parents[3]` in orbit/rewards broke when the
flatten changed the file's depth; the anchor silently pointed at the wrong
directory.

Rule: for every `Path(__file__)...parents[N]` or chained `.parent` on
`__file__`, compute the anchor from the file's ACTUAL path in the repo and
require it to still be inside the repo (never above the repo root). When the
expression is immediately joined with string literals (`anchor / "a" / "b.py"`),
require the resulting path to EXIST in the working tree. Report the file, the
line, the computed anchor and the missing target.

## Constraints for all three

- Must pass clean on the current tree (that is the acceptance gate).
- Runtime budget: the whole family under ~20s in the fast suite; parse each file
  once and share the AST across guards where convenient.
- Use the 3.12 interpreter (/lustre/fast/fast/zqiu/clthegoat-orbit/uv_env_build/orbit-cu130-v1/bin/python);
  the harness python3 is 3.10 and fails on PEP 695 `type X = ...`.
- Set PYTHONDONTWRITEBYTECODE=1; Lustre is at its inode quota, create no stray
  files (scratch work goes in the session scratchpad, not the repo).
- If a guard finds a REAL inconsistency in the codebase, do NOT fix it
  unilaterally — report it with file:line and the evidence; the main session
  decides. Only exception: a fix that is unambiguously a typo restoring
  obviously-intended behavior, which must still be reported.
- Do NOT commit. Do NOT regenerate the purity manifest or any golden.
