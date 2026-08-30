# Phase 4: merge miles upstream dbbab1566 — plan and conflict dispositions

Mechanics: git 2.34 lacks `merge-tree --merge-base`, so the merge runs on a
scaffold commit (miles-isolation's tree with `refs/miles/base` as parent) in a
node-local scratch worktree (/tmp/phase4-merge); the resolved result is grafted
back as a two-parent commit (parents: miles-isolation HEAD, refs/miles/upstream)
and fast-forwarded — no history rewriting, no destructive ops.

## Measured conflict census (real merge, 2026-08-30)

Auto-resolved: 1,278 upstream adds, 72 modifies, 27 deletes, 7 renames.
Conflicted: 392 paths in six categories, with dispositions:

| Cat | n | Meaning | Disposition |
|:--|--:|:--|:--|
| DU | 197 | we deleted, upstream modified | KEEP DELETED (our deliberate drop list); bulk `git rm` |
| UA | 66 | upstream added (mostly model_args/*.py) | TAKE THEIRS; bulk add |
| UD | 43 | we modified, upstream deleted | split: model_args/*.sh KEEP OURS (launchers source them; upstream's .py registry is additive for us — harmonize later); rollout.py + tracking_utils.py = STRUCTURAL, see below; CI files case-by-case |
| AU | 19 | our files mapped into upstream-renamed dirs (examples/low_precision -> examples/infra_features/low_precision) | adopt upstream's path, our content |
| AA | 1 | both added tools/convert_torch_dist_to_hf_ray.py | content-merge by hand |
| UU | 66 | both modified | content merges; our side is stamped seams, so conflict hunks localize around ORBIT-SEAM marks; slice to agents |

## Structural re-anchoring (the real Phase-4 work)

Upstream #899 decomposed `miles/ray/rollout.py` into the `miles/ray/rollout/`
package, and `miles/utils/tracking_utils.py` into a package. Keeping our
single-file versions would shadow the packages (module vs package name
collision) — so both are ADOPTED as upstream's decomposition, and every orbit
seam hunk (stamped, enumerable via `git diff refs/miles/base -- <old file>`)
is re-applied into the decomposed file that now holds its surrounding base
code. Orbit-only symbols that had moved to the orbit/ home already (teacher
pool, phase stats) reduce this to genuine seam lines.

## Post-merge gates (in order)

1. `MILES_BASE` in tools/miles_purity.py bumps to dbbab1566; fetch line in the
   docstring updated; manifest regenerated against the NEW base. Purity
   expectation: pristine count grows a lot (all 1,278 upstream adds + files
   whose only delta was upstream-side drift).
2. Import verifier; compileall; args-surface golden regenerated (upstream added
   args — review the diff, only upstream-attributable changes allowed).
3. Fast suite failure-set triage vs the 7 known (upstream drift may add
   failures; each triaged: env-pin drift vs seam breakage).
4. GPU reference smoke (0.5B LoRA-PPO adapter-critic) green with abs_diff at
   the reference floor.
5. Numerical-equivalence campaign (docs/reports/2026-08-27 pattern) remains
   the final qualification before this becomes orbit-main — user-owned.

Env note: upstream expects sglang 0.5.12; orbit pins the 0.5.18-line head and
owns pyproject — our pins stand; API drift surfaces in gate 3.
