# Phase 5: nest the orbit home into miles/orbit/

Decision (2026-08-30): the repo keeps ONE code tree. The orbit home layer moves
from top-level `orbit/` to `miles/orbit/`; the import name changes from
`orbit.*` to `miles.orbit.*`. `miles_plugins/` stays at top level (it is
upstream's own second tree; moving it would break upstream-path-verbatim
vendoring). The vendored base guarantee is untouched: upstream has nothing at
`miles/orbit/` (verified against dbbab1566), so the subtree is add-only in
every future upstream merge.

## Invariants that must survive

1. All 1,525 pristine files stay byte-identical (they contain zero orbit
   references, so no rewrite may touch them — any pristine-file diff is a bug).
2. Budgeted seam files change ONLY on their existing `orbit` import/reference
   lines (already inside stamped hunks); delta-line counts stay ~constant.
3. `git mv orbit miles/orbit` preserves history (one rename, no content change
   in the move commit step).
4. Guard suite green at the end: purity ratchet (with updated home rule),
   import integrity, args golden, seam stamping.

## Rewrite rules (apply to every in-scope file)

R1. Module references: `orbit.` preceded by neither `[A-Za-z0-9_.]`
    → `miles.orbit.`  — applies to import statements
    (`from orbit.x import y`, `from orbit import x`, `import orbit.x as y`)
    AND dotted strings that name package objects: mock/monkeypatch targets,
    `custom_rm_path="orbit...."`, argparse default strings, `python -m orbit.x`,
    importlib strings, `--custom-*-path orbit....` flags in .sh/.yaml.
R2. Filesystem path references: `orbit/<known subpkg or file>` →
    `miles/orbit/<...>` in comments, docstrings, string literals, asserts, and
    shell scripts (e.g. `$REPO/orbit/sglang/compat_site`). Known subpackages:
    megatron rewards opd transport sglang critic rollout utils merge
    true_on_policy ultra audit arguments.py __init__.py.
R3. NEVER touch: the word `orbit` alone (product name in prose), `orbit_`/
    `_orbit` identifiers (flag dests like use_orbit_router, function names),
    `ORBIT-SEAM` stamps, `ORBIT_*` env vars, dashed tokens (`orbit-build-wheels`,
    `--use-orbit-router`, repo names, wheel URLs), `load_cuda13_2_orbit_env.sh`,
    uv cache names (`uv_cu13_orbit`), pyproject `name = "orbit"` (dist name
    stays), docs/ prose and historical reports, results/.
R4. Idempotency: the lookbehind in R1 makes `miles.orbit.` a non-match; never
    produce `miles.miles.orbit` or `miles.orbit.orbit`.
R5. Packaging (setup.py + pyproject): drop `"orbit*"` from the find-packages
    include lists (now covered by `miles*`); in isort config replace `"orbit"`
    with `"miles"` in known_first_party and src_paths.

## Execution split (subagents edit; they never commit, never regen manifest/golden)

- Agent A (opus): `miles/orbit/**` (52 files with imports + any dotted/path
  refs), the 29 `miles/**` seam files importing orbit,
  `miles_plugins/models/qwen3_5.py`, `train.py`, `train_async.py`,
  `setup.py`, `pyproject.toml` (R5).
- Agent B (sonnet): `tests/**` (73 import files + 49 dotted-string refs +
  path refs in docstrings/asserts, incl. the literal-path assert in
  tests/test_lora_regret_rl_launcher.py) and `tools/**` (12 files) — EXCEPT
  reserved: tools/miles_purity.py, tools/check_import_integrity.py,
  tools/dump_args_surface.py, tests/fast/test_miles_purity_ratchet.py,
  tests/fast/test_import_integrity.py.
- Agent C (sonnet): `scripts/**` + `examples/**` (.py/.sh/.yaml; ~50 sh files
  carry `--custom-*-path orbit....` or PYTHONPATH/path refs). If any
  examples/**/README.md changes, regenerate the docs mirror via
  `scripts/tools/sync_example_docs.py`.
- Fable (main): the reserved guard files — miles_purity.py home_violations
  gains the `miles/orbit/` home-layer exemption + docstring update;
  check_import_integrity ROOTS/resolution; ratchet-test docstrings;
  dump_args_surface import — then manifest regen, golden regen (review diffs:
  only default-value strings `orbit.…` → `miles.orbit.…` may appear),
  repo-wide both-forms scan, fast suite, commit, consolidated suite + GPU
  smoke (reference abs_diff mean 0.012841 / max 0.33313, rc=0).

## Verification gates

G1. Both-forms scan: zero remaining `(?<![\w.])orbit\.` / `(?<![\w.])orbit/`
    hits outside R3 exclusions; zero `miles\.miles\.orbit|miles\.orbit\.orbit`.
G2. Pristine untouched: `git diff --name-only` ∩ manifest.pristine = ∅.
G3. Guard tests 7/7 green; fast suite green minus known env failures.
G4. GPU smoke: same recipe as Phase 4 gate
    (examples/high_precision/run-qwen2_5-0_5b-bf16-math-lora-ppo-adapter-critic-smoke.sh,
    seed 1234, 1 train GPU + 2 rollout GPUs), rc=0, abs_diff at reference level.
