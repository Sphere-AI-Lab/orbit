# Upstream sync design — miles + thirdparty submodules

**Status**: miles-sync + miles-upstream-prs + sglang-sync implemented. ACTIVE/UPSTREAM sglang-pin model implemented 2026-06-02 (pins.env restructure + WHEELS_STACK in extract_pins.py + install_env.sh fail-closed guards) on branch `install-tooling-sglang-pin-model`; sglang-sync skill + miles-sync sync-together wiring added the same day. megatron-lm-sync, megatron-bridge-sync NOT yet built — design captured here for later.

**Git-tracked** (since 2026-07): this file lives under `scripts/slurm/docs/sync-records/`, the committed sync-history tree (see its README). It moved here from the gitignored `debug-notes/`.

**Per-sync folder convention** (added 2026-05-30): each `/miles-sync` invocation writes all its artifacts into a single `sync-records/miles-sync-YYYY-MM-DD/` folder containing:

```
miles-sync-YYYY-MM-DD/
├── prs.md            ← /miles-upstream-prs report (always)
├── pr-body.md        ← /miles-sync drafted PR body (only when sync runs)
├── divergence.patch  ← git diff vs upstream after sync — LOCAL-ONLY (gitignored)
└── divergence.stat   ← git diff --stat companion (only when sync runs; tracked)
```

This keeps sync-records/ tidy as more syncs accumulate. Standalone `/miles-upstream-prs` runs (no sync) create the folder with only `prs.md`. Future `/sglang-sync` and similar workflows should follow the same pattern (`sglang-sync-YYYY-MM-DD/`, etc.).

---

## Context

`impossible-inc/miles-imp` is a fork of `radixark/miles` with substantial local additions:

- `scripts/slurm/{setup,docs,lib}` — slurm launcher + env setup (the "no docker" path)
- `.claude/skills/{slurm-launch,rl-monitor-loop,miles-sync,miles-upstream-prs,mechanical-refactor-verify}` — workflow skills
- `examples/vagen/`, recipes under `scripts/experiments/`
- Submodules under `thirdparty/{Megatron-LM,sglang,Megatron-Bridge}` — these exist only in miles-imp; upstream installs them via Dockerfile pip git-installs

Three independent sync axes:

1. **miles**: `radixark/miles → impossible-inc/miles-imp` (this design doc; skill built)
2. **sglang**: `sgl-project/sglang + radixark/sglang (sglang-miles branch) → impossible-inc/sglang` (most volatile; skill deferred)
3. **Megatron-LM, Megatron-Bridge**: similar to sglang but lower-frequency (skill deferred; no local changes yet)

---

## Topology decisions

### Why 2-tier (not slime's 3-tier)

slime uses THUDM (true upstream) ← public-fork ← private-fork. The public-fork acts as a sanitization layer where merge-conflict resolutions are staged before reaching the private fork.

We don't need that:
- `radixark/miles` is public and is the direct upstream we track.
- No need for a separate "upstream-fork" mirror — `git fetch upstream main` works directly.

→ Skill collapses to single `upstream` remote.

### Push URL guard

`git remote -v` for the miles-imp project shows:
```
upstream  git@github.com:radixark/miles.git  (fetch)
upstream  DISABLE_PUSH_TO_UPSTREAM           (push)
```

The sentinel `DISABLE_PUSH_TO_UPSTREAM` is intentional — prevents accidental `git push upstream`. Both skills explicitly preserve it (verify fetch URL only, never `git remote set-url --push`).

### Submodule conflict policy

Upstream radixark/miles installs Megatron-LM/sglang/Megatron-Bridge via Dockerfile `pip install git+...@branch`. It does NOT track them as submodules. miles-imp promoted them to submodules.

When `git merge upstream/main` runs, git sometimes flags conflicts on the `thirdparty/<name>` paths even though upstream doesn't really have anything there — they're artifacts of the merge driver.

**Resolution policy: STOP and surface, never auto-resolve.** Even though "keep ours" is the right fix in essentially every case (upstream doesn't track these as submodules; upstream syncs never move our submodule pins), the user wants to look at every conflict first. The skill suggests:

```bash
git checkout --ours -- thirdparty/<name> && git add thirdparty/<name>
```

…as the typical fix, but does not run it until the user OKs.

The earlier version of this doc allowed auto-resolve. Changed 2026-05-29 per explicit user preference: "always let me take a look first."

---

## The Dockerfile-as-pin-source pattern

We don't run Docker. But `docker/Dockerfile` (radixark) is upstream's *source of truth* for tool versions, and `scripts/slurm/setup/extract_pins.py` scrapes specific values out of it into `pins.env`, which `install_env.sh` consumes.

Mapping:

| `pins.env` variable | Source | Owner |
|---|---|---|
| `TORCH_VERSION` | `thirdparty/sglang/python/pyproject.toml` | extracted (submodule) |
| `TE_VERSION`, `MBRIDGE_COMMIT`, `TMS_COMMIT`, `CUDNN_CU12_VERSION`, `FLASH_ATTN_INTERFACE_COMMIT`, `MILES_WHEELS_REPO` | `docker/Dockerfile` | extracted (Dockerfile) |
| `MOONCAKE_VERSION` | `thirdparty/sglang/docker/Dockerfile` | extracted (Dockerfile) |
| `MILES_SGLANG_SOURCE_VERSION` | preserved from pins.env | **sglang-sync only** |
| `MILES_WHEELS_TAG` | **preserved** from pins.env | **sglang-sync only** |
| `MILES_WHEELS_TORCH_VERSION`, `MILES_WHEELS_SGLANG_VERSION`, `SGLANG_ROUTER_VERSION` | `WHEELS_STACK[MILES_WHEELS_TAG]` | derived |
| `UPSTREAM_SGLANG_IMAGE_TAG`, `UPSTREAM_WHEELS_TAG` | `docker/Dockerfile` | extracted (target) |

After a miles sync, `extract_pins.py --check` detects drift; `--write` regenerates. Most fields are fully automated. The sglang stack is the exception — see the next section.

**What's NOT automated**: new `RUN pip install <package>` lines in upstream Dockerfile. If upstream adds a new dep that needs to be in our conda env, the corresponding `$UV` line must be added to `install_env.sh` by hand. `miles-sync` flags these via:

```bash
git diff ${MB}..HEAD -- docker/Dockerfile | grep -E '^[+-](RUN|ARG|ENV) '
```

…and asks the user to verify install_env.sh manually. Deliberate choice — auto-editing bash with regex-on-Dockerfile is brittle.

---

## The sglang stack: ACTIVE vs UPSTREAM_TARGET (the core model)

The mistake the 2026-05-29 sync surfaced: `MILES_WHEELS_TAG` was blindly mirrored from the upstream Dockerfile. The wheels release is a **torch-ABI bundle** — `cu129-x86_64-v0.5.12` ships flash-attn/apex compiled against torch 2.11's C++ ABI. Mirroring just the tag (while the `thirdparty/sglang` submodule stays on v0.5.10 / torch 2.9.1) produced `TORCH_VERSION=2.9.1 + MILES_WHEELS_TAG=…v0.5.12`, which a fresh `install_env.sh` would turn into an ABI-mismatched env (torch-2.11 flash-attn into torch-2.9.1 → `ImportError`/segfault), while the old CUDA-tag preflight (cu129==cu129) waved it through. The bundle's SGLang label is release metadata, not a requirement that source stay on that exact version; source may advance while reusing the bundle when torch remains compatible.

Fix — two views of the bundle, reconciled by sglang-sync:

- **ACTIVE** = what `install_env.sh` actually installs. `MILES_SGLANG_SOURCE_VERSION` records the `thirdparty/sglang` source line; `MILES_WHEELS_TAG` selects its torch-ABI wheel bundle, whose `torch`/release-`sglang`/`router` metadata is **derived** through `WHEELS_STACK`. The source and bundle labels may differ when torch matches. **Only sglang-sync advances these hand-owned pins.** miles-sync must never auto-bump them.
- **UPSTREAM_TARGET** = where `docker/Dockerfile` points (`UPSTREAM_SGLANG_IMAGE_TAG` / `UPSTREAM_WHEELS_TAG`). Recorded, never auto-applied. The destination sglang-sync advances ACTIVE to.

`UPSTREAM_SGLANG_IMAGE_TAG` is therefore the source-version **destination**, `MILES_SGLANG_SOURCE_VERSION` records the current source line, and **sglang-sync is the action that drives one to the other**. `UPSTREAM_WHEELS_TAG` separately describes upstream's ABI bundle choice.

### WHEELS_STACK mapping (single source of truth for "what a tag means")

Lives in `extract_pins.py`. Maps a wheels tag → `{sglang base tag, torch ABI, sglang_router version}`, read off the miles-wheels release's own name on GitHub. `--write` resolves the derived pins.env fields from it; `install_env.sh` reads the resolved values (no bash-side mapping). Add a row here as part of every sglang-sync.

```python
WHEELS_STACK = {
    "cu129-x86_64":         {"sglang": "v0.5.10", "torch": "2.9.1",  "router": "0.3.2"},
    "cu129-x86_64-v0.5.12": {"sglang": "v0.5.12", "torch": "2.11.0", "router": "0.3.2"},
}
```

### Three-category wheel taxonomy (what the ABI guard actually guards)

Not every artifact in the release is torch-ABI-bound. The fail-closed guard only covers category 1:

| Category | Members | torch-ABI-bound? | Handling |
|---|---|---|---|
| Prebuilt + links libtorch C++ ABI | `flash_attn`, `flash_attn_3`, `apex`, `fake_int4_quant_cuda` | **yes** | **fail-closed**: `MILES_WHEELS_TORCH_VERSION == TORCH_VERSION` |
| sglang-version-coupled, not torch-linked | `sglang_router` (Rust/abi3), sglang editable itself | no | pin version (`SGLANG_ROUTER_VERSION`), track sglang line; no torch guard |
| compiled-in-place / host exception | TE (`--no-build-isolation`, compiles vs local torch), `sgl-model-gateway` (Rust binary, GLIBC<2.38 skip) | no | TE always ABI-correct by construction; gateway is an explicit host exception |

(TE was wrongly lumped into "the atomic set" in an earlier draft — it isn't, because we compile it in place; the release's `transformer_engine-*-py3-none-any.whl` is for the Docker path, which we don't use.)

### Enforcement (defense in depth)

- `extract_pins.py` exit-code contract (shared by `--check` and `--write`):
  - **exit 0** = consistent, **or** only `[sglang-sync pending]` (`MILES_SGLANG_SOURCE_VERSION` differs from `UPSTREAM_SGLANG_IMAGE_TAG` — deferrable, must NOT block CI/install/miles-sync).
  - **exit 1** = drift (committed ≠ extracted; run `--write`) or pins.env missing.
  - **exit 2** = torch-ABI inconsistency (`WHEELS_STACK[ACTIVE].torch != TORCH_VERSION`) or unknown wheels tag. `--write` **refuses** (won't materialize a bad bundle). Distinct from 1 so miles-sync / CI can stop on danger instead of blindly regenerating.
- `install_env.sh`, independent of extract_pins (it sees real env state + runtime overrides):
  - **Re-derives** `MILES_WHEELS_TORCH_VERSION` / `MILES_WHEELS_SGLANG_VERSION` / `SGLANG_ROUTER_VERSION` from the *effective* `MILES_WHEELS_TAG` via `extract_pins.py --resolve` right after sourcing pins.env. This is the fix for the **runtime-override hole**: pins.env's baked derived fields are only a snapshot, and `MILES_WHEELS_TAG` is independently overridable (`MILES_WHEELS_TAG=… bash install_env.sh`); without re-derivation the guard would pass on stale torch while `_fetch_miles_wheel` pulls a mismatched-ABI wheel set. The tag is authoritative; WHEELS_STACK (in extract_pins.py) is the single mapping — no duplicated bash mapping.
  - Preflight torch-ABI guard: `MILES_WHEELS_TORCH_VERSION == TORCH_VERSION` (now using the re-derived value).
  - Post-submodule-init: source/bundle version differences are reported as a warning, while submodule pyproject torch == `TORCH_VERSION` remains mandatory.

Why not "hard lockstep" (block miles-sync until sglang-sync done)? Three reasons: sglang base bumps are frequent; the `sglang-miles` patch-stack rebase onto a new base is an **external dependency** (radixark must publish it first — we can't unilaterally do it); torch major jumps are heavyweight and deserve their own validation window. So source/target divergence is a **loud, deferrable pending**, not a blocker. Safety is held by the install-time torch-ABI fail-closed guard, not by blocking.

---

## What miles-sync does (high level)

1. Verify remotes (preserve `DISABLE_PUSH_TO_UPSTREAM`).
2. Pre-analysis via `/miles-upstream-prs merge-base`.
3. Create `sync-upstream-YYYYMMDD` branch.
4. `git merge upstream/main --no-edit`. On any conflict: STOP, surface to the user, suggest typical fix (for `thirdparty/*` paths: `git checkout --ours`), wait for instructions before staging anything.
5. Refresh install scripts:
   - Show upstream Dockerfile diff.
   - `extract_pins.py --check || --write` for pins.env.
   - Surface RUN/ARG/ENV diff for manual install_env.sh review.
   - Emit forward-compat banner if `thirdparty/sglang/python/pyproject.toml` changed (this hints at a needed sglang-sync).
6. Single commit for our local changes on top of merge.
7. Generate divergence diff vs upstream → `sync-records/miles-sync-YYYY-MM-DD/divergence.{patch,stat}`.
8. Draft PR body to `sync-records/miles-sync-YYYY-MM-DD/pr-body.md`. **STOP. Show user. Wait.**
9. **Only after explicit "push it":** `git push -u origin <branch>` + `gh pr create --body-file <draft>`.
10. Report PR URL + summary; remind about merge-commit mode.
11. Team-notification template.

Invariants:
- Merge mode (no squash/rebase) preserves upstream SHAs for future merge-base detection.
- Single commit for our changes keeps the PR reviewable.
- `git add` by name, never `.` — avoids unrelated untracked files.

---

## sglang-sync workflow (BUILT — `.claude/skills/sglang-sync/`)

### Topology (validated 2026-06-02, NOT what the earlier draft guessed)

The miles sglang line is the **`sglang-miles` branch hosted directly on
`sgl-project/sglang`** (the official sglang repo) — there is no radixark sglang
fork. So the chain is two-tier, not three:

```
sgl-project/sglang @ sglang-miles   ← the miles RL sglang line (patches on a release tag)
    ↓ (we mirror the branch tip)
impossible-inc/sglang @ sglang-miles  ← our fork (.gitmodules origin; private https)
    ↓ (tracked via gitlink)
miles-imp:thirdparty/sglang
```

Submodule remotes: `origin` = `impossible-inc/sglang`, `upstream` = `sgl-project/sglang`.
`impossible-inc/sglang` is private https, so non-interactive pushes use a
token-injected URL: `https://x-access-token:$GH_TOKEN@github.com/impossible-inc/sglang.git`.

Snapshot at build time: pin `v0.5.10-40-g4d795356c` (torch 2.9.1); upstream
`sgl-project/sglang@sglang-miles` already at `v0.5.12-23-g3102015ca` (torch
2.11.0), +1434 commits — exactly the line miles' Dockerfile UPSTREAM target
(`v0.5.12-cu129` / `cu129-x86_64-v0.5.12`) points at. So the bundle is ready to
sync together.

### Contract: advance ACTIVE to UPSTREAM_TARGET, in lockstep with miles

sglang-sync's whole job is to make `MILES_SGLANG_SOURCE_VERSION` match
`UPSTREAM_SGLANG_IMAGE_TAG` while selecting a torch-compatible wheels bundle:
fetch `upstream/sglang-miles`, re-apply local mirror patches on a rebased target,
publish the review branch, bump the gitlink and source pin, add a `WHEELS_STACK`
row only if a new bundle is needed, then run `extract_pins.py --write`.
`--check` must finish with no `[sglang-sync pending]`. The wheels tag need not
equal upstream's tag when CUDA deployment differs or a same-torch bundle is
intentionally reused. See the skill for exact mirror landing mechanics; the
install-time guards and `--check`/`--write` ABI refusal backstop every step.

### Default: sync TOGETHER with miles-sync

Decided 2026-06-02 (supersedes the earlier "its own skill, never folded in"
stance): because the bundle is atomic, **the default is to advance miles and
sglang in the same PR.** miles-sync's Step 5d, on detecting `[sglang-sync pending]`,
invokes `/sglang-sync <UPSTREAM_WHEELS_TAG>` on the same branch; its staged
changes fold into miles-sync's single "our changes" commit, and the outward pushes
(impossible-inc/sglang mirror FIRST, then the miles-imp PR) share miles-sync's
final approval gate. Running new miles code against an old sglang is the very
mismatch we're trying to avoid, so together-by-default is correct.

sglang-sync is still a **standalone** skill (sglang-only bumps, or hotfixes), and
**deferral is the fallback**: if `sgl-project/sglang@sglang-miles` hasn't rebased
to the version miles wants yet, or the miles-wheels release for the tag is missing,
sglang-sync stops, ACTIVE source stays put, `--check` keeps emitting `[sglang-sync pending]`,
and the install-time ABI guard keeps a fresh build safe until upstream is ready.

### Hard things specific to sglang

- **Patches over a release tag**: `sglang-miles` is patches on top of a sgl-project tag (v0.5.10 → v0.5.12). The rebase onto a new base is done UPSTREAM (on sgl-project's branch); we just fast-forward our mirror. If we ever carry *local* sglang patches on `impossible-inc/sglang`, the `--ff-only` advance fails → STOP and resolve (merge/rebase) by hand.
- **torch jump blast radius**: v0.5.12 pins torch 2.11.0 (vs our 2.9.1). A torch bump can break Megatron-LM / TE / apex / flash-attn — always a full `install_env.sh` rebuild + `verify_env.py` + smoke launch. This is the main reason it's a deliberate, approved step, not silent.
- **CUDA tag coupling**: `MILES_WHEELS_TAG` (cu129 vs cu130) must match `TORCH_INDEX_URL` / `FLASHINFER_INDEX_URL`; `install_env.sh` preflight guards tag agreement. A torch needing cu130 flips the whole wheel set.
- **mooncake-transfer-engine version**: from `thirdparty/sglang/docker/Dockerfile`; may also move on an sglang sync (`extract_pins.py --write` picks it up).

---

## Planned megatron-{lm,bridge}-sync (deferred)

Similar to sglang-sync but lower priority:

- We have **no local modifications** to Megatron-LM or Megatron-Bridge yet. The submodules are pure mirrors of `radixark/Megatron-LM` (`miles-main` branch) and `radixark/Megatron-Bridge` (`bridge` branch).
- When we do start modifying them, the workflow looks like sglang-sync above but with simpler patch story (no equivalent of "patches over sgl-project tags" — we just track radixark directly).
- For now, manual bumps via `python scripts/slurm/setup/track_submodules.py --bump <path>` are sufficient.

---

## Open questions for later

1. **CI integration**: should miles-sync produce a PR that triggers a CI run, and if CI fails, should the skill auto-amend (no — one-commit invariant) or surface failures back to the user?
2. **`verify_env.py` automation**: it needs a GPU salloc. Could detect `$SLURM_JOB_ID` and conditionally run, but cross-cluster portability hurts. Current decision: leave to operator via the PR's test-plan checklist.
3. **Multiple sync branches**: if a sync attempt is abandoned, should subsequent invocations clean up the stale `sync-upstream-YYYYMMDD` branch? Currently skill says "add numeric suffix" — fine for v1.
4. **Drift baseline**: ~~should we keep `divergence.patch` files around for trending?~~ RESOLVED 2026-07-21: full patches are NOT tracked (gitignored; they were most of the record bulk). They're derivable on demand — `git diff <merge-base>..<sync-tip>` with the SHAs from `pr-body.md` — since sync PRs merge with preserved upstream SHAs. Only `divergence.stat` ships, so folders stay small and no pruning is needed.

---

## See also

- [`/miles-sync`](../../../../.claude/skills/miles-sync/SKILL.md) — the sync skill.
- [`/miles-upstream-prs`](../../../../.claude/skills/miles-upstream-prs/SKILL.md) — the pre-analysis skill.
- [`scripts/slurm/setup/extract_pins.py`](../../setup/extract_pins.py) — Dockerfile → pins.env extraction.
- [`scripts/slurm/setup/track_submodules.py`](../../setup/track_submodules.py) — manual submodule bump helper.
- [`scripts/slurm/setup/install_env.sh`](../../setup/install_env.sh) — what consumes pins.env.
