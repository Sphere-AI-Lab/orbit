---
name: miles-sync
description: "Sync miles-imp with upstream radixark/miles. Creates sync-upstream-YYYYMMDD branch, merges upstream/main preserving original SHAs, refreshes pins.env from upstream Dockerfile via extract_pins.py, surfaces RUN-line diffs for human review against install_env.sh, folds local adjustments into a single follow-up commit, drafts a PR body — and STOPS. When upstream bumps the sglang line, drives /sglang-sync to advance it in the SAME PR (sync-together default; defer fallback). Hard rules: (1) STOP at any merge conflict and let the user inspect before resolving anything; (2) push and PR happen ONLY after explicit user approval."
---

# miles-sync — upstream → miles-imp sync workflow

Sync `impossible-inc/miles-imp` with `radixark/miles`. Built on top of `/miles-upstream-prs` for the pre-analysis. Produces a draft PR locally; **never pushes or opens a PR without explicit user approval**.

## ⚠️⚠️⚠️ HARD RULES ⚠️⚠️⚠️

1. **STOP at any merge conflict.** Do NOT auto-resolve anything — not even `thirdparty/*` gitlink conflicts that look "obviously ours." Surface the conflict list to the user, suggest the typical fix (see Step 4), and wait for instructions. The user always looks first.
2. **No `git push` until the user says push.** "Looks good" / silence / acknowledgment ≠ approval. Wait for an explicit instruction like "push it", "submit the PR", "ok send it."
3. **Merge mode MUST be "Create a merge commit"** when the PR is eventually merged on GitHub. NOT squash, NOT rebase. Upstream SHAs must survive — future `merge-base` detection depends on it.
4. **One commit for our changes** on top of the merge commit. No multiple commits for local adjustments — bundle pins.env regen + install_env.sh tweaks + conflict fixes into a single commit. (The sync-record folder is the one exception: it goes in its own `[docs] sync record` commit at Step 9, so the code commit stays clean.)
5. **Stage conflicts by name**, never `git add .` — the user often has unrelated untracked files (e.g. `examples/vagen/docs/plan-notes/`).

## Topology

- `origin` → `git@github.com:impossible-inc/miles-imp.git` (where the sync PR opens)
- `upstream` → `git@github.com:radixark/miles.git` (fetch only; push URL is `DISABLE_PUSH_TO_UPSTREAM` — never overwrite)
- Submodules under `thirdparty/{Megatron-LM,sglang,Megatron-Bridge}` exist **only in miles-imp**. Upstream syncs never advance their gitlinks.

## Workflow

### Step 0 — Ensure remotes are configured

```bash
git remote get-url upstream >/dev/null 2>&1 \
    || git remote add upstream git@github.com:radixark/miles.git

# Verify fetch URL only. Do NOT touch push URL (DISABLE_PUSH_TO_UPSTREAM is intentional).
fetch_url=$(git remote get-url upstream)
if [[ "$fetch_url" != "git@github.com:radixark/miles.git" ]] && \
   [[ "$fetch_url" != "https://github.com/radixark/miles.git" ]]; then
    git remote set-url upstream git@github.com:radixark/miles.git
fi
```

### Step 1 — Ensure clean working directory

```bash
git status --porcelain
```

If there are uncommitted/untracked changes that aren't pre-existing junk (e.g. `examples/vagen/docs/plan-notes/`), ask the user to commit or stash before proceeding.

### Step 2 — Pre-analysis

**First, read the history — but ONLY the latest record**: skim
`scripts/slurm/docs/sync-records/README.md` (the index) and the single newest
`miles-sync-*/` folder (at minimum its `pr-body.md` and any `*-findings.md` /
`*-env-test.md`). That's where the still-live context is: recurring conflict spots,
install regressions, deferred items. Do NOT read older event folders up front —
they describe superseded states (old pins, fixed bugs, dead workarounds) and
pollute the sync context. Dig into an older record only when the newest one or the
README index explicitly points there for a problem you're actually hitting.

Then invoke `/miles-upstream-prs merge-base`. This:
- Computes `MB=$(git merge-base HEAD upstream/main)`.
- Writes the report to `scripts/slurm/docs/sync-records/miles-sync-YYYY-MM-DD/prs.md`.
- Highlights watchlist hits (pin-source files) and PRs touching files we've modified.

All artifacts for this sync event live under one folder: `scripts/slurm/docs/sync-records/miles-sync-YYYY-MM-DD/` containing `prs.md`, `divergence.{patch,stat}` (Step 7), and `pr-body.md` (Step 8) — plus freeform notes for anything debugged along the way. The folder is **git-tracked** and ships in this sync's PR (Step 9), so the record survives for the next operator.

Show the user the report summary (total PRs, flagged PRs, watchlist hits) and **ask if they want to proceed**. Stop here if they don't.

### Step 3 — Create sync branch

```bash
BRANCH_NAME="sync-upstream-$(date +%Y%m%d)"
# If branch exists, add a numeric suffix: sync-upstream-YYYYMMDD-2, -3, ...
git checkout -b $BRANCH_NAME
```

### Step 4 — Merge upstream

```bash
git merge upstream/main --no-edit
```

Merge preserves each upstream commit's original SHA — they appear individually in `git log`, alongside the merge commit at the tip.

**On conflict — STOP. Do not auto-resolve. Let the user look first.**

1. List the conflicted files:
   ```bash
   git diff --name-only --diff-filter=U
   ```
2. Categorize and surface to the user — show the list, then for each path note the typical fix as a suggestion (NOT an action):
   - **`thirdparty/<name>` paths**: typical fix is `git checkout --ours -- thirdparty/<name> && git add thirdparty/<name>`. Rationale: upstream doesn't track these as submodules; the "conflict" is structural noise. But confirm with the user — they may want to inspect the conflicted gitlink first.
   - **Shared source files** (`miles/*`, `tests/*`, `docker/Dockerfile`, etc.): show the conflict markers (`git diff <file>` after the failed merge) and ask the user how to proceed.
   - **Files we own exclusively** (`scripts/slurm/*`, `.claude/skills/*`, `examples/vagen/*`): shouldn't conflict in practice, but if one does, definitely pause.
3. **Wait for the user's instruction before staging anything.** Once they approve a resolution path, apply it, then stage by name (NOT `git add .`), then `git commit --no-edit`.

### Step 5 — Refresh install scripts from upstream Dockerfile

This is the miles-imp-specific replacement for slime's "update Docker base image" step. We don't run Docker, but we DO derive `pins.env` and `install_env.sh` from upstream's `docker/Dockerfile`. After a merge, the Dockerfile may have new values or new RUN lines that need to flow into our install scripts.

#### 5a. Surface the upstream pin-source diff

```bash
MB=$(git merge-base HEAD@{1} upstream/main)   # @{1} = pre-merge HEAD; or save MB from Step 2
echo "=== upstream pin-source changes between merge-base and tip ==="
git diff ${MB}..HEAD -- \
    docker/Dockerfile \
    requirements.txt \
    thirdparty/sglang/docker/Dockerfile \
    thirdparty/sglang/python/pyproject.toml
```

Show the user this diff verbatim.

#### 5b. Regenerate pins.env

Branch on the exit code — do NOT use a blind `--check || --write`, because exit 1
(drift, safe to regenerate) and exit 2 (torch-ABI danger, must NOT regenerate) are
distinct:

```bash
python scripts/slurm/setup/extract_pins.py --check; rc=$?
case $rc in
  0) : ;;  # consistent, or only [sglang-sync pending] — fine, continue
  1) python scripts/slurm/setup/extract_pins.py --write ;;  # drift — refresh extracted + UPSTREAM_* fields
  2) echo "STOP: torch-ABI inconsistency — do NOT --write. Surface to user."; exit 1 ;;
esac
```

`extract_pins.py` will **never** auto-bump the ACTIVE
`MILES_SGLANG_SOURCE_VERSION` or `MILES_WHEELS_TAG`, only refresh the
purely-extracted pins (TE, mbridge, cudnn, mooncake, …) and the `UPSTREAM_*`
target fields. So a normal sync that bumps upstream's sglang base will hit exit
1 (drift in `UPSTREAM_*`); `--write` then leaves ACTIVE untouched, moves
`UPSTREAM_WHEELS_TAG`/`UPSTREAM_SGLANG_IMAGE_TAG` forward, and prints
`[sglang-sync pending]`. Exit-code contract:

- **exit 0** = consistent, **or** only `[sglang-sync pending]` (ACTIVE source differs from the UPSTREAM image target — deferrable, NOT a blocker).
- **exit 1** = drift (run `--write`) or pins.env missing.
- **exit 2** = torch-ABI inconsistency / unknown wheels tag (DANGER). `--write` also refuses this. On a normal miles-sync it should never happen (ACTIVE stays put); if it does, STOP and surface to the user — the submodule/tag are out of sync and need a real sglang-sync, not a regenerate.

If `pins.env` changed, `git diff scripts/slurm/setup/pins.env` shows what moved (usually just `UPSTREAM_*` + the small scalar pins).

#### 5c. Flag new RUN/ARG/ENV lines for human review

```bash
git diff ${MB}..HEAD -- docker/Dockerfile \
    | grep -E '^[+-](RUN|ARG|ENV) '
```

This is a focusing aid — **the skill does NOT auto-edit install_env.sh**. Present the matching lines to the user with the prompt:

> Upstream `docker/Dockerfile` added/removed these install/ARG/ENV lines. Check whether `scripts/slurm/setup/install_env.sh` still mirrors them, and edit it manually if needed.

#### 5d. sglang gate — sync together (default), or defer

The sglang source and its torch-compatible prebuilt wheels form one deployment
unit, although the wheel release's SGLang label may lag the source when torch
matches. When upstream's target leads ACTIVE, the **default is to advance them
together in this same PR** — the miles code you just merged was written against
the newer sglang, so shipping the two together avoids running new miles code on
an old source line. Deferring is the fallback, only when the new line isn't
ready.

After 5b, detect the move:

```bash
# extract_pins.py prints [sglang-sync pending] when ACTIVE != UPSTREAM.
python scripts/slurm/setup/extract_pins.py --check 2>&1 | grep -F '[sglang-sync pending]' && PENDING=1 || PENDING=0
```

If `PENDING=1`, tell the user what upstream now wants (`UPSTREAM_SGLANG_IMAGE_TAG`
/ `UPSTREAM_WHEELS_TAG` + the torch jump) and **recommend syncing together**.

**Default — sync together:** invoke **`/sglang-sync <UPSTREAM_WHEELS_TAG>`** now.
It runs on the same sync branch: fetches `sgl-project/sglang@sglang-miles`,
fast-forwards our mirror, bumps the `thirdparty/sglang` gitlink, adds the
`WHEELS_STACK` row if needed, sets the ACTIVE source to the image target, selects
a torch-compatible wheels bundle, and `--write`s — leaving `extract_pins.py
--check` at exit 0 with **no** pending. Its staged changes
(`thirdparty/sglang` + `pins.env` + `extract_pins.py`) fold into this sync's
single "our changes" commit (Step 6). The combined PR then carries the miles
merge + the sglang bump as one bundle. Its outward pushes (the
`impossible-inc/sglang` mirror push, then the miles-imp PR) wait for the same
final approval as Step 9 — push the sglang mirror FIRST so the gitlink resolves.

**Fallback — defer** (only if `/sglang-sync` reports the line isn't ready: the
miles-wheels release for the target tag is missing, or `sgl-project/sglang@sglang-miles`
hasn't rebased to the version miles wants yet):

1. Do NOT bump ACTIVE or the submodule. Leave the consistent ACTIVE source and
   wheels bundle.
2. `extract_pins.py --write` already refreshed `UPSTREAM_*`; commit that delta in Step 6.
3. Add a prominent PR-body item:

   > ⚠️ **sglang-sync deferred** — upstream wants sglang `<UPSTREAM_SGLANG_IMAGE_TAG>`
   > / `<UPSTREAM_WHEELS_TAG>` (torch `<new>`) but the line isn't ready yet.
   > ACTIVE source held at `<MILES_SGLANG_SOURCE_VERSION>` with wheels
   > `<MILES_WHEELS_TAG>` (torch `<current>`); run `/sglang-sync` once
   > upstream publishes. install_env.sh fails closed on any ABI mismatch meanwhile.

See `scripts/slurm/docs/sync-records/upstream-sync-design.md` for the ACTIVE vs
UPSTREAM_TARGET model and the `sglang-sync` contract ("advance ACTIVE to
UPSTREAM_TARGET").

### Step 6 — Single commit for our local changes

Stage ONLY the files you intentionally modified in Steps 4–5:

```bash
git add scripts/slurm/setup/pins.env           # if regen ran
git add scripts/slurm/setup/install_env.sh     # if you edited it
# + any conflict resolutions on shared files
git commit -m "[chore] refresh pins.env + install_env.sh for upstream sync"
```

Do NOT use `git add .` and do NOT create multiple commits. The PR should look like:

```
[docs] miles-sync <date> record               ← Step 9 (the sanctioned exception in HARD RULE 4)
<our single commit>                           ← all local adjustments
Merge upstream/main                           ← merge commit (preserves SHAs)
<upstream commit N>                           ← original SHA preserved
...
<upstream commit 1>                           ← original SHA preserved
<previous main HEAD>
```

If Step 5 didn't produce any pins/install changes and no conflicts needed fixing, **skip this step** — no empty commits.

### Step 7 — Generate divergence diff

```bash
SYNC_DATE=$(date +%Y-%m-%d)   # dashed event date — distinct from Step 3's compact branch date
EVENT_DIR=scripts/slurm/docs/sync-records/miles-sync-${SYNC_DATE}
mkdir -p "$EVENT_DIR"
# Exclude sync-records itself: records describe drift, they aren't drift.
git diff upstream/main -- ./ ':(exclude)scripts/slurm/docs/sync-records' > "$EVENT_DIR/divergence.patch"
git diff --stat upstream/main -- ./ ':(exclude)scripts/slurm/docs/sync-records' > "$EVENT_DIR/divergence.stat"
```

`divergence.patch` is a LOCAL review aid — gitignored, never committed (it would be
most of the record's bulk, and it's derivable later: `git diff <merge-base>..<sync-tip>`
with the SHAs from `pr-body.md`). Only the small `divergence.stat` ships in the record
(Step 9's `git add` picks up exactly that automatically).

Show the `--stat` to the user — this is our "drift surface" against upstream after the sync. Note: `$EVENT_DIR` should already exist from Step 2 (where `/miles-upstream-prs` wrote `prs.md` to the same folder); `mkdir -p` is defensive.

### Step 8 — Draft PR body (do NOT push yet)

Write the PR body to `$EVENT_DIR/pr-body.md` (i.e. `scripts/slurm/docs/sync-records/miles-sync-${SYNC_DATE}/pr-body.md`). Pull content from:
- The `/miles-upstream-prs` report at `$EVENT_DIR/prs.md` (Step 2) for the upstream PR list.
- The `--stat` at `$EVENT_DIR/divergence.stat` (Step 7) for the divergence section.
- Any noteworthy pins/install changes from Step 5.

When linking inside the body, use **relative paths** (e.g. `[prs.md](prs.md)`, `[divergence.stat](divergence.stat)`) — files are siblings in the same folder. Do NOT link `divergence.patch` (untracked, local-only).

Template:

```markdown
## Summary

Sync with upstream `radixark/miles` — N upstream commits merged.

## Upstream PRs merged

<paste the /miles-upstream-prs report — full list, sorted by PR number asc>

## Pin / install script changes

- `pins.env`: <list which pinned versions moved, e.g. TE_VERSION 2.10.0 → 2.11.0>
- `install_env.sh`: <list any RUN-line mirror edits you made; or "no changes" if none>

## ⚠️ Attention items

<from the watchlist section of the report — breaking changes, new deps, etc.>

## Divergence from upstream after sync

<paste the git diff --stat upstream/main from Step 7>

## Test plan

- [ ] Run `bash scripts/slurm/setup/install_env.sh` in a fresh GPU salloc.
- [ ] Run `python scripts/slurm/setup/verify_env.py` and confirm all checks pass.
- [ ] Sanity-launch a recipe (`bash scripts/slurm/submit.sh scripts/experiments/qwen3-4B-disagg-1node.sh`) and let it reach the first eval.

⚠️ **Merge mode**: this PR MUST be merged via "Create a merge commit". Squash or rebase will break future `merge-base` detection.
```

**Show the draft body to the user**. Tell them:
> Draft PR body written to `$EVENT_DIR/pr-body.md`. Branch `${BRANCH_NAME}` is committed locally. **Not pushed.** Reply "push it" to push and open the PR.

### Step 9 — Push & PR (ONLY after explicit user approval)

⛔ **DO NOT proceed past this line without an explicit user instruction to push.** Acknowledgments, "looks good," or silence do NOT count.

When approved, first commit the sync record so it ships in the PR (its own commit —
the code commit from Step 6 stays clean). Re-derive the event folder from disk: a
delayed approval can land in a fresh shell without Step 7's variables, and "today"
may no longer be the event date.

```bash
EVENT_DIR=$(ls -d scripts/slurm/docs/sync-records/miles-sync-* | sort | tail -1)
SYNC_DATE=${EVENT_DIR##*miles-sync-}
git add "$EVENT_DIR"
git commit -m "[docs] miles-sync ${SYNC_DATE} record"
```

Then push and open the PR:

```bash
git push -u origin $BRANCH_NAME

gh pr create --repo impossible-inc/miles-imp \
    --base main \
    --title "[sync] upstream radixark/miles $(date +%Y-%m-%d)" \
    --body-file "$EVENT_DIR/pr-body.md"
```

The `--body-file` form uses the exact draft the user just approved — no last-minute edits, no placeholders.

#### Review/CI repairs after publication

HARD RULE 4 governs the branch prepared before the first push. Once the PR is
public, do not rewrite its history merely to preserve the single-commit shape.
For a real issue found by CI or review:

1. Make the smallest focused follow-up commit and run the relevant validation.
2. Update the checked-in `pr-body.md` change table and validation claims so
   every follow-up is represented.
3. Regenerate `divergence.stat` from the recorded upstream tip, excluding the
   sync-record directory as in Step 7.
4. Push normally and update the live PR body from `pr-body.md`.

Record that the public review-repair protocol was used. Do not claim the
single-commit invariant still holds, and do not force-push unless the user
explicitly requests a history rewrite.

### Step 10 — Report

Print:
- The PR URL.
- Number of upstream commits merged.
- Conflicts resolved (and how — list each `thirdparty/*` auto-resolution).
- Watchlist hits and what they implied for install scripts.
- ⚠️ Reminder: merge via "Create a merge commit" mode.

### Step 11 — Prepare team notification (after PR is merged)

Compose a short message for the user to share. Template:

```
🔄 miles upstream sync (<date>) — <N> upstream commits merged (#<sync PR>)

- <change group 1 summary> (#<upstream PRs>).
- <change group 2 summary> (#<upstream PRs>).

⚠️ Attention items (if any):
- <breaking changes, dep bumps, new requirements>

Links:
- impossible-inc/miles-imp#<sync PR>: <URL>
- radixark/miles#<PR 1>: https://github.com/radixark/miles/pull/<N>
- radixark/miles#<PR 2>: https://github.com/radixark/miles/pull/<N>
- ... (sorted by PR number ascending)
```

Show this to the user — they share it manually. The skill does not post.

## See also

- [`/miles-upstream-prs`](../miles-upstream-prs/SKILL.md) — Step 2's pre-analysis.
- [`scripts/slurm/setup/extract_pins.py`](../../../scripts/slurm/setup/extract_pins.py) — pin regen.
- [`scripts/slurm/setup/install_env.sh`](../../../scripts/slurm/setup/install_env.sh) — mirrors upstream Dockerfile RUN lines.
- [`scripts/slurm/docs/sync-records/upstream-sync-design.md`](../../../scripts/slurm/docs/sync-records/upstream-sync-design.md) — design rationale, sglang-sync forward plan.
- [`scripts/slurm/docs/sync-records/README.md`](../../../scripts/slurm/docs/sync-records/README.md) — the tracked sync-history layout + index of past syncs.
