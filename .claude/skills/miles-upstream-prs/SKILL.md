---
name: miles-upstream-prs
description: "Analyze merged PRs on radixark/miles (upstream) since miles-imp's last sync. Read-only — produces a markdown report flagging PRs that touch files we've locally modified, plus a first-class watchlist for pin-source files (docker/Dockerfile, requirements.txt, thirdparty/sglang/{Dockerfile,pyproject.toml}). Invoked as a pre-step by /miles-sync, or on its own to scan upstream activity. Accepts merge-base (default), a YYYY-MM-DD date, or a commit SHA as argument."
---

# miles-upstream-prs — upstream PR analyzer for miles → miles-imp

Read-only inspection of `radixark/miles` activity since `miles-imp` last synced. Produces a markdown report (cached under `scripts/slurm/docs/debug-notes/`) that `/miles-sync` consumes when drafting the sync PR body.

## When to use this skill

- As the **pre-step** invoked by `/miles-sync` (Step 2 there).
- On its own when the user wants to see "what's new upstream" without committing to a sync.

## Usage

```
/miles-upstream-prs                # since merge-base (recommended; auto-detects last sync)
/miles-upstream-prs merge-base     # explicit merge-base mode
/miles-upstream-prs YYYY-MM-DD     # since a specific date
/miles-upstream-prs <sha>          # since a specific upstream commit
```

## Topology

- `origin` → `git@github.com:impossible-inc/miles-imp.git` (our private fork)
- `upstream` → `git@github.com:radixark/miles.git` (the radixark/miles fork that we track)
- Upstream's push URL is intentionally set to the literal sentinel `DISABLE_PUSH_TO_UPSTREAM`. **Never rewrite it.**
- Unlike slime's 3-tier setup, there is no `upstream-fork` intermediary. `radixark/miles` is public; `gh` queries it directly with no fallback chain.

## Workflow

### Step 0 — Ensure remotes are configured

```bash
git remote get-url upstream >/dev/null 2>&1 \
    || git remote add upstream git@github.com:radixark/miles.git

# Verify the fetch URL — do NOT touch push URL (DISABLE_PUSH_TO_UPSTREAM is intentional).
fetch_url=$(git remote get-url upstream)
if [[ "$fetch_url" != "git@github.com:radixark/miles.git" ]] && \
   [[ "$fetch_url" != "https://github.com/radixark/miles.git" ]]; then
    git remote set-url upstream git@github.com:radixark/miles.git
fi
```

### Step 1 — Parse argument and determine mode

1. **No argument** or `merge-base` → commit-based, using `git merge-base HEAD upstream/main`. **Default.**
2. `YYYY-MM-DD` → date-based query (may include already-synced PRs).
3. Plain SHA → commit-based since that SHA.

### Step 2 — Fetch and detect

```bash
git fetch upstream main

MB=$(git merge-base HEAD upstream/main)
echo "merge-base: $MB ($(git log -1 --format='%h %ad %s' --date=short $MB))"

# Commits on upstream not yet in our HEAD, oldest first
git log --oneline --reverse ${MB}..upstream/main

# Extract PR numbers (commit subjects on radixark/miles consistently include #NNNN)
PR_NUMBERS=$(git log --oneline ${MB}..upstream/main \
    | grep -oE '#[0-9]+' | tr -d '#' | sort -un)
echo "PRs to analyze: $PR_NUMBERS"
```

### Step 3 — Compute the "our touched" file set

This is the **primary** signal — files we have modified relative to merge-base:

```bash
git diff --name-only ${MB}..HEAD > /tmp/miles-our-touched.txt
wc -l /tmp/miles-our-touched.txt
```

Use it to flag upstream PRs whose touched files intersect with ours.

### Step 4 — Always-watch list (independent of `OUR_TOUCHED`)

These files drive `scripts/slurm/setup/pins.env` and `install_env.sh`. Any upstream PR touching them must be flagged, even if we haven't modified them locally:

- `docker/Dockerfile` — primary pins source for `extract_pins.py`
- `requirements.txt`
- `thirdparty/sglang/docker/Dockerfile`
- `thirdparty/sglang/python/pyproject.toml` (`TORCH_VERSION` comes from here)

### Step 5 — Fetch PR metadata

For each PR number from Step 2:

```bash
gh pr view $N --repo radixark/miles \
    --json number,title,author,mergedAt,body,labels,files,additions,deletions
```

`radixark/miles` is public, so this works for everyone. No fallback chain needed.

### Step 6 — (Optional secondary signal) scan for `miles.*` imports

If anyone has spun out user-facing code that imports `miles.*` as a library:

```bash
python .claude/skills/miles-upstream-prs/scripts/scan_imports.py <path>
```

This is **secondary**. miles-imp itself IS miles, with modifications — we mostly modify `miles/` in place, so the diff-vs-merge-base signal in Step 3 covers the common case. The import scanner is here for forward-compat if anyone builds a downstream consumer.

### Step 7 — Generate report

Write to `scripts/slurm/docs/debug-notes/miles-sync-YYYY-MM-DD/prs.md` (gitignored). Create the folder if it doesn't exist — this is the same folder `/miles-sync` writes `pr-body.md` and `divergence.{patch,stat}` to for the same date, so a sync event's artifacts stay grouped. When running standalone (not as part of a sync), `prs.md` may be the only file in the folder; that's fine.

Report structure:

```markdown
# miles upstream PRs report

**Period**: since `<merge-base SHA>` (`<merge-base date>`)
**Total PRs**: N
**Flagged (touch files we've modified)**: F
**Watchlist hits (touch pin-source files)**: W

---

## ⚠️ Watchlist hits (pin sources)

For each PR touching `docker/Dockerfile` / `requirements.txt` / sglang pyproject:

### PR #1164 — Bump sglang to v0.5.12
**Merged**: 2026-05-22 | **Author**: alice | **Files**: `docker/Dockerfile` (4+/4-)

**Summary**: <2-3 sentence AI summary>

**Pin impact**:
- `docker/Dockerfile`: `transformer_engine[pytorch]==2.10.0` → `2.11.0` ⇒ rerun `extract_pins.py --write` ⇒ `TE_VERSION` in pins.env will change.
- New `RUN pip install ...` line at L142: needs to be mirrored into `install_env.sh` if it's not already there.

---

## PRs that touch files we've modified

For each flagged PR, show overlapping files and a brief impact note.

---

## Other PRs (no overlap detected)

| PR | Title | Merged | One-line summary |
|----|-------|--------|------------------|
| #1180 | ci: display improvement | 2026-05-21 | CI-only change, no runtime impact |
```

### Step 8 — Show the user

Print the report path, summary stats, and the watchlist section to stdout so the user can decide whether to proceed with `/miles-sync`.

## Notes

- The skill is **read-only**: never branches, never commits, never pushes.
- Reports are local-only (gitignored). If you want to share one, copy it manually.
- Date-based mode is provided for "what's been happening upstream lately?" exploratory use, but `merge-base` mode is what `/miles-sync` uses and what you should default to.

## See also

- [`/miles-sync`](../miles-sync/SKILL.md) — invokes this skill as its Step 2.
- [`scripts/slurm/setup/extract_pins.py`](../../../scripts/slurm/setup/extract_pins.py) — what consumes Dockerfile changes.
- [`scripts/slurm/docs/debug-notes/upstream-sync-design.md`](../../../scripts/slurm/docs/debug-notes/upstream-sync-design.md) — design rationale, sglang-sync forward plan.
