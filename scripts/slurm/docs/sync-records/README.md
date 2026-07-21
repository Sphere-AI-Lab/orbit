# sync-records — upstream-sync history

Git-tracked records of every upstream sync of this fork. Each sync event gets one
folder here; the folder is committed **in the sync PR itself** so anyone running
`/miles-sync` or `/sglang-sync` later has the full history: what merged, what
conflicted, what broke during env validation, and how it was fixed.

Unlike `../debug-notes/` and `../plan-notes/` (gitignored scratch space), everything
under this directory is tracked — including raw install logs (`.gitignore` carries an
explicit un-ignore for this tree, so the global `*.log` rule does not apply here).

## Layout

```
sync-records/
  README.md                     ← this file
  upstream-sync-design.md       ← the ACTIVE vs UPSTREAM pin model + sglang topology
                                   (referenced by /miles-sync and /sglang-sync)
  miles-sync-YYYY-MM-DD/        ← one folder per miles upstream-sync event
  sglang-sync-YYYY-MM-DD/       ← standalone sglang bumps only; a combined
                                   miles+sglang sync records into the miles-sync dir
```

## Standard files in an event folder

Produced by the skills (`/miles-upstream-prs`, `/miles-sync`; `/sglang-sync` produces
only freeform files — mirror-commit classification, publish script, re-apply notes):

| file | producer | content |
|---|---|---|
| `prs.md` | `/miles-upstream-prs` | INPUT — pre-merge report: upstream PRs since last sync, watchlist hits |
| `divergence.patch` | `/miles-sync` step 7 | full drift vs upstream — generated locally as a review aid, **NOT committed** (gitignored; regenerate with `git diff <merge-base>..<sync-tip>`, SHAs in `pr-body.md`) |
| `divergence.stat` | `/miles-sync` step 7 | the `--stat` summary of the above |
| `pr-body.md` | `/miles-sync` step 8 | OUTPUT — the body of OUR sync PR, as merged |

Everything else is freeform: debug notes for issues hit during the sync
(`install-findings.md`, `*-env-test.md`, incident writeups), one-off scripts used for
the landing (`publish-sglang-mirror.sh`), probe logs, etc. Keep names descriptive and
dated — the next sync's operator reads these cold.

## Conventions

- **Record folders are committed in the sync PR** as a separate `[docs] sync record`
  commit on top of the single code/pins commit — the code commit stays clean.
- **Full divergence patches are not committed** — they were most of the record's
  bulk and are derivable from git (the sync PRs merge with preserved upstream SHAs,
  and `pr-body.md` records merge-base and tip). Only `divergence.stat` ships.
- **Files > 1000 KB must be gzipped** (`gzip -9`) if you do need to track one — the
  repo's `check-added-large-files` pre-commit hook caps files at 1000 KB (`--maxkb=1000`).
- **The divergence diff excludes this directory** (`':(exclude)scripts/slurm/docs/sync-records'`)
  — records describe drift, they aren't drift.
- Notes added *after* the sync PR merged (post-merge incidents, validation follow-ups)
  go into the same event folder via a normal docs PR.

## Reading order for a new sync

Operators starting a sync read **only the newest event folder** (plus this index).
Older records describe superseded states — old pins, fixed bugs, dead workarounds —
and reading them up front pollutes the context of the sync you're about to run.
Go older only for a targeted lookup the newest record or this index points you to.

## Index of past syncs

| event | outcome |
|---|---|
| `miles-sync-2026-05-29/` | First recorded sync. Merge landed; env work deferred. |
| `miles-sync-2026-06-02/` | The env-install saga: torch 2.11/cu129 wall (cu13-linked kernel wheels, TE build, router GLIBC). `install-findings.md` is the canonical writeup; sglang bump deferred, ACTIVE held at v0.5.10 bundle. `pr-body.md`/`prs.md` are the as-shipped versions (absorbed from the old tracked `docs/sync/` dir); the pre-wall drafts survive as `*.draft.md`. |
| `miles-sync-2026-06-30/` | The v0.5.13 line (miles-imp PR #18). Router spawn regression, WekaFS wedge incident, `/begin_weight_update` 404 root-cause, full cu129 bare-metal validation (`v0513-cu129-env-test.md`), sglang mirror publish (impossible-inc/sglang PR #2), pause-aware flush_cache restore. |
