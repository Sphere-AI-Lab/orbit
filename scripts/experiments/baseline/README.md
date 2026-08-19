# baseline — pre-merge regression gate runs

Curated wrappers over the standing recipes, run BEFORE a sync/feature PR merges
to main. A sync PR merges only after these curves show no regression against
the previous entries in the same wandb project.

## Conventions

- **Run name**: `sync<YYYYMMDD>-<content>-<task>-<setting>` — content is one of
  `opd` / `r3moe` / `rl`; the date is the sync event being gated.
- **wandb**: every run lands in **`M3TRL/baseline`**
  (`api.run("M3TRL/baseline/<run_id>")`), grouped by its run name.
- Wrappers only pin naming/destination and then source the standing recipe —
  the workload definition stays single-sourced and cannot drift.
- Compare on the stable window against the prior baseline entries (see
  `scripts/slurm/docs/sync-records/`'s per-sync notes for reference run ids).

## Observability notes (2026-08-18 onward)

- **`rollout/rollout_train_kl/{k1,k2,k3}/{mean,min,max}`** (rollout panel):
  training/inference mismatch — q = rollout engine, p = trainer recompute.
  Emitted on ordinary RL runs (no reference model needed); k-estimator family
  with true global extrema. Cross-validates the coarser train-panel
  `train/{train,current}_rollout_kl` (k3 mean only). New curves — no
  historical baseline entries carry them yet.
- The fully-async health metrics use upstream names since this sync:
  `rollout/fully_async/{queue_size, aborted_groups_filtered,
  stale_groups_filtered, avg_staleness, max_staleness, buffer_*_staleness}`
  (old fork names `fully_async/accepted_staleness/*` and `over_cap_ratio` are
  gone — map by semantics when comparing against pre-sync runs).

## Fully-async knob migration (2026-08-18 sync)

The class-based rewrite (#1716/#1717/#2522) changed how the standing knobs are
expressed. Same-semantics cheat sheet:

| pre-sync (legacy worker) | post-sync | same semantics |
|---|---|---|
| `--rollout-function-path examples.fully_async...generate_rollout_fully_async` | `--fully-async` (requires an empty function path) | recipes migrated |
| `--fully-async-prefetch-batches N` (window = batch x N groups) | kept: derives `--async-max-concurrent-samples = batch x N x n_samples_per_prompt`; mutually exclusive with the explicit knob; N=1 == upstream default | keep the knob |
| `--max-weight-staleness N` (+ fork fail-closed filter, router-polled version) | same flag; filtering moved into the data buffer, current version passed from the trainer (#2244), fail-closed via the default `FailClosedDataBuffer` | keep the flag |
| aborted/stale groups always recycled (hardcoded) | `--async-unused-samples-handler {retry,drop}` — **defaults to `drop`** | **pass `retry` explicitly** (both gate recipes do since `f3aefa8e`; any other fully-async recipe with a staleness bound needs the same treatment) |
| `--fully-async-max-completed-queue-groups` (soft cap: stop launching) | inert (warns); `--async-data-buffer-capacity-factor F` bounds the finished buffer at F x batch with a **blocking put** (default 2.0) | accept the new mechanism |
| (fixed group window) | `--rollout-submission-granularity {group,sample}`; fully-async defaults to `sample` (per-sample backfill, better saturation) | accept `sample` (an upgrade); `group` reproduces the old windowing |
| eval unsupported (raised) | shared-engine pause-the-world or a dedicated eval fleet (#1740) | free upgrade |

Gate-run history note: attempts 1-3 died on infra/bad nodes plus a shim env
bug; attempt 4 ran with an inert prefetch knob and the fail-open buffer;
attempt 5 ran without retry semantics. **Attempt 6 (jobs 42956/42957,
`f3aefa8e`) is the first fully comparable pair** — earlier curves in
M3TRL/baseline from 2026-08-19 should be ignored.

## 2026-08-18 sync gate

| run | wraps | status |
|---|---|---|
| `sync20260818-opd-geo3k-mm-mt-fullyasync-200step` | `OPD/multimodal/baseline/baseline-geo3k-multimodal-multiturn-fully-async-200step.sh` | pending |
| `sync20260818-rl-geo3k-mt-fullyasync-prefetch2-3node` | `async/geo3k-vlm-multi-turn-fully-async-prefetch2-3node.sh` | pending |
| `r3moe` slot | — | SKIPPED this round: the R3 route-plane work (int16/binary, MOE recipes) lives on `feature/moe_multimodal`; the sync branch carries only upstream's int32 base R3. Gate it when that branch rebases onto the synced main. |
