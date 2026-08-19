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

## 2026-08-18 sync gate

| run | wraps | status |
|---|---|---|
| `sync20260818-opd-geo3k-mm-mt-fullyasync-200step` | `OPD/multimodal/baseline/baseline-geo3k-multimodal-multiturn-fully-async-200step.sh` | pending |
| `sync20260818-rl-geo3k-mt-fullyasync-prefetch2-3node` | `async/geo3k-vlm-multi-turn-fully-async-prefetch2-3node.sh` | pending |
| `r3moe` slot | — | SKIPPED this round: the R3 route-plane work (int16/binary, MOE recipes) lives on `feature/moe_multimodal`; the sync branch carries only upstream's int32 base R3. Gate it when that branch rebases onto the synced main. |
