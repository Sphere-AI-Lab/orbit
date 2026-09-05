# Dataset pipeline

Covers `build_env_dataset.py` (offline builder) and `data_source.py` (in-memory
loader). The dataset is the single source of truth — same `samples.jsonl`
feeds both training and orbit eval.

## Row schema

`build_env_dataset` emits one row per `(env, seed)`:

```
{
  "input":  "vagen_placeholder",
  "images": [],
  "metadata": {
    "vagen": {
      "env_name":                 "Sokoban",
      "seed":                     10001,
      "config":                   {...},
      "max_turns":                5,
      "response_length_per_turn": 512,
      "env_uuid":                 "md5 of the saved PNG bytes",
      "image_path":               "images/seed_00010001_<spec>.png",
      "split":                    "train" | "eval" | "eval_heldout",
      "heldout":                  true if --exclude-data was used,
      "source_format":            "samples_jsonl",
      "drift_check_required":     true for vision rows
    }
  }
}
```

`input` and `images` satisfy orbit' default eval `Dataset` loader keys
(`--input-key=input`, `--multimodal-keys=images`); both are overridden inside
`rollout.generate` after `env.reset`, so their stored values are placeholders.

## Train vs eval consumption

- **Train**: `VagenEnvSpecDataSource` reads `samples.jsonl` and lifts each
  row's `metadata.vagen` straight onto a `Sample`. `n_samples_per_prompt`
  deepcopies form a GRPO group sharing the same `(env, seed)`, so the group
  baseline measures rollout variance, not env-difficulty variance.
- **Eval**: orbit eval loads the jsonl via `--eval-prompt-data`. Same
  `rollout.generate`.

Both paths use the LIVE `env.reset` render as the turn-0 input; the saved
PNG is only used for drift detection (see below).

## Two input modes for the data source

`VagenEnvSpecDataSource._load_samples` dispatches on suffix:

- `*.jsonl` → prebuilt artifact (default for sokoban-main / frozenlake-main).
- anything else → treated as an EnvSpec yaml, materialized in-process via
  VAGEN's `_generate_seeds_for_spec`. The yaml path preserves the original
  VAGEN-MVP semantics; it does NOT attach `env_uuid`, so rollout skips drift
  detection for these rows.

The data source bypasses `RolloutDataSource.__init__` because `prompt_data`
here is jsonl/yaml, not the parquet prompt table the base class expects.
`args.rollout_global_dataset` must stay `True` — multiple orbit paths assert
it (`inference_rollout_train.py`, `sglang_rollout.py`, `ray/rollout.py`,
`ray/placement_group.py`).

## Drift detection (not avoidance)

Each vision row stamps `env_uuid = md5(PNG bytes of turn-0 obs)`.
`rollout.generate` hashes the LIVE `env.reset` render and asserts equality —
any divergence (VAGEN env code change, PIL bump, encoder bump) fails the run
instead of silently training on a drifted dataset.

This *detects* drift; it does not *avoid* it. We still rely on
`env.reset(seed)` being deterministic given current VAGEN code (the
`seeding.py` thread-lock fix + `_stable_next_seed` retry orbit in the patched
Sokoban env). Full decoupling would require saving the env's underlying state
(`room_fixed` / `room_state` / `player_position` / …) and restoring it
instead of calling `env.reset` — out of scope.

`env_uuid` is REQUIRED for any vision row loaded from jsonl. A vision row
missing it means a malformed row got past the loader, which would silently
disable this safety property — the loader fails loudly instead.

## Map-heldout splits via `--exclude-data`

Sokoban's small state space (`dim_room=6x6` / `num_boxes=1` /
`min_solution_steps in [1,5]`) plus the acceptance-test retry orbit makes
`seed -> final_map` many-to-one. Empirically ~76% of 256-seed val maps in r6
also appeared in the 10k-seed train pool. FrozenLake has the same problem
via its BFS-reachability acceptance test.

To get a true map-heldout eval set:

1. Build `train` and an `eval_pool` yaml (a LARGER candidate pool than the
   target N, since most candidates will be filtered).
2. Re-run `build_env_dataset` on the pool yaml with
   `--exclude-data <train samples.jsonl> --target-kept N`.
3. Resulting `samples.jsonl` is map-disjoint from train.

`--target-kept N` means "exactly N rows or fail": the build short-circuits
once N rows survive the exclude filter, and fails if fewer than N survive
after walking the whole pool. Size the candidate yaml so N is comfortably
reachable after filtering. `--dedup-within` further drops candidates whose
`env_uuid` duplicates one already kept by THIS build, promoting "N rows" to
"N unique maps" — recipes set it for eval splits where measurement quality
matters.

## Output layout & idempotence

```
data/<dataset>/<split>/
  samples.jsonl
  images/seed_<NNNNNNNN>_<spec>.png
  dataset_meta.json    # provenance for idempotence checks
```

The meta stamps `yaml_md5 + base_seed + exclude_data_md5 + target_kept +
dedup_within + split`. A re-run with matching inputs short-circuits unless
`--force` (or `FORCE=1`) is passed. Build streams to `samples.jsonl.tmp` and
only `os.replace`s to the final filename after the `target_kept` floor
check — the launcher's `-s samples.jsonl` test cannot see a partial build.
A forced rebuild wipes stale PNGs so `images/` stays in lockstep with
`samples.jsonl`.

## Determinism prerequisites

VAGEN's Sokoban env uses a global-RNG `set_seed` context. For stable output:

- `vagen/envs/sokoban/utils/seeding.py` needs the thread-lock fix (avoids
  intra-process races; `build_env_dataset` only calls sequentially so this
  is belt-and-suspenders for it, but rollout hits the same code).
- `_stable_next_seed` in `patch_sokoban_env.py` must replace the
  PYTHONHASHSEED-dependent retry-orbit fallback.

`--base-seed 0` matches VAGEN's default (`config.get("base_seed", 0)`); the
recipe must pass `--seed 0` to orbit so train's seed expansion matches.

## CLI

```
env -u LD_LIBRARY_PATH conda run -n orbit \
    python -m examples.vagen.build_env_dataset \
        --yaml       examples/vagen/configs/sokoban_train_env.yaml \
        --output-dir data/sokoban-main/train \
        --split      train \
        --base-seed  0
```

See `examples/vagen/scripts/{sokoban,frozenlake}-main.sh` for the production
recipes (train + heldout-eval in one script).
