# debug_dump

Permanent debug helper: writes one JSON file per `Sample` after each
rollout step, for offline inspection of multi-turn trajectories. Wire via:

```
--rollout-all-samples-process-path examples.vagen.debug_dump.dump_samples
```

The hook signature is the orbit contract `fn(args, all_samples, data_source)`
(verified in `arguments.py`, `inference_rollout_train.py`, and
`sglang_rollout.py`). It fires once per rollout step after all
samples for that step are generated, so we get a clean per-iter snapshot
without touching `rollout.generate`.

## File layout

Under `args.save` (launcher points at `$RUN_DIR/traces`):

```
train/                                         # is_eval=False
  step<NNNN>/
    prompt<PPPPP>_rollout<RR>/
      record.json
      turn0_obs.png    # image the model conditioned on for turn 0
      turn1_obs.png    # image after turn 0's env.step (input to turn 1)
      ...
      final_obs.png    # post-last-env.step obs (terminal frame)
eval/                                          # is_eval=True
  step<NNNN>/
    prompt<PPPPP>_rollout<RR>/
      {same files as train}
```

- `step` — the `rollout_id` the dump was triggered at. Train increments by
  1 per call; eval at `--eval-interval 20` advances 0, 20, 40, …
  → `train/step0020` and `eval/step0020` align on the same wall-clock
  checkpoint.
- `prompt_idx = sample.index // n_samples_per_prompt` — same value for every
  rollout sharing a prompt; a GRPO group's K rollouts cluster together when
  you `ls`. For eval (`n_samples_per_eval_prompt=1`) the divisor is 1.
- `r_in_group = sample.index % n_samples_per_prompt` — 0..K-1 for train,
  always 0 for eval.

Why `index // n_per_group` instead of `sample.group_index`: eval samples
don't carry `group_index` (eval skips GRPO grouping and assigns
`sample.index = 0..N-1`), so the unified divmod formula is the only one
that works on both paths without branching.

The intermediate `step/` dir caps inode-per-dir at ~256 (one GRPO rollout
batch / one val set) so `ls` and tab-completion stay fast across a
400-step run.

**Multi-dataset eval caveat**: the dump folder doesn't namespace by dataset
name. If `eval_datasets` ever grows past one entry, later datasets will
overwrite earlier ones for the same `step<NNNN>/` dir. Single-dataset
assumption is the MVP scope.

## record.json contents

```
{
  "ids":        {step, group_index, sample_index, epoch_id},
  "env":        {name, seed, max_turns, config},     # enough to env.reset() and replay
  "outcome":    {status, reward, env_reward, traj_success, num_turns, per_turn[]},
  "counts":     {response_length, loss_mask_len, loss_mask_sum,
                 rollout_log_probs_len, tokens_len, n_images},
  "mm_audit":   {...},                                # see below
  "trajectory": {prompt, response}                    # full chat-templated text
}
```

`prompt` is the initial chat-templated system+user message; `response` is
the full multi-turn concatenation of assistant turns and env-obs suffixes
interleaved. Per-turn obs PNGs let you visualize the exact frames the model
saw without re-running the env.

## Multimodal alignment audit (`mm_audit`)

Cross-checks between three independent multimodal views of the sample:

- **A. `sample.tokens`** — what sglang saw at rollout time AND what the
  training forward will see.
- **B. `sample.multimodal_inputs`** — the PIL list (sglang's view).
- **C. `sample.multimodal_train_inputs`** — concatenated processor outputs
  across turns: `{pixel_values, image_grid_thw}` — what the training
  forward unpacks as kwargs.

Four invariants:

1. `#vision spans in (A)` == `image_grid_thw.shape[0]` in (C).
2. Every vision span in (A) sits under `loss_mask == 0`. (If any image-pad
   token had `loss_mask == 1`, we'd backprop through the placeholder, which
   has no gradient meaning and just noises up the signal.)
3. `#PILs in (B)` == `#vision spans in (A)` == `#rows in image_grid_thw (C)`.
4. `sum(t·h·w / merge_size²)` over `image_grid_thw` rows in (C) equals the
   count of image-pad tokens in (A); AND `sum(t·h·w)` over rows equals
   `pixel_values.shape[0]`.

The audit returns raw counts and a single `ok` flag. Each `dump_samples`
call logs a step-level rollup:

```
mm_audit step=N  ok=X/Y  count_mismatch=A  mask_dirty=B  patch_mismatch=C
```

So `grep mm_audit run.log` summarizes alignment health across a full run
without scanning JSONs.

`_get_image_token_id` resolves the image-placeholder token id from the
tokenizer at `args.hf_checkpoint`, cached once per process. Qwen2.5-VL uses
`<|image_pad|>` (id 151655); Qwen3-VL uses the same surface string but a
different id — we don't hardcode either. Returns None on resolution failure
(text-only model, missing tokenizer files), and the audit degrades
gracefully.

`loss_mask` covers the RESPONSE portion only (`sample.tokens` is
prompt+response). The mask-clean check subtracts the prompt offset before
indexing.

## Per-turn rollup

After writing per-sample records, `dump_samples` logs a `turn_stats` line:

```
traj_success=...   N_trajs reaching goal / N_trajs
traj_any_format=... N_trajs with ≥1 well-formed turn / N_trajs
avg_turns=...       sum(num_turns) / N_trajs
turn_format=...     N_turns parsed cleanly / N_turns
turn_action_valid=... N_turns with ALL parsed actions ∈ env ACTION_LOOKUP / N_turns
turn_action_effective=... N_turns that changed env state / N_turns
avg_n_actions=...   sum(actions_parsed) / N_turns
```

Lets you tell apart "reward goes up because of format compliance" from
"reward goes up because of success" at a glance.

Reward mean is intentionally NOT recomputed here — orbit already logs
`rollout/raw_reward` for that.

## Wandb mirror

Same aggregates as the stdout `turn_stats` line:

- **Train**: `rollout/vagen/*`, keyed at `rollout/step` (matches orbit' train
  rollout axis).
- **Eval**: `eval/<eval_dataset_name>/vagen/*`, keyed at `eval/step` — the
  `<eval_dataset_name>/` middle level matches orbit' eval metric convention
  (`eval/<ds>/reward`, `eval/<ds>/response_len/...` in
  `orbit/ray/rollout.py`), so vagen aggregates plot alongside orbit'
  built-in eval metrics on the same `eval/step` axis.

The matching `step_key` per family is critical: wandb's
`define_metric("eval/*", step_metric="eval/step")`
(`wandb_utils.py`) globs by family, and a mismatched step would pile
eval points at the last-known `eval/step` instead of the actual rollout id.

Per-turn time series intentionally NOT logged: turn count varies per
trajectory (1..max_turns) and wandb can't cleanly plot variable-cardinality
series. The rollup ratios above capture the same signal at fixed cardinality
per step.

`rollout_id` is forwarded from orbit' rollout-hook wrappers. When absent
(legacy callers, smoke tests) we fall back to a module-local counter; the
filenames still include `group_index` + `sample_index` so the global
identity of each trajectory remains unambiguous even if `step` resets.

## Auto-cleanup

`dump_samples` writes into `args.save` (typically a scratch dir under
`/tmp/vagen-mvp/<tag>/`) — never inside the repo. Cleanup is automatic
when the scratch dir is wiped.
