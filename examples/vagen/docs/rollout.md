# Rollout pipeline

Covers `rollout.py` (custom multi-turn generate fn) and `env_adapter.py`
(thin VAGEN ↔ miles bridge).

## generate() flow

`rollout.generate` wires miles' inference rollout
(`miles.rollout.inference_rollout`) to VAGEN's `GymImageEnv` async protocol.
Each call:

1. **Build env** from `sample.metadata['vagen']`.
2. **Initial prompt**: `env.reset` → `system_prompt`, then encode
   `[sys_msg, user_msg]` via the HF processor (NOT the tokenizer — Qwen3-VL
   needs the right image-placeholder token count). VAGEN's order is
   reset-before-system_prompt (`gym_agent_loop.py`); we preserve it.
3. **Drift check** (vision rows only): md5 the live turn-0 PIL and compare
   against `meta['env_uuid']`. See [dataset.md](dataset.md) for the design.
4. **Per-turn loop** (`for _ in range(max_turns)`):
   - inference (HTTP to SGLang router) with `min(global, per_turn)` cap
   - `env.step` wrapped in try/except — a parser failure or env blow-up
     ends the episode with `reward=0`, matching VAGEN
     (`gym_agent_loop.py`)
   - log per-turn audit (reward / format_correct / actions / success / done)
   - terminate on `done` / `success` / `is_last_turn` / `budget_exhausted`
   - else encode the next obs as a user suffix (`loss_mask=0`) using the
     placeholder-system prefix trim (see below)
5. **Finalize** `sample.reward`, `sample.response`, `sample.multimodal_*`,
   and stash `per_turn` log + obs PILs on `metadata['vagen']` for
   `debug_dump`.

## Key invariants

- `args.rollout_global_dataset` is True (miles asserts). Our
  `VagenEnvSpecDataSource` bypasses HF-Dataset construction, so this
  rollout must not touch `state.args.prompt_data` as a parquet path.
- Image content blocks are `{"type": "image"}` markers with no inline PIL —
  PILs go through `processor(images=...)` separately, IN THE SAME ORDER as
  placeholders in `obs_str`. Reordering would break HF's processor alignment.
- `processor` is REQUIRED when env returns images. Fail loud on `None`
  rather than silently falling back to tokenizer encoding (would corrupt
  Qwen3-VL placeholder accounting).
- `partial_rollout` is NOT supported. Assert at the top of `generate()`.

## Placeholder-system prefix trim

When encoding turn-N obs as a user suffix, we need the user-message ids
*as the template would emit them when they follow the existing system
prompt*. Running the processor over just `[user_msg]` would not produce
chat-template-compatible ids (no system prefix context).

The trick (VAGEN `gym_agent_loop.py`): run the processor over
`[_placeholder_sys, user_msg]` with `add_generation_prompt=True`, then slice
off the leading `sys_prefix_len` ids precomputed once via
`_compute_system_prompt_prefix`. NOT a BOS strip — it's a chat-template-
aware splice.

`_PLACEHOLDER_SYSTEM = {"role": "system", "content": "placeholder"}` is a
sentinel; its exact content doesn't matter, only that it occupies the system
slot so the chat template emits the same kind of leading ids as the real
session.

## Termination ordering

A correct loop terminates BEFORE encoding the next obs:

- `env.step` runs first (collects reward + done)
- then we check termination conditions
- only if we're continuing do we encode the obs as a user suffix

Skipping a terminal turn's obs-encoding avoids a dangling "no model answer
for this obs" tail polluting `loss_mask` and `sample.tokens`. We DO snapshot
the post-step obs PIL on terminal turns (handy for debug visualization of
the env's last frame), even though it's not fed back to the model.

We deliberately do NOT short-circuit on `finish_reason == "length"` before
`env.step`: with a per-turn cap, "length" often just means this turn hit
its answer cap, not that the global budget is gone. VAGEN collects the env
reward first, then decides termination. The `response_budget_exhausted`
flag is checked AFTER `env.step` for the same reason.

## Budget computation

`_compute_budget` prefers `rollout_max_context_len` for a literal full-
context cap. The `max_new_tokens` branch mirrors geo3k's convention of
subtracting the current prompt length — so the response budget is
undercounted by `prompt_len`. We keep this for MVP compatibility but set
`--rollout-max-context-len` in launchers to avoid the ambiguity.

## env_adapter local copy

The VAGEN helpers we need (`_normalize_images`, `convert_obs_to_content`,
`extract_success`) all live in `vagen.agent_loop.gym_agent_loop`, which
imports VERL at the top. VERL is not installed
in the miles conda env, so we cannot import that module here.

`env_adapter.py` keeps a local copy of these three helpers inside a
clearly-marked `--- Begin/End local copy ---` block. When VAGEN factors
them into a dependency-light util module we can switch back to
`from vagen...` imports and delete the block.

## Per-turn audit fields

Each entry in `meta['vagen']['per_turn']` carries:

- `reward` — env.step's raw reward this turn (format bonus + success bonus
  + per-step shaping).
- `format_correct` — parsed `<think>…</think><answer>…</answer>` regex hit.
- `n_actions_parsed` — len of parsed `actions` list (≤ `max_actions`).
- `action_is_valid` — all parsed actions ∈ env's `ACTION_LOOKUP` (also
  gated on `format_correct` inside the VAGEN env).
- `action_is_effective` — env state actually changed (player moved or box
  pushed); detects no-ops.
- `success` — `info["success"]` (sticky-or per env); true iff goal reached.
- `done` — terminal turn.

`debug_dump.dump_samples` rolls these up across each rollout step. See
[debug_dump.md](debug_dump.md).

## PIL stash convention

`generate()` stashes per-turn and final obs PILs under underscore-prefixed
keys (`_per_turn_obs_pils`, `_final_obs_pil`) on `metadata['vagen']`. The
leading underscore signals "not serializable" — `debug_dump` pops these
before `json.dump` and writes them as `turn{N}_obs.png` / `final_obs.png`.
Length of `_per_turn_obs_pils` matches `per_turn` exactly.
