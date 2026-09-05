---
title: "VAGEN example"
description: "Multi-turn visual-game RL (Sokoban/FrozenLake) against VAGEN-style environment servers."
# Generated from examples/vagen/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Multi-turn vision-language RL on VAGEN environments (Sokoban, FrozenLake)
trained with orbit' GRPO. Each rollout drives a `vagen.envs.GymImageEnv`
through an SGLang-served VLM for up to N turns; rewards come from the env's
per-turn `format_reward` + terminal `success_reward`.

## What's in here

```
examples/vagen/
├── README.md                 ← you are here
├── configs/                  ← EnvSpec yamls used by the offline builder
│   ├── sokoban_train_env.yaml
│   ├── sokoban_val_env.yaml      (heldout-eval candidate pool)
│   ├── frozenlake_train_env.yaml
│   └── frozenlake_val_env.yaml
├── scripts/                  ← offline dataset builders
│   ├── sokoban-main.sh
│   └── frozenlake-main.sh
├── docs/                     ← design notes (read these for the why)
│   ├── dataset.md            (row schema, drift detection, heldout)
│   ├── rollout.md            (multi-turn loop, key invariants)
│   ├── debug_dump.md         (per-rollout JSONs + mm_audit)
│   └── launch_recipe.md      (VAGEN→orbit knob mapping, eval cadence)
├── build_env_dataset.py      ← offline (env, seed) → samples.jsonl
├── data_source.py            ← VagenEnvSpecDataSource (jsonl / yaml loader)
├── rollout.py                ← custom multi-turn generate function
├── env_adapter.py            ← VAGEN ↔ orbit bridge (local copy of helpers)
├── debug_dump.py             ← --rollout-all-samples-process-path hook
└── tests/env_dynamics_probe.py
```

The experiment launchers live under `scripts/experiments/vagen-*.sh`.

## Quick start

### 0. Install VAGEN into the orbit env

The example imports `vagen.envs.*` directly. VAGEN ships its own training
stack (vLLM/sglang/Megatron) but here we only need the Python package
importable in the orbit conda env:

```bash
git clone https://github.com/impossible-inc/VAGEN.git
cd VAGEN
conda activate orbit
pip install -e .
```

Editable install — pulling new VAGEN code is just `git pull` in that
checkout. Verify with `python -c "import vagen; print(vagen.__file__)"`.

VAGEN's own `scripts/install_vllm_sglang_mcore.sh` is for VAGEN's training
entry points — skip it; the orbit env already has SGLang.

> **On a freshly-built env** (e.g. orbit' `install_env.sh`, which pulls the
> latest `setuptools`): also run `pip install "setuptools<81"`. `gym-sokoban`
> 0.0.6 does `import pkg_resources` at import time, and setuptools ≥81 removed
> `pkg_resources` — without this the **Sokoban** env silently fails to register
> (`Unknown env name: Sokoban`); FrozenLake is unaffected. Prefer
> `pip install -e . --no-deps` followed by
> `pip install gym-sokoban gymnasium "gymnasium[toy-text]" "setuptools<81"` so
> VAGEN's `uvicorn<0.41` pin does not downgrade the env's SGLang-side uvicorn.

### 1. Build the dataset

Each environment has one offline-builder shell script that produces two
splits — `train/samples.jsonl` and `eval/samples.jsonl` (heldout, map-
disjoint from train by default). Run once per environment:

```bash
# Sokoban
env -u LD_LIBRARY_PATH conda run -n orbit \
    examples/vagen/scripts/sokoban-main.sh

# FrozenLake
env -u LD_LIBRARY_PATH conda run -n orbit \
    examples/vagen/scripts/frozenlake-main.sh
```

Each call writes:

```
data/<dataset>/
  train/samples.jsonl    10k seeds [1,10000]
  eval/samples.jsonl     256 map-disjoint-from-train seeds
  {train,eval}/images/seed_<NNNNNNNN>_<spec>.png
  {train,eval}/dataset_meta.json
```

The build is idempotent: a second run with the same inputs short-circuits.
Force a rebuild with `FORCE=1`. To skip the heldout filter (use the val
yaml directly), edit the eval step in the build script — see comments in
`examples/vagen/scripts/<dataset>-main.sh`.

See `docs/dataset.md` for the row schema, the drift-detection design
(live-render `env_uuid` md5 cross-check at rollout time), and the heldout
strategy (`--exclude-data` + `--target-kept N`).

### 2. Run an experiment

Two reference recipes live at the repo's `scripts/experiments/`:

**FrozenLake on Qwen3-VL-2B-Instruct (1 node × 8 GPUs):**

```bash
bash scripts/slurm/submit.sh vagen-frozenlake-main-qwen3vl2b-colocate-1node
```

**Sokoban on Qwen2.5-VL-3B-Instruct (1 node × 8 GPUs):**

```bash
bash scripts/slurm/submit.sh vagen-sokoban-main-qwen25vl3b-colocate-1node
```

Both:

- Read the prebuilt train/eval jsonl from `data/{frozenlake,sokoban}-main/`
  (the launcher errors with a build hint if missing).
- Train for `--num-rollout 400`, `--rollout-batch-size 32`,
  `--n-samples-per-prompt 8`, GRPO with `lr=1e-6 / kl=0 / eps_clip=0.2`.
- Eval every 20 rollouts (`--eval-interval 20`).
- Write per-rollout debug JSONs + per-turn obs PNGs under `$RUN_DIR/traces/`.

The FrozenLake-qwen3vl2b recipe also opts into a `<think>`→`<thinking>`
tag swap via `VAGEN_THINK_TAG=thinking` (the Qwen3-VL Instruct never saw
the `<think>` added_tokens during training — see `docs/launch_recipe.md`).

See `docs/launch_recipe.md` for the full VAGEN→orbit knob mapping, eval
cadence, perf-args rationale, and overrides.

### Common overrides

```bash
# Use a different eval split (default is heldout)
VAGEN_EVAL_DATA=/path/to/eval.jsonl \
bash scripts/slurm/submit.sh <recipe>

# Enable DAPO-style dynamic sampling
ORBIT_VAGEN_DAPO=1 bash scripts/slurm/submit.sh <recipe>

# Scale knobs
ORBIT_SCRIPT_NUM_ROLLOUT=200 \
ORBIT_SCRIPT_ROLLOUT_BATCH_SIZE=16 \
bash scripts/slurm/submit.sh <recipe>
```

## Inspecting a run

The `debug_dump` hook fires once per rollout step and writes:

```
$RUN_DIR/traces/{train,eval}/step<NNNN>/prompt<P>_rollout<R>/
  record.json    # env metadata, outcome, mm_audit, prompt+response text
  turn0_obs.png  # frame the model saw at turn 0
  turn1_obs.png  # frame after turn 0's env.step (= turn 1's input)
  ...
  final_obs.png  # post-last-env.step frame
```

`grep mm_audit run.log` summarizes multimodal-alignment health across the
run (4 invariants: vision-span count parity, image-pad token masking,
PIL/grid_thw row alignment, patch-count match). See `docs/debug_dump.md`.

### Visualize with trace-viewer

For an interactive view of multi-turn trajectories (system / user-obs /
assistant blocks separated, per-turn images inlined), use the local web
viewer at &lt;https://github.com/impossible-inc/trace-viewer>. It reads the
exact `traces/` layout this example produces.

```bash
# one-time setup (its own isolated .venv — don't share with orbit)
git clone https://github.com/impossible-inc/trace-viewer.git
cd trace-viewer
uv venv .venv --python 3.11
uv pip install -e ".[dev]"

# launch — point TRACES_ROOT at the parent dir of your <run-name>/<timestamp>/ tree
TRACES_ROOT=/path/to/runs \
    .venv/bin/uvicorn trace_viewer.server:app --reload --port 8080
```

Then open &lt;http://localhost:8080/>. See the trace-viewer repo's README
for the full expected directory shape.

## Other recipes

In this PR (under `scripts/experiments/`):

- `vagen-sokoban-main-qwen25vl3b-colocate-1node-global_bsz32.sh` — overlay
  on the sokoban recipe with `--global-batch-size 32` (8 PPO updates per
  rollout instead of 1).

Planned variants (follow-up PRs):

- `vagen-frozenlake-main-qwen25vl7b-colocate-2node.sh` — 7B on 2 nodes.
- `vagen-frozenlake-main-colocate-1node.sh` — Qwen2.5-VL-3B baseline.
- `vagen-frozenlake-{filter,legacy,smoke}-colocate-1node.sh` — variants.

## Notes

- The dataset is the single source of truth: same `samples.jsonl` feeds
  both training and orbit' eval pipeline. No parquet, no HF Dataset.
- VAGEN's `env.reset(seed)` is global-state; the Sokoban patches
  (`_stable_next_seed`, seeding-lock) are required for determinism. The
  baked `env_uuid` is a tripwire — any divergence fails the run.
- `examples/vagen/data_source.py` has a tiny CLI:
  `python -m examples.vagen.data_source <samples.jsonl>` peeks at the rows.
