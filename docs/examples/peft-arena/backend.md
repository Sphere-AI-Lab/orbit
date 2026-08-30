---
title: "PEFT-Arena (vendored)"
description: "PEFT-Arena evaluation backend and graders."
# Generated from examples/peft_arena/backend/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Trimmed, pinned snapshot of [PEFT-Arena](https://github.com/peft-arena)
math evaluation pipeline. Used by `examples/peft_arena/eval/eval-math-peft-arena.sh`.

## Provenance

- Upstream branch: `sglang-eval-backend`
- Upstream commit: `1527012` ("fix(eval_math): pass absolute --data_dir
  so callers from outside the repo work")
- Local patches over upstream `main` (`01fd317`):
  - `9c34a62` — add SGLang backend (`--use_sglang`) alongside vLLM
  - `22f9d90` — drop unused field import; collapse double from_sglang call
  - `467d315` — thread `--use_sglang` and `--num_gpus` to math_eval.py
  - `1527012` — pass absolute `--data_dir`

## Layout

```
examples/peft_arena/backend/
├── LICENSE                      # upstream MIT
├── README.md                    # this file
├── eval/
│   └── eval_math.sh             # math benchmark orchestrator (vLLM | SGLang)
├── tools/
│   ├── merge_peft.py            # merges a PEFT adapter into the base model
│   └── prepare_eval_checkpoint.py
│                                # detects checkpoint layout; for adapter dirs
│                                # short-circuits as a passthrough.
│                                # Inlined a 60-line metadata helper module so
│                                # we don't need to vendor train/peft_arena_verl/.
└── third_party/
    └── math_eval/
        ├── data/
        │   ├── math500/test.jsonl
        │   ├── aime24/test.jsonl
        │   └── amc23/test.jsonl
        ├── latex2sympy/         # runtime-only subset of the upstream package
        │   ├── __init__.py
        │   ├── latex2sympy2.py
        │   └── gen/             # ANTLR-generated lexer + parser
        ├── data_loader.py
        ├── evaluate.py
        ├── examples.py
        ├── grader.py
        ├── math_eval.py
        ├── model_utils.py
        ├── parser.py
        ├── python_executor.py
        ├── trajectory.py
        └── utils.py
```

## What was deliberately omitted

- `train/`, `run.py`, `setup_env.sh`, `setup_third_party.sh`,
  `configs/`, `tests/`, `assets/`, `docs/`.
- `eval/eval_{broad_general,coding,commonsense,general,med}.sh`,
  `eval/{summarize_results,summarize_eval,…}.py`, `eval/opencompass_configs/`.
- `tools/{compare_model_norms,plot_*,spectral_*}.py`.
- `third_party/{verl,opencompass,med_eval}/`.
- `third_party/math_eval/data/{aqua,asdiv,gsm8k,olympiadbench,…}/`.
- `third_party/math_eval/latex2sympy/{antlr-*-complete.jar,gen.bak/,
  sandbox/, tests/, scripts/, setup.py, *.in, *.txt, PS.g4,
  asciimath_printer.py}`.

## Refreshing the vendor

When upstream advances and you want to pick up new patches:

```bash
UPSTREAM=${ORBIT_WORKSPACE_ROOT}/ORBIT_ORG/orbit-workspace/PEFT-Arena
ORBIT_ROOT=$(git rev-parse --show-toplevel)
cd "$ORBIT_ROOT"

# 1. copy the same set of paths listed under "Layout" above
cp "$UPSTREAM/eval/eval_math.sh"               examples/peft_arena/backend/eval/
cp "$UPSTREAM/tools/merge_peft.py"             examples/peft_arena/backend/tools/
cp "$UPSTREAM/tools/prepare_eval_checkpoint.py" \
                                               examples/peft_arena/backend/tools/
# … (other paths from Layout)

# 2. re-apply the inline patch in tools/prepare_eval_checkpoint.py:
#    drop the peft_arena_verl import, inline DEFAULT_BASE_MODEL_NAME_OR_PATH,
#    normalize_model_reference, load_checkpoint_metadata.

# 3. re-run the smoke
ITER_DIR=/path/to/orbit_ckpts/<run>/iter_NNNNNNN/adapter \
    bash examples/peft_arena/eval/eval-math-peft-arena.sh

# 4. update the "Provenance" block above with the new commit SHA
```

## Training with PEFT-Arena data (optional)

PEFT-Arena's training data lives in a separate HuggingFace repo,
[`SphereLab/PEFTArena-data`](https://huggingface.co/datasets/SphereLab/PEFTArena-data),
not vendored here. There's a dedicated launcher for the math split
(`data/openr1-50k`, 50k filtered OpenR1 rows) that lazy-runs the
converter on first use:

```bash
# 1. Train (lazily converts data on first run)
bash examples/high_precision/run-qwen3-4b-instruct-2507-bf16-peft-arena-openr1-oft.sh

# 2. Eval the resulting adapter against the vendored math test sets
ITER_DIR="$(pwd)/orbit_ckpts/Qwen3-4B-Instruct-2507-BF16_peft_arena_openr1_50k_oft/iter_NNNNNNN/adapter" \
    bash examples/peft_arena/eval/eval-math-peft-arena.sh
```

The launcher is a thin wrapper around `run-qwen3-4b-instruct-2507-bf16-math-oft.sh`
with `DATASET=peft_arena_openr1_50k` and `TRAIN_JSONL`/`TEST_JSONL` pointed
at the converted data — same RL/GRPO recipe and hyperparameters, different
data. To call the converter directly (e.g. for medical data or a custom
output dir):

```bash
python tools/convert_peftarena_data.py \
    --subset openr1-50k \
    --output_dir ${ORBIT_DATA_ROOT}/data/peft_arena_openr1_50k
```

Notes:

- Orbit's `examples/high_precision/*` launchers run RL/GRPO, not SFT.
  PEFT-Arena's reference SFT recipe (`configs/train/sft_lora_qwen_r8_math.yaml`,
  upstream) won't match the RL launcher's hyperparameters; tune as needed.
- The converter appends Orbit's standard `\boxed{}` instruction to user
  prompts by default so the model emits boxed answers the eval grader can
  extract. Pass `--no-append-instruction` to leave prompts as-is.
- For medical training data, swap `--subset openr1-50k` for `--subset medthink-23k`.

### Parallel eval on a separate GPU

The PEFT-Arena launcher saves a checkpoint every 20 steps by default
(`SAVE_INTERVAL=20`). If you have spare GPUs while training is running, you
can score the intermediate checkpoints in parallel — no GPU contention with
training, full pass@k numbers on math500/aime24/amc23, builds the real
training curve retroactively.

In a separate shell on the spare GPU:

```bash
# Pin to one or more spare GPUs (training uses 0-3 by default).
CUDA_VISIBLE_DEVICES=4 \
SAVE_DIR=$(pwd)/orbit_ckpts/Qwen3-4B-Instruct-2507-BF16_peft_arena_openr1_50k_oft \
    bash tools/eval_checkpoints_loop.sh
```

The loop polls every 60s, runs the eval wrapper against each new
`iter_*/adapter` it finds, skips iters whose `metrics.json` already
exists (resume-safe), and logs each eval to
`logs/eval_loop_<run>_<iter>.log`. Results land at
`eval_results/<run>/<iter>/.../math/<dataset>/*metrics.json`.

## Regenerating `latex2sympy/gen/`

The vendor ships pre-generated ANTLR output. To regenerate from `PS.g4`
in upstream, install `antlr4` and run upstream's `latex2sympy/scripts/`
build script. We do not vendor the JARs (~6.8 MB) since regen is rare.

## In-loop Orbit Math Eval

This tree also supports running the `math500`, `aime24`, and `amc23`
benchmarks directly inside Orbit's training loop with the existing Orbit
inference backend:

- SGLang stays the inference backend.
- The live PEFT adapter stays mounted in-place.
- No merge-to-HF step is required during training-time eval.

Two evaluation paths are supported.

### 1. `orbit515` path

This preserves the original `orbit-515` semantics:

- user prompt = original problem text
- optional suffix:
  `Please reason step by step, and put your final answer within \boxed{}`
- reward/judging path = Orbit's existing `math` / `boxed_math` logic

Use this mode when you want continuity with the existing `orbit-515`
dashboard and prompt style.

### 2. `math_alignment` path

This keeps Orbit's runtime behavior, but aligns the evaluation semantics more
closely with the PEFT-Arena math benchmark:

- prompt becomes:
  `Question: ...\nAnswer:`
- judge uses a dedicated `math_alignment` rule-based grader
- ground-truth handling keeps `answer`, and for `math500` also keeps
  `solution` so the canonical answer can be extracted from the reference
  solution
- per-dataset metrics include:
  - `acc`
  - `pass_acc`
  - `pass@k`

Use this mode when you want training-time eval that is closer to the
PEFT-Arena benchmark semantics without changing the inference backend or
merging adapters.

### Data preparation

Convert the vendored math-eval JSONLs into Orbit eval JSONL with:

```bash
python tools/convert_math_eval_to_orbit.py \
  --mode orbit515 \
  --output_dir /path/to/eval_orbit515

python tools/convert_math_eval_to_orbit.py \
  --mode math_alignment \
  --output_dir ${ORBIT_DATA_ROOT}/data/math_eval
```

Outputs are:

- `orbit515`:
  - `prompt`
  - `label`
- `math_alignment`:
  - `prompt`
  - `label`
  - `metadata`
    - `rm_type=math_alignment`
    - `dataset_name`
    - `answer`
    - `solution` when present

### Training-time configuration

Provide multiple eval datasets through `EVAL_DATASETS`:

```bash
EVAL_DATASETS="\
math500:/path/to/math500.jsonl \
aime24:/path/to/aime24.jsonl \
amc23:/path/to/amc23.jsonl"
```

Relevant knobs:

```bash
N_SAMPLES_PER_EVAL_PROMPT=64
EVAL_PASS_K_VALUES=1,2,4,8,16
LOG_PASSRATE=1
```

`EVAL_PASS_K_VALUES` is optional. The shared metric helper still defaults to
all power-of-two values up to the group size for training passrate logging,
while the PEFT-Arena launcher defaults eval reporting to:

```text
1, 2, 4, 8, 16
```

filtered by `N_SAMPLES_PER_EVAL_PROMPT`. For example, if
`N_SAMPLES_PER_EVAL_PROMPT=8`, Orbit will only emit:

```text
pass@1, pass@2, pass@4, pass@8
```

You can still request larger sets explicitly, e.g.:

```bash
EVAL_PASS_K_VALUES=1,2,4,8,16,32,64
```

### Reported metrics

For every dataset `X`, Orbit logs:

- common:
  - `eval/X`
  - `eval/X/response_len/*`
  - `eval/X-truncated_ratio`

- `orbit515` path:
  - `eval/X-pass@k`
  - `eval/X/pass@k`

- `math_alignment` path:
  - `eval/X/acc`
  - `eval/X/pass_acc`
  - `eval/X/pass@k`

When more than one eval dataset is configured, Orbit also logs:

```text
eval/avg
```

as the simple average of the per-dataset scalar score used for each dataset's
main `eval/X`.

### Notes

- `LOG_PASSRATE=1` is enabled by default on this tree so training-time eval
  will report pass@k without extra launcher work.
- The in-loop eval path is intended for no-merge adapter evaluation during
  training. If you want merged-checkpoint post-hoc evaluation, continue to use
  the external wrapper path documented above.
