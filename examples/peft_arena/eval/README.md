# Orbit eval wrappers

Manual one-shot wrappers that point external benchmark suites at Orbit
training output. Each wrapper validates the adapter directory's format,
resolves the base model, then shells out to the benchmark's own runner.

## eval-math-peft-arena.sh

Runs PEFT-Arena's math suite (`math500`, `aime24`, `amc23`) against an
Orbit OFT/LoRA adapter, using SGLang as the inference backend.

### Prerequisites

- Orbit's venv (`${ORBIT_WORKSPACE_ROOT}/orbit-workspace/.venv`) with
  `peft`, `sglang`, `transformers`, and `safetensors` installed.
- A PEFT-Arena checkout. By default the wrapper uses the vendored copy
  at `examples/peft_arena/backend/` (trimmed snapshot of `sglang-eval-backend
  @ 1527012`, see `examples/peft_arena/backend/README.md`). Set `PEFT_ARENA_ROOT`
  to override with an external clone — useful when developing against
  upstream changes.
- An adapter directory written under the post-fix Orbit save path (HF-PEFT
  canonical key shape, `base_model_name_or_path` populated). Pre-fix
  directories are not supported — rerun training.
- The HF base model on disk under `${ORBIT_DATA_ROOT}/hf_models/<NAME>`
  (or wherever the launcher's `HF_CKPT` points).

### Usage

```bash
ITER_DIR=/path/to/orbit_ckpts/<run>/iter_NNNNNNN/adapter \
    bash examples/peft_arena/eval/eval-math-peft-arena.sh
```

To evaluate all saved adapter checkpoints in a run directory once and write a
curve-friendly CSV:

```bash
SAVE_DIR=/path/to/orbit_ckpts/<run> \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_GPUS=8 \
N_SAMPLING=16 \
bash tools/eval_checkpoints_once.sh
```

The once-through script skips checkpoints whose requested datasets already have
metrics, prints progress and ETA after each checkpoint, and writes
`eval_results/<run>/summary.csv` with `acc`, `pass@1`, `pass@2`, `pass@4`,
`pass@8`, and `pass@16`.

### Layer 3 manual smoke (debug-grade verification)

1. Run the qwen3-4b-instruct-2507 OFT launcher for one short iteration:

   ```bash
   cd ${ORBIT_WORKSPACE_ROOT}/ORBIT_ORG/orbit-workspace/orbit_peft_arena
   TOTAL_EPOCHS=1 \
   bash examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft.sh
   ```

2. Locate the produced iter dir, e.g.
   `orbit_ckpts/Qwen3-4B-Instruct-2507-BF16_math_oft/iter_0000000/adapter/`.

3. Run the wrapper:

   ```bash
   ITER_DIR=$(pwd)/orbit_ckpts/Qwen3-4B-Instruct-2507-BF16_math_oft/iter_0000000/adapter \
       bash examples/peft_arena/eval/eval-math-peft-arena.sh
   ```

4. Expected:
   - The wrapper prints a base-model path and an output dir under
     `eval_results/.../math/`.
   - SGLang loads the merged HF model, evaluates each dataset, and writes
     score files in `OUTPUT_DIR/<dataset>/`.
   - `math500` accuracy below ~0.05 indicates a merge or prompt-formatting
     bug. Above that, treat scores as opaque for debug purposes.

### Knobs

| Var                      | Default                                            |
| ------------------------ | -------------------------------------------------- |
| `OUTPUT_DIR`             | `${ORBIT_ROOT}/eval_results/<run>/<iter>/math`     |
| `NUM_GPUS`               | `1`                                                |
| `DATA_NAMES`             | `math500,aime24,amc23`                             |
| `N_SAMPLING`             | `16`                                               |
| `TEMPERATURE`            | `0.6`                                              |
| `MAX_TOKENS_PER_CALL`    | `8192`                                             |
| `GPU_MEMORY_UTILIZATION` | `0.7`                                              |
| `MAX_MODEL_LEN`          | `9216`                                             |
| `PYTHON_BIN`             | `${ORBIT_WORKSPACE_ROOT}/orbit-workspace/.venv/bin/python` |

### Known limitations

- If SGLang crashes mid-evaluation, the merged HF model directory at
  `<iter_dir>_merged` is left on disk (matches existing PEFT-Arena
  behavior; clean up with `rm -rf` manually).
- Pre-fix Orbit adapters (no `base_model_name_or_path` in the JSON, raw
  `model.layers.X.<mod>.oft_R` keys) are unsupported. The wrapper exits
  with a remediation hint.
- Multi-GPU SGLang configurations are not exercised; `NUM_GPUS=1` is the
  tested path.
- PEFT-Arena's `third_party/math_eval` requires `pebble`,
  `word2number`, `timeout-decorator`, and `func-timeout` in the active
  venv. Install with
  `VIRTUAL_ENV=${ORBIT_WORKSPACE_ROOT}/orbit-workspace/.venv uv pip install pebble word2number timeout-decorator func-timeout`
  (one-time, until those deps land in the project's lockfile).

### Verification status

| Layer | Status | Evidence |
| ----- | ------ | -------- |
| Layer 1 — Save-format unit tests | Passing | `tests/fast/backends/megatron_utils/test_peft_save_format.py` (7 tests) |
| Layer 2 — Synthetic round-trip merge | Passing | `tests/peft_arena_eval/test_save_format_loadable.py` (loads orbit-saved adapter via peft, asserts no missing keys + non-zero lora_B post-load) |
| Layer 2 — Live launcher producing HF adapter | Verified manually | qwen2.5-0.5B LoRA launcher with `SAVE_INTERVAL=1` produced an HF-PEFT-canonical adapter dir (336 keys, format `base_model.model.<orig>.<adapter>.weight`) |
| Layer 3 — Full eval smoke | Passing | qwen2.5-0.5B LoRA `iter_0000001` adapter → wrapper → `merge_peft` → SGLang → math500 (10 problems × N_SAMPLING=2): `acc=25.0`, `pass@2=30.0`, run-time 7.5 s after init (2026-05-10). End-to-end command in the **Layer 3 manual smoke** section above; metrics JSON at `eval_results/peft_arena_smoke_qwen0_5b_lora/iter_0000001/.../math/math500/test_cot_10_seed0_t0.6_s0_e-1_metrics.json`. |
| Layer 3 — Smoke against vendored copy | Passing | Same wrapper invocation with `PEFT_ARENA_ROOT` unset (defaults to `examples/peft_arena/backend/`): `acc=25.0`, `pass@2=30.0`, run-time 7.5 s after init (2026-05-10). Metrics JSON at `eval_results/peft_arena_vendor_smoke_qwen0_5b_lora/iter_0000001/.../math/math500/test_cot_10_seed0_t0.6_s0_e-1_metrics.json`. Vendored `parser.py`/`utils.py` show up in the log's SyntaxWarning paths under `examples/peft_arena/backend/third_party/math_eval/`, confirming the vendored modules are loaded. |
