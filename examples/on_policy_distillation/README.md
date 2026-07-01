# On-Policy Distillation (OPD)

On-policy distillation trains a student on its own sampled tokens using a
teacher's log-probs. Orbit supports two objective forms, which share the same
teacher-producer infrastructure (`teacher_log_probs`) and differ only in how the
teacher signal enters the advantage:

- **Pure MOPD** (reward-free): `adv_t = teacher_logp_t - student_logp_t`.
  Selected by `--advantage-estimator on_policy_distillation`.
- **Blend** (RL reward + distillation): a base estimator's advantage minus
  `opd_kl_coef * (student_logp - teacher_logp)`. Enabled by `--use-opd
  --opd-kl-coef <λ>` on top of a reward estimator (`grpo`, `gspo`, `ppo`, ...).

Pure MOPD and blend are **mutually exclusive** (the arg validator errors if both
are set): the blend is meant to sit on top of a reward estimator, not on top of
pure distillation.

The teacher's `teacher_log_probs` come from a teacher-forcing forward pass (the
teacher does **not** generate). The producer is chosen with `--opd-type`:

- `--opd-type megatron`: a second full Megatron model loaded on the training
  GPUs, scored like the `ref` model. Requires `--opd-teacher-load <megatron
  checkpoint>` (optionally `--opd-teacher-ckpt-step <iter>`).
- `--opd-type sglang`: scoring via an external SGLang teacher server (no
  in-process checkpoint). Added in a later phase.

## Megatron teacher recipe

`run-qwen3-4B-opd-megatron.sh` runs pure MOPD with an in-process Megatron
teacher. The student is Qwen3-4B; point `OPD_TEACHER_LOAD` at a (typically
larger/better) teacher Megatron checkpoint.

```bash
HF_CKPT=/path/to/hf/Qwen3-4B-Instruct-2507 \
MEGATRON_LOAD=/path/to/megatron/Qwen3-4B-Instruct-2507 \
OPD_TEACHER_LOAD=/path/to/megatron/teacher-checkpoint \
TRAIN_JSONL=/path/to/math/train.jsonl \
TEST_JSONL=/path/to/math/test.jsonl \
bash examples/on_policy_distillation/run-qwen3-4B-opd-megatron.sh
```

Optional: `OPD_TEACHER_CKPT_STEP=<iter>` selects a specific teacher iteration.

### CPU-free argv inspection

`ORBIT_DRY_RUN_ARGV=1` assembles and prints the python argv, then exits before
starting Ray (it does not run the python arg parser/validation):

```bash
ORBIT_DRY_RUN_ARGV=1 DISABLE_EVAL=1 ENABLE_WANDB=0 TRAIN_ROWS=1 \
HF_CKPT=/path/to/hf/Qwen3-4B-Instruct-2507 \
MEGATRON_LOAD=/path/to/megatron/Qwen3-4B-Instruct-2507 \
OPD_TEACHER_LOAD=/path/to/megatron/teacher-checkpoint \
TRAIN_JSONL=/path/to/math/train.jsonl \
bash examples/on_policy_distillation/run-qwen3-4B-opd-megatron.sh
```

Arg parsing + validation is unit-tested separately in `tests/test_opd_args.py`.
A full GPU training run is a manual smoke test (out of unit-test scope).

## Blend variant

To blend distillation onto a reward-based estimator, drop
`--advantage-estimator on_policy_distillation`, use a reward estimator, and add
`--use-opd`:

```
RL_ARGS=(
    --advantage-estimator grpo
    --use-opd
    --opd-kl-coef 1.0
    --opd-type megatron
    --opd-teacher-load "${OPD_TEACHER_LOAD}"
    ...
)
```

## Constraints

- The Megatron teacher requires full fine-tuning (`--peft-method none`) and the
  CPU weights backuper (enabled by default). It adds a second full-model CPU
  backup, so account for the extra host memory.
