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
  in-process checkpoint). Requires `--opd-teacher-url <http://host:port/generate>`,
  plus `--custom-rm-path orbit.rollout.opd_sglang.reward_func
  --custom-reward-post-process-path orbit.rollout.opd_sglang.post_process` to
  wire the scoring call into orbit's reward pipeline (see below).

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

## SGLang teacher recipe

`run-qwen3-4B-opd-sglang.sh` runs pure MOPD with an external SGLang teacher
server -- the teacher is not loaded on the training GPUs. Start a separate
SGLang server hosting the teacher checkpoint first (e.g. `python -m
sglang.launch_server --model-path <teacher-hf-checkpoint> --port <port>`),
then point `OPD_TEACHER_URL` at its `/generate` endpoint:

```bash
HF_CKPT=/path/to/hf/Qwen3-4B-Instruct-2507 \
MEGATRON_LOAD=/path/to/megatron/Qwen3-4B-Instruct-2507 \
OPD_TEACHER_URL=http://<teacher-host>:<port>/generate \
TRAIN_JSONL=/path/to/math/train.jsonl \
TEST_JSONL=/path/to/math/test.jsonl \
bash examples/on_policy_distillation/run-qwen3-4B-opd-sglang.sh
```

At each rollout, the trainer POSTs the student's sampled token sequence to
`OPD_TEACHER_URL` for prefill-only scoring (`max_new_tokens=0,
return_logprob=True, temperature=0` -- the teacher does not generate), then
trims the returned per-token log-probs to the response span and stores them on
`teacher_log_probs`. This is implemented in `orbit/rollout/opd_sglang.py` and
wired through **two** hooks (both required):

- `--custom-rm-path orbit.rollout.opd_sglang.reward_func`: performs the
  scoring POST per sample during rollout generation and returns `0.0` (pure
  distillation has no task reward).
- `--custom-reward-post-process-path orbit.rollout.opd_sglang.post_process`:
  extracts and trims the teacher log-probs and sets `sample.teacher_log_probs`.

(`reward_func` stashes the raw teacher response in `sample.metadata` rather
than `sample.reward`, because orbit computes zero-std-reward rollout metrics
from `sample.reward` *before* the post-process hook runs, and those metrics
assume a numeric reward.)

Same CPU-free argv inspection and mutual-exclusion rules as the Megatron
recipe apply here.

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
- `orbit.rollout.opd_sglang.reward_func` always returns `0.0` and occupies the
  single `--custom-rm-path` slot. Combining `--opd-type sglang` with the
  **blend** form (which needs a real task reward from a base estimator) is not
  wired up by this recipe -- it would require a `reward_func` that both scores
  the teacher and computes the task reward.
