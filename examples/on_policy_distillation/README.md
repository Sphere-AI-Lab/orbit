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

- `--opd-type megatron`: trainer-side scoring. `--opd-teacher load:<ckpt>`
  (legacy `--opd-teacher-load <megatron checkpoint>`, optionally
  `--opd-teacher-ckpt-step <iter>`) loads a second full Megatron model on the
  training GPUs, scored like the `ref` model; same-base specs
  (`--opd-teacher base/adapter:<path>/self:*`, see "Teacher-as-Adapter-Slot"
  below) score without a second model.
- `--opd-type sglang`: rollout-side scoring (no in-process second
  checkpoint). With `--opd-teacher-url <http://host:port/generate>` an
  external SGLang teacher server scores the samples; that mode requires
  `--custom-rm-path miles.orbit.opd.opd_sglang.reward_func
  --custom-reward-post-process-path miles.orbit.opd.opd_sglang.post_process` to
  wire the scoring call into orbit's reward pipeline (see below). Without a
  URL (local mode), the rollout engine scores a same-base teacher itself and
  those hooks must be left unset (see "Teacher-as-Adapter-Slot" below).

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
`teacher_log_probs`. This is implemented in `miles/orbit/opd/opd_sglang.py` and
wired through **two** hooks (both required):

- `--custom-rm-path miles.orbit.opd.opd_sglang.reward_func`: performs the
  scoring POST per sample during rollout generation and returns `0.0` (pure
  distillation has no task reward).
- `--custom-reward-post-process-path miles.orbit.opd.opd_sglang.post_process`:
  extracts and trims the teacher log-probs and sets `sample.teacher_log_probs`.

(`reward_func` stashes the raw teacher response in `sample.metadata` rather
than `sample.reward`, because orbit computes zero-std-reward rollout metrics
from `sample.reward` *before* the post-process hook runs, and those metrics
assume a numeric reward.)

**Eval-accuracy/pass-rate is not meaningful in this external-URL hook mode.**
`reward_func` always returns `0.0`, and orbit shares this hook between train
and eval, so any
task-accuracy or pass-rate metric derived from `sample.reward` (`eval/<dataset>`,
`--eval-pass-k-values`, `--log-passrate`) reports 0 regardless of student
quality -- the actual training signal is `teacher_log_probs`, not reward.
`run-qwen3-4B-opd-sglang.sh` therefore disables eval by default
(`DISABLE_EVAL=${DISABLE_EVAL:-1}`) and drops `--eval-pass-k-values`/
`--log-passrate` entirely; `TEST_JSONL` is only consulted if you explicitly set
`DISABLE_EVAL=0`. Contrast with `run-qwen3-4B-opd-megatron.sh`, which uses a
real `--rm-type math` reward, so its eval-accuracy numbers are meaningful.
(In sglang local-teacher mode — no `--opd-teacher-url`, see
"Teacher-as-Adapter-Slot" below — scoring is a built-in rollout stage, the
reward hook stays real, and eval metrics are meaningful again.)

Same CPU-free argv inspection and mutual-exclusion rules as the Megatron
recipe apply here.

## Top-k distributional scoring (Rethinking OPD)

`--opd-log-prob-top-k 0` (the default) scores only the *sampled* token — a
high-variance single-point estimate of the reverse KL. Setting it above zero
switches the sglang teacher to the top-k recipe from
[Rethinking On-Policy Distillation](https://arxiv.org/abs/2604.13016): the
student's own top-k logprobs are harvested during rollout generation, the
teacher is scored at the same sequence, and a weighted reverse-KL estimate is
aggregated over a selected token set per response position. The result ships
as one scalar per token in `sample.opd_reverse_kl`, which the trainer consumes
directly (both pure MOPD and the blend) — training-side cost is unchanged.

The token set is controlled by `--opd-top-k-strategy`:

| Strategy | Token set |
|----------|-----------|
| `only-student` | Student top-k tokens, with teacher logprobs queried for those IDs. |
| `only-teacher` | Teacher top-k tokens, with student logprobs queried for those IDs. |
| `intersection` | Tokens appearing in both top-k sets. |
| `union` | Tokens appearing in either top-k set, with duplicates removed. |
| `xor` | Tokens appearing in exactly one top-k set. |

`--opd-reward-weight-mode` weights each selected token by student probability
(`student_p`, default), teacher probability (`teacher_p`), or uniformly
(`none`). Weights are softmax-normalized over the set except for `xor`.

`--opd-kl-type` selects the KL direction (mirroring NeMo-RL's distillation
`kl_type`): `reverse` (default) weights by the student distribution and is
mode-seeking; `forward` weights by the teacher distribution and is
mass-covering; `mixed` is the convex combination with
`--opd-mixed-kl-weight` on the forward term (0.5 matches NeMo-RL's default
recipe). `--opd-reward-weight-mode` applies to the reverse term only — the
forward term is always teacher-weighted (its natural measure). Forward and
mixed require `--opd-log-prob-top-k > 0`; the sampled-token path is
reverse-only. Both directions compose with `--opd-topk-tail-bucket` (the
forward tail term penalizes teacher mass the student's support misses).

```
RL_ARGS=(
    --advantage-estimator on_policy_distillation
    --opd-type sglang
    --opd-log-prob-top-k 16
    --opd-top-k-strategy only-student
    ...
)
```

### Tail-mass bucket (exact truncated KL)

The default weighting renormalizes over the selected token set, so the
estimate cannot see probability mass the student moves *outside* the top-k.
`--opd-topk-tail-bucket` instead treats the position as k+1 buckets that sum
to 1 — the selected ids at their exact full-softmax probabilities, plus one
aggregated tail bucket — and computes the exact reverse KL over that
partition. The tail term penalizes the student for pushing mass off the
support. Requires `--opd-reward-weight-mode student_p` and
`--opd-top-k-strategy only-student` or `intersection`: the bucket partition
is only exact when all student logprobs at the selected ids come from a
single softmax (the rollout harvest). (This is the same idea as NeMo-RL's
`zero_outside_topk` distillation-loss correction, computed rollout-side.)

## Multi-teacher routing and ensembles

`--opd-teacher-urls NAME=URL[@W][,URL[@W]...]` (sglang mode only) routes each
sample to a named teacher group instead of the single `--opd-teacher-url`:

- **Routing**: each sample is sent to the group named by
  `sample.metadata[--opd-teacher-key]` (default key: `opd_teacher`, populated
  from the dataset's metadata column). The reserved name `default` is the
  fallback for samples with a missing or unknown name; without a `default`,
  such samples fail loudly — silently distilling from the wrong teacher is
  worse than failing the rollout.
- **Ensembles**: a name mapping to several comma-separated URLs scores the
  sample against every member in parallel (wall clock = max latency, not the
  sum) and combines the teachers as a weighted mixture in probability space
  (logsumexp of weighted logprobs — the logprob of the mixture teacher, not a
  geometric mean). Per-URL weights default to 1.0. With
  `--opd-log-prob-top-k > 0`, ensembles require
  `--opd-top-k-strategy only-student` so every member is scored at the same
  student token ids.

```bash
--opd-teacher-urls \
    math=http://h1:30001/generate \
    code=http://h2:30002/generate@2,http://h3:30003/generate \
    default=http://h1:30001/generate
```

Scoring robustness: `--opd-scoring-timeout-secs` bounds each teacher/student
scoring request (teachers are often much larger and slower than the student);
transient failures (timeout, connection error, HTTP 5xx) get one automatic
jittered retry, 4xx responses never retry.

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

## Async correction (ICE-POP)

With asynchronous / off-policy rollouts the acting policy drifts from the
current student, biasing the OPD advantage. `--opd-icepop` (default off) applies
orbit's existing ICE-POP truncated-importance-sampling gate to the OPD
advantage: per token it computes the train/rollout importance ratio (the same
ratio the policy-gradient `icepop`/TIS path uses), reweights in-band tokens by
that ratio, and zeroes tokens whose ratio falls outside `[--tis-clip-low,
--tis-clip]`. It reuses those existing thresholds (no new knobs) and applies to
both pure MOPD and blend. This mirrors NeMo-RL's MOPD ICE-POP correction.

```
RL_ARGS=(
    --advantage-estimator on_policy_distillation
    --opd-type megatron
    --opd-teacher-load "${OPD_TEACHER_LOAD}"
    --opd-icepop
    --tis-clip-low 0.2
    --tis-clip 5.0
    ...
)
```

`--opd-icepop` needs the train-recomputed log-probs to differ from the rollout
log-probs, so it is rejected together with `--use-rollout-logprobs` (which would
make the ratio identically 1).

## Constraints

- The Megatron teacher requires full fine-tuning (`--peft-method none`) only
  for `--opd-teacher load:<ckpt>` (the second-full-model path, same as legacy
  `--opd-teacher-load`); that path also requires the CPU weights backuper
  (enabled by default) and adds a second full-model CPU backup, so account
  for the extra host memory. Same-base specs
  (`--opd-teacher base/adapter:<path>/self:*`) require PEFT instead and load
  no second model.
- With an external `--opd-teacher-url`, `miles.orbit.opd.opd_sglang.reward_func`
  always returns `0.0` and occupies the single `--custom-rm-path` slot.
  Combining that external-teacher mode with the **blend** form (`--use-opd`)
  is rejected by `_validate_opd_args` with a `ValueError`: it would require a
  `reward_func` that both scores the teacher and computes the task reward,
  which is not wired up by this recipe. The external-URL sglang teacher
  supports only pure MOPD (`--advantage-estimator on_policy_distillation`);
  for the blend use `--opd-type megatron` or a same-base local teacher (no
  `--opd-teacher-url`), where scoring is a built-in rollout stage and the
  `--custom-rm-path` slot stays free for a real task reward.
- In external-URL sglang-teacher mode, task-accuracy/pass-rate eval is **not
  meaningful**: `reward_func` always returns `0.0` and is shared between train
  and eval, so any `eval/<dataset>`, pass@k, or `--log-passrate` metric reports
  0 regardless of student quality -- the training signal is
  `teacher_log_probs`, not reward. `run-qwen3-4B-opd-sglang.sh` disables eval
  by default accordingly. The Megatron recipe and sglang local-teacher mode
  use a real reward (`--rm-type math` here), so their eval works as expected.

## Teacher-as-Adapter-Slot (same-base teachers)

When the teacher shares the student's frozen base, no second model and no
teacher server are needed — the teacher is a named adapter:

- `--opd-teacher base`: the frozen base itself (with `--kl-coef`/KL on, the
  ref forward is reused: the teacher is literally free).
- `--opd-teacher adapter:<path>`: base + a frozen adapter checkpoint (SFT /
  expert / RL-trained). Trainer-side scoring swaps the adapter tensors in for
  the teacher forward; sglang-side scoring targets the engine's reserved
  `orbit_teacher` slot via per-request `lora_path`.
- `--opd-teacher self:ema` / `self:lag`: an EMA (`--opd-ema-decay`) or lagged
  (`--opd-self-teacher-interval`) snapshot of the student adapter. With
  `--opd-type sglang`, add `--opd-promote-interval N` to push the buffer to
  the engine slot every N steps (the EMA updates once per rollout training
  step). This enables mean-teacher MOPD and iterated self-distillation at
  adapter cost.

Same-base specs require PEFT (`--peft-method != none`); with full fine-tuning
use `--opd-teacher load:<ckpt>` (the legacy second-model path,
`--opd-teacher-load` is equivalent). Local sglang teachers specifically require
OFT: unified LoRA is single-active, so it cannot route the student, frozen base,
and teacher independently. OFT `self:*` teachers also require
`--adapter-double-buffer` to stay disabled because double buffering has only one
fixed active adapter slot; use Ray transport for a distributed rollout or IPC
for a colocated rollout instead. Frozen `base` and `adapter:<path>` teachers
remain supported with double buffering. In sglang local mode (no
`--opd-teacher-url`), scoring is a built-in rollout stage: `--custom-rm-path`
stays free, so real task rewards compose with distillation (`--use-opd`
blend now works with sglang teachers) and eval accuracy is meaningful again.
