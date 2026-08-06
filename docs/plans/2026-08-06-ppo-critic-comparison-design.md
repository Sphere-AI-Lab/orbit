# PPO Full-Critic vs Adapter-Critic Comparison

## Objective

Compare Orbit's dense, separate full critic with its in-actor adapter critic without mixing learning-quality and systems-efficiency claims. The benchmark therefore has two explicitly labeled panels:

1. **Controlled learning panel:** hold the actor, rollout capacity, data order, sampling, PPO configuration, and evaluation fixed. Compare policy quality and critic optimization dynamics against rollouts, samples, and generated tokens.
2. **Fixed-budget panel:** occupy the same four B200 GPUs and allow the adapter configuration to spend its freed critic GPU on rollout. Compare end-to-end throughput, GPU-hours, and time-to-quality.

These panels answer different questions and must be reported separately.

## Model and task

The main benchmark uses:

- Qwen2.5-3B-Instruct in BF16 for both actor and critic trunks.
- Canonical OFT for the actor in both critic modes.
- OpenR1-style exact-answer math JSONL for PPO training.
- Math500 as the primary held-out evaluation set.
- AIME 2024 and AMC 2023 as secondary evaluation sets.
- Orbit's deterministic math reward, avoiding a learned reward model or tool-use confounder.

Qwen2.5-0.5B-Instruct and a small GSM8K-style subset are reserved for short launcher smoke tests; their results are not benchmark results.

The validated cluster inputs are:

- HF model: `/fast/groups/ei-slm/hf_models/Qwen2.5-3B-Instruct`;
- training candidate: `/fast/groups/ei-slm/data/peft_arena_openr1_50k/train.jsonl`;
- aligned evaluation directory: `/fast/groups/ei-slm/data/peft_arena_eval_math_alignment`.

The training candidate has 49,990 string labels and 10 null labels. A deterministic filtered copy was prepared at `/lustre/fast/fast/lechen/clthegoat/ppo_critic_benchmark_data/openr1_49990/train.jsonl` (49,990 rows; SHA-256 `29608e7b64328af1215dca3971b84dbd2e1c39e0614d076ccf4168e86307ad25`). The aligned evaluation files include the `math_alignment` metadata required for dataset-specific grading; the similarly named `peft_arena_eval_orbit` files do not and should not be substituted.

The validated Megatron `torch_dist` conversion is at `/lustre/fast/fast/lechen/clthegoat/ppo_critic_benchmark_models/Qwen2.5-3B-Instruct_torch_dist` (5.8 GB; iteration 0; 16 distributed shards; filename/size manifest SHA-256 `a2c8e1ad824f1ba6b899e9956c4cb68105cbdec0b2c6ce56d4e11dc12b7938d0`). It was produced from the HF model above with `tools/convert_hf_to_torch_dist.py` in BF16. The launcher's per-run metadata records the live manifest again, so a later artifact change is visible.

## Experimental matrix

| Panel | Critic | Actor GPUs | Critic GPUs | Rollout GPUs | Total occupied GPUs |
|---|---|---:|---:|---:|---:|
| Controlled | Full | 1 | 1 | 2 | 4 |
| Controlled | Adapter | 1 | 0 | 2 | 3 (1 deliberately idle) |
| Fixed budget | Full | 1 | 1 | 2 | 4 |
| Fixed budget | Adapter | 1 | 0 | 3 | 4 |

The controlled panel intentionally leaves one B200 idle for the adapter critic. This keeps rollout service capacity identical and isolates sample efficiency. The fixed-budget panel uses all four B200s and measures the practical benefit of removing the separate critic worker.

The two full-critic rows have the same topology and recipe. A completed full-critic run for a seed can therefore serve both panels; the two wrappers remain for a symmetric operator interface and distinct output identities when independent runs are desired.

Run at least three matched seeds for claims about learning. A single seed is acceptable only for launcher qualification and performance debugging.

## Launcher structure

The launcher suite consists of one shared benchmark recipe and four thin entry points:

- `ppo_critic_compare_common.sh`
- `run-qwen2_5-3b-math-oft-ppo-full-critic-controlled.sh`
- `run-qwen2_5-3b-math-oft-ppo-adapter-critic-controlled.sh`
- `run-qwen2_5-3b-math-oft-ppo-full-critic-budget.sh`
- `run-qwen2_5-3b-math-oft-ppo-adapter-critic-budget.sh`

The wrappers select only `PPO_CRITIC_MODE`, `PPO_COMPARISON_PANEL`, and the approved resource layout. The common recipe owns every scientific hyperparameter. This makes configuration drift visible and keeps each pair mechanically comparable.

All dataset and checkpoint paths are supplied through environment variables. Run identity, output directory, log filename, and W&B group include panel, critic mode, and seed. Environment overrides support small smoke settings without changing the benchmark defaults.

A main launch has the following shape:

```bash
cd /lustre/fast/fast/lechen/clthegoat/orbit-ppo-critic-benchmark
export HF_CKPT=/fast/groups/ei-slm/hf_models/Qwen2.5-3B-Instruct
export MEGATRON_LOAD=/lustre/fast/fast/lechen/clthegoat/ppo_critic_benchmark_models/Qwen2.5-3B-Instruct_torch_dist
export TRAIN_JSONL=/lustre/fast/fast/lechen/clthegoat/ppo_critic_benchmark_data/openr1_49990/train.jsonl
export EVAL_ORBIT_DIR=/fast/groups/ei-slm/data/peft_arena_eval_math_alignment
export SAVE_ROOT=/lustre/fast/fast/lechen/clthegoat/ppo_critic_benchmark_runs
export SEED=1234

bash examples/high_precision/run-qwen2_5-3b-math-oft-ppo-full-critic-controlled.sh
```

Select another wrapper to change the declared panel/mode. To resume, provide the same scientific inputs and set `RESUME_DIR` to that run's directory; do not set `SAVE_DIR` or `CRITIC_LOAD` independently.

Smoke mode uses different model arguments, so it must use the 0.5B checkpoints rather than inheriting the 3B exports. The following qualification commands disable evaluation; alternatively, provide a compatible `TEST_JSONL` and omit `DISABLE_EVAL`:

```bash
cd /lustre/fast/fast/lechen/clthegoat/orbit-ppo-critic-benchmark
export HF_CKPT=/fast/groups/ei-slm/hf_models/Qwen2.5-0.5B-Instruct
export MEGATRON_LOAD=/fast/groups/ei-slm/hf_models/Qwen2.5-0.5B-Instruct_torch_dist
export TRAIN_JSONL=/fast/groups/ei-slm/data/lora_regret/gsm8k_train.jsonl
export SAVE_ROOT=/lustre/fast/fast/lechen/clthegoat/ppo_critic_benchmark_qualification
export SEED=260806
export SMOKE=1
export DISABLE_EVAL=1

bash examples/high_precision/run-qwen2_5-3b-math-oft-ppo-full-critic-controlled.sh
bash examples/high_precision/run-qwen2_5-3b-math-oft-ppo-adapter-critic-controlled.sh
```

## Controlled variables

Both critic modes use the same:

- actor base-checkpoint path and manifest (this recipe has no reference-policy worker);
- model architecture, BF16 precision, OFT targets, OFT block size, and initialization;
- prompt file, prompt order, rollout-shuffle seed, and training seed;
- rollout batch size, samples per prompt, response limit, temperature, sampling parameters, and deterministic inference mode;
- the one-pass-per-rollout update schedule, global/micro batch settings, KL shaping, GAE, clipping, loss normalization, and optimizer schedule;
- evaluation datasets, evaluation cadence, save cadence, and total rollout horizon.

Only the critic architecture and the resource differences declared in the matrix may vary. The full critic receives the same base checkpoint explicitly and uses a critic worker. The adapter critic sets `--critic-mode adapter`, uses no separate critic worker, and saves its value adapter/head sidecar alongside the actor checkpoint under the same run directory.

Exact rollout equality is expected only before the learned policies diverge. Across a training run, fairness means matched random streams and prompt schedules, not identical generated trajectories.

## Resume and reproducibility

Benchmark runs save actor/critic parameters, optimizer, scheduler, and dataset state. The native full critic also preserves its RNG state. Orbit's current PEFT actor and adapter-critic sidecars do not preserve RNG state, so a resumed run is not promised to be bitwise identical to an uninterrupted run. A resume must continue into the same run identity and reject an incompatible critic mode, panel, seed, schedule, input fingerprint, or recipe revision unless the operator explicitly starts a new run.

Each launch logs the post-normalization command and a manifest containing the code revision/status/diff, launcher and entry-point hashes, dataset content hashes, model-checkpoint file manifests, reward timeout, Ray/SGLang settings, resources, and schedule. Main runs refuse a dirty or untracked worktree unless the operator explicitly sets `ALLOW_DIRTY_BENCHMARK=1`. The launchers validate non-null prompt/label records and require dataset-specific `math_alignment` metadata in all three main evaluation files.

Each run also uses a stable W&B ID across resumes and a canonical, checkpoint-adjacent writer lock. A SIGKILL or node loss can leave `<SAVE_DIR>.launch-lock`; inspect its `owner.tsv` and verify that no writer remains before manually removing that stale directory. The scripts require existing local checkpoint and dataset paths rather than silently choosing local data. Model directory manifests cover relative filenames and sizes; they are drift detectors, not cryptographic claims about every weight byte.

## Measurements

The initial launcher work uses Orbit's existing metrics:

- reward, raw reward, main-benchmark pass@k, response length, and truncation;
- on-policy critic value loss, value clipping fraction, critic gradient norm, and learning rate;
- policy loss, entropy, policy clipping fraction, PPO KL, and log-probability mismatch;
- critical-path rollout duration and per-rollout-GPU token throughput;
- externally measured end-to-end wall time and successful checkpoint/resume behavior.

For analysis:

- plot learning quality against rollout number, sampled responses, and generated tokens for the controlled panel;
- plot quality against wall time and GPU-hours for the fixed-budget panel;
- multiply the existing per-rollout-GPU throughput by the rollout-GPU count when reporting aggregate rollout throughput;
- measure process wall time outside Orbit because `progress/elapsed_seconds` excludes startup and restarts after resume;
- treat critic losses as optimization diagnostics, not held-out critic-quality estimates;
- do not directly compare the existing `timing_s/actor_train` value across modes, because full critic training overlaps actor training while adapter-critic training is sequential on the actor worker.

Critic explained variance, held-out return RMSE, equivalent per-phase timing, peak per-rank VRAM, utilization, and trainable parameter counts are desirable follow-up instrumentation. They are deliberately outside the first launcher-only change.

## Validation

Before any long benchmark run:

1. Run `bash -n` on the common recipe and all wrappers.
2. Resolve/dry-run all four commands and verify pairwise argument parity, allowing differences only in identity, critic mode/checkpoint, and declared resources.
3. Verify required paths, resource counts, unique run identities, and resumable checkpoint flags.
4. Exercise fresh preparation, incompatible-artifact rejection, and synthetic adapter resume without starting Ray.
5. Run a two-rollout Qwen2.5-0.5B smoke test for full and adapter critics on the available B200s. Smoke evaluation uses one sample and therefore does not report pass@k; evaluation may be disabled for the shortest hardware qualification.
6. Inspect logs for finite losses, checkpoint creation, clean process shutdown, and—when enabled—successful evaluation.

Only after these checks should the Qwen2.5-3B multi-seed matrix be launched.

## Non-goals

This change does not alter PPO math, adapter-critic implementation, reward semantics, or production metric instrumentation. It prepares reproducible launchers and validation checks for the comparison.
