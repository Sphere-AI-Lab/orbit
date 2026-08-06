# PPO Full-Critic vs Adapter-Critic Comparison

## Objective

Compare Orbit's dense, separate full critic with its in-actor adapter critic without mixing learning-quality and systems-efficiency claims. The benchmark therefore has two explicitly labeled panels:

1. **Controlled learning panel:** hold the actor, rollout capacity, data order, sampling, PPO configuration, and evaluation fixed. Compare reward and critic quality against rollouts, samples, and generated tokens.
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

## Experimental matrix

| Panel | Critic | Actor GPUs | Critic GPUs | Rollout GPUs | Total occupied GPUs |
|---|---|---:|---:|---:|---:|
| Controlled | Full | 1 | 1 | 2 | 4 |
| Controlled | Adapter | 1 | 0 | 2 | 3 (1 deliberately idle) |
| Fixed budget | Full | 1 | 1 | 2 | 4 |
| Fixed budget | Adapter | 1 | 0 | 3 | 4 |

The controlled panel intentionally leaves one B200 idle for the adapter critic. This keeps rollout service capacity identical and isolates sample efficiency. The fixed-budget panel uses all four B200s and measures the practical benefit of removing the separate critic worker.

Run at least three matched seeds for claims about learning. A single seed is acceptable only for launcher qualification and performance debugging.

## Launcher structure

Create one shared benchmark recipe and four thin entry points:

- `ppo_critic_compare_common.sh`
- `run-qwen2_5-3b-math-oft-ppo-full-critic-controlled.sh`
- `run-qwen2_5-3b-math-oft-ppo-adapter-critic-controlled.sh`
- `run-qwen2_5-3b-math-oft-ppo-full-critic-budget.sh`
- `run-qwen2_5-3b-math-oft-ppo-adapter-critic-budget.sh`

The wrappers select only `CRITIC_MODE`, `COMPARISON_PANEL`, and the approved resource layout. The common recipe owns every scientific hyperparameter. This makes configuration drift visible and keeps each pair mechanically comparable.

All dataset and checkpoint paths are supplied through environment variables. Run identity, output directory, log filename, and W&B group include panel, critic mode, and seed. Environment overrides support small smoke settings without changing the benchmark defaults.

## Controlled variables

Both critic modes use the same:

- actor and reference checkpoint bytes;
- model architecture, BF16 precision, OFT targets, OFT block size, and initialization;
- prompt file, prompt order, rollout-shuffle seed, and training seed;
- rollout batch size, samples per prompt, response limit, temperature, and sampling parameters;
- PPO epochs, global/micro batch settings, KL shaping, GAE, clipping, loss normalization, and optimizer schedule;
- evaluation datasets, evaluation cadence, save cadence, and total rollout horizon.

Only the critic architecture and the resource differences declared in the matrix may vary. The full critic receives its explicit initial checkpoint and a critic worker. The adapter critic sets `--critic-mode adapter`, uses no separate critic worker, and saves its value adapter/head state with the actor checkpoint.

Exact rollout equality is expected only before the learned policies diverge. Across a training run, fairness means matched random streams and prompt schedules, not identical generated trajectories.

## Resume and reproducibility

Benchmark runs save optimizer, scheduler, RNG, dataset, actor, and critic state. A resume must continue into the same run identity and reject an incompatible critic mode, panel, seed, or schedule unless the operator explicitly starts a new run.

Each launch logs the resolved command and key experiment metadata. The scripts require explicit checkpoint and dataset paths rather than silently choosing local data.

## Measurements

The initial launcher work uses Orbit's existing metrics:

- reward, raw reward, pass@k, response length, and truncation;
- critic value loss, value clipping fraction, critic gradient norm, and learning rate;
- policy loss, entropy, policy clipping fraction, PPO KL, and log-probability mismatch;
- critical-path rollout duration and rollout token throughput;
- end-to-end wall time and successful checkpoint/resume behavior.

For analysis:

- plot learning quality against rollout number, sampled responses, and generated tokens for the controlled panel;
- plot quality against wall time and GPU-hours for the fixed-budget panel;
- do not directly compare the existing `timing_s/actor_train` value across modes, because full critic training overlaps actor training while adapter-critic training is sequential on the actor worker.

Critic explained variance, held-out return RMSE, equivalent per-phase timing, peak per-rank VRAM, utilization, and trainable parameter counts are desirable follow-up instrumentation. They are deliberately outside the first launcher-only change.

## Validation

Before any long benchmark run:

1. Run `bash -n` on the common recipe and all wrappers.
2. Resolve/dry-run all four commands and verify pairwise argument parity, allowing differences only in identity, critic mode/checkpoint, and declared resources.
3. Verify required paths, resource counts, unique run identities, and resumable checkpoint flags.
4. Run a two-rollout Qwen2.5-0.5B smoke test for full and adapter critics on the available B200s.
5. Inspect logs for finite losses, successful evaluation, checkpoint creation, and clean process shutdown.

Only after these checks should the Qwen2.5-3B multi-seed matrix be launched.

## Non-goals

This change does not alter PPO math, adapter-critic implementation, reward semantics, or production metric instrumentation. It prepares reproducible launchers and validation checks for the comparison.
