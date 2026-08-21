# Clean env2 E4 rerun

This directory is the isolated launcher set for the 2026-08-21 rerun. It does
not reuse the old ledgers, per-arm logs, local W&B files, or checkpoints. The
exported `E4_ENV2_SCHEDULER_DIR` is the destination for the separate scheduler
submission layer; these training wrappers do not create scheduler logs.

Each `lr1`–`lr7` wrapper keeps the original FullFT learning rate and shifts the
LoRA grid down by one old column:

| New column | FullFT LR | LoRA r1/r16/r256 LR |
|---|---:|---:|
| `lr1` | `5e-8` | `2e-6` (old `lr0`) |
| `lr2` | `1e-7` | `5e-6` (old `lr1`) |
| `lr3` | `3e-7` | `1e-5` (old `lr2`) |
| `lr4` | `7e-7` | `3e-5` (old `lr3`) |
| `lr5` | `2e-6` | `7e-5` (old `lr4`) |
| `lr6` | `4e-6` | `2e-4` (old `lr5`) |
| `lr7` | `1e-5` | `4e-4` (old `lr6`) |

`env.sh` activates `/fast/zqiu/orbit-iclr/orbit_env_v2` and defaults every
artifact to:

```text
/lustre/fast/fast/zqiu/orbit-iclr/experiment-runs/env2-rerun-20260821/
  results/
  logs/lora_regret/
  wandb/
  orbit_ckpts/lora_regret/
  scheduler/
```

Run one whole-node column with, for example:

```bash
bash scripts/lora_regret/env2_rerun/run_e4_gsm8k_lr1_8gpu.sh
```

The script runs one FullFT arm followed by LoRA ranks 1, 16, and 256. Repeating
the command resumes from its new ledger. To upload only this rerun's offline
W&B files from a host with egress:

```bash
bash scripts/lora_regret/env2_rerun/sync_wandb.sh
```

## Run length

Every env2 runner exports `NUM_ROLLOUT` per dataset from `columns.sh` before
calling the campaign: **MATH 150, GSM8K 200**. One rollout is one optimizer
step under the E4 protocol, so these are step counts. An explicit
`NUM_ROLLOUT=...` in the calling shell still overrides both, and each runner
prints `rollouts=` so the value in force is in the log.

## Per-method FullFT and LoRA sweeps

The column wrappers above run FullFT and all three LoRA ranks into one ledger,
which is right for one node per column. To put one *method* on one node
instead, the per-method launchers run all seven columns sequentially and keep
a separate ledger per (method, rank, column), so a FullFT node and three LoRA
nodes can run the same column at the same time without sharing a file:

```bash
bash scripts/lora_regret/env2_rerun/run_e4_math_ft_lr1_lr7_8gpu.sh
bash scripts/lora_regret/env2_rerun/run_e4_gsm8k_ft_lr1_lr7_8gpu.sh
bash scripts/lora_regret/env2_rerun/run_e4_math_lora_r1_lr1_lr7_8gpu.sh     # also r16, r256
bash scripts/lora_regret/env2_rerun/run_e4_gsm8k_lora_r1_lr1_lr7_8gpu.sh    # also r16, r256
```

Ledgers land in `results/` as `e4_<dataset>_ft_lr<N>.jsonl` and
`e4_<dataset>_lora_r<rank>_lr<N>.jsonl`; the learning rate of column `N` is
the same as in the table above. A single column is

```bash
bash scripts/lora_regret/env2_rerun/run_ft_column.sh math 4
bash scripts/lora_regret/env2_rerun/run_lora_column.sh gsm8k 16 4
```

These ledgers are separate from the `e4_<dataset>_lr<N>.jsonl` ones written by
`run_column.sh`, so an arm finished under one launcher family is not skipped by
the other. Pick one family per (dataset, column). The column tables are in
`columns.sh`, shared by every `run_*_column.sh`.

## OFT grid

The OFT wrappers use block size 128 on all target modules. The seven-column
grid puts the historical MATH optimum at its midpoint:

| Column | OFT LR |
|---|---:|
| `lr1` | `5e-7` |
| `lr2` | `1e-6` |
| `lr3` | `3e-6` |
| `lr4` | `7e-6` |
| `lr5` | `2e-5` |
| `lr6` | `4e-5` |
| `lr7` | `1e-4` |

Each MATH or GSM8K wrapper runs one OFT arm into its own resumable ledger:

```bash
bash scripts/lora_regret/env2_rerun/run_e4_math_oft_lr4_8gpu.sh
bash scripts/lora_regret/env2_rerun/run_e4_gsm8k_oft_lr4_8gpu.sh
```

To run all seven columns sequentially on one allocated 8-GPU node:

```bash
bash scripts/lora_regret/env2_rerun/run_e4_math_oft_lr1_lr7_8gpu.sh
bash scripts/lora_regret/env2_rerun/run_e4_gsm8k_oft_lr1_lr7_8gpu.sh
```

These aggregate launchers call the per-column wrappers above, so they share the
same ledgers. Re-running an aggregate launcher skips completed columns safely.

## HTCondor submission

`condor/` holds one submit file per aggregate launcher, all requesting a whole
H100 node (`request_gpus = 8`, `request_cpus = 32`, `request_memory = 1000000`,
`request_disk = 1000G`, `CUDADeviceName == "NVIDIA H100 80GB HBM3"`):

```text
condor/e4_{math,gsm8k}_ft_lr1_lr7.sub
condor/e4_{math,gsm8k}_lora_r{1,16,256}_lr1_lr7.sub
condor/e4_{math,gsm8k}_oft_lr1_lr7.sub
```

Submit with the cluster's bid wrapper, one node per file:

```bash
condor_submit_bid 35 scripts/lora_regret/env2_rerun/condor/e4_math_ft_lr1_lr7.sub
# or several at once, same bid:
bash scripts/lora_regret/env2_rerun/condor/submit.sh e4_math_ft_lr1_lr7 e4_gsm8k_ft_lr1_lr7
```

Every job writes into `E4_ENV2_SCHEDULER_DIR` (the submit files spell out the
same default path as `env.sh`; change both together):

```text
scheduler/<name>.<ClusterId>.stdout.log   launcher output, stderr merged in
scheduler/<name>.<ClusterId>.stderr.log   HTCondor's own stderr (normally empty)
scheduler/<name>.<ClusterId>.condor.log   HTCondor job event log
scheduler/<name>.<ClusterId>.status       state, commit, host, exit code
```

`condor/job.sh` is the executable: it sources `env.sh`, writes the status file,
refuses to start unless `nvidia-smi` reports exactly eight H100 80GB HBM3
devices (`state=allocation_unhealthy`, exit 3), then runs the launcher.
`getenv` imports only `HOME, USER, PATH, LANG`, so a `NUM_ROLLOUT` left in the
submit shell cannot reach the job. The ledgers are the resumable state, so a
job that dies can simply be resubmitted. Do not submit an OFT file while the
same sweep is still running interactively: both would append to one ledger.

