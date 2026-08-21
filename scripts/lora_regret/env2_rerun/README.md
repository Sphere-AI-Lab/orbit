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
