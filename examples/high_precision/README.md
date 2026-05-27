# High-Precision Launchers

These launchers run BF16 or high-precision Orbit training recipes. Each file is
an independent entrypoint and sources shared Orbit launcher libraries from
`scripts/lib/`.

Common smoke overrides:

```bash
NUM_ROLLOUT=1 TOTAL_EPOCHS=1 TRAIN_ROWS=1 \
ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=1 GLOBAL_BATCH_SIZE=1 \
DISABLE_EVAL=1 ENABLE_WANDB=0
```
