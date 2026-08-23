# Adapter-first experiment program launchers

Launchers for the adapter-first experiment program
(`docs/plans/2026-08-17-adapter-first-experiments-design.md`,
`docs/superpowers/plans/2026-08-19-adapter-first-phase0-phase1.md`). Each script
drives `tools/adapter_runtime_compare/run_compare.py` for one model rung with
the constraint-8 arm assignment: OFT runs `sync` and double-buffer `async_db`
only (the engine rejects single-slot OFT on the distributed path), the
single-slot NCCL arm is LoRA `async`, and the full-model broadcast control is
`async_fullft`.

`env.sh` (sourced by every launcher) activates the workspace env through
`uv_env_build/activate.sh` (`ORBIT_ENV=cu130` by default), sets
`ORBIT_PEFT_ADAPTER_TRANSPORT=cpu_gather` (constraint 10: CUDA IPC is denied on
this cluster), wires the harness branch variables, and provides the checkpoint
and data defaults per rung. Every value is an environment variable you can
export before running. `NUM_ROLLOUT` (default 4) and `CAMPAIGN` are also
overridable; `NUM_ROLLOUT=1` is the cheapest launch test that still crosses Ray
placement, SGLang engine start, Megatron `torch_dist` load, the first weight sync
and one PPO step.

| launcher | rung / arms | GPUs | status (cu130 env, 4xB200, 2026-08-23) |
|---|---|---|---|
| `phase0-q25-oft-arms.sh` | 0.5B: oft/sync, oft/async_db, fullft/async | 4 | verified, 4 rollouts: 3/3 ok |
| `phase0-q25-lora-async.sh` | 0.5B: lora/async | 4 | verified, 4 rollouts: ok |
| `phase1-q25-3b-oft-arms.sh` | 3B: oft/sync, oft/async_db | 4 | launch-verified, 1 rollout: 2/2 ok |
| `phase1-q3-4b-arms.sh` | 4B bf16: oft ×2, lora ×3 | 4 | launch-verified, 1 rollout: 5/5 ok (oft 233 s/~160 s, lora 178/158/165 s) |
| `phase1-q3-30b-arms.sh` | 30B-A3B bf16: oft ×2, lora ×3 | **8** | NOT verified: needs an 8-GPU allocation |

The harness's 4B LoRA cases used to point at the OFT launcher (which hardcodes
`--peft-method oft`), so they silently trained OFT; `run-qwen3-4b-instruct-2507-bf16-math-lora.sh`
now exists and the case table references it.

Not covered: the 4B INT4 cases (`examples/low_precision/run-qwen3-4b-int4-math-oft.sh`)
need a W4A16 checkpoint that does not exist yet; the 30B rung needs 8 GPUs.

Outputs land under `logs/adapter_runtime_compare/<campaign>/<run_id>/`
(`console.log`, `status.json` with `returncode` and `wall_s`, `run.json`).

## Checkpoints

`env.sh` defaults: 0.5B and 3B `torch_dist` conversions exist on disk; the 4B
`torch_dist` must be produced once (33 s on one B200):

```bash
source examples/adapter_first/env.sh
python tools/convert_hf_to_torch_dist.py --hf-checkpoint "$Q3_4B_HF" --save "$Q3_4B_TORCH_DIST"
```

Write it to node-local disk first if `/lustre/fast` is quota-throttled, then
copy it to the group path (or export `Q3_4B_TORCH_DIST` to wherever it lives).
