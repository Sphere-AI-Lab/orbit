# Phase-0 launcher qualification — 0.5B slice (2026-08-21, i305, 4×B200)

Scope: the launchers Phase 1 consumes, run at Qwen2.5-0.5B-Instruct on the node's four B200s
(plan: `docs/superpowers/plans/2026-08-19-adapter-first-phase0-phase1.md`, Task 8). The 4B/30B
pieces — harness `q3_4b`/`q3_30b`, the 4B fully-async launcher — need 8 GPUs and the 4B/30B
`torch_dist` conversions and are still pending. Logs: `/lustre/home/zqiu/log/phase0-*.log`
(driver stdout) and `logs/adapter_runtime_compare/<campaign>/<run_id>/console.log` (harness runs).

## Result ledger

| Launcher / arm | Result | Evidence |
|---|---|---|
| harness `pilot` (0.5B LoRA r32, async single-slot, 2+2) | PASS, 265 s | `pilot_20260821_022845/…lora_async_g0123`: all five `perf/update_weights_*` keys; warm update 0.086 s, payload 31.1 MB, pause 0.069 s |
| harness `q25` OFT `sync` (colocated, ipc/cpu_gather) | PASS, 195 s | `oft6a/…oft_sync_g0123`: warm update 0.20 s, payload 61.9 MB (4 engines × 15.5 MB — per-rank sum), pause 0.19 s |
| harness `q25` OFT `async` (single-slot, NCCL) | **unsupported by design** | engine: "distributed non-double-buffer OFT adapter sync … not supported; enable --adapter-double-buffer" → constraint 8 |
| harness `q25` OFT `async_db` (double-buffer, NCCL) | PASS, 171 s | `oft5b/…oft_async_db_g0123`: warm update 0.077–0.086 s, payload 15.5 MB, pause 0.07–0.08 s |
| harness `q25` full-FT `async_fullft` (broadcast) | PASS, 228 s | `q25_20260821_023310/…none_async_fullft_g0123`: warm update 0.124 s; payload/pause keys absent (emission for this path landed afterwards in `2c354ab`) |
| OPD free-teacher (`--opd-teacher base`) | PASS | `phase0-opd-free.log`; saves `actor/iter_*/adapter/adapter_megatron_tp0_pp0.pt` with `EXTRA_TRAIN_ARGS="--save-interval 1"` |
| OPD `self:ema` (sglang-local teacher, OFT) | PASS | `phase0-opd-ema-6.log`: 15 streamed payloads each to `orbit_oft` and `orbit_teacher` |
| OPD mopd (`--opd-teacher-load`) | PASS | `phase0-opd-mopd.log` |
| OPD served full-vocab (`--opd-serve-teacher`) | PASS | `phase0-opd-served-3.log` (after sglang `3748b2494`) |
| OPD adapter-swap (`--opd-teacher adapter:`) | PASS | `phase0-opd-adapter.log`, teacher = free-teacher's saved LoRA-16 adapter |
| 4B fully-async; harness `q3_4b`, `q3_30b` | NOT RUN | 8 GPUs + 4B/30B torch_dist paths pending |

Standing guard metric `train/train_rollout_logprob_abs_diff` sat at 0.010–0.014 in every adapter run
(LoRA, OFT single-slot colocated, OFT double-buffer, EMA), i.e. every pushed adapter reached the
engine intact.

## Defects surfaced and fixed

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Ray: "start raylet with 6 GPU, but CUDA_VISIBLE_DEVICES contains [0,1,2,3]" | launchers hardcoded `GPUS_PER_NODE`/`ROLLOUT_NUM_GPUS`/`COLOCATE_ARGS`, shadowing the harness's topology env | orbit `620fb89`: 11 launchers honor the env with their literals as defaults; `tests/fast/test_launcher_topology_env.py` |
| 2 | `'TritonOFTBackend' object has no attribute 'batch_info'` at prefill CUDA-graph capture | the `breakable` capture path never bound the PEFT batch_info (the tc-piecewise path did) | sglang `e679123c5` |
| 3 | full-vocab teacher: "expected meta_info['hidden_states'] to have exactly 1 entry, got 0" | new `hs[:finished_len]` truncation drops the prefill block when `max_new_tokens=0` | sglang `3748b2494` |
| 4 | OFT async: `input_dim (448) must be divisible by block_size (128)` | harness's `OFT_BLOCK_SIZE=64` ignored (hardcoded PEFT flags) and per-engine TP = all rollout GPUs | orbit `d33c967`: `OFT_BLOCK_SIZE` honored; `Case.rollout_gpus_per_engine` knob |
| 5 | `async_db` arm identical to `async` | nothing passed `--adapter-double-buffer` | orbit `d33c967`: `ADAPTER_DOUBLE_BUFFER=1` → flag, generic in `launcher.sh` |
| 6 | OFT async: 400 on `/update_adapter_from_distributed` | engine requires double-buffer for OFT over NCCL (by design) | no code fix; constraint 8 + arm reassignment (A1/A2) |
| 7 | OFT + prefill graphs: NaN at first sample; colocated: "Breakable CUDA graph is not compatible with memory saver mode"; `tc_piecewise`: torch.compile error in OFT layers | sglang v0.5.16 enabled prefill CUDA graphs by default | orbit `fe9ab9b`: default `--sglang-cuda-graph-backend-prefill disabled`, rejected under OFT; constraint 9; follow-up I-8 |
| 8 | colocated sync: `pidfd_getfd: Operation not permitted` | CUDA IPC denied on this cluster | `ORBIT_PEFT_ADAPTER_TRANSPORT=cpu_gather` (env.sh default; activate.sh does not set it); constraint 10 |
| 9 | full-FT async arm emitted only `update_weights_time` | distributed broadcast path recorded payload but never emitted metrics or timeline markers | orbit `2c354ab` + `tests/fast/test_distributed_update_weights_sync_metrics.py` |
| 10 | latent: streamed OFT loader rebinds `adapter_id` while scanning refs | loop-variable shadowing | sglang `9a6b12d5b` |
| 11 | EMA smoke: `assert oft_adapter is not None` with IPC transport | IPC push reported success without registering the adapter (silent) | masked by cpu_gather; follow-up I-9: surface engine-side IPC failures |

Also learned: `run_compare.py --profile pilot` is LoRA-async-only (a plumbing smoke); the four-arm
0.5B qualification is `--profile q25 --pefts oft --modes sync,async_db,async_fullft` plus
`--pefts lora --modes async` for the single-slot NCCL arm. The full-vocab smoke needs
`OPD_SERVE_TEACHER=1 OPD_TEACHER_HF_CKPT=… ROLLOUT_NUM_GPUS=1` on a 4-GPU node.

## Follow-ups opened

- **I-8** — make the sglang prefill CUDA-graph replay apply OFT adapters (then the `disabled` default can be revisited for throughput).
- **I-9** — the IPC PEFT transport must fail loudly when the engine does not register the adapter.
- Payload accounting caption for A1: colocated points report per-rank sums (engines × adapter), broadcast/NCCL points the logical payload once.
