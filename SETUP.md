# Setup — `feat/mopd`

Environment recipe for this branch (on-policy distillation: sampled-token + full-vocab
teacher score modes, managed teacher serving, teacher pools, one-trunk PPO).

## Repo pins

| Component | Clone | Branch | Install |
|---|---|---|---|
| orbit | `Sphere-AI-Lab/orbit-develop` | `feat/mopd` | `pip install -e .` |
| SGLang | `Sphere-AI-Lab/sglang-develop` | `feat/dev` | `pip install -e python/` |
| Megatron-LM | `Sphere-AI-Lab/Megatron-LM` | `orbit-main` | `pip install -e .` (provides `megatron-core`) |
| Megatron-Bridge | `Sphere-AI-Lab/Megatron-Bridge` | `orbit-main` | `pip install -e .` |

`sglang-develop @ feat/dev` = the v0.5.9 Sphere base plus the tensor hidden-states
encoding. Plain `main` also works for full-vocab, but teachers fall back to a slow
nested-JSON encoding (orbit warns once per process).

## Environment

- Python 3.12, torch 2.11 (cu13x), and a CUDA 13.x toolkit on the machine.
  Export `CUDA_HOME` to it — some extensions (e.g. deep_ep) need it at import time.
- Install the four repos editable into one venv (order doesn't matter).
- Ray workers import orbit by path: the launchers prepend the repo checkout to
  `PYTHONPATH`. If you run trainers by hand, do the same —
  `export PYTHONPATH=/path/to/orbit-develop` — or a stale editable install wins.
- `MEGATRON_PATH` (the full Megatron-LM tree, not just core) is auto-inferred from
  the editable install by `scripts/lib/ray.sh`; export it explicitly to override.
- Cluster proxies: if `http_proxy` is set, make sure `no_proxy` covers `127.0.0.1`,
  the hostname, and the node IPs, or SGLang server warmup kills itself through the
  proxy. The launchers' own preflight is proxy-immune.
- `WANDB_API_KEY` in the environment if you use `--use-wandb`.

## Full-vocab OPD in one paragraph

The teacher is an SGLang server scored prefill-only with hidden states returned;
the trainer reconstructs full-vocab teacher logits as `hidden @ lm_head.T`
(vocab-sharded for TP) and trains the generalized JSD (GKD Eq. 1). Key flags:
`--teacher-score-mode full_vocab --loss-type opd_jsd_loss --opd-jsd-beta 0.5
--teacher-hf-checkpoint <teacher HF dir>` (the trainer loads the teacher LM head
from there; tied-embedding models and teacher-wider-than-student vocab padding are
handled automatically).

Teacher serving, either way:

- **Managed** (recommended): `--opd-serve-teacher --opd-teacher-num-gpus N`
  (`--colocate` shares training GPUs). Orbit launches the server with the required
  flags and publishes the URL itself.
- **External**: run the server yourself with `--enable-return-hidden-states
  --disable-radix-cache --chunked-prefill-size -1`, then pass
  `--opd-teacher-url http://host:port/generate`. Orbit validates the server
  config at decode time and fails loud if a flag is missing.

Multiple teachers: `--opd-teacher-pool pool.yaml` (kinds `url`/`served`, weights,
metadata routing) — sampled-token mode only; full-vocab is single-teacher.

## Quickstart

```bash
# 2-GPU smoke (0.5B student, LoRA, managed 0.5B teacher):
OPD_SERVE_TEACHER=1 bash examples/on_policy_distillation/run-qwen2_5-0_5b-opd-full-vocab-smoke.sh

# The 7B->0.5B GSM8K science run (4 GPUs; reproduced 48.2% -> 55.0% pass@1):
bash examples/on_policy_distillation/run-qwen2_5-0_5b-opd-full-vocab-gsm8k.sh
```

Both are env-tunable (`COLOCATE`, `OPD_TEACHER_NUM_GPUS`, `OPD_JSD_BETA`,
`GLOBAL_BATCH_SIZE`, and for the science run `LORA_RANK`, `LR`, `EVAL_INTERVAL`);
see the script headers. `ORBIT_DRY_RUN_ARGV=1` prints the resolved argv without
touching GPUs. The other launchers in the same directory cover sampled-token MOPD,
EMA self-teachers, teacher pools, and PPO blends.

## Known limits on these pins

- `--optimizer pion` / `pion_msign` is not available (the kernels live in an
  unpublished Megatron commit); use the default optimizer.
- Full-vocab mode requires OPD hooks and is mutually exclusive with
  advantage-based losses (`compute_advantages_and_returns` must stay off);
  the arg validation enforces the legal combinations and says why.
- On 4-GPU nodes, the adapter-critic smokes need `CRITIC_NUM_GPUS_PER_NODE=0`
  (the GPU budget helper otherwise reserves critic GPUs the node doesn't have).
