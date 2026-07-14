# On-Policy Distillation (OPD) Recipes

The active baseline is `math_qwen3_32b_8b_3nodes/`: Qwen3-8B student,
Qwen3-32B SGLang teacher, and one dedicated node each for the teacher, actor,
and rollout workers. It runs pure sampled-token distillation by default:
`--opd-kl-coef 1.0`, task reward 0, reward/loss KL 0, entropy 0, and no
`--normalize-advantages`.

```text
math_qwen3_32b_8b_3nodes/   Active 3-node baseline. Whole-node TP=8 teacher
                            on the head node, 8-GPU Megatron actor, and 8-GPU
                            SGLang rollout worker.
archive/                    Historical 1-node smoke recipes; not active
                            baselines or current validation targets.
```

The active recipe defaults `OPD_TOP_K=0`. This is sampled-token OPD: teacher
scoring is a compact prefill request and the distillation advantage depends on
the sampled token. `OPD_TOP_K>0` remains experimental. The current top-k path
precomputes one divergence value per response position and injects it through
sampled-token PPO/GRPO advantage; it is not a differentiable teacher-top-k
`[T,K]` distillation loss. It also incurs slower rollout decoding and an
`O(length x response-wide candidate union)` scoring payload.

The recipe is a quick-check configuration and intentionally omits checkpoint
saving. W&B uses entity `M3TRL`, project `OPD`.

Launch (on the slinky cluster, use `HF_CACHE_DIR=/data/shared`; the default
`/data/shared/hf_cache` is read-only):

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh OPD/math_qwen3_32b_8b_3nodes/qwen3-8B
```

The teacher model must already exist on disk:

```bash
hf download Qwen/Qwen3-32B --local-dir $HF_CACHE_DIR/models/Qwen3-32B
```

The launcher converts the student torch_dist artifact on first use if it is
missing. See `docs/advanced/on-policy-distillation.md` for the core OPD path.
