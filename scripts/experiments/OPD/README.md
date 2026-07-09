# On-Policy Distillation (OPD) Recipes

`submit.sh`-style ports of `examples/on_policy_distillation/`, wrapped in the
recipe contract (pure config, no side effects; the orchestrator does the I/O).
The SGLang baseline intentionally defaults `--opd-log-prob-top-k` to 2 instead
of the example's 16, and the quick-check baselines omit checkpoint saving.
Doc: `docs/advanced/on-policy-distillation.md`.

```text
megatron_teacher_baseline/  1 node. Teacher loaded into Megatron via
                            --opd-teacher-load (default: the student's own
                            torch_dist = self-distillation smoke config,
                            exactly like the upstream example).
sglang_teacher_baseline/    1 node. Qwen3-32B teacher served by SGLang on
                            head-node GPU 7 (TP=1); student queries it during
                            rollout via the OPD custom reward fn. Defaults to
                            --opd-log-prob-top-k 2 — the paper's 16 wedges a
                            TP=1 teacher at 16k response length (see the
                            recipe header for the full story).
math_qwen3_32b_8b_3nodes/   3 nodes: whole-node TP=8 teacher (head) + 8-GPU
                            Megatron actor node + 8-GPU rollout node. Fail-fast
                            scoring (no retry), effectively uncapped in-flight.
```

The baselines are QUICK-CHECK configs (1 node, 2 GPUs Megatron actor + 4 GPUs
SGLang rollout, **no checkpoint saving**). W&B: entity `M3TRL`, project `OPD`.

Launch (note `HF_CACHE_DIR=/data/shared` on the slinky cluster — the default
`/data/shared/hf_cache` is read-only):

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh OPD/megatron_teacher_baseline/qwen3-8B
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh OPD/sglang_teacher_baseline/qwen3-8B
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh OPD/math_qwen3_32b_8b_3nodes/qwen3-8B
```

The SGLang recipes require the teacher model on disk first (checkpoints/data
are owner-managed, not auto-downloaded):

```bash
hf download Qwen/Qwen3-32B --local-dir $HF_CACHE_DIR/models/Qwen3-32B
```

The student torch_dist artifact is auto-converted by the launcher on first
run if missing.
