# CP2 on Qwen3-VL needs `calculate_per_token_loss` — diagnosis & fix

Status: **diagnosed, fix NOT yet applied** (decide later). The cp2 recipe already
sets `--calculate-per-token-loss`, but that alone is insufficient — see below.

Recipe: `experimental/geo3k-vlm-multi-turn-fully-async-cp2-3node.sh` (TP4 × **CP2** × PP1 + `--allgather-cp`).

## Symptom

Every trainer actor dies at model init → job goes `CLUSTER_DEAD` before any
training step (looks like it's stuck in "warmup"):

```
File ".../megatron/bridge/models/qwen_vl/modelling_qwen3_vl/model.py", line 203, in __init__
    assert self.config.calculate_per_token_loss, (
AssertionError: Qwen3-VL model only supports context parallelism with calculate_per_token_loss enabled
```

Seen on jobs **21102** (no flag set) and **21105** (flag set in the recipe but it
still asserted — that is the real bug below).

## Root cause

Qwen3-VL's Megatron-Bridge model hard-requires `config.calculate_per_token_loss == True`
whenever `cp.size() > 1` (`model.py:203`). The flag is reaching the *args* but not
the *model config*, because orbit builds the config two different ways:

- **Non-bridge path** (`orbit/backends/megatron_utils/model_provider.py:181`):
  `config = core_transformer_config_from_args(args)` — copies **all** Megatron
  training-time args (incl. `calculate_per_token_loss`) into the `TransformerConfig`. ✅
- **Bridge path** (VLM; `model_provider.py:90-93`):
  ```python
  bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
  provider = bridge.to_megatron_provider(load_weights=False)   # -> GPTModelProvider
  ```
  `GPTModelProvider` subclasses `TransformerConfig` (`gpt_provider.py:127`) but is
  built from the **HF `config.json`**, which only describes model *architecture*
  (layers, hidden size, heads…). It has no notion of a training-time switch like
  `calculate_per_token_loss`, so that field stays at the dataclass default **`False`**.

So: `args.calculate_per_token_loss == True` (parsed correctly, and used in the loss
math under `orbit/backends/training_utils/...`), but `provider.calculate_per_token_loss
== False` (what the model actually sees) → assert fires under CP>1.

Confirmed on 21105: arg dump shows `calculate_per_token_loss ... True`, yet the
`assert self.config.calculate_per_token_loss` still tripped.

## Fix (one line, in the bridge branch)

After `provider = bridge.to_megatron_provider(...)` (`model_provider.py:~93`), propagate
the training-time arg onto the provider (which *is* a `TransformerConfig`):

```python
provider = bridge.to_megatron_provider(load_weights=False)
provider.calculate_per_token_loss = args.calculate_per_token_loss
```

Then `provider.provide()` builds the model with `config.calculate_per_token_loss == True`
and the assert passes.

- **Safe for everyone**: default is `False`, so existing bridge runs that don't set
  the flag are unchanged; it only activates when `--calculate-per-token-loss` is passed.
- **Correct general fix, not a cp2 hack**: the bridge path *should* inherit training-time
  config from args, not only architecture from HF.
- **Scope**: `calculate_per_token_loss` is the only dropped field that *asserts*. Other
  training-time `TransformerConfig` fields the bridge drops would silently use defaults
  (no crash); audit separately if needed.

## Important caveat — cp2 is a CP-path probe, not a loss comparison

`calculate_per_token_loss=True` changes loss **normalization** (per-token vs the default
per-sample used by every other recipe — none of the async recipes set this flag). So once
fixed, treat cp2 as a **CP-path correctness/perf probe** (allgather-cp alignment,
`train_rollout_logprob_abs_diff`, throughput/memory) — **not** a loss-curve comparison
against the CP1 prefetch recipes.

## To resume later

1. Apply the `model_provider.py` one-liner above.
2. Resubmit: `JOB_NAME=geo3k-async-mt-cp2-3node ... bash scripts/slurm/submit.sh async/experimental/geo3k-vlm-multi-turn-fully-async-cp2-3node`
3. Confirm it passes init (reaches `train/step >= 1`), then point `verify_alignment.py`'s
   lag-correlation check at it to validate the `--allgather-cp` path is aligned.
