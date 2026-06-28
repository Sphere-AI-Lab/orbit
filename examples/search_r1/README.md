# Search-R1 PPO

This example adds Search-R1 style reasoning plus retrieval rollouts to Orbit.
Use it through `--custom-generate-function-path examples.search_r1.generate_with_search.generate`
and `--custom-rm-path examples.search_r1.generate_with_search.reward_func`.

The custom rollout uses Orbit's `/generate` payload helper for every model call,
so PPO rollout logprobs and LoRA/OFT adapter request fields are handled by the
same path as other Orbit rollouts.

Default productization settings:

- `--search-r1-backend local`
- `--search-r1-topk 3`
- `--search-r1-max-turns 2`
- `--n-samples-per-prompt 8` or higher for effectiveness runs
- `--target-modules all-linear` for LoRA and OFT
- `--oft-block-size 32`

The local backend expects a retrieval server compatible with Search-R1's
`/retrieve` API:

```json
{"queries": ["query"], "topk": 3, "return_scores": false}
```

and a response shaped like:

```json
{"result": [[{"document": {"contents": "\"Title\"\nPassage text"}}]]}
```

## Launchers

The Qwen2.5-3B PPO launchers are:

```bash
bash examples/search_r1/run-qwen2_5-3b-bf16-search-r1-ppo-full.sh
bash examples/search_r1/run-qwen2_5-3b-bf16-search-r1-ppo-lora.sh
bash examples/search_r1/run-qwen2_5-3b-bf16-search-r1-ppo-oft.sh
```

Required paths:

```bash
export HF_CKPT=/path/to/Qwen2.5-3B-Instruct
export MEGATRON_LOAD=/path/to/Qwen2.5-3B-Instruct-torchdist
export TRAIN_DATA=/path/to/search_r1_train.parquet
export TEST_DATA=/path/to/search_r1_eval.parquet
```

Useful overrides:

```bash
export SEARCH_R1_LOCAL_URL=http://127.0.0.1:8000/retrieve
export NUM_ROLLOUT=3000
export ROLLOUT_BATCH_SIZE=32
export N_SAMPLES_PER_PROMPT=8
export GLOBAL_BATCH_SIZE=256
```
