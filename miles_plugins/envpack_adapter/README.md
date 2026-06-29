# envpack adapter for IMP-Miles

This plugin lets IMP-Miles sample from envpack environments without moving
environment state into Miles. It is intentionally a thin sampler-side adapter:

```text
Miles sampler/trainer:
  tokenizer, processor, SGLang requests, logprobs, R3 guard, Sample assembly

envpack repo/server:
  reset, step, finalize, env state, parser/rubric/credit, multimodal bytes
```

The current experiment recipes are HTTP/server-first. In-process mode still
exists for local debugging and dataset building, but the server_train scripts
default to `envpack_adapter.api=session`.

## Files

```text
miles_plugins/envpack_adapter/
  config.py             # envpack yaml parsing and Miles feature guards
  runtime.py            # LocalEnvpackClient vs RemoteEnvpackClient selection
  data_source.py        # Miles RolloutDataSource from envpack JSONL/EnvSpec
  generate.py           # custom_generate loop; Miles calls SGLang, envpack steps env
  renderer.py           # SemanticObservation -> Miles chat messages/media bytes
  build_env_dataset.py  # offline seed/jsonl materialization and env_uuid hashing
  recipes/common.sh     # shell helper used by thin experiment recipes
```

The Slurm experiment recipes import shared envpack launcher plumbing from:

```text
miles_plugins/envpack_adapter/recipes/common.sh
```

That file owns envpack repo discovery, adapter YAML generation, managed server
command exports, HTTP retry/refill settings, and common envpack rollout args.
Individual experiment scripts should stay focused on model choice, dataset
paths, rollout/training hyperparameters, and the env/profile/pool tuple.
Repo discovery defaults to `$MILES_REPO/thirdparty/envpack`; use
`ENVPACK_REPO` only for an explicit override.

There is no `cli/` package here. IMP-Miles already has its own trace/debug
viewer path, so envpack-specific UI export code was removed.

## Plugin Boundary

The adapter is meant to be a self-contained Miles plugin. It exposes three
surfaces:

```text
Python plugin API:
  data_source.py
  generate.py
  renderer.py
  config.py
  runtime.py

Shell recipe API:
  recipes/common.sh

Offline dataset builder:
  python -m miles_plugins.envpack_adapter.build_env_dataset
```

Experiment scripts under `scripts/experiments/server_train/` should stay thin:
they select the model, dataset paths, training hyperparameters, and the
env/profile/pool tuple, then source
`miles_plugins/envpack_adapter/recipes/common.sh` for the envpack plumbing.
This keeps envpack server/session behavior owned by the plugin instead of
copy-pasted into every launch recipe.

## Runtime Modes

HTTP session mode:

```yaml
envpack_adapter:
  api: session
  server: http://env-node-private-ip:18081
  http:
    timeout_s: 60
    max_retries: 3
    retry_backoff_s: 0.25
    auth_token_env: ENVPACK_AUTH_TOKEN
  refill:
    max_attempts: 3
    backoff_s: 0.5
  pools:
    - env: sokoban
      profile: vision_free_think_local
      pool_id: sokoban-vision
  rollout:
    max_turns: 5
    response_length_per_turn: 512
```

In session mode, the envpack server owns env profile defaults and runtime
capacity. The Miles adapter sends the `pool_id` plus per-sample `env_config`
overrides from the JSONL row. It does not tell the server how many instances to
start.

The HTTP retry settings apply to envpack create/step/finalize/cancel requests.
Retries are safe for transient transport errors and retryable HTTP statuses
because envpack session requests carry stable `request_id` / `episode_id`
idempotency keys.

The server_train recipes expose these through:

```bash
ENVPACK_HTTP_TIMEOUT_S=60
ENVPACK_HTTP_MAX_RETRIES=3
ENVPACK_HTTP_RETRY_BACKOFF_S=0.25
```

Normal experiment budgets are grouped in Miles-style arrays in the training
recipe:

```bash
INTERACTION_BUDGET_ARGS=(
  --max-env-turns-per-sample 5
  --max-model-tokens-per-turn 512
)
```

`recipes/common.sh` maps those values into `envpack_adapter.rollout` when it
generates `$RUN_DIR/envpack_adapter_config.yaml`.

If `ENVPACK_AUTH_TOKEN` is set, the adapter sends it as a bearer token on every
envpack HTTP request. The managed server recipes generate a job-local token
automatically when they auto-launch the envpack server. For an external server,
set the same `ENVPACK_AUTH_TOKEN` in both the Miles job and the envpack server
environment. In multi-node Ray runs, the token must be present in every rollout
worker environment, not only in the launcher shell; the provided Slurm launcher
uses `--export=ALL` for Ray and envpack `srun` calls so the generated token is
propagated inside the job.

In-process mode:

```yaml
envpack_adapter:
  api: in_process
  pools:
    - env: sokoban
      profile: vision_free_think_local
      pool_id: sokoban-vision
```

Use this only for local parity/debug runs or offline dataset construction. The
checked-in server_train recipes are HTTP/session recipes.

## System-Failure Refill

Envpack treats model mistakes as normal training data: invalid actions, wrong
answers, time limits, and non-solved trajectories should return a reward/rubric
and continue into Miles.

Infrastructure failures are different. If the envpack server/client reports a
retryable transport/capacity/state-lost failure after the request retry budget is
exhausted, the adapter does not turn that failure into reward 0. It resets the
same Miles sample and tries the same prompt/env seed/env config again with a
fresh `episode_id`:

```text
same sample index / group index
same env_name / seed / env_config / prompt row
new episode_id
new env reset on an available instance
```

The shipped recipes expose the retry budget as:

```bash
ENVPACK_REFILL_MAX_ATTEMPTS=3
ENVPACK_REFILL_BACKOFF_S=0.5
```

Successful refills are recorded under `sample.metadata["envpack"]["refill"]`.
If all attempts fail, the sample error is re-raised so the rollout can fail or be
requeued by the launcher/watchdog. This keeps system failures out of GRPO/DAPO
reward normalization.

Refill is deliberately narrow. It does not catch tokenizer/processor errors,
schema mismatches, drift/hash failures, prompt rendering bugs, generic
`RuntimeError`s, or SGLang sampler failures. Those should surface as real
adapter/runtime bugs or sampler infrastructure issues instead of being hidden by
env reset.

## Static Dataset Build

Yes, the existing static data generation still works with HTTP server mode.
The JSONL is not a baked environment trajectory. It is a Miles-owned prompt
index containing:

```text
env_name
seed
pool_id
profile
env_config overrides
env_uuid for split/dedup/drift checks
solver_metrics / bucket_name when the env builder provides them
```

Rollout and model-generation budgets are not dataset identity. They live in the
adapter runtime config generated by the Miles recipe:

```yaml
envpack_adapter:
  rollout:
    max_turns: 20
    response_length_per_turn: 512
```

Environment rules such as Sokoban `max_steps`, board size, number of boxes, and
prompt format remain in `metadata.envpack.env_config` because envpack needs
them to reset and step the environment.

At rollout time, the live environment still comes from envpack:

```text
Miles EnvpackDataSource -> sample.metadata["envpack"]
Miles generate.py       -> RemoteEnvpackClient
envpack server          -> reset/step/finalize real env session
```

Build data before training:

```bash
scripts/experiments/server_train/build-envpack-main.sh sokoban
scripts/experiments/server_train/build-envpack-main.sh frozenlake
```

For strict environment separation, run the build in an envpack build/server
environment that has simulator dependencies installed, then point Miles at the
resulting `samples.jsonl`. The training-side Miles environment only needs the
lightweight envpack client/core imports plus image decoding; it does not need to
import simulator backends when `api=session`.

## Conda Environment Split

Recommended layout:

```text
miles-train env:
  IMP-Miles
  model/tokenizer/processor stack
  SGLang client/runtime pieces needed by Miles
  editable envpack install for core/client contracts

envpack-sokoban-server env:
  envpack[server]
  sokoban simulator dependencies
  starts: python -m envpack.server --env sokoban:vision_free_think_local:sokoban-vision ...

envpack-frozenlake-server env:
  envpack[server]
  frozenlake/gymnasium dependencies
  starts: python -m envpack.server --env frozenlake:vision_free_think_local:frozenlake-vision ...
```

For Sokoban/FrozenLake you can run both pools in one server environment if the
dependencies are compatible. For future coding/browser/sandbox envs, prefer one
server environment per env family so dependency and resource failures do not
pollute Miles or other envs.

## Running The HTTP Recipes

The checked-in Slurm recipes normally launch the envpack server for you. On a
same-node run, `recipes/common.sh` exports `ENVPACK_LOCAL_SERVER_CMD` and the
launcher starts it on `127.0.0.1:$ENVPACK_SERVER_PORT`. On a two-node run, the
launcher reserves the envpack node, fills `ENVPACK_SERVER_URL`, and starts
`ENVPACK_REMOTE_SERVER_CMD` there.

For manual debugging, you can still start a compatible server yourself:

```bash
python3 -m envpack.server \
  --env sokoban:vision_free_think_local:sokoban-vision \
  --desired-concurrency 256 \
  --host 127.0.0.1 \
  --port 18081
```

Then launch Miles while pointing the adapter at that server:

```bash
ENVPACK_API=session \
ENVPACK_SERVER_URL=http://127.0.0.1:18081 \
bash scripts/slurm/submit.sh server_train/envpack-sokoban-main-qwen25vl3b-colocate-1node
```

The server_train scripts use W&B project `vagen` with run names prefixed by
`http-`.

## Slurm Two-Node Remote Mode

For remote-server validation inside one Slurm job, use the `remote-2node`
recipes:

```bash
bash scripts/slurm/submit.sh server_train/envpack-sokoban-main-qwen25vl3b-remote-2node
bash scripts/slurm/submit.sh server_train/envpack-frozenlake-main-qwen3vl2b-remote-2node
```

The launcher allocates two homogeneous GPU nodes, reserves the final healthy
node for envpack, and excludes that node from the Ray cluster:

```text
node 0:
  Miles + Ray head + SGLang + trainer

node 1:
  envpack server only
  python3 -m envpack.server --host 0.0.0.0 --port 18081 ...
```

At launch time `launch_miles.sbatch` resolves the envpack node IP, sets
`ENVPACK_SERVER_URL=http://<env-node-ip>:18081`, starts the server via `srun`,
waits on `/v1/health`, then starts Ray only on the Miles nodes. The Miles recipe
still keeps `--actor-num-nodes 1` and `--rollout-num-gpus 8`; the second node is
not available to Miles.

The remote recipe enables bearer-token auth by default through
`ENVPACK_AUTH_TOKEN`. If you replace the launcher with a different scheduler,
make sure that token reaches both the envpack server process and all Ray rollout
workers that construct `RemoteEnvpackClient`.

After startup, the launcher runs a small envpack health watchdog. If
`/v1/health` fails for `ENVPACK_SERVER_WATCHDOG_FAILURES` consecutive polls, the
launcher exits with the same temporary-failure code used for requeueable
infrastructure failures. The default poll interval is controlled by
`ENVPACK_SERVER_WATCHDOG_INTERVAL=15`.

Set `ENVPACK_SERVER_ENV_NAME` if the server uses a separate conda env:

```bash
ENVPACK_SERVER_ENV_NAME=envpack-sokoban-server \
bash scripts/slurm/submit.sh server_train/envpack-sokoban-main-qwen25vl3b-remote-2node
```

This mode is intentionally a first remote validation path. It uses homogeneous
Slurm nodes and may waste GPUs on the envpack node. CPU-only env nodes,
heterogeneous Slurm allocations, and multi-envpack-node load balancing should be
added after the single remote server path is stable.

## Compatibility Notes

- Existing non-envpack VAGEN scripts continue to use `examples.vagen.*`.
- The envpack adapter still writes VAGEN-shaped debug metadata because current
  IMP-Miles debug dump tooling expects it. Canonical envpack provenance lives
  under `sample.metadata["envpack"]`.
- LoRA payload routing is preserved.
- `partial_rollout`, `group_rm`, external reward models, `rollout_external`,
  and Miles `RadixTreeMiddleware` are rejected for now instead of silently
  producing incompatible samples.
- Non-envpack Miles R3 is unaffected. Envpack adapter R3 pass-through is not
  implemented yet, so `--use-rollout-routing-replay` fails at startup on the
  envpack custom-generate path only.
