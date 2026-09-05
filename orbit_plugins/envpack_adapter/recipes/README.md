# Envpack recipe helpers

This directory contains shell helpers that belong to the envpack Orbit plugin,
not to a single experiment recipe.

Experiment scripts under `scripts/experiments/server_train/` should source
`common.sh` and remain thin. They should define:

- model and checkpoint selection
- dataset paths
- rollout and optimizer hyperparameters
- the envpack env/profile/pool tuple

`common.sh` owns:

- envpack repo discovery and `PYTHONPATH`
- managed same-node or remote envpack server command exports
- bearer token generation for managed server jobs
- adapter YAML generation for `--custom-config-path`
- common envpack rollout args

By default, repo discovery uses only `$ORBIT_REPO/thirdparty/envpack`. Set
`ENVPACK_REPO` explicitly if you need to test a different checkout. There is no
implicit fallback to a sibling `../envpack`, so the experiment normally runs
against the envpack submodule/pointer recorded by Orbit.

Typical use:

```bash
ENVPACK_ADAPTER_DIR=${ENVPACK_ADAPTER_DIR:-"$ORBIT_REPO/orbit_plugins/envpack_adapter"}
source "$ENVPACK_ADAPTER_DIR/recipes/common.sh"

envpack_resolve_repo
envpack_require_dataset "$ENVPACK_TRAIN_DATA" "$ENVPACK_EVAL_DATA" "$_BUILD_HINT"
envpack_prepare_adapter_config sokoban vision_free_think_local sokoban-vision 20 512
envpack_set_rollout_args
```

The plugin helper deliberately does not set model, optimizer, W&B, or Slurm
layout flags. Those remain experiment-level decisions.

Training recipes should expose human-facing Orbit-style arrays such as
`TRAINING_SCHEDULE_ARGS`, `INTERACTION_BUDGET_ARGS`, and
`ENVPACK_SERVER_ARGS`. `common.sh` provides small parsing helpers and maps the
resulting values into the generated `envpack_adapter.rollout` YAML and managed
server launch command, so users do not need to edit low-level `ENVPACK_*`
variables for normal experiments.
