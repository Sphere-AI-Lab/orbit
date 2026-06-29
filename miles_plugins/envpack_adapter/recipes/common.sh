#!/bin/bash
#
# Shared envpack recipe plumbing for server_train experiments.
#
# Source this file from an experiment recipe after MILES_REPO and RECIPE_NAME
# are defined. The recipe should still own model/training hyperparameters; this
# file only owns envpack repository discovery, adapter YAML generation, managed
# server command exports, and the common envpack rollout args.

envpack_resolve_repo() {
    ENVPACK_REPO=${ENVPACK_REPO:-"$MILES_REPO/thirdparty/envpack"}
    if [[ ! -f "$ENVPACK_REPO/pyproject.toml" ]]; then
        echo "error: envpack repo not found at $ENVPACK_REPO" >&2
        echo "       initialize thirdparty/envpack or set ENVPACK_REPO explicitly" >&2
        exit 1
    fi
    export ENVPACK_REPO
    export PYTHONPATH="$ENVPACK_REPO${PYTHONPATH:+:$PYTHONPATH}"
}

envpack_require_dataset() {
    local train_data=${1:?train data path is required}
    local eval_data=${2:?eval data path is required}
    local build_hint=${3:?build hint is required}
    if [[ ! -s "$train_data" ]]; then
        echo "error: missing train data: $train_data" >&2
        echo "       $build_hint" >&2
        exit 1
    fi
    if [[ ! -s "$eval_data" ]]; then
        echo "error: missing eval data: $eval_data" >&2
        echo "       $build_hint" >&2
        exit 1
    fi
}

envpack_ensure_auth_token() {
    if [[ -z "${ENVPACK_AUTH_TOKEN:-}" ]]; then
        ENVPACK_AUTH_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
        export ENVPACK_AUTH_TOKEN
    fi
}

envpack_recipe_arg_value() {
    local flag=${1:?flag is required}
    local default_value=${2:?default value is required}
    shift 2
    while (($#)); do
        if [[ "$1" == "$flag" ]]; then
            if (($# < 2)); then
                echo "error: missing value after $flag" >&2
                exit 64
            fi
            printf '%s\n' "$2"
            return 0
        fi
        shift
    done
    printf '%s\n' "$default_value"
}

envpack_prepare_adapter_config() {
    local env_name=${1:?env name is required}
    local profile=${2:?profile is required}
    local pool_id=${3:?pool id is required}
    local default_max_turns=${4:?default max turns is required}
    local default_response_length=${5:?default response length is required}

    ENVPACK_ADAPTER_MAX_TURNS=${ENVPACK_ADAPTER_MAX_TURNS:-${MAX_ENV_TURNS_PER_SAMPLE:-$default_max_turns}}
    ENVPACK_ADAPTER_RESPONSE_LENGTH_PER_TURN=${ENVPACK_ADAPTER_RESPONSE_LENGTH_PER_TURN:-${MAX_MODEL_TOKENS_PER_TURN:-$default_response_length}}
    ENVPACK_CONFIG_DIR=${RUN_DIR:-/tmp/envpack-mvp/${RECIPE_NAME}}
    ENVPACK_CONFIG_PATH=${ENVPACK_CONFIG_PATH:-"$ENVPACK_CONFIG_DIR/envpack_adapter_config.yaml"}
    mkdir -p "$(dirname "$ENVPACK_CONFIG_PATH")"

    ENVPACK_API=${ENVPACK_API:-session}
    ENVPACK_SERVER_PORT=${ENVPACK_SERVER_PORT:-18081}
    ENVPACK_SERVER_NODE_COUNT=${ENVPACK_SERVER_NODE_COUNT:-0}
    ENVPACK_HTTP_TIMEOUT_S=${ENVPACK_HTTP_TIMEOUT_S:-60}
    ENVPACK_HTTP_MAX_RETRIES=${ENVPACK_HTTP_MAX_RETRIES:-3}
    ENVPACK_HTTP_RETRY_BACKOFF_S=${ENVPACK_HTTP_RETRY_BACKOFF_S:-0.25}
    ENVPACK_REFILL_MAX_ATTEMPTS=${ENVPACK_REFILL_MAX_ATTEMPTS:-3}
    ENVPACK_REFILL_BACKOFF_S=${ENVPACK_REFILL_BACKOFF_S:-0.5}

    local server_config=""
    if [[ "$ENVPACK_API" == "session" ]]; then
        unset ENVPACK_LOCAL_SERVER_CMD ENVPACK_LOCAL_SERVER_HEALTH
        unset ENVPACK_REMOTE_SERVER_CMD ENVPACK_REMOTE_SERVER_HEALTH
        if ! [[ "$ENVPACK_SERVER_NODE_COUNT" =~ ^[0-9]+$ ]]; then
            echo "error: ENVPACK_SERVER_NODE_COUNT must be an integer, got '$ENVPACK_SERVER_NODE_COUNT'" >&2
            exit 64
        fi
        if (( ENVPACK_SERVER_NODE_COUNT > 0 )); then
            envpack_ensure_auth_token
            if [[ -z "${ENVPACK_SERVER_URL:-}" ]]; then
                if [[ -n "${ENVPACK_SERVER_IP:-}" ]]; then
                    ENVPACK_SERVER_URL="http://${ENVPACK_SERVER_IP}:${ENVPACK_SERVER_PORT}"
                else
                    ENVPACK_SERVER_URL="http://envpack-server-set-by-launcher:${ENVPACK_SERVER_PORT}"
                fi
            fi
            export ENVPACK_REMOTE_SERVER_CMD="python3 -m envpack.server --env ${env_name}:${profile}:${pool_id} --desired-concurrency ${ENVPACK_DESIRED_CONCURRENCY:-256} --host 0.0.0.0 --port ${ENVPACK_SERVER_PORT}"
            export ENVPACK_REMOTE_SERVER_HEALTH="${ENVPACK_SERVER_URL%/}/v1/health"
        elif [[ -z "${ENVPACK_SERVER_URL:-}" ]]; then
            if [[ "${ENVPACK_SERVER_LOCAL:-1}" == "1" ]]; then
                envpack_ensure_auth_token
                ENVPACK_SERVER_URL="http://127.0.0.1:${ENVPACK_SERVER_PORT}"
                export ENVPACK_LOCAL_SERVER_CMD="python3 -m envpack.server --env ${env_name}:${profile}:${pool_id} --desired-concurrency ${ENVPACK_DESIRED_CONCURRENCY:-256} --host 127.0.0.1 --port ${ENVPACK_SERVER_PORT}"
                export ENVPACK_LOCAL_SERVER_HEALTH="http://127.0.0.1:${ENVPACK_SERVER_PORT}/v1/health"
            else
                echo "error: ENVPACK_SERVER_URL is required when ENVPACK_API=session and ENVPACK_SERVER_LOCAL=0" >&2
                exit 64
            fi
        fi
        server_config="  server: $ENVPACK_SERVER_URL"
    fi

    local runtime_config=""
    if [[ "$ENVPACK_API" != "session" ]]; then
        if [[ -n "${ENVPACK_POOL_CAPACITY:-}" && -z "${ENVPACK_MAX_ACTIVE_EPISODES_PER_INSTANCE:-}" ]]; then
            ENVPACK_NUM_INSTANCES=${ENVPACK_NUM_INSTANCES:-1}
            ENVPACK_MAX_ACTIVE_EPISODES_PER_INSTANCE=$ENVPACK_POOL_CAPACITY
        fi
        if [[ -n "${ENVPACK_NUM_INSTANCES:-}" || -n "${ENVPACK_MAX_ACTIVE_EPISODES_PER_INSTANCE:-}" ]]; then
            ENVPACK_NUM_INSTANCES=${ENVPACK_NUM_INSTANCES:-1}
            ENVPACK_MAX_ACTIVE_EPISODES_PER_INSTANCE=${ENVPACK_MAX_ACTIVE_EPISODES_PER_INSTANCE:-1}
            runtime_config=$(cat <<EOF_RUNTIME
      runtime_config:
        num_instances: $ENVPACK_NUM_INSTANCES
        max_active_episodes_per_instance: $ENVPACK_MAX_ACTIVE_EPISODES_PER_INSTANCE
EOF_RUNTIME
)
        fi
    fi

    cat > "$ENVPACK_CONFIG_PATH" <<EOF
envpack_adapter:
  api: $ENVPACK_API
$server_config
  http:
    timeout_s: $ENVPACK_HTTP_TIMEOUT_S
    max_retries: $ENVPACK_HTTP_MAX_RETRIES
    retry_backoff_s: $ENVPACK_HTTP_RETRY_BACKOFF_S
    auth_token_env: ENVPACK_AUTH_TOKEN
  refill:
    max_attempts: $ENVPACK_REFILL_MAX_ATTEMPTS
    backoff_s: $ENVPACK_REFILL_BACKOFF_S
  pools:
    - env: $env_name
      profile: $profile
      pool_id: $pool_id
$runtime_config
  rollout:
    max_turns: $ENVPACK_ADAPTER_MAX_TURNS
    response_length_per_turn: $ENVPACK_ADAPTER_RESPONSE_LENGTH_PER_TURN
EOF
}

envpack_set_rollout_args() {
    ROLLOUT_ARGS=(
        --data-source-path miles_plugins.envpack_adapter.data_source.EnvpackDataSource
        --prompt-data       "$ENVPACK_TRAIN_DATA"
        --custom-generate-function-path miles_plugins.envpack_adapter.generate.generate
        --rollout-all-samples-process-path examples.vagen.debug_dump.dump_samples
        --custom-config-path "$ENVPACK_CONFIG_PATH"
        --rollout-shuffle
        --seed                    0
        --num-rollout            "$NUM_ROLLOUT"
        --rollout-batch-size      "$ROLLOUT_BATCH_SIZE"
        --n-samples-per-prompt    "$N_SAMPLES_PER_PROMPT"
        --rollout-max-context-len "$ROLLOUT_MAX_CONTEXT_LEN"
        --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
        --global-batch-size       "$GLOBAL_BATCH_SIZE"
    )
}
