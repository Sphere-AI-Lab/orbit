#!/bin/bash
#
# Shared envpack recipe plumbing for server_train experiments.
#
# Source this file from an experiment recipe after ORBIT_REPO and RECIPE_NAME
# are defined. The recipe should still own model/training hyperparameters; this
# file only owns envpack repository discovery, adapter YAML generation, managed
# server command exports, and the common envpack rollout args.

envpack_resolve_repo() {
    ENVPACK_REPO=${ENVPACK_REPO:-"$ORBIT_REPO/thirdparty/envpack"}
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

envpack_curriculum_yaml() {
    if [[ "${ENVPACK_CURRICULUM_ENABLED:-0}" != "1" ]]; then
        return 0
    fi
    if ! declare -p ENVPACK_CURRICULUM_STAGES >/dev/null 2>&1 || ((${#ENVPACK_CURRICULUM_STAGES[@]} == 0)); then
        echo "error: ENVPACK_CURRICULUM_ENABLED=1 requires ENVPACK_CURRICULUM_STAGES" >&2
        exit 64
    fi

    cat <<EOF_CURRICULUM_HEAD
  curriculum:
    enabled: true
    stages:
EOF_CURRICULUM_HEAD
    local stage until steps steps_yaml
    for stage in "${ENVPACK_CURRICULUM_STAGES[@]}"; do
        if [[ "$stage" != *:* ]]; then
            echo "error: curriculum stage must be '<until>:<comma-separated-steps>', got '$stage'" >&2
            exit 64
        fi
        until=${stage%%:*}
        steps=${stage#*:}
        steps=${steps// /}
        if [[ -z "$steps" ]]; then
            echo "error: curriculum stage has empty solve_steps: '$stage'" >&2
            exit 64
        fi
        steps_yaml="[${steps//,/, }]"
        if [[ "$until" == "end" || "$until" == "*" || "$until" == "null" ]]; then
            until="null"
        fi
        cat <<EOF_CURRICULUM_STAGE
      - until: $until
        solve_steps: $steps_yaml
EOF_CURRICULUM_STAGE
    done
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

    # --- auto-size server concurrency from the real rollout demand ----------
    # The server pool must hold one open episode per concurrently-rolled-out
    # sample. Peak demand = max(train batch x n_samples, eval prompts x n_eval).
    # If ENVPACK_DESIRED_CONCURRENCY is set explicitly but can't cover that, FAIL
    # LOUD (otherwise the pool 429s mid-run, e.g. eval). If unset, auto-derive it
    # with headroom so we never silently undersize the server again.
    if [[ -n "${ROLLOUT_BATCH_SIZE:-}" && -n "${N_SAMPLES_PER_PROMPT:-}" ]]; then
        local _train_batch=${EFFECTIVE_ROLLOUT_BATCH_SIZE:-$ROLLOUT_BATCH_SIZE}
        local _train_demand=$(( _train_batch * N_SAMPLES_PER_PROMPT ))
        local _eval_prompts=0
        if [[ -f "${ENVPACK_EVAL_DATA:-}" ]]; then
            _eval_prompts=$(grep -c . "$ENVPACK_EVAL_DATA" 2>/dev/null || echo 0)
        fi
        local _eval_nsamp=${N_SAMPLES_PER_EVAL_PROMPT:-1}
        local _eval_demand=$(( _eval_prompts * _eval_nsamp ))
        local _required=$_train_demand
        (( _eval_demand > _required )) && _required=$_eval_demand
        if [[ -n "${ENVPACK_DESIRED_CONCURRENCY:-}" ]]; then
            if (( ENVPACK_DESIRED_CONCURRENCY < _required )); then
                echo "[envpack] FATAL: ENVPACK_DESIRED_CONCURRENCY=$ENVPACK_DESIRED_CONCURRENCY < required concurrency $_required" \
                     "(train ${_train_batch}x${N_SAMPLES_PER_PROMPT}=${_train_demand}, eval ${_eval_prompts}x${_eval_nsamp}=${_eval_demand})." \
                     "The server pool would 429 mid-run. Raise ENVPACK_DESIRED_CONCURRENCY (>= $_required) or shrink the rollout/eval batch." >&2
                exit 78
            fi
            echo "[envpack] server concurrency=$ENVPACK_DESIRED_CONCURRENCY (explicit) >= required=$_required (train=$_train_demand eval=$_eval_demand)"
        else
            local _margin=$(( _required / 4 )); (( _margin < 32 )) && _margin=32
            export ENVPACK_DESIRED_CONCURRENCY=$(( _required + _margin ))
            echo "[envpack] server concurrency auto-derived=$ENVPACK_DESIRED_CONCURRENCY" \
                 "(required=$_required + margin=$_margin; train ${_train_batch}x${N_SAMPLES_PER_PROMPT}=${_train_demand}, eval ${_eval_prompts}x${_eval_nsamp}=${_eval_demand})"
        fi
    else
        echo "[envpack] WARN: ROLLOUT_BATCH_SIZE/N_SAMPLES_PER_PROMPT unset; cannot auto-size server" \
             "concurrency — falling back to ENVPACK_DESIRED_CONCURRENCY=${ENVPACK_DESIRED_CONCURRENCY:-256}" >&2
    fi

    # --- assert the rollout turn budget fits inside the env's own step limit -
    # max_turns x max_actions_per_step must not exceed the env's max_steps, else
    # the env terminates before the turn budget is spent (silently shorter
    # episodes). Best-effort: reads the env_config from the first train sample.
    if [[ -f "${ENVPACK_TRAIN_DATA:-}" ]]; then
        local _envcfg
        _envcfg=$(python3 -c "import json,sys; c=json.loads(sys.stdin.readline()).get('metadata',{}).get('envpack',{}).get('env_config',{}); print(c.get('max_steps') or 0, c.get('max_actions_per_step') or 1)" < "$ENVPACK_TRAIN_DATA" 2>/dev/null || echo "")
        if [[ -n "$_envcfg" ]]; then
            local _max_steps _max_act
            read -r _max_steps _max_act <<<"$_envcfg"
            local _turns=${ENVPACK_ADAPTER_MAX_TURNS:-0}
            if (( _max_steps > 0 && _turns * _max_act > _max_steps )); then
                echo "[envpack] FATAL: rollout budget ${_turns} turns x ${_max_act} actions/turn =" \
                     "$(( _turns * _max_act )) env steps > env max_steps=${_max_steps}. The env will terminate" \
                     "before the turn budget is used (silently shorter episodes). Lower --max-env-turns-per-sample" \
                     "or raise the env's max_steps." >&2
                exit 78
            fi
            (( _max_steps > 0 )) && echo "[envpack] turn budget ok: ${_turns} turns x ${_max_act} = $(( _turns * _max_act )) <= env max_steps=${_max_steps}"
        fi
    fi

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

    local env_config=""
    if [[ "$env_name" == "sokoban" ]]; then
        ENVPACK_SOKOBAN_RENDER_STYLE=${ENVPACK_SOKOBAN_RENDER_STYLE:-sprite}
        ENVPACK_SOKOBAN_TINY_SCALE=${ENVPACK_SOKOBAN_TINY_SCALE:-16}
        ENVPACK_SOKOBAN_RAW_PLANE_SCALE=${ENVPACK_SOKOBAN_RAW_PLANE_SCALE:-16}
        env_config=$(cat <<EOF_ENV_CONFIG
      env_config:
        sokoban_render_style: $ENVPACK_SOKOBAN_RENDER_STYLE
        tiny_scale: $ENVPACK_SOKOBAN_TINY_SCALE
        raw_plane_scale: $ENVPACK_SOKOBAN_RAW_PLANE_SCALE
EOF_ENV_CONFIG
)
    fi
    local curriculum_config=""
    curriculum_config=$(envpack_curriculum_yaml)

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
$curriculum_config
  pools:
    - env: $env_name
      profile: $profile
      pool_id: $pool_id
$env_config
$runtime_config
  rollout:
    max_turns: $ENVPACK_ADAPTER_MAX_TURNS
    response_length_per_turn: $ENVPACK_ADAPTER_RESPONSE_LENGTH_PER_TURN
EOF
}

envpack_set_rollout_args() {
    local rollout_log_func=${ENVPACK_CUSTOM_ROLLOUT_LOG_FUNCTION_PATH:-orbit_plugins.envpack_adapter.logging.log_rollout_data}
    local eval_rollout_log_func=${ENVPACK_CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH:-orbit_plugins.envpack_adapter.logging.log_eval_rollout_data}
    local all_samples_process_func=${ENVPACK_ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-orbit_plugins.envpack_adapter.logging.process_all_samples}

    ROLLOUT_ARGS=(
        --data-source-path orbit_plugins.envpack_adapter.data_source.EnvpackDataSource
        --prompt-data       "$ENVPACK_TRAIN_DATA"
        --custom-generate-function-path orbit_plugins.envpack_adapter.generate.generate
        --custom-rollout-log-function-path "$rollout_log_func"
        --custom-eval-rollout-log-function-path "$eval_rollout_log_func"
        --rollout-all-samples-process-path "$all_samples_process_func"
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
