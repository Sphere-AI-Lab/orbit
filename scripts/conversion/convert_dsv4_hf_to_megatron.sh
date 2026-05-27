#!/usr/bin/env bash
# Convert a DeepSeek V4 HF checkpoint to Megatron torch_dist so future orbit
# training/rollout runs skip the slow per-startup HF -> Megatron conversion.
#
# Same code path orbit training boot would otherwise run inline
# (AutoBridge.from_hf_pretrained -> to_megatron_model -> save_megatron_model);
# scales to DSV4-Pro (~800 GB) by raising NPROC / TP / EP at launch.
#
# Usage:
#   MEGATRON_BRIDGE_ROOT=/path/to/Megatron-Bridge \
#   MEGATRON_LM_PATH=/path/to/Megatron-LM \
#   HF_PATH=/path/to/dsv4-hf \
#   OUTPUT_PATH=/path/to/dsv4-megatron \
#   bash scripts/conversion/convert_dsv4_hf_to_megatron.sh
#
# Env knobs:
#   NPROC=1                Number of GPUs. Must be divisible by TP*PP*CP and
#                          by ETP*EP*PP.
#   TP=1 PP=1 CP=1         Model/pipeline/context parallel sizes
#   EP=1 ETP=1             Expert parallel + expert TP sizes (live in the DP group)
#   DEFAULT_OUTPUT_ROOT    Override output root (default $ORBIT_ROOT/orbit_ckpts)
#   DSV4_OFFICIAL_CONVERT  auto/1/0. In auto mode, run DeepSeek's official
#                          inference/convert.py unless the input already has
#                          model0-mp${DSV4_OFFICIAL_MP}.safetensors.
#   DSV4_OFFICIAL_STAGE_PATH
#                          Where to write/read the official mp staging dir.
#                          By default, local reservation inputs stage beside
#                          the HF copy; other inputs stage under DEFAULT_OUTPUT_ROOT.
#   DSV4_OFFICIAL_MP=1     Official DeepSeek converter model-parallel size.
#   DSV4_OFFICIAL_EXPERT_DTYPE=fp4
#                          Keep routed experts in FP4 for V4 Flash/Pro.
#   DSV4_PATCH_CHAT_TEMPLATE=1
#                          Add Orbit-compatible chat_template to the staged
#                          tokenizer_config.json when it is missing.
#   DSV4_CHAT_TEMPLATE_PATH
#                          Template file to inject; defaults to the repo-owned
#                          DeepSeek V4 template in scripts/conversion.
#   DSV4_PATCH_HF_CONFIG=1 Add configuration_deepseek_v4.py + auto_map to the
#                          staged HF config when missing.
#   DSV4_DROP_MTP=1        Disable MTP metadata in the staged config and trim
#                          compress_ratios to decoder layers. Current Orbit V4
#                          RL does not use MTP.
#   DSV4_CONFIGURATION_PY_PATH
#                          Config shim file to copy when the source lacks one.
#   DSV4_DEFAULT_MAX_SEQ_LEN
#                          Optional explicit top-level max_seq_len when the
#                          source omits it. By default the alias is derived
#                          from rope_scaling.original_max_position_embeddings.
#   DSV4_DEFAULT_MAX_BATCH_SIZE=1
#                          Top-level max_batch_size written when omitted.
#   RDZV_PORT              torchrun rendezvous port (default 29501)
#
# DSV4 needs the cuda-13.2 module + nk-env-cu13 venv (V4 attention uses
# tilelang); the wrapper pins those just like the V4 training launchers.
#
# Scaling examples:
#   # DSV4-Pro on 8 B200s with TP=8:
#   NPROC=8 TP=8 HF_PATH=/path/to/DSV4-Pro-HF \
#       OUTPUT_PATH=/path/to/dsv4_pro_megatron_tp8 \
#       bash scripts/conversion/convert_dsv4_hf_to_megatron.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# DSV4 needs CUDA 13.2 + the same environment used by public V4 launchers.
export ORBIT_LOAD_CUDA_MODULES=${ORBIT_LOAD_CUDA_MODULES:-0}
DSV4_PYTHON_ENV=${DSV4_PYTHON_ENV:-}
if [[ -n "${DSV4_PYTHON_ENV}" && -d "${DSV4_PYTHON_ENV}/bin" ]]; then
    export PATH="${DSV4_PYTHON_ENV}/bin:${PATH}"
fi

if command -v module >/dev/null 2>&1; then
    module load cuda/13.2 2>&1 | tail -1 || true
    module load nccl 2>&1 | tail -1 || true
fi

source "${SCRIPT_DIR}/../lib/tool_env.sh"
if [[ -z "${CUDA_HOME:-}" || ! -d "${CUDA_HOME}" ]]; then
    echo "Missing CUDA 13.2 root. Set CUDA_HOME or ORBIT_CUDA_HOME_DEFAULT." >&2
    exit 2
fi

# Bridge + Megatron-LM checkouts the V4 stack expects.
: "${MEGATRON_BRIDGE_ROOT:?set MEGATRON_BRIDGE_ROOT to a Megatron-Bridge checkout}"
: "${MEGATRON_LM_PATH:?set MEGATRON_LM_PATH to a Megatron-LM checkout}"
if [[ ! -f "${MEGATRON_BRIDGE_ROOT}/src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py" ]]; then
    echo "MEGATRON_BRIDGE_ROOT does not contain deepseek_v4_bridge.py: ${MEGATRON_BRIDGE_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${MEGATRON_LM_PATH}/megatron/core" ]]; then
    echo "MEGATRON_LM_PATH does not contain megatron/core: ${MEGATRON_LM_PATH}" >&2
    exit 1
fi
export PYTHONPATH="${MEGATRON_BRIDGE_ROOT}/src:${MEGATRON_LM_PATH}:${PYTHONPATH:-}"

export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DEFAULT_OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT:-${ORBIT_ROOT}/orbit_ckpts}"
DRIVER="${ORBIT_ROOT}/tools/convert_dsv4_hf_to_megatron.py"
DSV4_OFFICIAL_CONVERT=${DSV4_OFFICIAL_CONVERT:-auto}
DSV4_OFFICIAL_MP=${DSV4_OFFICIAL_MP:-1}
DSV4_OFFICIAL_EXPERT_DTYPE=${DSV4_OFFICIAL_EXPERT_DTYPE:-fp4}
DSV4_PATCH_CHAT_TEMPLATE=${DSV4_PATCH_CHAT_TEMPLATE:-1}
DSV4_CHAT_TEMPLATE_PATH=${DSV4_CHAT_TEMPLATE_PATH:-${SCRIPT_DIR}/deepseek_v4_chat_template.jinja}
DSV4_PATCH_HF_CONFIG=${DSV4_PATCH_HF_CONFIG:-1}
DSV4_DROP_MTP=${DSV4_DROP_MTP:-1}
DSV4_CONFIGURATION_PY_PATH=${DSV4_CONFIGURATION_PY_PATH:-${SCRIPT_DIR}/configuration_deepseek_v4.py}
DSV4_DEFAULT_MAX_BATCH_SIZE=${DSV4_DEFAULT_MAX_BATCH_SIZE:-1}
DSV4_DEFAULT_MAX_SEQ_LEN=${DSV4_DEFAULT_MAX_SEQ_LEN:-}

# DSV4 attention tilelang kernels read these; keep them in sync with the V4
# training launchers.
export MEGATRON_USE_KV_QAT=${MEGATRON_USE_KV_QAT:-1}

CACHE_ROOT="${MEGATRON_BRIDGE_FP4_CACHE_ROOT:-${ORBIT_CACHE_ROOT:-/tmp/orbit_dsv4_convert}}"
mkdir -p "${CACHE_ROOT}"/{hf,hub,datasets,torch,tmp,xdg,xdg_config,matplotlib,flashinfer_workspace}
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CACHE_ROOT}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_ROOT}/hub}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${CACHE_ROOT}/xdg_config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/matplotlib}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${CACHE_ROOT}/flashinfer_workspace}"

usage() {
    echo "Usage: HF_PATH=/path/to/hf OUTPUT_PATH=/path/to/megatron bash scripts/conversion/convert_dsv4_hf_to_megatron.sh" >&2
    echo "       (NPROC, TP, PP, CP, EP, ETP env vars control parallelism)" >&2
    echo "       MEGATRON_BRIDGE_ROOT and MEGATRON_LM_PATH must point to checkouts." >&2
}

resolve_official_dsv4_convert_py() {
    local source_path="$1"
    if [[ -n "${DSV4_OFFICIAL_CONVERT_PY:-}" ]]; then
        if [[ -f "${DSV4_OFFICIAL_CONVERT_PY}" ]]; then
            printf '%s\n' "${DSV4_OFFICIAL_CONVERT_PY}"
            return 0
        fi
        echo "DSV4_OFFICIAL_CONVERT_PY does not exist: ${DSV4_OFFICIAL_CONVERT_PY}" >&2
        return 1
    fi

    local candidate
    for candidate in \
        "${source_path}/inference/convert.py"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    echo "Could not find DeepSeek V4 official inference/convert.py; set DSV4_OFFICIAL_CONVERT_PY." >&2
    return 1
}

resolve_dsv4_n_experts() {
    local source_path="$1"
    python - "${source_path}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for cfg_path in (root / "config.json", root / "inference" / "config.json"):
    if not cfg_path.exists():
        continue
    cfg = json.loads(cfg_path.read_text())
    for key in ("n_routed_experts", "num_routed_experts", "n_experts"):
        value = cfg.get(key)
        if value is not None:
            print(int(value))
            raise SystemExit(0)
print(f"Could not resolve n_routed_experts from {root}/config.json or inference/config.json", file=sys.stderr)
raise SystemExit(1)
PY
}

copy_dsv4_metadata() {
    local source_path="$1"
    local stage_path="$2"
    local file
    for file in \
        config.json \
        configuration_deepseek_v4.py \
        generation_config.json \
        tokenizer.json \
        tokenizer_config.json \
        tokenizer.model \
        special_tokens_map.json \
        README.md \
        LICENSE; do
        if [[ -f "${source_path}/${file}" ]]; then
            cp -f "${source_path}/${file}" "${stage_path}/${file}"
        fi
    done
}

default_dsv4_official_stage_path() {
    local source_path="$1"
    local source_parent
    local model_name

    source_parent="$(cd "$(dirname "${source_path}")" && pwd)"
    model_name="$(basename "${source_path}")"

    if [[ -n "${DSV4_OFFICIAL_STAGE_ROOT:-}" ]]; then
        printf '%s/%s-inference-mp%s\n' "${DSV4_OFFICIAL_STAGE_ROOT}" "${model_name}" "${DSV4_OFFICIAL_MP}"
        return 0
    fi

    printf '%s/dsv4_official_mp/%s-inference-mp%s\n' "${DEFAULT_OUTPUT_ROOT}" "${model_name}" "${DSV4_OFFICIAL_MP}"
}

ensure_dsv4_hf_shim_config() {
    local checkpoint_path="$1"
    local config_path="${checkpoint_path}/config.json"
    local config_py_path="${checkpoint_path}/configuration_deepseek_v4.py"
    case "${DSV4_PATCH_HF_CONFIG}" in
        1|true|TRUE|yes|YES|on|ON) ;;
        0|false|FALSE|no|NO|off|OFF)
            echo "Skipping DSV4 HF config shim patch: DSV4_PATCH_HF_CONFIG=${DSV4_PATCH_HF_CONFIG}"
            return 0
            ;;
        *)
            echo "DSV4_PATCH_HF_CONFIG must be 1/0, got: ${DSV4_PATCH_HF_CONFIG}" >&2
            exit 1
            ;;
    esac

    if [[ ! -f "${config_path}" ]]; then
        echo "DSV4 HF config shim patch needs config.json: ${config_path}" >&2
        exit 1
    fi
    if [[ ! -f "${config_py_path}" ]]; then
        if [[ ! -f "${DSV4_CONFIGURATION_PY_PATH}" ]]; then
            echo "DSV4_CONFIGURATION_PY_PATH does not exist: ${DSV4_CONFIGURATION_PY_PATH}" >&2
            exit 1
        fi
        cp -f "${DSV4_CONFIGURATION_PY_PATH}" "${config_py_path}"
        echo "Added DSV4 configuration shim: ${config_py_path}"
    fi

    python - "${config_path}" "${DSV4_DEFAULT_MAX_BATCH_SIZE}" "${DSV4_DEFAULT_MAX_SEQ_LEN}" "${DSV4_DROP_MTP}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
default_max_batch_size = int(sys.argv[2])
default_max_seq_len = int(sys.argv[3]) if sys.argv[3] else None
drop_mtp = sys.argv[4].lower() in {"1", "true", "yes", "on"}
data = json.loads(config_path.read_text())
changed = False


def fill_missing(key, value):
    global changed
    if value is None:
        return
    if data.get(key) is None:
        data[key] = value
        changed = True


def first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def add_if_int(lhs, rhs):
    if lhs is None or rhs is None:
        return None
    return int(lhs) + int(rhs)


DEFAULT_HEAD_DIM = 512
DEFAULT_ROPE_HEAD_DIM = 64

auto_map = data.get("auto_map")
if not isinstance(auto_map, dict):
    auto_map = {}
if auto_map.get("AutoConfig") != "configuration_deepseek_v4.DeepseekV4Config":
    auto_map["AutoConfig"] = "configuration_deepseek_v4.DeepseekV4Config"
    data["auto_map"] = auto_map
    changed = True

rope_scaling = data.get("rope_scaling")
if not isinstance(rope_scaling, dict):
    rope_scaling = {}

# Native aliases read directly by SGLang's DeepSeek V4 implementation.
rope_head_dim = first_non_none(
    data.get("rope_head_dim"),
    data.get("qk_rope_head_dim"),
    data.get("qk_pos_emb_head_dim"),
    DEFAULT_ROPE_HEAD_DIM,
)
head_dim = first_non_none(
    data.get("head_dim"),
    data.get("qk_head_dim"),
    data.get("kv_lora_rank"),
    data.get("v_head_dim"),
    add_if_int(data.get("qk_nope_head_dim"), rope_head_dim),
    DEFAULT_HEAD_DIM,
)
fill_missing("dim", data.get("hidden_size"))
fill_missing("n_layers", data.get("num_hidden_layers"))
fill_missing("n_heads", data.get("num_attention_heads"))
fill_missing("head_dim", head_dim)
fill_missing("kv_lora_rank", head_dim)
fill_missing("qk_head_dim", head_dim)
fill_missing("qk_nope_head_dim", head_dim)
fill_missing("v_head_dim", head_dim)
fill_missing("qk_pos_emb_head_dim", rope_head_dim)
fill_missing("rope_head_dim", rope_head_dim)
fill_missing("norm_eps", data.get("rms_norm_eps"))
fill_missing("window_size", data.get("sliding_window"))
fill_missing("score_func", data.get("scoring_func"))
fill_missing("route_scale", data.get("routed_scaling_factor"))
fill_missing("moe_inter_dim", data.get("moe_intermediate_size", data.get("intermediate_size")))
fill_missing("inter_dim", data.get("intermediate_size", data.get("moe_intermediate_size")))
fill_missing("n_activated_experts", data.get("num_experts_per_tok"))
fill_missing("n_hash_layers", data.get("num_hash_layers"))
if drop_mtp:
    for key in ("num_nextn_predict_layers", "n_mtp_layers"):
        if data.get(key) not in (0, None):
            data[key] = 0
            changed = True
        else:
            fill_missing(key, 0)
    num_layers = data.get("num_hidden_layers", data.get("n_layers"))
    compress_ratios = data.get("compress_ratios")
    if isinstance(compress_ratios, list) and num_layers is not None:
        num_layers = int(num_layers)
        if len(compress_ratios) > num_layers:
            data["compress_ratios"] = compress_ratios[:num_layers]
            changed = True
else:
    fill_missing("n_mtp_layers", data.get("num_nextn_predict_layers"))
fill_missing("rope_factor", rope_scaling.get("factor"))
fill_missing("beta_fast", rope_scaling.get("beta_fast"))
fill_missing("beta_slow", rope_scaling.get("beta_slow"))
original_seq_len = first_non_none(data.get("original_seq_len"), rope_scaling.get("original_max_position_embeddings"))
fill_missing("original_seq_len", original_seq_len)
fill_missing("max_batch_size", default_max_batch_size)
max_seq_len = first_non_none(default_max_seq_len, original_seq_len)
if max_seq_len is not None:
    if data.get("max_seq_len") is None or (
        default_max_seq_len is None
        and data.get("max_seq_len") == data.get("max_position_embeddings")
        and data.get("max_seq_len") != max_seq_len
    ):
        data["max_seq_len"] = max_seq_len
        changed = True

if changed:
    config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Patched DSV4 HF config aliases in {config_path}")
else:
    print(f"DSV4 HF config aliases already present: {config_path}")
PY
}

ensure_dsv4_chat_template() {
    local checkpoint_path="$1"
    case "${DSV4_PATCH_CHAT_TEMPLATE}" in
        1|true|TRUE|yes|YES|on|ON) ;;
        0|false|FALSE|no|NO|off|OFF)
            echo "Skipping DSV4 chat_template patch: DSV4_PATCH_CHAT_TEMPLATE=${DSV4_PATCH_CHAT_TEMPLATE}"
            return 0
            ;;
        *)
            echo "DSV4_PATCH_CHAT_TEMPLATE must be 1/0, got: ${DSV4_PATCH_CHAT_TEMPLATE}" >&2
            exit 1
            ;;
    esac

    if [[ ! -f "${DSV4_CHAT_TEMPLATE_PATH}" ]]; then
        echo "DSV4_CHAT_TEMPLATE_PATH does not exist: ${DSV4_CHAT_TEMPLATE_PATH}" >&2
        exit 1
    fi

    python - "${checkpoint_path}" "${DSV4_CHAT_TEMPLATE_PATH}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint_path = Path(sys.argv[1])
template_path = Path(sys.argv[2])
tokenizer_config_path = checkpoint_path / "tokenizer_config.json"
template = template_path.read_text()

data = {}
if tokenizer_config_path.exists():
    data = json.loads(tokenizer_config_path.read_text())

if data.get("chat_template"):
    print(f"tokenizer_config already has chat_template: {tokenizer_config_path}")
    raise SystemExit(0)

data["chat_template"] = template
tokenizer_config_path.write_text(
    json.dumps(data, indent=2, ensure_ascii=False) + "\n",
)
print(f"Added DSV4 chat_template to {tokenizer_config_path}")
PY
}

run_official_dsv4_convert() {
    local source_path="$1"
    local stage_path="$2"
    local convert_py="$3"
    local expected_model="${stage_path}/model0-mp${DSV4_OFFICIAL_MP}.safetensors"

    mkdir -p "${stage_path}"
    copy_dsv4_metadata "${source_path}" "${stage_path}"

    if [[ -f "${expected_model}" && "${DSV4_OFFICIAL_FORCE:-0}" != "1" ]]; then
        echo "Reusing official DeepSeek mp${DSV4_OFFICIAL_MP} staging dir: ${stage_path}"
        ensure_dsv4_hf_shim_config "${stage_path}"
        ensure_dsv4_chat_template "${stage_path}"
        return 0
    fi

    if [[ "${DSV4_OFFICIAL_FORCE:-0}" == "1" ]]; then
        rm -f "${stage_path}"/model*-mp*.safetensors
    fi

    local n_experts
    n_experts="$(resolve_dsv4_n_experts "${source_path}")"

    echo "Running official DeepSeek V4 convert.py staging"
    echo "  convert.py:   ${convert_py}"
    echo "  source:       ${source_path}"
    echo "  stage:        ${stage_path}"
    echo "  n_experts:    ${n_experts}"
    echo "  mp:           ${DSV4_OFFICIAL_MP}"
    echo "  expert dtype: ${DSV4_OFFICIAL_EXPERT_DTYPE}"
    python "${convert_py}" \
        --hf-ckpt-path "${source_path}" \
        --save-path "${stage_path}" \
        --n-experts "${n_experts}" \
        --model-parallel "${DSV4_OFFICIAL_MP}" \
        --expert-dtype "${DSV4_OFFICIAL_EXPERT_DTYPE}"

    if [[ ! -f "${expected_model}" ]]; then
        echo "Official DeepSeek convert did not produce expected file: ${expected_model}" >&2
        exit 1
    fi
    copy_dsv4_metadata "${source_path}" "${stage_path}"
    ensure_dsv4_hf_shim_config "${stage_path}"
    ensure_dsv4_chat_template "${stage_path}"
}

if [[ $# -gt 2 ]]; then
    usage
    exit 1
fi

HF_PATH="${1:-${HF_PATH:-}}"
OUTPUT_PATH="${2:-${OUTPUT_PATH:-}}"
: "${HF_PATH:?set HF_PATH to the source Hugging Face checkpoint}"
: "${OUTPUT_PATH:?set OUTPUT_PATH to the output Megatron torch_dist checkpoint}"
if [[ ! -d "${HF_PATH}" ]]; then
    echo "HF_PATH does not exist: ${HF_PATH}" >&2
    exit 1
fi
if [[ ! -f "${HF_PATH}/config.json" ]]; then
    echo "HF_PATH missing config.json: ${HF_PATH}" >&2
    exit 1
fi

RUN_OFFICIAL_CONVERT=0
case "${DSV4_OFFICIAL_CONVERT}" in
    auto)
        if [[ ! -f "${HF_PATH}/model0-mp${DSV4_OFFICIAL_MP}.safetensors" ]]; then
            RUN_OFFICIAL_CONVERT=1
        fi
        ;;
    1|true|TRUE|yes|YES|on|ON)
        RUN_OFFICIAL_CONVERT=1
        ;;
    0|false|FALSE|no|NO|off|OFF)
        RUN_OFFICIAL_CONVERT=0
        ;;
    *)
        echo "DSV4_OFFICIAL_CONVERT must be auto/1/0, got: ${DSV4_OFFICIAL_CONVERT}" >&2
        exit 1
        ;;
esac

BRIDGE_HF_PATH="${HF_PATH}"
if [[ "${RUN_OFFICIAL_CONVERT}" -eq 1 ]]; then
    OFFICIAL_CONVERT_PY="$(resolve_official_dsv4_convert_py "${HF_PATH}")"
    DSV4_OFFICIAL_STAGE_PATH="${DSV4_OFFICIAL_STAGE_PATH:-$(default_dsv4_official_stage_path "${HF_PATH}")}"
    run_official_dsv4_convert "${HF_PATH}" "${DSV4_OFFICIAL_STAGE_PATH}" "${OFFICIAL_CONVERT_PY}"
    BRIDGE_HF_PATH="${DSV4_OFFICIAL_STAGE_PATH}"
else
    ensure_dsv4_hf_shim_config "${BRIDGE_HF_PATH}"
    ensure_dsv4_chat_template "${BRIDGE_HF_PATH}"
fi

if [[ ! -f "${BRIDGE_HF_PATH}/config.json" ]]; then
    echo "Bridge HF path missing config.json: ${BRIDGE_HF_PATH}" >&2
    exit 1
fi
if ! grep -q '"num_hidden_layers"' "${BRIDGE_HF_PATH}/config.json"; then
    echo "${BRIDGE_HF_PATH}/config.json looks DSV4-native (no num_hidden_layers)." >&2
    echo "Provide/copy a local HF-shim config.json before Bridge conversion." >&2
    exit 1
fi

MEGATRON_PATH="${OUTPUT_PATH}"

NPROC="${NPROC:-1}"
TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
EP="${EP:-1}"
ETP="${ETP:-1}"

base_model_size=$((TP * PP * CP))
if [[ $((NPROC % base_model_size)) -ne 0 ]]; then
    echo "NPROC=${NPROC} must be divisible by TP*PP*CP=${base_model_size}" >&2
    exit 1
fi
expert_model_size=$((ETP * EP * PP))
if [[ $((NPROC % expert_model_size)) -ne 0 ]]; then
    echo "NPROC=${NPROC} must be divisible by ETP*EP*PP=${expert_model_size}" >&2
    exit 1
fi
DATA_PARALLEL_SIZE=$((NPROC / base_model_size))
EXPERT_DATA_PARALLEL_SIZE=$((NPROC / expert_model_size))

mkdir -p "$(dirname "${MEGATRON_PATH}")"

echo "======================================"
echo "DSV4 HF -> Megatron torch_dist conversion"
echo "======================================"
echo "Source:        ${HF_PATH}"
echo "Bridge input:  ${BRIDGE_HF_PATH}"
echo "Destination:   ${MEGATRON_PATH}"
echo "Driver:        ${DRIVER}"
echo "Bridge root:   ${MEGATRON_BRIDGE_ROOT}"
echo "Megatron-LM:   ${MEGATRON_LM_PATH}"
echo "Venv python:   $(command -v python)"
echo "Cache root:    ${CACHE_ROOT}"
echo "NPROC:         ${NPROC}  (TP=${TP} PP=${PP} CP=${CP} DP=${DATA_PARALLEL_SIZE} EP=${EP} ETP=${ETP} EDP=${EXPERT_DATA_PARALLEL_SIZE})"
echo "Chat template: ${DSV4_PATCH_CHAT_TEMPLATE} (${DSV4_CHAT_TEMPLATE_PATH})"
echo "HF config shim: ${DSV4_PATCH_HF_CONFIG} (${DSV4_CONFIGURATION_PY_PATH})"
echo "======================================"

TORCHRUN="$(command -v torchrun)"
if [[ -z "${TORCHRUN}" ]]; then
    echo "torchrun not on PATH; DSV4_PYTHON_ENV=${DSV4_PYTHON_ENV} must contain bin/torchrun" >&2
    exit 1
fi

"${TORCHRUN}" \
    --nproc-per-node="${NPROC}" \
    --nnodes=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint="localhost:${RDZV_PORT:-29501}" \
    "${DRIVER}" \
    --hf-path "${BRIDGE_HF_PATH}" \
    --megatron-path "${MEGATRON_PATH}" \
    --tp "${TP}" --pp "${PP}" --cp "${CP}" \
    --ep "${EP}" --etp "${ETP}"

echo "======================================"
echo "Done. Megatron torch_dist saved at: ${MEGATRON_PATH}"
echo
echo "Orbit launch hints:"
echo "  export HF_CKPT=${BRIDGE_HF_PATH}"
echo "  export MEGATRON_LOAD=${MEGATRON_PATH}"
echo "  export LOAD_CKPT=\"\${MEGATRON_LOAD}\""
echo "  export REQUIRE_MEGATRON_LOAD=1"
echo "  bash examples/low_precision/run-dsv4-mxfp4-openr1-oft-flash.sh"
echo "======================================"
