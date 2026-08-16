#!/usr/bin/env bash
# Download the base models the LoRA-without-regret reproduction needs.
#
# Every repo here is a BASE model. Instruct variants of several of them are already
# on disk and are NOT interchangeable: Qwen3-4B != Qwen3-4B-Instruct-2507, and
# Llama-3.1-8B != Llama-3.1-8B-Instruct. The blog uses base models throughout.
#
# meta-llama/Llama-3.1-8B is GATED (gated=manual): the license must be accepted on
# the HuggingFace account and a token must exist at ~/.cache/huggingface/token, or
# the download 401s. Its `original/*` consolidated .pth weights are a ~16G duplicate
# of the safetensors and are skipped.
#
# Qwen3-30B-A3B (the MoE arm of the layer study) is already local and is the base
# model -- Qwen3MoeForCausalLM, 128 experts, top-8 -- so it is not listed here.
set -euo pipefail

HF_MODELS_DIR=${HF_MODELS_DIR:-/lustre/fast/fast/zqiu/hf_models}
mkdir -p "${HF_MODELS_DIR}"

for repo in meta-llama/Llama-3.1-8B Qwen/Qwen3-4B Qwen/Qwen3-1.7B; do
    name="${repo#*/}"
    dest="${HF_MODELS_DIR}/${name}"
    if [[ -f "${dest}/config.json" ]]; then
        echo "skip ${name}: already at ${dest}"
        continue
    fi
    echo "downloading ${repo} -> ${dest}"
    huggingface-cli download "${repo}" --local-dir "${dest}" \
        --exclude "original/*" "*.pth"
done

echo "done. hidden sizes:"
for name in Llama-3.1-8B Qwen3-4B Qwen3-1.7B; do
    python -c "import json,sys;c=json.load(open('${HF_MODELS_DIR}/${name}/config.json'));print('${name}', c['hidden_size'], c['intermediate_size'])"
done
