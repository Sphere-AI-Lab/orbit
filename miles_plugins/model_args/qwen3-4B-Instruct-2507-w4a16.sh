source "$(dirname -- "${BASH_SOURCE[0]}")/qwen3-4B-Instruct-2507.sh"
MODEL_ARGS+=(--untie-embeddings-and-output-weights)
