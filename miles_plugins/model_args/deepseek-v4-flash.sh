#!/usr/bin/env bash
# DeepSeek V4 Flash architecture.
# Runtime launch policy lives in examples/high_precision/*.sh.

NLAYERS="${MODEL_ARGS_NUM_LAYERS:-43}"
FIRST_K_DENSE_REPLACE="${MODEL_ARGS_FIRST_K_DENSE_REPLACE:-0}"

arr=()
for ((i = 0; i < NLAYERS; i++)); do
    if (( i < FIRST_K_DENSE_REPLACE )); then
        arr+=(0)
    else
        arr+=(1)
    fi
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

MODEL_ARGS=(
    --disable-bias-linear
    --num-layers "${NLAYERS}"
    --hidden-size 4096
    --ffn-hidden-size 2048
    --num-attention-heads 64
    --kv-channels 512
    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 129280

    --multi-latent-attention
    --q-lora-rank 1024
    --kv-lora-rank 512
    --qk-head-dim 512
    --qk-pos-emb-head-dim 64
    --v-head-dim 512
    --qk-layernorm
    --rotary-scaling-factor 16
    --rotary-base 10000
    --mscale 1.0
    --mscale-all-dim 1.0
    --attention-softmax-in-fp32
    --no-rope-fusion

    --num-experts 256
    --moe-layer-freq "${MOE_LAYER_FREQ}"
    --moe-ffn-hidden-size 2048
    --moe-router-topk 6
    --moe-shared-expert-intermediate-size 2048
    --moe-router-pre-softmax
    --moe-router-score-function sqrtsoftplus
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0
    --moe-grouped-gemm
    --moe-router-topk-scaling-factor 1.5
    --moe-router-dtype fp32
    --moe-permute-fusion
)
