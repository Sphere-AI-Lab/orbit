"""The base models the campaign runs on, and everything a launcher needs to know.

One source of truth. Before this existed, `--hidden-size`, `--ffn-size` and
`--num-layers` were three independent CLI arguments an operator could get wrong
without the model being run changing, and a wrong `--num-layers` makes every
`adapter_params` in the ledger wrong by a constant factor.

`qkv_output_size` is a field rather than a derivation from `hidden_size` because
GQA makes the two differ: Llama-3.1-8B fuses 32 query and 2x8 key/value heads at
128 channels into 6144, against a 4096 hidden size. E3's and E5's
matched-parameter arithmetic is wrong without it.

Every field is checked against the `miles_plugins/model_args/*.sh` plugin it
names by `tests/fast/utils/test_lora_regret_models.py`, so the registry cannot
drift from the plugin that actually configures the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HF_MODELS_DIR = "/lustre/fast/fast/zqiu/hf_models"
# Still under the *old* repo's path -- a cross-repo dependency rather than a
# break, which is why preflight checks it rather than assuming it.
MEGATRON_CKPT_DIR = "/lustre/fast/fast/zqiu/orbit-infra/orbit/checkpoints"
PINNED_LLAMA_TEMPLATE = "orbit/utils/chat_template_utils/templates/llama3.1_pinned.jinja"

# One H100. `HEADROOM_GB` is what a FullFT arm needs for activations, the
# rollout engine's share and allocator fragmentation on top of optimizer state.
# 20 GB is chosen so the formula reproduces the SFT launcher's existing
# hardcoded `>= 4` for Llama-3.1-8B exactly; changing it moves every model's
# GPU floor, so it is a campaign-wide constant rather than a tuning knob.
GPU_GB = 80.0
HEADROOM_GB = 20.0
# Megatron shards optimizer state across DP but not weights or grads, so only
# the second term divides. Candidate DP sizes are powers of two because that is
# what the launcher's placement supports.
DP_CANDIDATES = (1, 2, 4, 8)


@dataclass(frozen=True)
class MoE:
    num_experts: int
    moe_ffn_size: int
    topk: int


@dataclass(frozen=True)
class Model:
    key: str
    hf_checkpoint: str
    megatron_load: str
    model_args_plugin: str
    hidden_size: int
    ffn_size: int
    num_layers: int
    qkv_output_size: int
    loss_mask_type: str
    param_billions: float
    # None means "the model ships its own and the launcher must not override
    # it". Llama-3.1-8B *base* ships none, which is why the campaign pins one.
    chat_template: str | None = None
    moe: MoE | None = None

    def per_gpu_fullft_gb(self, dp: int) -> float:
        """bf16 weights + bf16 grads replicated; fp32 master + Adam moments sharded.

        2 + 2 bytes/param replicated = 4*P GB per GPU for P in billions;
        4 + 8 bytes/param sharded = 12*P/N.
        """
        return 4.0 * self.param_billions + 12.0 * self.param_billions / dp

    def min_gpus_fullft(self) -> int:
        """Smallest DP size whose optimizer state leaves room for activations.

        Raises rather than returning a number that does not exist: for
        Qwen3-30B-A3B no supported DP size fits, and returning 8 would let an
        arm start and OOM twenty minutes into a reserved node.
        """
        budget = GPU_GB - HEADROOM_GB
        for dp in DP_CANDIDATES:
            if self.per_gpu_fullft_gb(dp) <= budget:
                return dp
        raise ValueError(
            f"{self.key} full fine-tuning does not fit: "
            f"{self.per_gpu_fullft_gb(max(DP_CANDIDATES)):.0f} GB/GPU at DP="
            f"{max(DP_CANDIDATES)} against a {budget:.0f} GB budget. "
            "Use a PEFT method, or more nodes than this formula models."
        )


MODELS: dict[str, Model] = {
    "llama3.1-8b": Model(
        key="llama3.1-8b",
        hf_checkpoint=f"{HF_MODELS_DIR}/Llama-3.1-8B",
        megatron_load=f"{MEGATRON_CKPT_DIR}/Llama-3.1-8B_torch_dist",
        # NOT llama3-8B.sh, which has the same six dimensions and differs in
        # --max-position-embeddings (8192 vs 131072) and, decisively,
        # --use-rope-scaling --rotary-scaling-factor 8.0. RoPE scaling changes
        # positional encoding, so it changes every NLL. This is the plugin the
        # SFT launcher has defaulted to all along; the registry records that
        # rather than quietly switching it. The "Instruct" in the filename names
        # the config, not the weights -- the architecture is identical.
        model_args_plugin="llama3.1-8B-Instruct.sh",
        hidden_size=4096, ffn_size=14336, num_layers=32, qkv_output_size=6144,
        loss_mask_type="llama3", param_billions=8.03,
        chat_template=PINNED_LLAMA_TEMPLATE,
    ),
    "qwen3-0.6b": Model(
        key="qwen3-0.6b",
        hf_checkpoint=f"{HF_MODELS_DIR}/Qwen3-0.6B",
        megatron_load=f"{MEGATRON_CKPT_DIR}/Qwen3-0.6B_torch_dist",
        model_args_plugin="qwen3-0.6B.sh",
        hidden_size=1024, ffn_size=3072, num_layers=28, qkv_output_size=4096,
        loss_mask_type="qwen", param_billions=0.752,
    ),
    "qwen3-1.7b": Model(
        key="qwen3-1.7b",
        hf_checkpoint=f"{HF_MODELS_DIR}/Qwen3-1.7B",
        megatron_load=f"{MEGATRON_CKPT_DIR}/Qwen3-1.7B_torch_dist",
        model_args_plugin="qwen3-1.7B.sh",
        hidden_size=2048, ffn_size=6144, num_layers=28, qkv_output_size=4096,
        loss_mask_type="qwen", param_billions=1.72,
    ),
    "qwen3-4b": Model(
        key="qwen3-4b",
        hf_checkpoint=f"{HF_MODELS_DIR}/Qwen3-4B",
        megatron_load=f"{MEGATRON_CKPT_DIR}/Qwen3-4B_torch_dist",
        model_args_plugin="qwen3-4B.sh",
        hidden_size=2560, ffn_size=9728, num_layers=36, qkv_output_size=6144,
        loss_mask_type="qwen", param_billions=4.02,
    ),
    "qwen3-8b": Model(
        key="qwen3-8b",
        hf_checkpoint=f"{HF_MODELS_DIR}/Qwen3-8B",
        megatron_load=f"{MEGATRON_CKPT_DIR}/Qwen3-8B_torch_dist",
        model_args_plugin="qwen3-8B.sh",
        hidden_size=4096, ffn_size=12288, num_layers=36, qkv_output_size=6144,
        loss_mask_type="qwen", param_billions=8.19,
    ),
    "qwen3-30b-a3b": Model(
        key="qwen3-30b-a3b",
        hf_checkpoint=f"{HF_MODELS_DIR}/Qwen3-30B-A3B",
        megatron_load=f"{MEGATRON_CKPT_DIR}/Qwen3-30B-A3B_torch_dist",
        model_args_plugin="qwen3-30B-A3B.sh",
        hidden_size=2048, ffn_size=6144, num_layers=48, qkv_output_size=5120,
        loss_mask_type="qwen", param_billions=30.5,
        moe=MoE(num_experts=128, moe_ffn_size=768, topk=8),
    ),
}

DEFAULT_MODEL = "llama3.1-8b"


def get(key: str) -> Model:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known: {sorted(MODELS)}") from None


def model_env(model: Model, repo_root: Path) -> dict[str, str]:
    """Environment overrides that point a launcher at this model.

    `CHAT_TEMPLATE_PATH` is the **empty string** for models that ship their own
    template, not an omitted key. The launcher reads it with the no-colon
    `${CHAT_TEMPLATE_PATH-default}` form, so empty means "omit the flag" while
    unset means "use the Llama default" -- the same distinction `LABEL_KEY`
    makes, and for the same reason: the colon form would re-default an
    intentionally empty value.
    """
    template = str(repo_root / model.chat_template) if model.chat_template else ""
    try:
        min_gpus = str(model.min_gpus_fullft())
    except ValueError:
        # A model with no viable FullFT DP size still runs PEFT arms. Passing a
        # floor larger than any node forces the launcher's own guard to refuse
        # a FullFT arm rather than letting it start.
        min_gpus = str(max(DP_CANDIDATES) + 1)
    return {
        "MODEL_KEY": model.key,
        "HF_CKPT": model.hf_checkpoint,
        "MEGATRON_LOAD": model.megatron_load,
        "MODEL_ARGS_FILE": str(repo_root / "miles_plugins" / "model_args" / model.model_args_plugin),
        "LOSS_MASK_TYPE": model.loss_mask_type,
        "CHAT_TEMPLATE_PATH": template,
        "MIN_GPUS_FULLFT": min_gpus,
    }
