"""HF AutoConfig shim for DeepSeek V4 checkpoints used by Orbit.

The official V4 config carries native fields that current Transformers releases
do not own as a built-in schema. This shim preserves those fields for SGLang and
adds the HF-style aliases that Orbit/Megatron validation reads. Alias values are
derived from the loaded config fields, not from the official inference dataclass
defaults.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig

_DSV4_DEFAULT_HEAD_DIM = 512
_DSV4_DEFAULT_ROPE_HEAD_DIM = 64


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


class DeepseekV4Config(PretrainedConfig):
    """DeepSeek V4 native config plus HF-compatible aliases."""

    model_type = "deepseek_v4"

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        max_batch_size: int | None = None,
        max_seq_len: int | None = None,
        max_position_embeddings: int | None = None,
        rms_norm_eps: float | None = None,
        hidden_act: str | None = None,
        attention_dropout: float | None = None,
        hidden_dropout: float | None = None,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        use_qk_norm: bool = True,
        initializer_range: float = 0.02,
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        qk_pos_emb_head_dim: int | None = None,
        qk_nope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        rope_theta: float | None = None,
        rope_scaling: dict | None = None,
        first_k_dense_replace: int | None = None,
        moe_intermediate_size: int | None = None,
        n_routed_experts: int | None = None,
        n_shared_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        scoring_func: str | None = None,
        routed_scaling_factor: float | None = None,
        aux_loss_alpha: float = 0.0,
        sliding_window: int | None = None,
        num_hash_layers: int | None = None,
        num_nextn_predict_layers: int | None = None,
        dim: int | None = None,
        n_layers: int | None = None,
        n_heads: int | None = None,
        head_dim: int | None = None,
        rope_head_dim: int | None = None,
        norm_eps: float | None = None,
        window_size: int | None = None,
        n_activated_experts: int | None = None,
        score_func: str | None = None,
        route_scale: float | None = None,
        moe_inter_dim: int | None = None,
        inter_dim: int | None = None,
        original_seq_len: int | None = None,
        rope_factor: float | None = None,
        beta_fast: int | None = None,
        beta_slow: int | None = None,
        n_hash_layers: int | None = None,
        n_mtp_layers: int | None = None,
        hc_mult: int | None = None,
        hc_eps: float | None = None,
        hc_sinkhorn_iters: int | None = None,
        index_head_dim: int | None = None,
        index_n_heads: int | None = None,
        index_topk: int | None = None,
        swiglu_limit: float | None = None,
        compress_rope_theta: float | None = None,
        topk_method: str | None = None,
        norm_topk_prob: bool | None = None,
        expert_dtype: str | None = None,
        torch_dtype: str | None = None,
        attn_impl: str | None = None,
        compress_ratios: list | None = None,
        quantization_config: dict | None = None,
        o_lora_rank: int | None = None,
        o_groups: int | None = None,
        **kwargs: Any,
    ) -> None:
        native_dtype = kwargs.pop("dtype", None)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        resolved_moe_intermediate = _first_not_none(
            moe_intermediate_size,
            moe_inter_dim,
            intermediate_size,
        )
        resolved_intermediate = _first_not_none(intermediate_size, resolved_moe_intermediate)
        resolved_rope_head_dim = _first_not_none(
            rope_head_dim,
            qk_rope_head_dim,
            qk_pos_emb_head_dim,
            _DSV4_DEFAULT_ROPE_HEAD_DIM,
        )
        inferred_head_dim = None
        if qk_nope_head_dim is not None and resolved_rope_head_dim is not None:
            inferred_head_dim = qk_nope_head_dim + resolved_rope_head_dim
        resolved_head_dim = _first_not_none(
            head_dim,
            qk_head_dim,
            kv_lora_rank,
            v_head_dim,
            inferred_head_dim,
            _DSV4_DEFAULT_HEAD_DIM,
        )
        resolved_kv_lora_rank = _first_not_none(kv_lora_rank, resolved_head_dim)
        resolved_qk_head_dim = _first_not_none(qk_head_dim, resolved_head_dim)
        resolved_qk_nope_head_dim = _first_not_none(qk_nope_head_dim, resolved_head_dim)
        resolved_v_head_dim = _first_not_none(v_head_dim, resolved_head_dim)
        resolved_num_hash_layers = _first_not_none(num_hash_layers, n_hash_layers)
        resolved_num_mtp_layers = _first_not_none(num_nextn_predict_layers, n_mtp_layers)
        resolved_original_seq_len = _first_not_none(
            original_seq_len,
            rope_scaling.get("original_max_position_embeddings") if isinstance(rope_scaling, dict) else None,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = resolved_intermediate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_batch_size = max_batch_size
        self.max_seq_len = _first_not_none(max_seq_len, resolved_original_seq_len)
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_qk_norm = use_qk_norm
        self.initializer_range = initializer_range
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = resolved_kv_lora_rank
        self.qk_head_dim = resolved_qk_head_dim
        self.qk_rope_head_dim = _first_not_none(qk_rope_head_dim, resolved_rope_head_dim)
        self.qk_pos_emb_head_dim = _first_not_none(qk_pos_emb_head_dim, resolved_rope_head_dim)
        self.qk_nope_head_dim = resolved_qk_nope_head_dim
        self.v_head_dim = resolved_v_head_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_intermediate_size = resolved_moe_intermediate
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.aux_loss_alpha = aux_loss_alpha
        self.sliding_window = sliding_window
        self.num_hash_layers = resolved_num_hash_layers
        self.num_nextn_predict_layers = resolved_num_mtp_layers

        self.dim = _first_not_none(dim, hidden_size)
        self.n_layers = _first_not_none(n_layers, num_hidden_layers)
        self.n_heads = _first_not_none(n_heads, num_attention_heads)
        self.head_dim = resolved_head_dim
        self.rope_head_dim = resolved_rope_head_dim
        self.norm_eps = _first_not_none(norm_eps, rms_norm_eps)
        self.window_size = _first_not_none(window_size, sliding_window)
        self.n_activated_experts = _first_not_none(n_activated_experts, num_experts_per_tok)
        self.score_func = _first_not_none(score_func, scoring_func)
        self.route_scale = _first_not_none(route_scale, routed_scaling_factor)
        self.moe_inter_dim = resolved_moe_intermediate
        self.inter_dim = _first_not_none(inter_dim, resolved_intermediate)
        self.original_seq_len = resolved_original_seq_len
        self.rope_factor = _first_not_none(
            rope_factor,
            rope_scaling.get("factor") if isinstance(rope_scaling, dict) else None,
        )
        self.beta_fast = _first_not_none(
            beta_fast,
            rope_scaling.get("beta_fast") if isinstance(rope_scaling, dict) else None,
        )
        self.beta_slow = _first_not_none(
            beta_slow,
            rope_scaling.get("beta_slow") if isinstance(rope_scaling, dict) else None,
        )
        self.n_hash_layers = _first_not_none(n_hash_layers, resolved_num_hash_layers)
        self.n_mtp_layers = _first_not_none(n_mtp_layers, resolved_num_mtp_layers)
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.swiglu_limit = swiglu_limit
        self.compress_rope_theta = compress_rope_theta
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.expert_dtype = expert_dtype
        self.attn_impl = attn_impl
        self.compress_ratios = compress_ratios
        if quantization_config is not None:
            self.quantization_config = quantization_config
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self._dsv4_torch_dtype = torch_dtype
        if torch_dtype is not None:
            self.dtype = torch_dtype

        if native_dtype is not None:
            self.dtype = native_dtype  # noqa: A003 - native DeepSeek V4 field name.

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        torch_dtype = output.pop("_dsv4_torch_dtype", None)
        if torch_dtype is not None:
            output.pop("dtype", None)
            output["torch_dtype"] = str(torch_dtype).removeprefix("torch.")
        return output
