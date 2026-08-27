from __future__ import annotations

import argparse


def _maybe_set_provider_attr(provider, attr_name: str, value) -> None:
    if value is not None and hasattr(provider, attr_name):
        setattr(provider, attr_name, value)


def apply_bridge_provider_overrides(
    provider,
    args: argparse.Namespace,
    *,
    variable_seq_lengths: bool | None = None,
    default_moe_token_dispatcher_type: str | None = None,
    default_moe_router_load_balancing_type: str | None = None,
) -> None:
    """Apply miles runtime overrides to a Megatron-Bridge provider."""

    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    provider.expert_model_parallel_size = args.expert_model_parallel_size
    provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    provider.sequence_parallel = args.sequence_parallel
    provider.context_parallel_size = args.context_parallel_size
    _maybe_set_provider_attr(
        provider,
        "virtual_pipeline_model_parallel_size",
        getattr(args, "virtual_pipeline_model_parallel_size", None),
    )
    _maybe_set_provider_attr(provider, "attention_softmax_in_fp32", getattr(args, "attention_softmax_in_fp32", None))
    _maybe_set_provider_attr(provider, "calculate_per_token_loss", getattr(args, "calculate_per_token_loss", None))
    # Propagate --[no-]gradient-accumulation-fusion to the provider: the bridge
    # GPTProvider defaults it via can_enable_gradient_accumulation_fusion()
    # independently of the arg, so without this the fused_weight_gradient_mlp_cuda
    # (APEX) extension is required even when --no-gradient-accumulation-fusion is set.
    _maybe_set_provider_attr(
        provider, "gradient_accumulation_fusion", getattr(args, "gradient_accumulation_fusion", None)
    )
    _maybe_set_provider_attr(provider, "recompute_method", getattr(args, "recompute_method", None))
    _maybe_set_provider_attr(provider, "recompute_granularity", getattr(args, "recompute_granularity", None))
    _maybe_set_provider_attr(provider, "recompute_num_layers", getattr(args, "recompute_num_layers", None))
    _maybe_set_provider_attr(
        provider,
        "gradient_accumulation_fusion",
        getattr(args, "gradient_accumulation_fusion", None),
    )
    _maybe_set_provider_attr(provider, "cuda_graph_impl", getattr(args, "cuda_graph_impl", None))
    cuda_graph_scope_val = getattr(args, "cuda_graph_scope", None)
    if cuda_graph_scope_val:  # argparse default is [] which is falsy
        _maybe_set_provider_attr(provider, "cuda_graph_scope", cuda_graph_scope_val)
    _maybe_set_provider_attr(provider, "use_te_rng_tracker", getattr(args, "te_rng_tracker", None))

    if variable_seq_lengths is not None:
        provider.variable_seq_lengths = variable_seq_lengths
    elif hasattr(args, "variable_seq_lengths"):
        provider.variable_seq_lengths = args.variable_seq_lengths

    moe_token_dispatcher_type = getattr(args, "moe_token_dispatcher_type", None)
    if moe_token_dispatcher_type is None:
        moe_token_dispatcher_type = default_moe_token_dispatcher_type
    _maybe_set_provider_attr(provider, "moe_token_dispatcher_type", moe_token_dispatcher_type)

    moe_router_load_balancing_type = getattr(args, "moe_router_load_balancing_type", None)
    if moe_router_load_balancing_type is None:
        moe_router_load_balancing_type = default_moe_router_load_balancing_type
    _maybe_set_provider_attr(provider, "moe_router_load_balancing_type", moe_router_load_balancing_type)

    _maybe_set_provider_attr(
        provider,
        "num_layers_in_first_pipeline_stage",
        getattr(args, "decoder_first_pipeline_num_layers", None),
    )
    _maybe_set_provider_attr(
        provider,
        "num_layers_in_last_pipeline_stage",
        getattr(args, "decoder_last_pipeline_num_layers", None),
    )
    _maybe_set_provider_attr(
        provider,
        "moe_router_bias_update_rate",
        getattr(args, "moe_router_bias_update_rate", None),
    )
    _maybe_set_provider_attr(provider, "moe_aux_loss_coeff", getattr(args, "moe_aux_loss_coeff", None))
    _maybe_set_provider_attr(provider, "moe_permute_fusion", getattr(args, "moe_permute_fusion", None))
    _maybe_set_provider_attr(provider, "moe_router_dtype", getattr(args, "moe_router_dtype", None))
    _maybe_set_provider_attr(provider, "moe_apply_probs_on_input", getattr(args, "moe_apply_probs_on_input", None))
    _maybe_set_provider_attr(provider, "dsv4_moe_dispatcher", getattr(args, "dsv4_moe_dispatcher", None))
