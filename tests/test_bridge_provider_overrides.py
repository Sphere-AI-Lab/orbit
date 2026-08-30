from types import SimpleNamespace

from miles.orbit.megatron.bridge_provider_overrides import apply_bridge_provider_overrides


def test_bridge_provider_overrides_gradient_accumulation_fusion():
    provider = SimpleNamespace(gradient_accumulation_fusion=True)
    args = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        variable_seq_lengths=False,
        attention_softmax_in_fp32=None,
        calculate_per_token_loss=None,
        recompute_method=None,
        recompute_granularity=None,
        recompute_num_layers=None,
        gradient_accumulation_fusion=False,
        cuda_graph_impl=None,
        cuda_graph_scope=[],
        te_rng_tracker=None,
    )

    apply_bridge_provider_overrides(provider, args)

    assert provider.gradient_accumulation_fusion is False
