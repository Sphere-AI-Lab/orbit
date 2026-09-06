from argparse import Namespace
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("args", "argv", "expected"),
    [
        (Namespace(), [], False),
        (Namespace(), ["--cp-comm-type", "p2p"], True),
        (Namespace(), ["--cp-comm-type=a2a"], True),
        (Namespace(cp_comm_type_explicit=False), ["--cp-comm-type", "a2a"], False),
        (Namespace(cp_comm_type_explicit=True), [], True),
    ],
)
def test_cp_comm_type_was_explicit_prefers_serialized_marker(args, argv, expected):
    from orbit.backends.megatron_utils.cp_contract import cp_comm_type_was_explicit

    assert cp_comm_type_was_explicit(args, argv=argv) is expected


def _bridge_args(cp_comm_type_canonical: str, *, explicit: bool = False) -> Namespace:
    return Namespace(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        context_parallel_size=2,
        cp_comm_type=[cp_comm_type_canonical],
        cp_comm_type_canonical=cp_comm_type_canonical,
        cp_comm_type_explicit=explicit,
        hierarchical_context_parallel_sizes=[1, 2] if cp_comm_type_canonical == "a2a+p2p" else None,
        calculate_per_token_loss=True,
        variable_seq_lengths=True,
        attention_softmax_in_fp32=False,
        fp32_residual_connection=False,
        deterministic_mode=False,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        recompute_modules=None,
        cpu_offloading_num_layers=0,
        distribute_saved_activations=False,
        tp_comm_overlap=False,
        fp8=None,
        fp8_recipe="delayed",
        attention_backend="auto",
        moe_token_dispatcher_type="alltoall",
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        moe_router_bias_update_rate=None,
        moe_aux_loss_coeff=None,
    )


@pytest.mark.parametrize("transport", ["p2p", "a2a", "all_gather", "a2a+p2p"])
def test_apply_bridge_runtime_config_propagates_explicit_cp_transport(transport):
    from orbit.backends.megatron_utils.model_provider import _apply_bridge_runtime_config

    provider = SimpleNamespace(cp_comm_type=None)

    _apply_bridge_runtime_config(provider, _bridge_args(transport, explicit=True))

    assert provider.cp_comm_type == transport
    assert provider.hierarchical_context_parallel_sizes == ([1, 2] if transport == "a2a+p2p" else None)


def test_apply_bridge_runtime_config_preserves_provider_transport_over_parser_default():
    from orbit.backends.megatron_utils.model_provider import _apply_bridge_runtime_config

    provider = SimpleNamespace(cp_comm_type="a2a")
    args = _bridge_args("p2p")

    _apply_bridge_runtime_config(provider, args)

    assert provider.cp_comm_type == "a2a"
    assert args.cp_comm_type == ["a2a"]
    assert args.cp_comm_type_canonical == "a2a"


def test_apply_bridge_runtime_config_rejects_explicit_provider_transport_conflict():
    from orbit.backends.megatron_utils.model_provider import _apply_bridge_runtime_config

    provider = SimpleNamespace(cp_comm_type="a2a")

    with pytest.raises(ValueError, match="explicit cp_comm_type=p2p.*provider requires a2a"):
        _apply_bridge_runtime_config(provider, _bridge_args("p2p", explicit=True))


def test_apply_bridge_runtime_config_accepts_explicit_provider_transport_match():
    from orbit.backends.megatron_utils.model_provider import _apply_bridge_runtime_config

    provider = SimpleNamespace(cp_comm_type="a2a")
    args = _bridge_args("a2a", explicit=True)

    _apply_bridge_runtime_config(provider, args)

    assert provider.cp_comm_type == "a2a"
    assert args.cp_comm_type == ["a2a"]
    assert args.cp_comm_type_canonical == "a2a"


def test_apply_bridge_runtime_config_syncs_initialized_parallel_state(monkeypatch):
    from orbit.backends.megatron_utils import model_provider

    state = SimpleNamespace(cp_comm_type="p2p")
    monkeypatch.setattr(model_provider, "is_parallel_state_initialized", lambda: True, raising=False)
    monkeypatch.setattr(model_provider, "get_parallel_state", lambda: state, raising=False)

    model_provider._apply_bridge_runtime_config(SimpleNamespace(cp_comm_type="a2a"), _bridge_args("p2p"))

    assert state.cp_comm_type == "a2a"


@pytest.fixture(params=["fullft", "peft"])
def apply_runtime_config(request):
    from orbit.backends.megatron_utils.bridge_provider_overrides import apply_bridge_provider_overrides
    from orbit.backends.megatron_utils.model_provider import _apply_bridge_runtime_config

    return _apply_bridge_runtime_config if request.param == "fullft" else apply_bridge_provider_overrides


@pytest.fixture
def qwen3_provider():
    import torch
    from megatron.bridge import AutoBridge
    from transformers import Qwen3Config

    config = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        num_hidden_layers=36,
        hidden_size=2560,
        intermediate_size=9728,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        dtype=torch.bfloat16,
    )
    return AutoBridge.from_hf_config(config).to_megatron_provider(load_weights=False)


RUNTIME_ARG_FIELDS = {
    "attention_backend": "attention_backend",
    "gradient_accumulation_fusion": "gradient_accumulation_fusion",
    "cuda_graph_impl": "cuda_graph_impl",
    "cuda_graph_scope": "cuda_graph_scope",
    "use_te_rng_tracker": "te_rng_tracker",
}


def test_bridge_runtime_settings_honor_explicit_recipe(apply_runtime_config, qwen3_provider):
    import torch
    from megatron.core.transformer.enums import AttnBackend, CudaGraphScope

    args = _bridge_args("p2p")
    args.tensor_model_parallel_size = 1
    args.context_parallel_size = 1
    args.sequence_parallel = False
    requested = {
        "attention_backend": AttnBackend.flash,
        "gradient_accumulation_fusion": False,
        "cuda_graph_impl": "local",
        "cuda_graph_scope": [CudaGraphScope.full_iteration],
        "use_te_rng_tracker": True,
    }
    for field, value in requested.items():
        setattr(args, RUNTIME_ARG_FIELDS[field], value)
    model_fields = (
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
        "num_query_groups",
        "kv_channels",
        "vocab_size",
        "bf16",
        "params_dtype",
        "attention_dropout",
        "hidden_dropout",
    )
    model_config = {field: getattr(qwen3_provider, field) for field in model_fields}
    # Parser model defaults must not override the architecture and dtype from HF.
    args.num_layers = 1
    args.hidden_size = 128
    args.bf16 = False
    args.params_dtype = torch.float32

    apply_runtime_config(qwen3_provider, args)

    assert {field: getattr(qwen3_provider, field) for field in requested} == requested
    assert {field: getattr(qwen3_provider, field) for field in model_fields} == model_config
    assert qwen3_provider.params_dtype is torch.bfloat16
    assert qwen3_provider.sequence_parallel is False


@pytest.mark.parametrize("unset", ["absent", "none", "empty_scope"])
def test_bridge_runtime_settings_preserve_unset_provider_values(apply_runtime_config, qwen3_provider, unset):
    from megatron.core.transformer.enums import AttnBackend, CudaGraphScope

    args = _bridge_args("p2p")
    del args.attention_backend
    expected = {
        "attention_backend": AttnBackend.fused,
        "gradient_accumulation_fusion": False,
        "cuda_graph_impl": "local",
        "cuda_graph_scope": [CudaGraphScope.attn],
        "use_te_rng_tracker": True,
    }
    for field, value in expected.items():
        setattr(qwen3_provider, field, value)
        if unset != "absent":
            setattr(args, RUNTIME_ARG_FIELDS[field], None)
    if unset == "empty_scope":
        args.cuda_graph_scope = []  # Native argparse default means no scope override.

    apply_runtime_config(qwen3_provider, args)

    assert {field: getattr(qwen3_provider, field) for field in expected} == expected


def test_bridge_runtime_settings_honor_explicit_false_rng(apply_runtime_config, qwen3_provider):
    args = _bridge_args("p2p")
    args.te_rng_tracker = False
    args.gradient_accumulation_fusion = True
    qwen3_provider.use_te_rng_tracker = True
    qwen3_provider.gradient_accumulation_fusion = False

    apply_runtime_config(qwen3_provider, args)

    assert qwen3_provider.use_te_rng_tracker is False
    assert qwen3_provider.gradient_accumulation_fusion is True
