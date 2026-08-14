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
    from miles.backends.megatron_utils.cp_contract import cp_comm_type_was_explicit

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
    from miles.backends.megatron_utils.model_provider import _apply_bridge_runtime_config

    provider = SimpleNamespace(cp_comm_type=None)

    _apply_bridge_runtime_config(provider, _bridge_args(transport, explicit=True))

    assert provider.cp_comm_type == transport
    assert provider.hierarchical_context_parallel_sizes == ([1, 2] if transport == "a2a+p2p" else None)


def test_apply_bridge_runtime_config_preserves_provider_transport_over_parser_default():
    from miles.backends.megatron_utils.model_provider import _apply_bridge_runtime_config

    provider = SimpleNamespace(cp_comm_type="a2a")
    args = _bridge_args("p2p")

    _apply_bridge_runtime_config(provider, args)

    assert provider.cp_comm_type == "a2a"
    assert args.cp_comm_type == ["a2a"]
    assert args.cp_comm_type_canonical == "a2a"


def test_apply_bridge_runtime_config_rejects_explicit_provider_transport_conflict():
    from miles.backends.megatron_utils.model_provider import _apply_bridge_runtime_config

    provider = SimpleNamespace(cp_comm_type="a2a")

    with pytest.raises(ValueError, match="explicit cp_comm_type=p2p.*provider requires a2a"):
        _apply_bridge_runtime_config(provider, _bridge_args("p2p", explicit=True))


def test_apply_bridge_runtime_config_accepts_explicit_provider_transport_match():
    from miles.backends.megatron_utils.model_provider import _apply_bridge_runtime_config

    provider = SimpleNamespace(cp_comm_type="a2a")
    args = _bridge_args("a2a", explicit=True)

    _apply_bridge_runtime_config(provider, args)

    assert provider.cp_comm_type == "a2a"
    assert args.cp_comm_type == ["a2a"]
    assert args.cp_comm_type_canonical == "a2a"


def test_apply_bridge_runtime_config_syncs_initialized_parallel_state(monkeypatch):
    from miles.backends.megatron_utils import model_provider

    state = SimpleNamespace(cp_comm_type="p2p")
    monkeypatch.setattr(model_provider, "is_parallel_state_initialized", lambda: True, raising=False)
    monkeypatch.setattr(model_provider, "get_parallel_state", lambda: state, raising=False)

    model_provider._apply_bridge_runtime_config(SimpleNamespace(cp_comm_type="a2a"), _bridge_args("p2p"))

    assert state.cp_comm_type == "a2a"
