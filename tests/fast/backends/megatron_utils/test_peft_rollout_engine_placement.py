"""Regression coverage for PEFT rollout-engine placement classification."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-fast")


from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


_MODULE = "orbit.backends.megatron_utils.update_weight.update_weight_from_tensor"


class _ExistingTransport:
    def __init__(self) -> None:
        self.connect = MagicMock()
        self.disconnect = MagicMock()
        self.runtime_mode = SimpleNamespace(log_line=lambda: "adapter_runtime test")


@patch(f"{_MODULE}.dist.get_rank", return_value=1)
@patch(f"{_MODULE}.get_parallel_state")
def test_non_colocated_peft_engine_is_always_distributed(mock_parallel_state, _mock_rank):
    """Offset zero is relative to the rollout placement group when not colocated."""
    from orbit.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

    mock_parallel_state.return_value = SimpleNamespace(
        intra_dp_cp=SimpleNamespace(rank=0),
        tp=SimpleNamespace(rank=1),
        pp=SimpleNamespace(rank=0),
    )
    engine = MagicMock(name="rollout_engine")
    transport = _ExistingTransport()
    updater = object.__new__(UpdateWeightFromTensor)
    updater.args = Namespace(
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        rollout_num_gpus_per_engine=2,
    )
    updater._peft_sync_spec = object()
    updater._peft_transport = transport
    updater._peft_transport_mode = "ipc"
    updater._model_update_groups = None
    updater._ipc_gather_group = object()
    updater._ipc_gather_src = 0
    updater._ipc_gather_layout = ((0, 2),)

    updater.connect_rollout_engines(
        rollout_engines=[engine],
        rollout_engine_lock=MagicMock(),
        engine_gpu_counts=[2],
        engine_gpu_offsets=[0],
    )

    assert updater.use_distribute is True
    assert updater.rollout_engines == []
    assert updater.distributed_rollout_engines == [engine]
    transport.disconnect.assert_called_once_with()
    transport.connect.assert_not_called()


@patch(f"{_MODULE}.connect_rollout_engines_from_distributed")
@patch(f"{_MODULE}.dist.get_rank", return_value=0)
@patch(f"{_MODULE}.get_parallel_state")
def test_ray_peft_does_not_create_legacy_nccl_group(
    mock_parallel_state,
    _mock_rank,
    mock_connect_distributed,
):
    """Ray transport must not initialize the legacy trainer-to-engine NCCL group."""
    from orbit.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

    mock_parallel_state.return_value = SimpleNamespace(
        intra_dp_cp=SimpleNamespace(rank=0),
        tp=SimpleNamespace(rank=0),
        pp=SimpleNamespace(rank=0),
    )
    engine = MagicMock(name="rollout_engine")
    transport = _ExistingTransport()
    updater = object.__new__(UpdateWeightFromTensor)
    updater.args = Namespace(
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        rollout_num_gpus_per_engine=1,
        peft_distributed_transport="ray",
    )
    updater._peft_args = updater.args
    updater._peft_sync_spec = object()
    updater._peft_transport = transport
    updater._peft_transport_mode = "ray"
    updater._model_update_groups = None
    updater._ipc_gather_group = object()
    updater._ipc_gather_src = 0
    updater._ipc_gather_layout = ((0, 1),)

    rollout_engine_lock = MagicMock()
    with patch(
        "orbit.backends.megatron_utils.peft_transport.build_peft_transport",
        return_value=transport,
    ):
        updater.connect_rollout_engines(
            rollout_engines=[engine],
            rollout_engine_lock=rollout_engine_lock,
            engine_gpu_counts=[1],
            engine_gpu_offsets=[0],
        )

    mock_connect_distributed.assert_not_called()
    transport.connect.assert_called_once_with([engine], rollout_engine_lock, [1])


@patch(f"{_MODULE}.connect_rollout_engines_from_distributed")
@patch(f"{_MODULE}.dist.get_rank", return_value=1)
@patch(f"{_MODULE}.get_parallel_state")
def test_distributed_peft_connects_one_source_per_pipeline_stage(
    mock_parallel_state,
    _mock_rank,
    mock_connect_distributed,
):
    """A nonzero PP stage owns different adapter tensors and must create its PEFT transport."""
    from orbit.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

    mock_parallel_state.return_value = SimpleNamespace(
        intra_dp_cp=SimpleNamespace(rank=0),
        tp=SimpleNamespace(rank=0),
        pp=SimpleNamespace(rank=1),
    )
    engine = MagicMock(name="rollout_engine")
    transport = _ExistingTransport()
    updater = object.__new__(UpdateWeightFromTensor)
    updater.args = Namespace(
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        rollout_num_gpus_per_engine=1,
        peft_distributed_transport="ray",
    )
    updater._peft_args = updater.args
    updater._peft_sync_spec = object()
    updater._peft_transport = None
    updater._peft_transport_mode = None
    updater._model_update_groups = None
    updater._ipc_gather_group = object()
    updater._ipc_gather_src = 0
    updater._ipc_gather_layout = ((0, 1),)

    rollout_engine_lock = MagicMock()
    with patch(
        "orbit.backends.megatron_utils.peft_transport.build_peft_transport",
        return_value=transport,
    ):
        updater.connect_rollout_engines(
            rollout_engines=[engine],
            rollout_engine_lock=rollout_engine_lock,
            engine_gpu_counts=[1],
            engine_gpu_offsets=[0],
        )

    assert updater._is_distributed_src_rank is True
    mock_connect_distributed.assert_not_called()
    transport.connect.assert_called_once_with([engine], rollout_engine_lock, [1])


@patch(f"{_MODULE}.dist.new_group")
@patch(f"{_MODULE}.dist.get_rank", return_value=1)
def test_colocated_peft_rebuilds_ipc_groups_for_heterogeneous_engine_layout(
    _mock_rank,
    mock_new_group,
):
    """Every rank must rebuild all IPC groups when the actual engine layout changes."""
    from orbit.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

    first_group = object()
    second_group = object()
    mock_new_group.side_effect = [first_group, second_group]
    engines = [MagicMock(name="tp1_engine"), MagicMock(name="tp3_engine")]
    transport = _ExistingTransport()
    updater = object.__new__(UpdateWeightFromTensor)
    updater.args = Namespace(
        colocate=True,
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        rollout_num_gpus_per_engine=2,
    )
    updater._peft_sync_spec = object()
    updater._peft_transport = transport
    updater._peft_transport_mode = "ipc"
    updater._model_update_groups = None
    updater._ipc_gather_group = object()
    updater._ipc_gather_src = 0
    updater._ipc_gather_layout = ((0, 2), (2, 2))

    rollout_engine_lock = MagicMock()
    updater.connect_rollout_engines(
        rollout_engines=engines,
        rollout_engine_lock=rollout_engine_lock,
        engine_gpu_counts=[1, 3],
        engine_gpu_offsets=[0, 1],
    )

    assert [call.kwargs for call in mock_new_group.call_args_list] == [
        {"ranks": [0], "backend": "gloo"},
        {"ranks": [1, 2, 3], "backend": "gloo"},
    ]
    assert updater._ipc_gather_layout == ((0, 1), (1, 3))
    assert updater._ipc_gather_group is second_group
    assert updater._ipc_gather_src == 1
    transport.connect.assert_called_once_with([engines[1]], rollout_engine_lock, [])
