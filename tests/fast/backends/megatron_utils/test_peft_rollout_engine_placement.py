"""Regression coverage for PEFT rollout-engine placement classification."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-fast")


from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"


class _ExistingTransport:
    def __init__(self) -> None:
        self.connect = MagicMock()
        self.disconnect = MagicMock()


@patch(f"{_MODULE}.dist.get_rank", return_value=1)
@patch(f"{_MODULE}.get_parallel_state")
def test_non_colocated_peft_engine_is_always_distributed(mock_parallel_state, _mock_rank):
    """Offset zero is relative to the rollout placement group when not colocated."""
    from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor

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
