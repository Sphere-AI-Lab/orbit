import pytest
import torch
import torch.distributed as dist

from tests.fast.dist_utils import init_gloo, run_multiprocess

from orbit.backends.training_utils import log_utils
from orbit.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _single_process_state() -> None:
    single = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(ParallelState(intra_dp=single, intra_dp_cp=single, cp=single, tp=single, is_pp_last_stage=True))


def test_aggregate_train_losses_preserves_min_and_max_across_microbatches(monkeypatch) -> None:
    _single_process_state()
    reduce_ops = []

    def _record_all_reduce(tensor, op, group):
        reduce_ops.append(op)

    monkeypatch.setattr(log_utils.dist, "all_reduce", _record_all_reduce)
    keys = ["loss", "gap_max", "opd_topk/teacher_mass_min"]
    losses = [
        {"keys": keys, "values": torch.tensor([2.0, 6.0, 0.8, 0.4])},
        {"keys": keys, "values": torch.tensor([3.0, 9.0, 0.9, 0.25])},
    ]

    result = log_utils.aggregate_train_losses(losses)

    assert result == {
        "loss": 3.0,
        "gap_max": pytest.approx(0.9),
        "opd_topk/teacher_mass_min": 0.25,
    }
    assert reduce_ops == [dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.MIN]


def _worker_remote_extrema(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        world = GroupInfo(
            rank=rank,
            size=world_size,
            group=dist.group.WORLD,
            gloo_group=dist.group.WORLD,
        )
        single = GroupInfo(rank=0, size=1, group=None)
        # Treat WORLD as CP as well: mean metrics receive the CP multiplier,
        # while extrema must remain raw global extrema.
        set_parallel_state(
            ParallelState(intra_dp=world, intra_dp_cp=world, cp=world, tp=single, is_pp_last_stage=True)
        )
        keys = ["loss", "gap_max", "opd_topk/teacher_mass_min"]
        local_values = torch.tensor([2.0, 6.0, 0.8, 0.4]) if rank == 0 else torch.tensor([3.0, 9.0, 0.9, 0.25])
        result = log_utils.aggregate_train_losses([{"keys": keys, "values": local_values}])

        assert result["loss"] == 6.0
        assert result["gap_max"] == pytest.approx(0.9)
        assert result["opd_topk/teacher_mass_min"] == 0.25
    finally:
        dist.destroy_process_group()


def test_aggregate_train_losses_reduces_remote_extrema_without_cp_normalization() -> None:
    run_multiprocess(_worker_remote_extrema, world_size=2)
