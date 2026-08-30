import pytest
import torch
import torch.distributed as dist

from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils import log_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.backends.training_utils.loss_hub.math_utils import VALUE_EV_METRIC_KEY, VALUE_EV_STAT_KEYS


def _single_process_state() -> None:
    single = GroupInfo(rank=0, size=1, group=None)
    # upstream's ParallelState gained required pp/ep/etp/indep_dp groups; trivial here.
    set_parallel_state(
        ParallelState(
            intra_dp=single,
            intra_dp_cp=single,
            cp=single,
            tp=single,
            pp=single,
            ep=single,
            etp=single,
            indep_dp=single,
            is_pp_last_stage=True,
        )
    )


def test_aggregate_train_losses_preserves_min_and_max_across_microbatches(monkeypatch) -> None:
    _single_process_state()
    reduce_ops = []

    # upstream routes the reduction through MultiPGUtil.all_reduce(tensor, groups, op)
    # instead of a bare dist.all_reduce, so the stub follows it.
    def _record_all_reduce(tensor, groups_inner_to_outer, op):
        reduce_ops.append(op)

    monkeypatch.setattr(log_utils.MultiPGUtil, "all_reduce", _record_all_reduce)
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


def test_aggregate_train_losses_finalizes_value_explained_var(monkeypatch) -> None:
    _single_process_state()
    monkeypatch.setattr(log_utils.MultiPGUtil, "all_reduce", lambda tensor, groups, op: None)

    # Token-level ground truth across two micro-batches of unequal token count:
    # returns r and errors d = r - v over the unmasked tokens.
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
    errors = torch.tensor([0.5, -0.5, 1.0, 0.0])

    def _stats(token_slice: slice) -> torch.Tensor:
        r = returns[token_slice]
        d = errors[token_slice]
        # values[0] is the per-sample count here (2 samples per micro-batch):
        # a normalization constant unrelated to the token count, which must
        # cancel inside the EV finalization.
        return torch.tensor(
            [2.0, 0.1, float(r.numel()), r.sum(), (r**2).sum(), d.sum(), (d**2).sum()]
        )

    keys = ["value_loss", *VALUE_EV_STAT_KEYS]
    losses = [
        {"keys": keys, "values": _stats(slice(0, 1))},
        {"keys": keys, "values": _stats(slice(1, 4))},
    ]

    result = log_utils.aggregate_train_losses(losses)

    expected_ev = 1.0 - errors.var(unbiased=False).item() / returns.var(unbiased=False).item()
    assert result[VALUE_EV_METRIC_KEY] == pytest.approx(expected_ev)
    assert not any(key in result for key in VALUE_EV_STAT_KEYS)
    # Ordinary metrics keep the existing sum/count normalization (0.2 / 4 samples).
    assert result["value_loss"] == pytest.approx(0.05)


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
            ParallelState(
                intra_dp=world,
                intra_dp_cp=world,
                cp=world,
                tp=single,
                pp=single,
                ep=single,
                etp=single,
                indep_dp=single,
                is_pp_last_stage=True,
            )
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
