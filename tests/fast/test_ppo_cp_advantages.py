from argparse import Namespace

import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from orbit.backends.training_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp
from orbit.backends.training_utils.loss import compute_advantages_and_returns
from orbit.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _parallel_state(rank: int = 0, world_size: int = 1) -> ParallelState:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    cp_group = dist.group.WORLD if world_size > 1 else None
    return ParallelState(
        intra_dp=trivial_group,
        intra_dp_cp=GroupInfo(rank=rank, size=world_size, group=cp_group),
        cp=GroupInfo(rank=rank, size=world_size, group=cp_group),
        tp=trivial_group,
    )


def _ppo_args(gamma: float = 0.0, lambd: float = 0.0, qkv_format: str = "thd") -> Namespace:
    return Namespace(
        advantage_estimator="ppo",
        use_rollout_logprobs=False,
        kl_coef=0.1,
        kl_loss_type="k1",
        gamma=gamma,
        lambd=lambd,
        qkv_format=qkv_format,
        use_opd=False,
        opd_icepop=False,
        normalize_advantages=False,
    )


def _ppo_rollout_data(
    log_probs: list[torch.Tensor],
    rewards: list[float],
    values: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> dict:
    return {
        "log_probs": log_probs,
        "ref_log_probs": [torch.zeros_like(lp) for lp in log_probs],
        "rewards": rewards,
        "values": values,
        "response_lengths": response_lengths,
        "loss_masks": loss_masks,
        "total_lengths": total_lengths,
        "max_seq_lens": max_seq_lens,
    }


def _run_ppo_case(rank: int, total_length: int, response_length: int, expected_local_sizes: list[int]) -> None:
    args = _ppo_args()
    full_kl = torch.arange(1, response_length + 1, dtype=torch.float32)
    full_values = torch.zeros(response_length)

    set_parallel_state(_parallel_state(rank=rank, world_size=2))
    local_kl = slice_log_prob_with_cp(full_kl, total_length, response_length)
    local_values = slice_log_prob_with_cp(full_values, total_length, response_length)
    assert local_kl.numel() == expected_local_sizes[rank]

    rollout_data = _ppo_rollout_data(
        log_probs=[local_kl.clone()],
        rewards=[10.0],
        values=[local_values.clone()],
        loss_masks=[torch.ones(response_length)],
        total_lengths=[total_length],
        response_lengths=[response_length],
    )
    compute_advantages_and_returns(args, rollout_data)
    cp_advantages = all_gather_with_cp(rollout_data["advantages"][0], total_length, response_length)
    cp_returns = all_gather_with_cp(rollout_data["returns"][0], total_length, response_length)

    set_parallel_state(_parallel_state())
    baseline_data = _ppo_rollout_data(
        log_probs=[full_kl.clone()],
        rewards=[10.0],
        values=[full_values.clone()],
        loss_masks=[torch.ones(response_length)],
        total_lengths=[total_length],
        response_lengths=[response_length],
    )
    compute_advantages_and_returns(args, baseline_data)

    expected = -0.1 * full_kl
    expected[-1] += 10.0
    torch.testing.assert_close(cp_advantages, expected)
    torch.testing.assert_close(cp_returns, expected)
    torch.testing.assert_close(cp_advantages, baseline_data["advantages"][0])
    torch.testing.assert_close(cp_returns, baseline_data["returns"][0])


def _worker_tail_on_rank_one(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        _run_ppo_case(rank, total_length=7, response_length=6, expected_local_sizes=[2, 4])
    finally:
        dist.destroy_process_group()


def _worker_empty_rank_zero(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        _run_ppo_case(rank, total_length=7, response_length=2, expected_local_sizes=[0, 2])
    finally:
        dist.destroy_process_group()


def test_ppo_terminal_reward_is_added_to_global_response_tail() -> None:
    run_multiprocess(_worker_tail_on_rank_one)


def test_ppo_terminal_reward_handles_empty_rank_zero_shard() -> None:
    run_multiprocess(_worker_empty_rank_zero)
