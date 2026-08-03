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


def _run_layout_case(rank: int, world_size: int, qkv_format: str) -> None:
    args = _ppo_args(qkv_format=qkv_format)
    total_lengths = [8, 11]
    response_lengths = [5, 6]
    max_seq_lens = [12, 12]
    rewards = [10.0, 20.0]
    full_log_probs = [
        torch.arange(1, response_length + 1, dtype=torch.float32) for response_length in response_lengths
    ]
    full_values = [torch.zeros_like(log_probs) for log_probs in full_log_probs]

    set_parallel_state(_parallel_state(rank=rank, world_size=world_size))
    local_log_probs = [
        slice_log_prob_with_cp(log_probs, total_length, response_length, qkv_format, max_seq_len)
        for log_probs, total_length, response_length, max_seq_len in zip(
            full_log_probs, total_lengths, response_lengths, max_seq_lens, strict=True
        )
    ]
    local_values = [torch.zeros_like(log_probs) for log_probs in local_log_probs]
    # chunk_size = ceil(12 / (2 * cp_size)) = 3 for both formats: sample 0
    # (prompt 3, logits span [2, 7)) puts 1 position on rank 0 and 4 on rank 1;
    # sample 1 (prompt 5, logits span [4, 10)) puts 1 on rank 0 and 5 on rank 1.
    expected_local_sizes = [[1, 1], [4, 5]][rank]
    assert [tensor.numel() for tensor in local_log_probs] == expected_local_sizes

    rollout_data = _ppo_rollout_data(
        log_probs=local_log_probs,
        rewards=rewards,
        values=local_values,
        loss_masks=[torch.ones(response_length) for response_length in response_lengths],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )
    compute_advantages_and_returns(args, rollout_data)
    cp_advantages = [
        all_gather_with_cp(advantage, total_length, response_length, qkv_format, max_seq_len)
        for advantage, total_length, response_length, max_seq_len in zip(
            rollout_data["advantages"], total_lengths, response_lengths, max_seq_lens, strict=True
        )
    ]
    cp_returns = [
        all_gather_with_cp(ret, total_length, response_length, qkv_format, max_seq_len)
        for ret, total_length, response_length, max_seq_len in zip(
            rollout_data["returns"], total_lengths, response_lengths, max_seq_lens, strict=True
        )
    ]

    set_parallel_state(_parallel_state())
    baseline_data = _ppo_rollout_data(
        log_probs=[tensor.clone() for tensor in full_log_probs],
        rewards=rewards,
        values=[tensor.clone() for tensor in full_values],
        loss_masks=[torch.ones(response_length) for response_length in response_lengths],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )
    compute_advantages_and_returns(args, baseline_data)

    for cp_advantage, cp_return, baseline_advantage, baseline_return in zip(
        cp_advantages, cp_returns, baseline_data["advantages"], baseline_data["returns"], strict=True
    ):
        torch.testing.assert_close(cp_advantage, baseline_advantage)
        torch.testing.assert_close(cp_return, baseline_return)


def _worker_bshd_layout_metadata(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        _run_layout_case(rank, world_size, qkv_format="bshd")
    finally:
        dist.destroy_process_group()


def _worker_padded_thd_layout_metadata(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        _run_layout_case(rank, world_size, qkv_format="thd")
    finally:
        dist.destroy_process_group()


def test_ppo_bshd_cp_uses_padded_layout_metadata() -> None:
    run_multiprocess(_worker_bshd_layout_metadata)


def test_ppo_padded_thd_cp_uses_padded_layout_metadata() -> None:
    run_multiprocess(_worker_padded_thd_layout_metadata)
