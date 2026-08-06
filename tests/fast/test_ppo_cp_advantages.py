from argparse import Namespace

import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from orbit.backends.training_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp
from orbit.backends.training_utils.loss import compute_advantages_and_returns
from orbit.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _parallel_state(
    rank: int = 0,
    world_size: int = 1,
    *,
    intra_dp_group: dist.ProcessGroup | None = None,
) -> ParallelState:
    trivial_group = GroupInfo(rank=0, size=1, group=intra_dp_group)
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


def _run_normalized_advantage_case(
    rank: int,
    world_size: int,
    intra_dp_group: dist.ProcessGroup,
    *,
    total_lengths: list[int],
    response_lengths: list[int],
    rewards: list[float],
    full_values: list[torch.Tensor],
    expected_local_sizes: tuple[list[int], list[int]],
) -> None:
    full_log_probs = [torch.zeros(length) for length in response_lengths]
    loss_masks = [torch.ones(length) for length in response_lengths]
    args = _ppo_args(gamma=0.0, lambd=0.0)
    args.kl_coef = 0.0
    args.normalize_advantages = True

    set_parallel_state(
        _parallel_state(
            rank=rank,
            world_size=world_size,
            intra_dp_group=intra_dp_group,
        )
    )
    local_log_probs = [
        slice_log_prob_with_cp(log_probs, total_length, response_length)
        for log_probs, total_length, response_length in zip(
            full_log_probs, total_lengths, response_lengths, strict=True
        )
    ]
    local_values = [
        slice_log_prob_with_cp(values, total_length, response_length)
        for values, total_length, response_length in zip(
            full_values, total_lengths, response_lengths, strict=True
        )
    ]
    assert [tensor.numel() for tensor in local_values] == expected_local_sizes[rank]

    rollout_data = _ppo_rollout_data(
        log_probs=local_log_probs,
        rewards=rewards,
        values=local_values,
        loss_masks=[mask.clone() for mask in loss_masks],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    )
    compute_advantages_and_returns(args, rollout_data)
    cp_advantages = [
        all_gather_with_cp(advantage, total_length, response_length)
        for advantage, total_length, response_length in zip(
            rollout_data["advantages"], total_lengths, response_lengths, strict=True
        )
    ]
    cp_returns = [
        all_gather_with_cp(ret, total_length, response_length)
        for ret, total_length, response_length in zip(
            rollout_data["returns"], total_lengths, response_lengths, strict=True
        )
    ]

    # Build the single-rank, unnormalized reference without entering another
    # distributed collective, then apply the exact global masked-whitening
    # formula used by distributed_masked_whiten.
    reference_args = _ppo_args(gamma=0.0, lambd=0.0)
    reference_args.kl_coef = 0.0
    set_parallel_state(_parallel_state())
    reference_data = _ppo_rollout_data(
        log_probs=[tensor.clone() for tensor in full_log_probs],
        rewards=rewards,
        values=[tensor.clone() for tensor in full_values],
        loss_masks=[mask.clone() for mask in loss_masks],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    )
    compute_advantages_and_returns(reference_args, reference_data)

    flat_advantages = torch.cat(reference_data["advantages"])
    flat_mask = torch.cat(loss_masks)
    count = flat_mask.sum()
    mean = (flat_advantages * flat_mask).sum() / count
    mean_square = (flat_advantages.square() * flat_mask).sum() / count
    variance = (mean_square - mean.square()) * count / (count - 1)
    expected_flat = (flat_advantages - mean) * torch.rsqrt(variance + 1e-8)
    expected_advantages = expected_flat.split(response_lengths)

    for actual, expected in zip(cp_advantages, expected_advantages, strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(cp_returns, reference_data["returns"], strict=True):
        torch.testing.assert_close(actual, expected)


def _worker_normalized_advantages_with_empty_cp_rank(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        assert world_size == 2
        # Model the real topology: DP excludes CP, so each CP rank has its own
        # singleton intra-DP group while intra_dp_cp and cp span WORLD.
        singleton_groups = [dist.new_group(ranks=[group_rank], backend="gloo") for group_rank in range(world_size)]
        intra_dp_group = singleton_groups[rank]

        # Rank 0 has an empty slice for sample 0 and the remaining slices are
        # uneven.  This distinguishes global DP+CP whitening from the old
        # per-CP-shard whitening over the singleton intra-DP groups.
        _run_normalized_advantage_case(
            rank,
            world_size,
            intra_dp_group,
            total_lengths=[7, 7],
            response_lengths=[2, 6],
            rewards=[2.0, -1.0],
            full_values=[torch.tensor([0.5, -0.5]), torch.tensor([1.0, -2.0, 0.25, 0.75, -1.5, 2.0])],
            expected_local_sizes=([0, 2], [2, 4]),
        )

        # Rank 0 has no local response tokens at all.  It must nevertheless
        # enter the combined-group whitening collective with empty tensors.
        _run_normalized_advantage_case(
            rank,
            world_size,
            intra_dp_group,
            total_lengths=[7, 7],
            response_lengths=[1, 2],
            rewards=[2.0, -1.0],
            full_values=[torch.tensor([0.5]), torch.tensor([1.0, -2.0])],
            expected_local_sizes=([0, 0], [1, 2]),
        )
    finally:
        dist.destroy_process_group()


def test_ppo_normalized_advantages_include_empty_cp_ranks_in_global_statistics() -> None:
    run_multiprocess(_worker_normalized_advantages_with_empty_cp_rank)


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


def _run_ppo_masked_case(rank: int) -> None:
    total_length, response_length = 7, 6
    loss_mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    for gamma, lambd in [(0.0, 0.0), (0.9, 0.8)]:
        args = _ppo_args(gamma=gamma, lambd=lambd)
        full_kl = torch.arange(1, response_length + 1, dtype=torch.float32)
        full_values = torch.tensor([0.5, -0.3, 0.7, 0.1, -0.2, 0.4])

        set_parallel_state(_parallel_state(rank=rank, world_size=2))
        local_kl = slice_log_prob_with_cp(full_kl, total_length, response_length).clone()
        local_values = slice_log_prob_with_cp(full_values, total_length, response_length).clone()

        rollout_data = _ppo_rollout_data(
            log_probs=[local_kl],
            rewards=[10.0],
            values=[local_values],
            loss_masks=[loss_mask.clone()],
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
            loss_masks=[loss_mask.clone()],
            total_lengths=[total_length],
            response_lengths=[response_length],
        )
        compute_advantages_and_returns(args, baseline_data)

        torch.testing.assert_close(cp_advantages, baseline_data["advantages"][0])
        torch.testing.assert_close(cp_returns, baseline_data["returns"][0])
        assert torch.all(cp_advantages[loss_mask == 0] == 0)
        assert torch.all(cp_returns[loss_mask == 0] == 0)

        if gamma == 0.0 and lambd == 0.0:
            # Terminal reward lands on the last trainable token (index 4), and
            # with gamma = 0 each trainable advantage is reward - value.
            expected = torch.zeros(response_length)
            expected[0] = -0.1 * 1.0 - 0.5
            expected[1] = -0.1 * 2.0 - (-0.3)
            expected[4] = -0.1 * 5.0 + 10.0 - (-0.2)
            torch.testing.assert_close(cp_advantages, expected)


def _worker_masked_case(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        _run_ppo_masked_case(rank)
    finally:
        dist.destroy_process_group()


def test_ppo_masked_gae_matches_single_rank_baseline() -> None:
    run_multiprocess(_worker_masked_case)
