from argparse import Namespace  # noqa: F401  (parity with sibling test files)

import pytest
import torch

from orbit.backends.training_utils.cp_utils import get_sum_of_sample_mean
from orbit.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


@pytest.fixture(autouse=True)
def _trivial_parallel_state() -> None:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial_group,
            intra_dp_cp=trivial_group,
            cp=trivial_group,
            tp=trivial_group,
        )
    )


def _make_batch() -> tuple[list[int], list[int], list[torch.Tensor], torch.Tensor]:
    torch.manual_seed(0)
    response_lengths = [3, 5, 2, 4, 6, 1]
    total_lengths = [length + 2 for length in response_lengths]
    loss_masks = [torch.randint(0, 2, (length,), dtype=torch.float32) for length in response_lengths]
    loss_masks[2] = torch.zeros(2)  # fully-masked sample: exercises clamp_min(., 1)
    x = torch.randn(sum(response_lengths))
    return total_lengths, response_lengths, loss_masks, x


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_reduction_is_microbatch_invariant(calculate_per_token_loss: bool) -> None:
    # The distillation/PPO losses reduce via this closure once per micro-batch
    # and sum across micro-batches (verl 594c51bc / 2eb020aa regression class):
    # partitioning the samples must not change the total. Megatron's outer
    # 1/num_microbatches scaling is applied uniformly on top and is out of
    # scope here.
    total_lengths, response_lengths, loss_masks, x = _make_batch()

    whole = get_sum_of_sample_mean(
        total_lengths, response_lengths, loss_masks, calculate_per_token_loss
    )(x)

    split_total = torch.zeros(())
    start_sample, start_token = 0, 0
    for micro_batch_size in (2, 3, 1):
        end_sample = start_sample + micro_batch_size
        n_tokens = sum(response_lengths[start_sample:end_sample])
        reduction = get_sum_of_sample_mean(
            total_lengths[start_sample:end_sample],
            response_lengths[start_sample:end_sample],
            loss_masks[start_sample:end_sample],
            calculate_per_token_loss,
        )
        split_total = split_total + reduction(x[start_token : start_token + n_tokens])
        start_sample, start_token = end_sample, start_token + n_tokens

    torch.testing.assert_close(split_total, whole)


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_reduction_gradient_is_microbatch_invariant(calculate_per_token_loss: bool) -> None:
    total_lengths, response_lengths, loss_masks, x = _make_batch()
    x_whole = x.clone().requires_grad_(True)
    x_split = x.clone().requires_grad_(True)

    get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, calculate_per_token_loss)(
        x_whole
    ).backward()

    total = torch.zeros(())
    start_sample, start_token = 0, 0
    for micro_batch_size in (2, 3, 1):
        end_sample = start_sample + micro_batch_size
        n_tokens = sum(response_lengths[start_sample:end_sample])
        total = total + get_sum_of_sample_mean(
            total_lengths[start_sample:end_sample],
            response_lengths[start_sample:end_sample],
            loss_masks[start_sample:end_sample],
            calculate_per_token_loss,
        )(x_split[start_token : start_token + n_tokens])
        start_sample, start_token = end_sample, start_token + n_tokens
    total.backward()

    torch.testing.assert_close(x_split.grad, x_whole.grad)
