from argparse import Namespace

import pytest
import torch

from orbit.peft.opd import teacher_lm_head as teacher_lm_head_module
from orbit.backends.training_utils.cp_utils import get_sum_of_sample_mean
from orbit.backends.training_utils.loss import loss_function
from orbit.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


_JSD_CHECKPOINT_KEY = "<test-loss-function-microbatch-invariance>"


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

    whole = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, calculate_per_token_loss)(x)

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

    get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, calculate_per_token_loss)(x_whole).backward()

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


def _make_jsd_inputs() -> tuple[torch.Tensor, dict]:
    generator = torch.Generator().manual_seed(91)
    response_lengths = [3, 5, 2, 4, 6, 1]
    prompt_lengths = [2, 3, 4, 2, 5, 3]
    total_lengths = [prompt + response for prompt, response in zip(prompt_lengths, response_lengths, strict=True)]
    vocab_size = 17
    loss_masks = [
        torch.randint(0, 2, (response,), generator=generator, dtype=torch.int64) for response in response_lengths
    ]
    loss_masks[2].zero_()  # Fully masked: it must remain partition-invariant.
    logits = torch.randn(1, sum(total_lengths), vocab_size, generator=generator)
    batch = {
        "unconcat_tokens": [torch.randint(0, vocab_size, (total,), generator=generator) for total in total_lengths],
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
        "loss_masks": loss_masks,
        # An identity teacher head turns these rows directly into teacher logits.
        "teacher_hidden_states": [
            torch.randn(response, vocab_size, generator=generator) for response in response_lengths
        ],
    }
    return logits, batch


def _jsd_args(*, calculate_per_token_loss: bool, global_batch_size: int) -> Namespace:
    return Namespace(
        loss_type="opd_jsd_loss",
        calculate_per_token_loss=calculate_per_token_loss,
        global_batch_size=global_batch_size,
        use_dynamic_global_batch_size=False,
        recompute_loss_function=False,
        qkv_format="thd",
        allgather_cp=False,
        opd_jsd_beta=0.35,
        rollout_temperature=1.0,
        opd_log_prob_min_clamp=-1e30,
        opd_loss_max_clamp=1e30,
        opd_jsd_pointwise_clip=None,
        opd_log_topk_overlap=False,
        opd_topk_overlap_ks=[],
        use_kl_loss=False,
        teacher_hf_checkpoint=_JSD_CHECKPOINT_KEY,
        log_probs_chunk_size=-1,
        true_on_policy_mode=False,
        vocab_size=17,
    )


def _slice_sample_batch(batch: dict, start: int, stop: int) -> dict:
    """Slice sample-aligned list fields; reusable by JSD and top-k OPD batches."""
    sample_count = len(batch["response_lengths"])
    return {
        key: value[start:stop] if isinstance(value, list) and len(value) == sample_count else value
        for key, value in batch.items()
    }


def _run_megatron_scaled_loss(
    args: Namespace,
    batch: dict,
    base_logits: torch.Tensor,
    microbatch_sizes: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce MCore's accumulation/final normalization around loss_function."""
    assert sum(microbatch_sizes) == len(batch["response_lengths"])
    logits = base_logits.detach().clone().requires_grad_(True)
    num_microbatches = len(microbatch_sizes)
    losses = []
    normalizers = []
    sample_start = 0
    token_start = 0
    for microbatch_size in microbatch_sizes:
        sample_stop = sample_start + microbatch_size
        token_stop = token_start + sum(batch["total_lengths"][sample_start:sample_stop])
        microbatch = _slice_sample_batch(batch, sample_start, sample_stop)
        loss, normalizer, _ = loss_function(
            args,
            microbatch,
            num_microbatches,
            logits[:, token_start:token_stop],
            apply_megatron_loss_scaling=True,
        )
        losses.append(loss)
        normalizers.append(normalizer)
        sample_start = sample_stop
        token_start = token_stop

    accumulated_loss = torch.stack(losses).sum()
    if args.calculate_per_token_loss:
        # finalize_model_grads divides accumulated gradients by this token count.
        objective = accumulated_loss / torch.stack(normalizers).sum()
    else:
        # MCore divides each microbatch loss by num_microbatches before backward.
        objective = accumulated_loss / num_microbatches
    objective.backward()
    return objective.detach(), logits.grad.detach()


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_opd_jsd_loss_function_is_microbatch_loss_and_gradient_invariant(
    calculate_per_token_loss: bool,
) -> None:
    logits, batch = _make_jsd_inputs()
    args = _jsd_args(
        calculate_per_token_loss=calculate_per_token_loss,
        global_batch_size=len(batch["response_lengths"]),
    )
    teacher_lm_head_module._TEACHER_LM_HEAD_CACHE[_JSD_CHECKPOINT_KEY] = torch.eye(logits.size(-1))
    teacher_lm_head_module._SHARDED.add(_JSD_CHECKPOINT_KEY)
    try:
        whole_loss, whole_grad = _run_megatron_scaled_loss(args, batch, logits, (6,))
        split_loss, split_grad = _run_megatron_scaled_loss(args, batch, logits, (2, 3, 1))
    finally:
        teacher_lm_head_module._TEACHER_LM_HEAD_CACHE.pop(_JSD_CHECKPOINT_KEY, None)
        teacher_lm_head_module._SHARDED.discard(_JSD_CHECKPOINT_KEY)

    assert whole_loss > 0
    assert whole_grad.abs().sum() > 0
    torch.testing.assert_close(split_loss, whole_loss, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(split_grad, whole_grad, atol=2e-6, rtol=2e-6)


def _make_ppo_value_inputs() -> tuple[torch.Tensor, dict]:
    generator = torch.Generator().manual_seed(2026)
    response_lengths = [2, 3]
    total_lengths = [3, 4]
    logits = torch.randn(1, sum(total_lengths), 1, generator=generator)
    batch = {
        "unconcat_tokens": [torch.zeros(total, dtype=torch.long) for total in total_lengths],
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
        "loss_masks": [torch.ones(response_lengths[0]), torch.zeros(response_lengths[1])],
        "values": [torch.zeros(response) for response in response_lengths],
        "returns": [torch.randn(response, generator=generator) for response in response_lengths],
    }
    return logits, batch


def _ppo_value_args(global_batch_size: int) -> Namespace:
    return Namespace(
        loss_type="value_loss",
        calculate_per_token_loss=True,
        global_batch_size=global_batch_size,
        use_dynamic_global_batch_size=False,
        recompute_loss_function=False,
        qkv_format="thd",
        allgather_cp=False,
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        value_clip=0.2,
    )


def test_ppo_value_per_token_gradient_ignores_fully_masked_sample() -> None:
    logits, batch = _make_ppo_value_inputs()
    valid_batch = _slice_sample_batch(batch, 0, 1)
    valid_token_count = batch["total_lengths"][0]

    valid_loss, valid_grad = _run_megatron_scaled_loss(
        _ppo_value_args(global_batch_size=1),
        valid_batch,
        logits[:, :valid_token_count],
        (1,),
    )
    augmented_loss, augmented_grad = _run_megatron_scaled_loss(
        _ppo_value_args(global_batch_size=2),
        batch,
        logits,
        (1, 1),
    )

    assert valid_loss > 0
    torch.testing.assert_close(augmented_loss, valid_loss)
    torch.testing.assert_close(augmented_grad[:, :valid_token_count], valid_grad)
    torch.testing.assert_close(
        augmented_grad[:, valid_token_count:],
        torch.zeros_like(augmented_grad[:, valid_token_count:]),
    )
