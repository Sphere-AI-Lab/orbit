import math
from argparse import Namespace

import numpy as np
import pytest
import torch

from miles.backends.training_utils import log_utils
from miles.backends.training_utils.loss import get_values, loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.backends.training_utils.loss_hub.math_utils import (
    VALUE_EV_METRIC_KEY,
    VALUE_EV_STAT_KEYS,
    compute_value_explained_var,
)


@pytest.fixture(autouse=True)
def _trivial_parallel_state(monkeypatch) -> None:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial_group,
            intra_dp_cp=trivial_group,
            cp=trivial_group,
            tp=trivial_group,
        )
    )
    # Single process: the DP/CP all-reduce inside aggregate_train_losses is a no-op.
    monkeypatch.setattr(log_utils.dist, "all_reduce", lambda tensor, op, group: None)


def _value_args(*, calculate_per_token_loss: bool, global_batch_size: int) -> Namespace:
    return Namespace(
        loss_type="value_loss",
        calculate_per_token_loss=calculate_per_token_loss,
        global_batch_size=global_batch_size,
        use_dynamic_global_batch_size=False,
        recompute_loss_function=False,
        qkv_format="thd",
        allgather_cp=False,
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        value_clip=0.2,
    )


def _make_value_batch(returns_fn=None) -> tuple[torch.Tensor, dict]:
    generator = torch.Generator().manual_seed(41)
    response_lengths = [3, 5, 2, 4, 6, 1]
    prompt_lengths = [2, 3, 4, 2, 5, 3]
    total_lengths = [prompt + response for prompt, response in zip(prompt_lengths, response_lengths, strict=True)]
    loss_masks = [
        torch.randint(0, 2, (response,), generator=generator, dtype=torch.float32) for response in response_lengths
    ]
    loss_masks[2] = torch.zeros(2)  # fully-masked sample must not contribute
    loss_masks[5] = torch.ones(1)  # guarantee unmasked tokens exist
    logits = torch.randn(1, sum(total_lengths), 1, generator=generator)
    if returns_fn is None:
        returns = [torch.randn(response, generator=generator) for response in response_lengths]
    else:
        returns = [returns_fn(response) for response in response_lengths]
    batch = {
        "unconcat_tokens": [torch.zeros(total, dtype=torch.long) for total in total_lengths],
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
        "loss_masks": loss_masks,
        "values": [torch.randn(response, generator=generator) for response in response_lengths],
        "returns": returns,
    }
    return logits, batch


def _slice_sample_batch(batch: dict, start: int, stop: int) -> dict:
    sample_count = len(batch["response_lengths"])
    return {
        key: value[start:stop] if isinstance(value, list) and len(value) == sample_count else value
        for key, value in batch.items()
    }


def _run_pipeline(args: Namespace, batch: dict, logits: torch.Tensor, microbatch_sizes: tuple[int, ...]) -> dict:
    """Drive loss_function per micro-batch and reduce like train_one_step does."""
    assert sum(microbatch_sizes) == len(batch["response_lengths"])
    num_microbatches = len(microbatch_sizes)
    losses_reduced = []
    sample_start = 0
    token_start = 0
    for microbatch_size in microbatch_sizes:
        sample_stop = sample_start + microbatch_size
        token_stop = token_start + sum(batch["total_lengths"][sample_start:sample_stop])
        _, _, log_dict = loss_function(
            args,
            _slice_sample_batch(batch, sample_start, sample_stop),
            num_microbatches,
            logits[:, token_start:token_stop],
            apply_megatron_loss_scaling=True,
        )
        losses_reduced.append(log_dict)
        sample_start, token_start = sample_stop, token_stop
    return log_utils.aggregate_train_losses(losses_reduced)


def _predicted_values(args: Namespace, batch: dict, logits: torch.Tensor) -> list[torch.Tensor]:
    return [
        value.flatten()
        for value in get_values(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
        )["values"]
    ]


def _numpy_reference_ev(args: Namespace, batch: dict, logits: torch.Tensor) -> float:
    """Direct whole-dataset EV over unmasked tokens, independent of the metric pipeline."""
    values = torch.cat(_predicted_values(args, batch, logits)).numpy()
    returns = torch.cat(batch["returns"]).numpy()
    mask = torch.cat(batch["loss_masks"]).numpy().astype(bool)
    err = (returns - values)[mask]
    ret = returns[mask]
    return float(1.0 - np.var(err) / np.var(ret))


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_value_explained_var_matches_whole_dataset_numpy(calculate_per_token_loss: bool) -> None:
    logits, batch = _make_value_batch()
    args = _value_args(calculate_per_token_loss=calculate_per_token_loss, global_batch_size=6)

    # Unequal micro-batch sizes with unequal token counts and means: naive
    # per-micro-batch EV averaging would be biased here.
    split = _run_pipeline(args, batch, logits, (2, 3, 1))
    whole = _run_pipeline(args, batch, logits, (6,))
    expected = _numpy_reference_ev(args, batch, logits)

    assert split[VALUE_EV_METRIC_KEY] == pytest.approx(expected, rel=1e-5, abs=1e-6)
    assert whole[VALUE_EV_METRIC_KEY] == pytest.approx(expected, rel=1e-5, abs=1e-6)
    # The sufficient statistics are internal and must not leak into the logs.
    assert not any(key in split for key in VALUE_EV_STAT_KEYS)
    assert "value_loss" in split and "value_clipfrac" in split


def test_value_explained_var_perfect_critic_is_one() -> None:
    logits, batch = _make_value_batch()
    args = _value_args(calculate_per_token_loss=False, global_batch_size=6)
    batch["returns"] = [value.detach().clone() for value in _predicted_values(args, batch, logits)]

    result = _run_pipeline(args, batch, logits, (2, 3, 1))

    assert result[VALUE_EV_METRIC_KEY] == pytest.approx(1.0)


def test_value_explained_var_constant_returns_reports_zero() -> None:
    logits, batch = _make_value_batch(returns_fn=lambda response: torch.full((response,), 1.7))
    args = _value_args(calculate_per_token_loss=False, global_batch_size=6)

    result = _run_pipeline(args, batch, logits, (2, 2, 2))

    assert result[VALUE_EV_METRIC_KEY] == 0.0
    assert all(math.isfinite(value) for value in result.values())


def test_value_explained_var_all_masked_reports_zero() -> None:
    logits, batch = _make_value_batch()
    batch["loss_masks"] = [torch.zeros_like(mask) for mask in batch["loss_masks"]]
    args = _value_args(calculate_per_token_loss=False, global_batch_size=6)

    result = _run_pipeline(args, batch, logits, (3, 3))

    assert result[VALUE_EV_METRIC_KEY] == 0.0
    assert math.isfinite(result[VALUE_EV_METRIC_KEY])


def test_compute_value_explained_var_degenerate_guards() -> None:
    # No trainable tokens.
    assert compute_value_explained_var(0.0, 0.0, 0.0, 0.0, 0.0) == 0.0
    # Constant returns: Var(returns) == 0.
    assert compute_value_explained_var(4.0, 8.0, 16.0, 1.0, 2.0) == 0.0
    # Non-finite statistics must never leak NaN/inf into the logs.
    assert compute_value_explained_var(float("nan"), 1.0, 1.0, 1.0, 1.0) == 0.0
    assert compute_value_explained_var(4.0, float("inf"), 1.0, 1.0, 1.0) == 0.0


def test_compute_value_explained_var_is_scale_invariant() -> None:
    # aggregate_train_losses divides every metric by the same count; the shared
    # factor must cancel inside the EV computation.
    base = (5.0, 2.0, 7.0, 1.0, 3.0)
    scaled = tuple(3.5 * stat for stat in base)
    expected = 1.0 - (3.0 / 5.0 - (1.0 / 5.0) ** 2) / (7.0 / 5.0 - (2.0 / 5.0) ** 2)
    assert compute_value_explained_var(*base) == pytest.approx(expected)
    assert compute_value_explained_var(*scaled) == pytest.approx(expected)
