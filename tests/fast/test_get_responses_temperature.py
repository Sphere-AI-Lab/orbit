from argparse import Namespace

import pytest
import torch

from orbit.backends.training_utils.loss import get_responses
from orbit.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from orbit.utils.arguments import validate_rollout_temperature


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


def _args(temperature: float) -> Namespace:
    return Namespace(
        rollout_temperature=temperature,
        qkv_format="thd",
        true_on_policy_mode=False,
    )


def _collect(logits: torch.Tensor, temperature: float) -> list[torch.Tensor]:
    total_length, response_length = 5, 3
    tokens = [torch.arange(total_length, dtype=torch.long)]
    return [
        chunk
        for chunk, _ in get_responses(
            logits.clone(),
            args=_args(temperature),
            unconcat_tokens=tokens,
            total_lengths=[total_length],
            response_lengths=[response_length],
        )
    ]


def test_value_logits_are_not_temperature_scaled() -> None:
    value_logits = torch.randn(1, 5, 1, dtype=torch.float32)
    scaled = _collect(value_logits, temperature=0.5)
    unscaled = _collect(value_logits, temperature=1.0)
    torch.testing.assert_close(scaled[0], unscaled[0])


def test_policy_logits_are_temperature_scaled() -> None:
    policy_logits = torch.randn(1, 5, 4, dtype=torch.float32)
    scaled = _collect(policy_logits, temperature=0.5)
    unscaled = _collect(policy_logits, temperature=1.0)
    torch.testing.assert_close(scaled[0], unscaled[0] / 0.5)


def test_non_positive_rollout_temperature_rejected() -> None:
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="rollout-temperature"):
            validate_rollout_temperature(Namespace(rollout_temperature=bad))
    validate_rollout_temperature(Namespace(rollout_temperature=1.0))
