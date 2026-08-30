from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.loss import get_responses
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.utils.arguments import validate_rollout_temperature


@pytest.fixture(autouse=True)
def _trivial_parallel_state() -> None:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial_group,
            intra_dp_cp=trivial_group,
            cp=trivial_group,
            tp=trivial_group,
            # upstream's ParallelState gained required pp/ep/etp/indep_dp groups; trivial here.
            pp=trivial_group,
            ep=trivial_group,
            etp=trivial_group,
            indep_dp=trivial_group,
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


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_non_finite_or_non_positive_rollout_temperature_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and > 0"):
        validate_rollout_temperature(Namespace(rollout_temperature=bad))


def test_positive_finite_rollout_temperature_accepted() -> None:
    validate_rollout_temperature(Namespace(rollout_temperature=1.0))
