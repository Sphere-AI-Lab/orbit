from types import SimpleNamespace

import pytest

from miles.utils.arguments import validate_async_off_policy_correction


def _args(**overrides) -> SimpleNamespace:
    defaults = dict(
        advantage_estimator="ppo",
        use_rollout_logprobs=False,
        use_tis=False,
        keep_old_actor=False,
        update_weights_interval=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_async_ppo_without_correction_raises() -> None:
    with pytest.raises(AssertionError, match="behavior-policy correction"):
        validate_async_off_policy_correction(_args())


@pytest.mark.parametrize("flag", ["use_rollout_logprobs", "use_tis", "keep_old_actor"])
def test_async_ppo_with_any_correction_passes(flag: str) -> None:
    validate_async_off_policy_correction(_args(**{flag: True}))


def test_keep_old_actor_alone_rejected_when_weight_updates_are_batched() -> None:
    # With interval 2, train_async prefetches one final W0 rollout, then publishes
    # W2 and snapshots W2 as old_actor. The saved W0 batch would therefore use a
    # W2 PPO denominator on the next iteration.
    with pytest.raises(AssertionError, match="update-weights-interval 1"):
        validate_async_off_policy_correction(_args(keep_old_actor=True, update_weights_interval=2))


@pytest.mark.parametrize("flag", ["use_rollout_logprobs", "use_tis"])
def test_explicit_logprob_corrections_allow_batched_weight_updates(flag: str) -> None:
    validate_async_off_policy_correction(_args(keep_old_actor=True, update_weights_interval=2, **{flag: True}))


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_async_rejects_nonpositive_or_noninteger_weight_update_interval(interval) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_async_off_policy_correction(_args(use_rollout_logprobs=True, update_weights_interval=interval))


def test_non_ppo_estimator_skips_validation() -> None:
    validate_async_off_policy_correction(
        _args(advantage_estimator="grpo", keep_old_actor=True, update_weights_interval=2)
    )
