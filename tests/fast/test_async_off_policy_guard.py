from types import SimpleNamespace

import pytest

from orbit.utils.arguments import validate_async_off_policy_correction


def _args(**overrides) -> SimpleNamespace:
    defaults = dict(
        advantage_estimator="ppo",
        use_rollout_logprobs=False,
        use_tis=False,
        keep_old_actor=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_async_ppo_without_correction_raises() -> None:
    with pytest.raises(AssertionError, match="behavior-policy correction"):
        validate_async_off_policy_correction(_args())


@pytest.mark.parametrize("flag", ["use_rollout_logprobs", "use_tis", "keep_old_actor"])
def test_async_ppo_with_any_correction_passes(flag: str) -> None:
    validate_async_off_policy_correction(_args(**{flag: True}))


def test_non_ppo_estimator_skips_validation() -> None:
    validate_async_off_policy_correction(_args(advantage_estimator="grpo"))
