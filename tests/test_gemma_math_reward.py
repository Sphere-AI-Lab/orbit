from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from orbit.rewards.gemma_math import get_gemma_math_reward


def test_gemma_reward_grades_text_after_channel_marker():
    assert get_gemma_math_reward("thinking... <channel|> The answer is \\boxed{42}", "42") == 1
    assert get_gemma_math_reward("thinking... <channel|> \\boxed{7}", "42") == 0


def test_gemma_reward_uses_last_channel_marker():
    # wrong answer before the marker, correct after -> graded on the tail
    assert get_gemma_math_reward("\\boxed{7} <channel|> \\boxed{42}", "42") == 1


def test_gemma_reward_without_marker_grades_whole_response():
    # unlike deepscaler (which needs </think> or ###Response), gemma grades the
    # whole response when the channel marker is absent
    assert get_gemma_math_reward("The answer is \\boxed{42}", "42") == 1


def test_deepscaler_reward_unchanged_by_refactor():
    # regression: the vendored grader is upstream's again (orbit's gemma reward
    # reuses it from orbit/rewards/gemma_math.py instead of splitting it)
    assert get_deepscaler_rule_based_reward("reasoning </think> \\boxed{42}", "42") == 1
    assert get_deepscaler_rule_based_reward("reasoning </think> \\boxed{7}", "42") == 0
    assert get_deepscaler_rule_based_reward("no marker \\boxed{42}", "42") == 0
