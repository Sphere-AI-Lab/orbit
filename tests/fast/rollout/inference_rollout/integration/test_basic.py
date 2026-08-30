import pytest
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import (
    MODULAR_ROLLOUT_BASE_ARGV,
    NO_ROLLOUT_LOG_PROBS,
    ROLLOUT_LOG_PROBS,
    expected_sample,
    load_and_call_train,
)

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function

_VARIANTS = [
    pytest.param(
        RolloutEnvConfig(
            extra_argv=[
                "--rollout-function-path",
                "miles.rollout.sglang_rollout.generate_rollout",
                "--eval-function-path",
                "miles.rollout.sglang_rollout.generate_rollout",
                "--custom-generate-function-path",
                "miles.rollout.sglang_rollout.generate",
            ]
        ),
        id="old_rollout_old_generate",
    ),
    pytest.param(
        RolloutEnvConfig(
            extra_argv=[
                "--rollout-function-path",
                "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn",
                "--custom-generate-function-path",
                "miles.rollout.sglang_rollout.generate",
            ]
        ),
        id="new_rollout_old_generate",
    ),
    pytest.param(
        RolloutEnvConfig(extra_argv=MODULAR_ROLLOUT_BASE_ARGV + extra_argv_for_variant("single_turn")),
        id="new_rollout_new_generate",
    ),
]


@pytest.mark.parametrize("rollout_env", _VARIANTS, indirect=True)
def test_train(rollout_env):
    env = rollout_env
    out = load_and_call_train(env.args, env.data_source)

    assert len(out.samples) == env.args.rollout_batch_size
    group = out.samples[0]
    assert len(group) == env.args.n_samples_per_prompt
    assert group[0] == expected_sample(group_index=0)


@pytest.mark.parametrize("rollout_env", _VARIANTS, indirect=True)
def test_eval(rollout_env):
    env = rollout_env
    fn = load_rollout_function(
        RolloutFnConstructorInput(args=env.args, data_source=env.data_source), env.args.eval_function_path
    )
    out = call_rollout_function(fn, RolloutFnEvalInput(rollout_id=0))

    assert "toy" in out.data
    rewards = out.data["toy"]["rewards"]
    samples = out.data["toy"]["samples"]
    assert len(rewards) == len(samples) == env.args.n_samples_per_eval_prompt
    assert rewards[0] == 1
    # orbit: eval does not ask the engine for logprobs unless args.eval_return_rollout_logprobs
    # is set (ORBIT-SEAM: should_request_rollout_logprobs in
    # miles/rollout/generate_utils/generate_endpoint_utils.py), so eval samples carry no
    # rollout logprobs. Upstream sets return_logprob unconditionally and expects them here;
    # that behaviour is covered under the opt-in by test_eval_returns_rollout_logprobs_when_opted_in.
    assert samples[0] == expected_sample(group_index=None, rollout_log_probs=NO_ROLLOUT_LOG_PROBS)


@pytest.mark.parametrize("rollout_env", _VARIANTS, indirect=True)
def test_eval_returns_rollout_logprobs_when_opted_in(rollout_env):
    """orbit: opting in restores upstream's always-on eval logprobs."""
    env = rollout_env
    env.args.eval_return_rollout_logprobs = True

    fn = load_rollout_function(
        RolloutFnConstructorInput(args=env.args, data_source=env.data_source), env.args.eval_function_path
    )
    out = call_rollout_function(fn, RolloutFnEvalInput(rollout_id=0))

    samples = out.data["toy"]["samples"]
    assert samples[0] == expected_sample(group_index=None, rollout_log_probs=ROLLOUT_LOG_PROBS)
