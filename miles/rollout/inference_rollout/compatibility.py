import inspect
from collections.abc import Callable

from miles.rollout.base_types import (
    GenerateFnInput,
    GenerateFnOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainOutput,
)
from miles.utils.async_utils import run
from miles.utils.misc import load_function


class LegacyRolloutFnAdapter:
    def __init__(self, input: RolloutFnConstructorInput, fn: Callable):
        self.args = input.args
        self.data_source = input.data_source
        self.fn = fn

    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        output = self.fn(self.args, input.rollout_id, self.data_source, evaluation=input.evaluation)

        # compatibility for legacy version
        if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
            output = RolloutFnEvalOutput(data=output) if input.evaluation else RolloutFnTrainOutput(samples=output)

        return output


def load_rollout_function(input: RolloutFnConstructorInput, path: str):
    fn = load_function(path)

    if inspect.isclass(fn):
        return fn(input)
    else:
        return LegacyRolloutFnAdapter(input, fn)


def call_rollout_function(fn, input: RolloutFnInput) -> RolloutFnOutput:
    output = fn(input)

    if inspect.iscoroutine(output):
        output = run(output)

    return output


class LegacyGenerateFnAdapter:
    def __init__(self, fn: Callable):
        self.fn = fn
        self._has_evaluation_param = "evaluation" in inspect.signature(fn).parameters

    async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
        if self._has_evaluation_param:
            output = await self.fn(input.args, input.sample, input.sampling_params, evaluation=input.evaluation)
        else:
            output = await self.fn(input.args, input.sample, input.sampling_params)

        if not isinstance(output, GenerateFnOutput):
            output = GenerateFnOutput(samples=output)

        return output


def load_generate_function(path: str):
    fn = load_function(path)
    if fn is None:
        return None

    if inspect.isclass(fn):
        return fn()
    elif _is_legacy_generate_fn(fn):
        return LegacyGenerateFnAdapter(fn)
    else:
        return fn


def _is_legacy_generate_fn(fn: Callable) -> bool:
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    return len(params) >= 3 and params[0] != "input"


def call_all_samples_process_fn(fn: Callable, args, samples, data_source, /, **kwargs) -> None:
    """Invoke the `--rollout-all-samples-process-path` hook, filtering kwargs
    to what the function accepts. Hooks that declare `**kwargs` receive
    everything; legacy `fn(args, samples, data_source)` impls without the
    new kwargs (is_eval / rollout_id / eval_dataset_name / n_samples_per_group)
    still work."""
    sig = inspect.signature(fn)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    accepted = kwargs if has_var_keyword else {k: v for k, v in kwargs.items() if k in sig.parameters}
    fn(args, samples, data_source, **accepted)
