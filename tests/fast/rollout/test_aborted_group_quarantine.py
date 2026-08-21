from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.utils.types import Sample


class _Progress:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _collector_args(*, partial_rollout: bool) -> Namespace:
    return Namespace(
        rollout_global_dataset=True,
        dynamic_sampling_filter_path=None,
        rollout_batch_size=1,
        n_samples_per_prompt=2,
        over_sampling_batch_size=1,
        partial_rollout=partial_rollout,
        rollout_submission_granularity=None,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )


def test_group_helpers_find_nested_abort_and_reject_nested_partial_retry():
    from miles.rollout.group_utils import group_has_aborted_sample, prepare_partial_retry_group

    completed = Sample(response="first", status=Sample.Status.COMPLETED)
    aborted = Sample(response="second", status=Sample.Status.ABORTED)

    assert group_has_aborted_sample([[completed, aborted]])
    with pytest.raises(ValueError, match="flat prompt group"):
        prepare_partial_retry_group([[completed, aborted]], rollout_id=3)

    assert prepare_partial_retry_group([completed, aborted], rollout_id=3) == [completed, aborted]
    assert completed.metadata["start_rollout_id"] == 3
    assert aborted.metadata["start_rollout_id"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("refactored", [False, True])
async def test_group_rm_skips_group_with_nested_aborted_leaf(monkeypatch, refactored):
    generated = {
        0: [Sample(status=Sample.Status.COMPLETED), Sample(status=Sample.Status.ABORTED)],
        1: Sample(status=Sample.Status.COMPLETED),
    }
    reward_calls = []
    group = [Sample(index=0), Sample(index=1)]
    args = Namespace(group_rm=True, sglang_enable_deterministic_inference=False)

    if refactored:
        from miles.rollout.inference_rollout import inference_rollout_common as rollout

        state = SimpleNamespace(args=args, aborted=False)

        async def generate(_state, sample, _sampling_params, evaluation=False):
            assert evaluation is False
            return generated[sample.index]

        async def group_rm(_args, samples, inplace_set_reward_field=False):
            reward_calls.append((samples, inplace_set_reward_field))

        monkeypatch.setattr(rollout, "policy_uses_routing_key", lambda _args: False)
        monkeypatch.setattr(rollout, "generate_and_rm", generate)
        monkeypatch.setattr(rollout, "batched_async_rm", group_rm)
        result = await rollout.generate_and_rm_group(state, group, {})
    else:
        from miles.rollout import sglang_rollout as rollout

        async def generate(_args, sample, _sampling_params, evaluation=False):
            assert evaluation is False
            return generated[sample.index]

        async def group_rm(_args, samples):
            reward_calls.append(samples)
            return [1.0] * len(samples)

        monkeypatch.setattr(rollout, "GenerateState", lambda _args: SimpleNamespace(aborted=False))
        monkeypatch.setattr(rollout, "policy_uses_routing_key", lambda _args: False)
        monkeypatch.setattr(rollout, "generate_and_rm", generate)
        monkeypatch.setattr(rollout, "batched_async_rm", group_rm)
        result = await rollout.generate_and_rm_group(args, group, {})

    assert result == [generated[0], generated[1]]
    assert reward_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_rollout", [False, True])
async def test_legacy_collector_quarantines_before_filter(monkeypatch, partial_rollout):
    from miles.rollout import sglang_rollout

    aborted_group = [
        Sample(index=1, response="partial", status=Sample.Status.ABORTED),
        Sample(index=2, response="peer", status=Sample.Status.COMPLETED),
    ]
    accepted_group = [
        Sample(index=3, prompt="p", response="a", reward=1.0, status=Sample.Status.COMPLETED),
        Sample(index=4, prompt="p", response="b", reward=0.0, status=Sample.Status.COMPLETED),
    ]
    groups = [aborted_group, accepted_group]
    filtered = []

    class _State:
        aborted = False
        remaining_batch_size = 0
        pendings = set()
        sampling_params = {}

        def submit_generate_tasks(self, submitted):
            async def complete(result):
                return result

            self.pendings.add(asyncio.create_task(complete(groups.pop(0))))
            self.remaining_batch_size += len(submitted)

        def reset(self):
            self.aborted = False
            self.remaining_batch_size = 0
            self.pendings = set()

    state = _State()

    async def no_op(*args, **kwargs):
        return None

    async def finish_abort(*args, **kwargs):
        return []

    def record_filter(_fn, _args, group):
        filtered.append(group)
        return SimpleNamespace(keep=True, reason=None)

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout.dumper_utils, "configure_sglang", no_op)
    monkeypatch.setattr(sglang_rollout, "abort", finish_abort)
    monkeypatch.setattr(sglang_rollout, "call_dynamic_filter", record_filter)
    monkeypatch.setattr(sglang_rollout, "recompute_samples_rollout_logprobs_via_prefill", no_op)
    monkeypatch.setattr(sglang_rollout, "maybe_log_all_samples_live_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(sglang_rollout, "tqdm", _Progress)

    output, retry_groups = await sglang_rollout.generate_rollout_async(
        _collector_args(partial_rollout=partial_rollout), 7, lambda _count: [[Sample(), Sample()]]
    )

    assert output.samples == [accepted_group]
    assert filtered == [accepted_group]
    assert retry_groups == ([aborted_group] if partial_rollout else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_rollout", [False, True])
async def test_refactored_collector_quarantines_before_filter(monkeypatch, partial_rollout):
    from miles.rollout.inference_rollout import inference_rollout_train

    aborted_group = [
        Sample(index=1, response="partial", status=Sample.Status.ABORTED),
        Sample(index=2, response="peer", status=Sample.Status.COMPLETED),
    ]
    accepted_group = [
        Sample(index=3, prompt="p", response="a", reward=1.0, status=Sample.Status.COMPLETED),
        Sample(index=4, prompt="p", response="b", reward=0.0, status=Sample.Status.COMPLETED),
    ]
    groups = [aborted_group, accepted_group]
    filtered = []
    args = _collector_args(partial_rollout=partial_rollout)
    state = SimpleNamespace(args=args, aborted=False, sampling_params={}, reset=lambda: None)

    def submit_tasks(_state, _samples, _sample_done_callback=None):
        async def complete(result):
            return result

        return [asyncio.create_task(complete(groups.pop(0)))]

    async def no_op(*args, **kwargs):
        return None

    async def finish_abort(*args, **kwargs):
        return []

    def record_filter(_fn, _args, group):
        filtered.append(group)
        return SimpleNamespace(keep=True, reason=None)

    monkeypatch.setattr(inference_rollout_train.dumper_utils, "configure_sglang", no_op)
    monkeypatch.setattr(inference_rollout_train, "submit_generate_tasks", submit_tasks)
    monkeypatch.setattr(inference_rollout_train, "abort", finish_abort)
    monkeypatch.setattr(inference_rollout_train, "call_dynamic_filter", record_filter)
    monkeypatch.setattr(inference_rollout_train, "recompute_samples_rollout_logprobs_via_prefill", no_op)
    monkeypatch.setattr(inference_rollout_train, "load_function", lambda _path: None)
    monkeypatch.setattr(inference_rollout_train, "initial_live_log_at", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference_rollout_train, "maybe_log_all_samples_live_diagnostics", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(inference_rollout_train, "tqdm", _Progress)

    output, retry_groups = await inference_rollout_train.generate_rollout_async(
        state, 8, lambda _count: [[Sample(), Sample()]]
    )

    assert output.samples == [accepted_group]
    assert filtered == [accepted_group]
    assert retry_groups == ([aborted_group] if partial_rollout else [])
