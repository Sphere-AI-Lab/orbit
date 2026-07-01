"""SGLang external-teacher scoring for On-Policy Distillation (OPD).

A separate SGLang server hosts the teacher. We POST the student's rollout
token sequence for prefill-only *scoring* (``max_new_tokens=0,
return_logprob=True, temperature=0`` -- no generation) to ``args.opd_teacher_url``
and extract the teacher's per-response-token log-probs from the response,
storing them on ``sample.teacher_log_probs``.

Wired via orbit's existing custom-reward hooks::

    --custom-rm-path orbit.rollout.opd_sglang.reward_func
    --custom-reward-post-process-path orbit.rollout.opd_sglang.post_process

Design note -- this differs from slime's ``slime/rollout/on_policy_distillation.py``:
slime's ``reward_func`` stores the raw sglang response dict directly on
``sample.reward``, and ``post_process_rewards`` reads it back via
``sample.get_reward_value(args)``. That does not carry over unmodified to
orbit: orbit computes zero-std-reward metrics from ``sample.reward``
(``orbit/ray/rollout.py::_compute_zero_std_metrics``, called from
``_log_rollout_data``) *before* ``_convert_samples_to_train_data``/
``post_process`` ever runs, and those metrics call
``round(sample.get_reward_value(args), 1)`` -- which raises on a dict. Orbit's
own ``--custom-rm-path`` docs also state the contract explicitly: "The
function should have the signature `def custom_rm(args, sample) -> float`"
(``orbit/utils/arguments.py``). So ``reward_func`` here keeps ``sample.reward``
numeric (``0.0`` -- pure distillation has no task reward) and stashes the raw
teacher response in ``sample.metadata`` instead, for ``post_process`` to read
back and discard.
"""

import aiohttp

from orbit.utils.types import Sample

TEACHER_RESPONSE_METADATA_KEY = "opd_teacher_response"


def _extract_teacher_log_probs(response: dict, response_length: int) -> list[float]:
    """Pure extraction/trim logic (no I/O) -- the unit-testable core.

    ``response`` is the JSON body of an sglang prefill-only scoring call
    (``max_new_tokens=0, return_logprob=True``): ``meta_info.input_token_logprobs``
    is a list of ``[logprob, token_id, ...]`` entries, one per input token
    (prompt followed by response). Trim to the last ``response_length``
    entries -- the response span -- and return their logprobs.
    """
    input_token_logprobs = response["meta_info"]["input_token_logprobs"]
    log_probs = [item[0] for item in input_token_logprobs]
    return log_probs[-response_length:]


async def _score_with_teacher(args, sample: Sample) -> dict:
    """POST the sample's full token sequence to the SGLang teacher server for
    prefill-only scoring. Kept separate from ``reward_func`` so tests can
    monkeypatch it and never hit the network.
    """
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(args.opd_teacher_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """``--custom-rm-path`` hook.

    Scores ``sample`` against the external SGLang teacher and stashes the raw
    response on ``sample.metadata`` for ``post_process`` to consume. Always
    returns ``0.0``: pure on-policy distillation has no task reward, and the
    learning signal comes entirely from the OPD (MOPD/blend) advantage term.
    """
    sample.metadata[TEACHER_RESPONSE_METADATA_KEY] = await _score_with_teacher(args, sample)
    return 0.0


def post_process(args, samples: list[Sample], **kwargs):
    """``--custom-reward-post-process-path`` hook.

    Extracts the teacher response ``reward_func`` stashed in each sample's
    metadata, trims it to the response span, and sets
    ``sample.teacher_log_probs``. Returns ``(raw_rewards, rewards)`` -- both
    all-zero, matching ``reward_func``'s task-reward-free contract -- in the
    ``(raw_rewards, rewards)`` shape expected by
    ``RolloutManager._convert_samples_to_train_data``.
    """
    for sample in samples:
        response = sample.metadata.pop(TEACHER_RESPONSE_METADATA_KEY)
        sample.teacher_log_probs = _extract_teacher_log_probs(response, sample.response_length)

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
