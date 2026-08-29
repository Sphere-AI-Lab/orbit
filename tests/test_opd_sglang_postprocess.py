import argparse

from orbit.peft.opd import opd_sglang
from orbit.utils.types import Sample


def _fake_response(logprobs, token_ids):
    return {"meta_info": {"input_token_logprobs": [[lp, tok] for lp, tok in zip(logprobs, token_ids, strict=True)]}}


def _make_sample(response_length, **overrides):
    defaults = dict(
        index=0,
        prompt="p",
        tokens=list(range(response_length + 3)),
        response="r",
        response_length=response_length,
        metadata={},
    )
    defaults.update(overrides)
    return Sample(**defaults)


# --- Task 2.1 Step 1: the pure extraction/trim core ---


def test_extract_teacher_log_probs_trims_to_response_span():
    response = _fake_response([None, 0.1, 0.2, 0.3, 0.4], [10, 11, 12, 13, 14])
    log_probs = opd_sglang._extract_teacher_log_probs(response, response_length=3)
    assert log_probs == [0.2, 0.3, 0.4]
    assert len(log_probs) == 3


def test_extract_teacher_log_probs_response_length_one():
    response = _fake_response([None, 0.1, 0.2], [10, 11, 12])
    log_probs = opd_sglang._extract_teacher_log_probs(response, response_length=1)
    assert log_probs == [0.2]


# --- post_process: reads the response stashed by reward_func, trims, stores ---


def test_post_process_sets_teacher_log_probs_from_stashed_response():
    response_length = 2
    response = _fake_response([None, 0.1, 0.2, 0.3, 0.4], [10, 11, 12, 13, 14])
    sample = _make_sample(response_length, metadata={opd_sglang.TEACHER_RESPONSE_METADATA_KEY: response})

    args = argparse.Namespace()
    raw_rewards, rewards = opd_sglang.post_process(args, [sample])

    assert sample.teacher_log_probs == [0.3, 0.4]
    assert len(sample.teacher_log_probs) == response_length
    assert raw_rewards == [0.0]
    assert rewards == [0.0]
    # consumed, not left dangling on metadata
    assert opd_sglang.TEACHER_RESPONSE_METADATA_KEY not in sample.metadata


def test_post_process_handles_multiple_samples_independently():
    r1 = _fake_response([None, 0.1, 0.2], [1, 2, 3])
    r2 = _fake_response([None, -0.5, -0.1, -0.2], [4, 5, 6, 7])
    s1 = _make_sample(1, metadata={opd_sglang.TEACHER_RESPONSE_METADATA_KEY: r1})
    s2 = _make_sample(2, metadata={opd_sglang.TEACHER_RESPONSE_METADATA_KEY: r2})

    args = argparse.Namespace()
    raw_rewards, rewards = opd_sglang.post_process(args, [s1, s2])

    assert s1.teacher_log_probs == [0.2]
    assert s2.teacher_log_probs == [-0.1, -0.2]
    assert raw_rewards == [0.0, 0.0]
    assert rewards == [0.0, 0.0]


# --- reward_func: network call kept behind _score_with_teacher, mocked here ---


async def test_reward_func_returns_zero_and_stashes_teacher_response(monkeypatch):
    fake_response = _fake_response([None, 0.1], [1, 2])

    async def fake_score(args, sample, targets=None):
        return fake_response

    monkeypatch.setattr(opd_sglang, "_score_with_teacher", fake_score)

    sample = _make_sample(1)
    args = argparse.Namespace(opd_teacher_url="http://fake-teacher/generate")

    reward = await opd_sglang.reward_func(args, sample)

    assert reward == 0.0
    assert sample.metadata[opd_sglang.TEACHER_RESPONSE_METADATA_KEY] is fake_response


def test_post_process_tolerates_unscored_sample():
    # A sample can reach post_process without the stashed teacher response
    # (aborted-then-recovered partial rollout, reward produced by another
    # path). One such sample must not KeyError the whole batch conversion;
    # it keeps teacher_log_probs=None while scored samples are extracted.
    args = argparse.Namespace()
    scored = _make_sample(2)
    scored.metadata[opd_sglang.TEACHER_RESPONSE_METADATA_KEY] = _fake_response(
        [None, 0.1, 0.2], [10, 11, 12]
    )
    unscored = _make_sample(2)

    raw_rewards, rewards = opd_sglang.post_process(args, [scored, unscored])

    assert scored.teacher_log_probs == [0.1, 0.2]
    assert unscored.teacher_log_probs is None
    assert raw_rewards == [0.0, 0.0] and rewards == [0.0, 0.0]
