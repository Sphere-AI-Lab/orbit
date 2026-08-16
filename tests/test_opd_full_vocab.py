"""Full-vocab OPD scoring transport: payload shape, hidden-state decode/slice
alignment, reward_func/post_process wiring, Sample bookkeeping, and the
--teacher-score-mode/--loss-type validation coupling."""

import argparse
import asyncio

import numpy as np
import pybase64
import pytest

import orbit.rollout.opd_sglang as opd_sglang
from orbit.rollout.opd_sglang import (
    TEACHER_RESPONSE_METADATA_KEY,
    _full_vocab_payload,
    _teacher_hidden_states_from_payload,
    post_process,
    reward_func,
    score_full_vocab_samples,
)
from orbit.utils.arguments import _validate_opd_args
from orbit.utils.types import Sample

HIDDEN = 8


def _hidden_payload(num_tokens: int, base64: bool = True) -> tuple[dict, np.ndarray]:
    """Fake sglang response whose row t is filled with the value t (sentinels)."""
    hidden = np.tile(np.arange(num_tokens, dtype=np.float32)[:, None], (1, HIDDEN))
    if base64:
        encoded = pybase64.b64encode(hidden.tobytes()).decode("ascii")
        return {"meta_info": {"hidden_states": [encoded]}}, hidden
    return {"meta_info": {"hidden_states": [hidden.tolist()]}}, hidden


@pytest.mark.parametrize("base64", [True, False])
def test_decode_slices_the_predicting_rows(base64):
    # 7 tokens = 4 prompt + 3 response; row t predicts token t+1, so the rows
    # scoring the response are exactly [3, 4, 5].
    payload, _ = _hidden_payload(7, base64=base64)
    rows = _teacher_hidden_states_from_payload(payload, num_tokens=7, response_length=3)
    assert rows.shape == (3, HIDDEN)
    assert rows[:, 0].tolist() == [3.0, 4.0, 5.0]


def test_decode_rejects_wrong_outer_batch():
    payload, _ = _hidden_payload(5)
    payload["meta_info"]["hidden_states"] = payload["meta_info"]["hidden_states"] * 2
    with pytest.raises(ValueError, match="exactly 1"):
        _teacher_hidden_states_from_payload(payload, num_tokens=5, response_length=2)


def test_decode_rejects_short_buffer():
    # 6 positions served for 7 sent, HIDDEN=5: 120 bytes % (7*4) != 0.
    hidden = np.zeros((6, 5), dtype=np.float32)
    payload = {"meta_info": {"hidden_states": [pybase64.b64encode(hidden.tobytes()).decode("ascii")]}}
    with pytest.raises(ValueError, match="not a whole number"):
        _teacher_hidden_states_from_payload(payload, num_tokens=7, response_length=3)


def test_decode_rejects_short_legacy_rows():
    payload, _ = _hidden_payload(5, base64=False)
    with pytest.raises(ValueError, match="expected"):
        _teacher_hidden_states_from_payload(payload, num_tokens=6, response_length=2)


def test_decode_requires_a_prompt_token():
    payload, _ = _hidden_payload(4)
    with pytest.raises(ValueError, match="at least one prompt token"):
        _teacher_hidden_states_from_payload(payload, num_tokens=4, response_length=4)


def test_full_vocab_payload_shape():
    payload = _full_vocab_payload([1, 2, 3])
    assert payload["input_ids"] == [1, 2, 3]
    assert payload["return_hidden_states"] is True
    assert payload["sampling_params"]["max_new_tokens"] == 0
    assert "return_logprob" not in payload


def _full_vocab_args(**overrides):
    defaults = dict(
        teacher_score_mode="full_vocab",
        opd_teacher_url="http://teacher:30001/generate",
        opd_teacher_urls=None,
        opd_teacher_key="opd_teacher",
        opd_log_prob_top_k=0,
        opd_scoring_timeout_secs=None,
        opd_defer_full_vocab_scoring=False,
        reward_key=None,
        rm_type="math",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _sample(num_tokens: int = 7, response_length: int = 3) -> Sample:
    return Sample(tokens=list(range(num_tokens)), response_length=response_length)


def test_reward_func_stashes_full_vocab_response(monkeypatch):
    seen = {}

    async def fake_post_json(
        url,
        payload,
        timeout_secs=None,
        max_response_bytes=None,
        trusted_local_response=False,
    ):
        seen.update(
            url=url,
            payload=payload,
            max_response_bytes=max_response_bytes,
            trusted_local_response=trusted_local_response,
        )
        return {"meta_info": {"hidden_states": ["canned"]}}

    monkeypatch.setattr(opd_sglang, "_post_json", fake_post_json)
    monkeypatch.setattr(opd_sglang, "_full_vocab_response_byte_limit", lambda args, n: 123456)
    args = _full_vocab_args()
    sample = _sample()
    sample.response = "The answer is \\boxed{72}."
    sample.label = "72"
    assert asyncio.run(reward_func(args, sample)) == 1
    assert seen["url"] == args.opd_teacher_url
    assert seen["payload"]["return_hidden_states"] is True
    assert seen["max_response_bytes"] == 123456
    assert seen["trusted_local_response"] is False
    assert sample.metadata[TEACHER_RESPONSE_METADATA_KEY] == {"meta_info": {"hidden_states": ["canned"]}}


def test_managed_full_vocab_scoring_dispatches_trusted_local_decoder(monkeypatch):
    seen = {}

    async def fake_post_json(
        url,
        payload,
        timeout_secs=None,
        max_response_bytes=None,
        trusted_local_response=False,
    ):
        seen["trusted_local_response"] = trusted_local_response
        return {"meta_info": {"hidden_states": ["canned"]}}

    monkeypatch.setattr(opd_sglang, "_post_json", fake_post_json)
    monkeypatch.setattr(opd_sglang, "_full_vocab_response_byte_limit", lambda args, n: 123456)
    args = _full_vocab_args(opd_serve_teacher=True)
    sample = _sample()

    asyncio.run(opd_sglang._score_full_vocab_sample(args, sample))

    assert seen["trusted_local_response"] is True
    assert sample.metadata[TEACHER_RESPONSE_METADATA_KEY] == {"meta_info": {"hidden_states": ["canned"]}}


def test_managed_full_vocab_fast_decode_keeps_hidden_state_validation(monkeypatch):
    async def fake_post_json(
        url,
        payload,
        timeout_secs=None,
        max_response_bytes=None,
        trusted_local_response=False,
    ):
        assert trusted_local_response is True
        malformed = np.zeros((1, HIDDEN), dtype=np.float32)
        encoded = pybase64.b64encode(malformed.tobytes()).decode("ascii")
        return {"meta_info": {"hidden_states": [encoded]}}

    monkeypatch.setattr(opd_sglang, "_post_json", fake_post_json)
    monkeypatch.setattr(opd_sglang, "_full_vocab_response_byte_limit", lambda args, n: 123456)
    args = _full_vocab_args(opd_serve_teacher=True)
    sample = _sample()
    sample.reward = 1.0

    asyncio.run(opd_sglang._score_full_vocab_sample(args, sample))

    with pytest.raises(ValueError, match="not a whole number"):
        post_process(args, [sample])
    # Validation failures retain the payload for debugging/retry.
    assert TEACHER_RESPONSE_METADATA_KEY in sample.metadata


def test_reward_func_rejects_full_vocab_ensembles():
    args = _full_vocab_args(opd_teacher_urls=["default=http://a/generate,http://b/generate"])
    with pytest.raises(ValueError, match="single teacher"):
        asyncio.run(reward_func(args, _sample()))


def test_post_process_sets_hidden_states_and_trims():
    args = _full_vocab_args()
    payload, _ = _hidden_payload(7)
    scored = _sample()
    scored.reward = 1.0
    scored.metadata[TEACHER_RESPONSE_METADATA_KEY] = payload
    empty = _sample(num_tokens=4, response_length=0)
    empty.reward = 0.0
    empty.metadata[TEACHER_RESPONSE_METADATA_KEY] = {"empty_response": True}

    raw, rewards = post_process(args, [scored, empty])
    assert raw == rewards == [1.0, 0.0]
    assert scored.teacher_hidden_states.shape == (3, HIDDEN)
    assert scored.teacher_hidden_states[:, 0].tolist() == [3.0, 4.0, 5.0]
    assert empty.teacher_hidden_states.shape == (0, 0)

    # Sample bookkeeping: truncation trims rows, retry reset clears the field.
    class _Tok:
        def decode(self, tokens):
            return ""

    scored.strip_last_output_tokens(1, _Tok())
    assert scored.teacher_hidden_states.shape == (2, HIDDEN)
    scored.reset_for_retry()
    assert scored.teacher_hidden_states is None


def test_reward_func_empty_response_still_computes_task_reward():
    args = _full_vocab_args()
    sample = _sample(num_tokens=4, response_length=0)
    sample.response = ""
    sample.label = "72"

    assert asyncio.run(reward_func(args, sample)) == 0
    assert sample.metadata[TEACHER_RESPONSE_METADATA_KEY] == {"empty_response": True}


def test_deferred_full_vocab_scoring_waits_for_batch_phase(monkeypatch):
    calls = []
    active = 0
    peak_active = 0

    async def fake_post_json(
        url,
        payload,
        timeout_secs=None,
        max_response_bytes=None,
        trusted_local_response=False,
    ):
        nonlocal active, peak_active
        assert trusted_local_response is False
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0)
        calls.append(payload["input_ids"])
        active -= 1
        return {"meta_info": {"hidden_states": ["canned"]}}

    monkeypatch.setattr(opd_sglang, "_post_json", fake_post_json)
    monkeypatch.setattr(opd_sglang, "_full_vocab_response_byte_limit", lambda args, n: 123456)
    args = _full_vocab_args(
        opd_defer_full_vocab_scoring=True,
        sglang_server_concurrency=1,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
    )
    samples = [_sample(), _sample(num_tokens=8, response_length=2)]
    for sample in samples:
        sample.response = "The answer is \\boxed{72}."
        sample.label = "72"

    # The custom RM now computes only the task reward during student rollout.
    assert asyncio.run(reward_func(args, samples[0])) == 1
    assert calls == []
    assert TEACHER_RESPONSE_METADATA_KEY not in samples[0].metadata

    # Teacher requests are issued only after the complete batch is available.
    aborted = _sample(num_tokens=9, response_length=2)
    aborted.status = Sample.Status.ABORTED
    asyncio.run(score_full_vocab_samples(args, [*samples, aborted]))
    assert calls == [samples[0].tokens, samples[1].tokens]
    assert peak_active == 1
    assert all(TEACHER_RESPONSE_METADATA_KEY in sample.metadata for sample in samples)
    assert TEACHER_RESPONSE_METADATA_KEY not in aborted.metadata


def _validate_args(**overrides):
    defaults = dict(
        advantage_estimator="grpo",
        use_opd=False,
        opd_type="sglang",
        opd_kl_coef=1.0,
        opd_teacher_load=None,
        opd_teacher_ckpt_step=None,
        opd_teacher_url="http://teacher:30001/generate",
        opd_icepop=False,
        use_rollout_logprobs=False,
        peft_method="lora",
        opd_teacher=None,
        opd_teacher_urls=None,
        opd_ema_decay=0.999,
        opd_self_teacher_interval=1,
        opd_promote_interval=None,
        custom_rm_path="orbit.rollout.opd_sglang.reward_func",
        custom_reward_post_process_path="orbit.rollout.opd_sglang.post_process",
        loss_type="opd_jsd_loss",
        teacher_score_mode="full_vocab",
        teacher_hf_checkpoint="/fake/teacher",
        compute_advantages_and_returns=True,
        opd_defer_full_vocab_scoring=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_validation_accepts_full_vocab_and_disables_advantages():
    args = _validate_args()
    _validate_opd_args(args)
    assert args.compute_advantages_and_returns is False


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(loss_type="policy_loss"), "must be used together"),
        (dict(teacher_score_mode="sampled_token"), "must be used together"),
        (dict(opd_teacher_url=None), "requires --opd-teacher-url"),
        (
            dict(opd_teacher_urls=["default=http://a/generate,http://b/generate"]),
            "routing/ensembles",
        ),
        (dict(opd_log_prob_top_k=1), "incompatible with --opd-log-prob-top-k"),
        (dict(teacher_hf_checkpoint=None), "requires --teacher-hf-checkpoint"),
        (dict(use_opd=True), "pure distillation loss"),
        (dict(opd_type="megatron"), "requires --opd-type sglang"),
        (
            dict(
                loss_type="policy_loss",
                teacher_score_mode="sampled_token",
                opd_defer_full_vocab_scoring=True,
            ),
            "requires --teacher-score-mode full_vocab",
        ),
    ],
)
def test_validation_rejects_bad_full_vocab_configs(overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate_opd_args(_validate_args(**overrides))


def test_reward_func_eval_bypass_uses_real_task_rm(monkeypatch):
    # Eval samples must get the real task reward, not the 0.0 transport return --
    # and must never ship hidden states (the teacher endpoint would be hit with
    # eval-length payloads for nothing).
    async def explode(*a, **k):
        raise AssertionError("teacher must not be scored for evaluation samples")

    monkeypatch.setattr(opd_sglang, "_post_json", explode)
    monkeypatch.setattr(opd_sglang, "post_json", explode)
    args = _full_vocab_args(custom_rm_path="orbit.rollout.opd_sglang.reward_func", rm_type="math")
    sample = _sample()
    sample.response = "The answer is \\boxed{72}."
    sample.label = "72"
    assert asyncio.run(reward_func(args, sample, evaluation=True)) == 1
    sample.label = "73"
    assert asyncio.run(reward_func(args, sample, evaluation=True)) == 0


def test_full_vocab_response_limit_scales_with_teacher_hidden(tmp_path):
    import json as json_mod

    ckpt = tmp_path / "teacher"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json_mod.dumps({"hidden_size": 3584}))
    args = _full_vocab_args(teacher_hf_checkpoint=str(ckpt))

    from orbit.rollout.scoring_client import SCORING_MAX_RESPONSE_BYTES

    big = opd_sglang._full_vocab_response_byte_limit(args, 1100)
    assert big > 20 * 1024 * 1024  # 1100 x 3584 x fp32 x base64 ~ 21MB
    small = opd_sglang._full_vocab_response_byte_limit(args, 10)
    assert small == SCORING_MAX_RESPONSE_BYTES  # generic cap stays the floor
