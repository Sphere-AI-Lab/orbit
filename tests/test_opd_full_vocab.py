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
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _sample(num_tokens: int = 7, response_length: int = 3) -> Sample:
    return Sample(tokens=list(range(num_tokens)), response_length=response_length)


def test_reward_func_stashes_full_vocab_response(monkeypatch):
    seen = {}

    async def fake_post_json(url, payload, timeout_secs=None):
        seen["url"], seen["payload"] = url, payload
        return {"meta_info": {"hidden_states": ["canned"]}}

    monkeypatch.setattr(opd_sglang, "_post_json", fake_post_json)
    args = _full_vocab_args()
    sample = _sample()
    assert asyncio.run(reward_func(args, sample)) == 0.0
    assert seen["url"] == args.opd_teacher_url
    assert seen["payload"]["return_hidden_states"] is True
    assert sample.metadata[TEACHER_RESPONSE_METADATA_KEY] == {"meta_info": {"hidden_states": ["canned"]}}


def test_reward_func_rejects_full_vocab_ensembles():
    args = _full_vocab_args(opd_teacher_urls=["default=http://a/generate,http://b/generate"])
    with pytest.raises(ValueError, match="single teacher"):
        asyncio.run(reward_func(args, _sample()))


def test_post_process_sets_hidden_states_and_trims():
    args = _full_vocab_args()
    payload, _ = _hidden_payload(7)
    scored = _sample()
    scored.metadata[TEACHER_RESPONSE_METADATA_KEY] = payload
    empty = _sample(num_tokens=4, response_length=0)
    empty.metadata[TEACHER_RESPONSE_METADATA_KEY] = {"empty_response": True}

    raw, rewards = post_process(args, [scored, empty])
    assert raw == rewards == [0.0, 0.0]
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
    ],
)
def test_validation_rejects_bad_full_vocab_configs(overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate_opd_args(_validate_args(**overrides))
