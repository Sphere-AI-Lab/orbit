import json
import logging
import os

import pytest
import torch
from examples.model_response_trace_viewer import response_log as model_response_log
from examples.model_response_trace_viewer.response_log import (
    model_response_row,
    save_model_response_log,
)
from tests.fast.examples.model_response_trace_viewer.conftest import make_sample
from tests.fast.ray.rollout.conftest import make_args

_STRUCTURED_MEDIA_ALIAS_CASES = (
    pytest.param(
        {
            "type": "input_audio",
            "input_audio": {"data": "typed-input-audio-sentinel", "format": "wav"},
        },
        {"type": "input_audio", "omitted": True},
        "typed-input-audio-sentinel",
        id="typed-input-audio",
    ),
    pytest.param(
        {"input_audio": {"data": "key-input-audio-sentinel", "format": "wav"}},
        {"input_audio": {"omitted": True}},
        "key-input-audio-sentinel",
        id="key-input-audio",
    ),
    pytest.param(
        {
            "type": "input_image",
            "input_image": {"data": "typed-input-image-sentinel"},
        },
        {"type": "input_image", "omitted": True},
        "typed-input-image-sentinel",
        id="typed-input-image",
    ),
    pytest.param(
        {"input_image": {"data": "key-input-image-sentinel"}},
        {"input_image": {"omitted": True}},
        "key-input-image-sentinel",
        id="key-input-image",
    ),
    pytest.param(
        {
            "type": "input_video",
            "input_video": {"data": "typed-input-video-sentinel"},
        },
        {"type": "input_video", "omitted": True},
        "typed-input-video-sentinel",
        id="typed-input-video",
    ),
    pytest.param(
        {"input_video": {"data": "key-input-video-sentinel"}},
        {"input_video": {"omitted": True}},
        "key-input-video-sentinel",
        id="key-input-video",
    ),
    pytest.param(
        {"type": "audio_url", "audio_url": "typed-audio-url-sentinel"},
        {"type": "audio_url", "omitted": True},
        "typed-audio-url-sentinel",
        id="typed-audio-url",
    ),
    pytest.param(
        {"audio_url": "key-audio-url-sentinel"},
        {"audio_url": {"omitted": True}},
        "key-audio-url-sentinel",
        id="key-audio-url",
    ),
    pytest.param(
        {"type": "video_url", "video_url": "typed-video-url-sentinel"},
        {"type": "video_url", "omitted": True},
        "typed-video-url-sentinel",
        id="typed-video-url",
    ),
    pytest.param(
        {"video_url": "key-video-url-sentinel"},
        {"video_url": {"omitted": True}},
        "key-video-url-sentinel",
        id="key-video-url",
    ),
)


def test_model_response_row_has_stable_versioned_schema_and_redacts_media():
    class ExplosiveKey:
        def __str__(self):
            raise AssertionError("non-JSON metadata keys must not be stringified")

        def __repr__(self):
            raise AssertionError("non-JSON metadata keys must not be represented")

    sample = make_sample(
        group_index=48,
        index=384,
        response_length=731,
        label="42",
        reward={"score": 1.0},
        weight_versions=[18, 19],
        prompt=[
            {"role": "user", "content": "geometry prompt"},
            {"type": "image", "image": "secret-image-payload"},
        ],
        metadata={
            "dataset": "geo3k",
            "messages": [{"role": "assistant", "content": "ignored fallback"}],
            "tensor": torch.tensor([1, 2]),
            "accumulated_token_ids": [101, 102, 103],
            "opd_student_top_logprobs": [[[-0.1, 101]]],
            ExplosiveKey(): "must be omitted",
        },
        response_turns=[
            {
                "turn": 1,
                "role": "assistant",
                "content": "first answer",
                "finish_reason": "stop",
                "weight_version": 18,
            },
            {
                "turn": 1,
                "role": "environment",
                "model_input_role": "user",
                "content": [
                    {"type": "image", "image": "secret-observation-payload"},
                    {"type": "text", "text": "tool feedback"},
                ],
            },
        ],
    )

    row = model_response_row(sample, rollout_id=12)

    assert row == {
        "schema_version": 1,
        "rollout_id": 12,
        "sample_index": 384,
        "group_index": 48,
        "status": "completed",
        "reward": {"score": 1.0},
        "label": "42",
        "response_length": 731,
        "weight_versions": [18, 19],
        "prompt": [
            {"role": "user", "content": "geometry prompt"},
            {"type": "image", "omitted": True},
        ],
        "metadata": {"dataset": "geo3k"},
        "turns": [
            {
                "turn": 1,
                "role": "assistant",
                "content": "first answer",
                "finish_reason": "stop",
                "weight_version": 18,
            },
            {
                "turn": 1,
                "role": "environment",
                "model_input_role": "user",
                "content": [
                    {"type": "image", "omitted": True},
                    {"type": "text", "text": "tool feedback"},
                ],
            },
        ],
    }
    encoded = json.dumps(row, allow_nan=False)
    assert "secret-image-payload" not in encoded
    assert "secret-observation-payload" not in encoded
    assert "tensor" not in encoded
    assert "accumulated_token_ids" not in encoded
    assert "opd_student_top_logprobs" not in encoded
    assert "messages" not in row["metadata"]


@pytest.mark.parametrize(("raw_media", "expected_marker", "sentinel"), _STRUCTURED_MEDIA_ALIAS_CASES)
def test_model_response_row_redacts_structured_media_aliases_in_prompt(raw_media, expected_marker, sentinel):
    sample = make_sample(prompt=[raw_media])

    row = model_response_row(sample, rollout_id=14)

    assert row["prompt"] == [expected_marker]
    assert sentinel not in json.dumps(row, allow_nan=False)


@pytest.mark.parametrize(("raw_media", "expected_marker", "sentinel"), _STRUCTURED_MEDIA_ALIAS_CASES)
def test_model_response_row_redacts_structured_media_aliases_in_message_fallback(raw_media, expected_marker, sentinel):
    sample = make_sample(
        metadata={
            "messages": [
                {"role": "assistant", "content": [raw_media]},
            ]
        }
    )

    row = model_response_row(sample, rollout_id=15)

    assert row["turns"] == [{"turn": 1, "role": "assistant", "content": [expected_marker]}]
    assert sentinel not in json.dumps(row, allow_nan=False)


@pytest.mark.parametrize(("raw_media", "expected_marker", "sentinel"), _STRUCTURED_MEDIA_ALIAS_CASES)
def test_model_response_row_redacts_structured_media_aliases_in_response_turns(raw_media, expected_marker, sentinel):
    sample = make_sample(
        response_turns=[
            {"turn": 1, "role": "assistant", "content": [raw_media]},
        ]
    )

    row = model_response_row(sample, rollout_id=16)

    assert row["turns"] == [{"turn": 1, "role": "assistant", "content": [expected_marker]}]
    assert sentinel not in json.dumps(row, allow_nan=False)


def test_model_response_row_omits_exact_internal_payload_keys_recursively():
    payload_keys = ("mask", "masks", "routed_experts", "indexer_topk")

    def internal_payloads(scope):
        return {key: f"{scope}-{key}-sentinel" for key in payload_keys}

    sample = make_sample(
        metadata={
            "safe_metadata": {
                "keep": "metadata-safe",
                "nested": internal_payloads("metadata"),
            }
        },
        prompt=[
            {
                "role": "user",
                "content": {
                    "keep": "prompt-safe",
                    "nested": [internal_payloads("prompt")],
                },
            }
        ],
        response_turns=[
            {
                "turn": 1,
                "role": "assistant",
                "content": {
                    "keep": "turn-safe",
                    "nested": internal_payloads("turn"),
                },
            }
        ],
    )

    row = model_response_row(sample, rollout_id=13)

    assert row["metadata"] == {"safe_metadata": {"keep": "metadata-safe", "nested": {}}}
    assert row["prompt"] == [
        {
            "role": "user",
            "content": {"keep": "prompt-safe", "nested": [{}]},
        }
    ]
    assert row["turns"] == [
        {
            "turn": 1,
            "role": "assistant",
            "content": {"keep": "turn-safe", "nested": {}},
        }
    ]
    encoded = json.dumps(row, allow_nan=False)
    for key in payload_keys:
        assert f'"{key}"' not in encoded
        for scope in ("metadata", "prompt", "turn"):
            assert f"{scope}-{key}-sentinel" not in encoded


def test_model_response_row_normalizes_metadata_messages_when_turn_buffer_is_absent():
    sample = make_sample(
        response="combined response",
        metadata={
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "call tool"},
                {
                    "role": "tool",
                    "name": "calc_score",
                    "tool_call_id": "call-1",
                    "content": "score: 0",
                },
                {"role": "assistant", "content": "final answer"},
            ]
        },
    )

    row = model_response_row(sample, rollout_id=3)

    assert row["turns"] == [
        {"turn": 1, "role": "assistant", "content": "call tool"},
        {
            "turn": 1,
            "role": "environment",
            "model_input_role": "tool",
            "content": "score: 0",
            "name": "calc_score",
            "tool_call_id": "call-1",
        },
        {"turn": 2, "role": "assistant", "content": "final answer"},
    ]


def test_model_response_row_falls_back_to_one_assistant_response():
    sample = make_sample(response="single answer")

    row = model_response_row(sample, rollout_id=9)

    assert row["turns"] == [{"turn": 1, "role": "assistant", "content": "single answer"}]


def test_save_model_response_log_is_noop_when_disabled(monkeypatch):
    args = make_args(save_model_response_log=None)

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("disabled response logging must perform no file I/O")

    monkeypatch.setattr(
        model_response_log,
        "_atomic_write_jsonl",
        unexpected_write,
    )

    save_model_response_log(args, [make_sample()], rollout_id=1)


def test_save_model_response_log_creates_parent_and_preserves_sample_order(tmp_path):
    template = str(tmp_path / "nested" / "{rollout_id}.jsonl")
    args = make_args(save_model_response_log=template)
    samples = [make_sample(index=7), make_sample(index=8)]

    save_model_response_log(args, samples, rollout_id=4)

    lines = (tmp_path / "nested" / "4.jsonl").read_text().splitlines()
    assert [json.loads(line)["sample_index"] for line in lines] == [7, 8]


def test_save_model_response_log_atomically_replaces_existing_rollout(tmp_path):
    path = tmp_path / "3.jsonl"
    args = make_args(save_model_response_log=str(tmp_path / "{rollout_id}.jsonl"))
    save_model_response_log(args, [make_sample(index=1)], rollout_id=3)

    save_model_response_log(args, [make_sample(index=9)], rollout_id=3)

    lines = path.read_text().splitlines()
    assert [json.loads(line)["sample_index"] for line in lines] == [9]
    assert list(tmp_path.glob(".3.jsonl.*.tmp")) == []


def test_atomic_writer_fsyncs_before_replace(tmp_path, monkeypatch):
    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def recording_replace(source, destination):
        events.append("replace")
        assert events[-2:] == ["fsync", "replace"]
        return real_replace(source, destination)

    monkeypatch.setattr(model_response_log.os, "fsync", recording_fsync)
    monkeypatch.setattr(model_response_log.os, "replace", recording_replace)
    path = tmp_path / "6.jsonl"

    model_response_log._atomic_write_jsonl(path, [{"sample_index": 1}])

    assert events == ["fsync", "replace"]
    assert json.loads(path.read_text()) == {"sample_index": 1}


def test_save_model_response_log_keeps_old_file_and_continues_on_replace_failure(tmp_path, monkeypatch, caplog):
    path = tmp_path / "5.jsonl"
    path.write_text('{"old":true}\n')
    args = make_args(save_model_response_log=str(tmp_path / "{rollout_id}.jsonl"))

    def fail_replace(_source, _destination):
        raise OSError("disk unavailable")

    monkeypatch.setattr(model_response_log.os, "replace", fail_replace)
    with caplog.at_level(logging.ERROR, logger=model_response_log.__name__):
        save_model_response_log(args, [make_sample(index=5)], rollout_id=5)

    assert path.read_text() == '{"old":true}\n'
    assert list(tmp_path.glob(".5.jsonl.*.tmp")) == []
    assert "rollout 5" in caplog.text
    assert str(path) in caplog.text
    assert "training will continue" in caplog.text
    assert any(record.exc_info is not None for record in caplog.records)


def test_save_model_response_log_cleans_up_and_continues_on_serialization_failure(tmp_path, monkeypatch, caplog):
    path = tmp_path / "10.jsonl"
    path.write_text('{"old":true}\n')
    args = make_args(save_model_response_log=str(tmp_path / "{rollout_id}.jsonl"))

    def fail_dump(*_args, **_kwargs):
        raise TypeError("cannot serialize row")

    monkeypatch.setattr(model_response_log.json, "dump", fail_dump)
    with caplog.at_level(logging.ERROR, logger=model_response_log.__name__):
        save_model_response_log(args, [make_sample(index=10)], rollout_id=10)

    assert path.read_text() == '{"old":true}\n'
    assert list(tmp_path.glob(".10.jsonl.*.tmp")) == []
    assert "rollout 10" in caplog.text
    assert str(path) in caplog.text
    assert "cannot serialize row" in caplog.text
    assert any(record.exc_info is not None for record in caplog.records)
