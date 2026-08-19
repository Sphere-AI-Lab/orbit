import asyncio
import copy
import itertools
import math
import queue
import threading
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from examples.geo3k_vlm.multi_turn import rollout as geo3k_rollout
from tests.ci.ci_register import register_cpu_ci

from miles.rollout import on_policy_distillation as opd
from miles.rollout.on_policy_distillation import (
    _compute_topk_reverse_kl,
    _per_position_ids,
    _score_payload,
    _teacher_url_for_sample,
    parse_teacher_urls,
)
from miles.utils.types import Sample

register_cpu_ci(est_time=60, suite="stage-a-cpu")


class _FakeResponse:
    request_info = SimpleNamespace(headers={"Content-Length": "128"})

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        return None

    async def read(self):
        return b'{"meta_info": {}}'

    async def json(self):
        return {"meta_info": {}}


class _FakeClientSession:
    def __init__(self, sessions, **kwargs):
        self.closed = False
        self.post_calls = []
        self.post_exceptions = []
        self.connection_reuse_events = []
        sessions.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.closed = True
        return False

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        if self.connection_reuse_events:
            kwargs["trace_request_ctx"].connection_reused = self.connection_reuse_events.pop(0)
        if self.post_exceptions:
            raise self.post_exceptions.pop(0)
        return _FakeResponse()


def _install_fake_http(monkeypatch):
    sessions = []
    monkeypatch.setattr(opd.aiohttp, "TCPConnector", lambda **kwargs: object())
    monkeypatch.setattr(opd.aiohttp, "ClientSession", lambda **kwargs: _FakeClientSession(sessions, **kwargs))
    return sessions


def _entry(prob: float, token_id: int):
    return [math.log(prob), token_id]


def _args(strategy: str, weight_mode: str = "student_p"):
    return Namespace(
        opd_top_k_strategy=strategy,
        opd_reward_weight_mode=weight_mode,
    )


def _sample():
    return Sample(
        tokens=[10, 11, 12],
        response_length=2,
        metadata={
            "opd_student_top_logprobs": [
                [_entry(0.6, 1), _entry(0.4, 2)],
                [_entry(0.7, 4), _entry(0.3, 5)],
            ]
        },
    )


def test_score_payload_materializes_only_the_response_window():
    input_ids = [10, 11, 12, 13, 14]

    payload = opd._score_payload(input_ids, response_length=2, top_k=4, token_ids=[21, 22])

    assert payload["input_ids"] is input_ids
    assert payload["logprob_start_len"] == 2
    assert payload["top_logprobs_num"] == 4
    assert payload["token_ids_logprob"] == [21, 22]
    assert "image_data" not in payload


def test_score_payload_uses_teacher_processed_prefix_and_exact_response_suffix_for_images():
    image_data = ["encoded-a", "encoded-b"]

    payload = opd._score_payload(
        [10, 11, 12],
        response_length=1,
        image_data=image_data,
        prompt="rendered multimodal prompt",
        use_exact_mm_scoring_suffix=True,
    )

    assert payload["image_data"] is image_data
    assert payload["text"] == "rendered multimodal prompt"
    assert payload["scoring_suffix_ids"] == [12]
    assert "input_ids" not in payload
    assert "logprob_start_len" not in payload


def test_score_payload_keeps_legacy_multimodal_path_when_exact_suffix_is_disabled():
    input_ids = [10, 11, 12]

    payload = opd._score_payload(
        input_ids,
        response_length=1,
        image_data=["encoded-image"],
        prompt="rendered multimodal prompt",
        use_exact_mm_scoring_suffix=False,
    )

    assert payload["input_ids"] is input_ids
    assert payload["logprob_start_len"] == 1
    assert "text" not in payload
    assert "scoring_suffix_ids" not in payload


def test_score_payload_requires_rendered_prompt_for_exact_multimodal_scoring():
    with pytest.raises(ValueError, match="rendered string prompt"):
        opd._score_payload(
            [10, 11, 12],
            response_length=1,
            image_data=["encoded-image"],
            prompt=None,
            use_exact_mm_scoring_suffix=True,
        )


def test_score_payload_rejects_empty_multimodal_response():
    with pytest.raises(ValueError, match="non-empty text response suffix"):
        opd._score_payload(
            [10, 11, 12],
            response_length=0,
            image_data=["encoded-image"],
            prompt="rendered multimodal prompt",
        )


def test_scoring_response_prompt_tokens_replace_request_side_token_estimate():
    response = {"meta_info": {"prompt_tokens": 137}}

    assert opd._payload_input_token_count({"scoring_suffix_ids": [41, 42]}) == 2
    assert opd._response_input_token_count(response) == 137


def test_score_payload_with_empty_response_starts_at_last_prompt_token():
    payload = opd._score_payload([10, 11, 12], response_length=0)

    assert payload["logprob_start_len"] == 2


@pytest.mark.parametrize("response_length", [-1, 4])
def test_score_payload_rejects_response_length_outside_input(response_length):
    with pytest.raises(ValueError, match="response_length must be between 0 and len"):
        opd._score_payload([10, 11, 12], response_length=response_length)


def test_score_payload_requires_at_least_one_prompt_token():
    with pytest.raises(ValueError, match="requires at least one prompt token"):
        opd._score_payload([10, 11, 12], response_length=3)


@pytest.mark.asyncio
async def test_scoring_post_retries_asyncio_timeout(monkeypatch):
    calls = 0

    async def fake_post_json(url, payload, timeout_s, *, persistent):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.TimeoutError
        return opd._PostJsonResult(
            response={
                "meta_info": {
                    "prompt_tokens": 137,
                    "input_token_logprobs": [None, [-0.5, 1]],
                }
            },
            request_body_bytes=128,
            response_body_bytes=256,
            body_read_s=0.01,
            json_decode_s=0.02,
            client_session_reused=True,
            connection_reused=True,
            transport_attempts=1,
            stale_connection_retry_count=0,
        )

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(opd, "_post_json", fake_post_json)
    monkeypatch.setattr(opd.asyncio, "sleep", no_sleep)
    args = Namespace(
        opd_scoring_timeout=1,
        opd_scoring_max_inflight=0,
        opd_scoring_retries=1,
    )

    result = await opd._scoring_post(
        args,
        "http://teacher/generate",
        {"input_ids": [1]},
        target="teacher",
        response_length=1,
    )

    assert result.response == {
        "meta_info": {
            "prompt_tokens": 137,
            "input_token_logprobs": [None, [-0.5, 1]],
        }
    }
    assert result.telemetry["attempts"] == 2
    assert result.telemetry["input_tokens"] == 137
    assert result.telemetry["response_tokens"] == 1
    assert result.telemetry["request_body_bytes"] == 128
    assert result.telemetry["response_body_bytes"] == 256
    assert result.telemetry["returned_positions"] == 2
    assert result.telemetry["persistent_session"] is True
    assert result.telemetry["client_session_reused"] is True
    assert result.telemetry["connection_reused"] is True
    assert result.telemetry["transport_attempts"] == 2
    assert result.telemetry["stale_connection_retry_count"] == 0
    assert calls == 2


@pytest.mark.asyncio
async def test_post_json_reuses_session_and_rebuilds_after_close(monkeypatch):
    sessions = _install_fake_http(monkeypatch)

    first = await opd._post_json("http://teacher/generate", {"input_ids": [1]}, 10, persistent=True)
    second = await opd._post_json("http://teacher/generate", {"input_ids": [2]}, 10, persistent=True)

    assert len(sessions) == 1
    assert first.client_session_reused is False
    assert second.client_session_reused is True
    assert first.connection_reused is False
    assert second.connection_reused is False
    assert sessions[0].closed is False

    await opd.close_scoring_transport()
    assert sessions[0].closed is True

    third = await opd._post_json("http://teacher/generate", {"input_ids": [3]}, 10, persistent=True)
    assert len(sessions) == 2
    assert third.client_session_reused is False

    await opd.close_scoring_transport()
    assert sessions[1].closed is True


@pytest.mark.asyncio
async def test_post_json_retries_stale_persistent_connection(monkeypatch):
    sessions = _install_fake_http(monkeypatch)

    first = await opd._post_json("http://teacher/generate", {"input_ids": [1]}, 10, persistent=True)
    assert first.client_session_reused is False

    # aiohttp confirms the failed request checked out a pooled connection; the
    # idempotent scoring request may then retry on a new connection.
    sessions[0].connection_reuse_events = [True, False]
    sessions[0].post_exceptions = [opd.aiohttp.ServerDisconnectedError("Server disconnected")]
    second = await opd._post_json("http://teacher/generate", {"input_ids": [2]}, 10, persistent=True)

    assert second.response == {"meta_info": {}}
    assert second.client_session_reused is True
    assert second.connection_reused is False
    assert second.transport_attempts == 2
    assert second.stale_connection_retry_count == 1
    assert len(sessions) == 1
    assert len(sessions[0].post_calls) == 3

    await opd.close_scoring_transport()


@pytest.mark.asyncio
async def test_post_json_stale_retries_share_one_timeout_budget(monkeypatch):
    sessions = _install_fake_http(monkeypatch)
    await opd._persistent_scoring_session()
    sessions[0].connection_reuse_events = [True, False]
    sessions[0].post_exceptions = [opd.aiohttp.ServerDisconnectedError("Server disconnected")]
    # opd.time IS the global time module, and the asyncio event loop reads
    # time.monotonic() for its own scheduling — an exact-length iterator gets
    # exhausted by those loop-internal calls. The scripted values cover the
    # seven in-band reads (they happen in one non-suspending await chain, in
    # order); everything after sees a frozen clock.
    clock = itertools.chain(iter([100.0, 101.0, 104.0, 105.0, 105.1, 105.2, 105.3]), itertools.repeat(105.3))
    monkeypatch.setattr(opd.time, "monotonic", lambda: next(clock))

    result = await opd._post_json("http://teacher/generate", {"input_ids": [1]}, 10, persistent=True)

    first_timeout = sessions[0].post_calls[0][1]["timeout"].total
    second_timeout = sessions[0].post_calls[1][1]["timeout"].total
    assert first_timeout == 9
    assert second_timeout == 6
    assert result.transport_attempts == 2

    await opd.close_scoring_transport()


@pytest.mark.asyncio
async def test_post_json_does_not_retry_disconnect_on_fresh_session(monkeypatch):
    sessions = _install_fake_http(monkeypatch)

    def session_factory(**kwargs):
        session = _FakeClientSession(sessions, **kwargs)
        session.post_exceptions = [opd.aiohttp.ServerDisconnectedError("Server disconnected")]
        return session

    monkeypatch.setattr(opd.aiohttp, "ClientSession", session_factory)

    with pytest.raises(opd.aiohttp.ServerDisconnectedError):
        await opd._post_json("http://teacher/generate", {"input_ids": [1]}, 10, persistent=True)

    assert len(sessions[0].post_calls) == 1

    await opd.close_scoring_transport()


@pytest.mark.asyncio
async def test_post_json_does_not_retry_fresh_connection_in_reused_session(monkeypatch):
    sessions = _install_fake_http(monkeypatch)

    await opd._post_json("http://teacher/generate", {"input_ids": [1]}, 10, persistent=True)
    sessions[0].connection_reuse_events = [False]
    sessions[0].post_exceptions = [opd.aiohttp.ServerDisconnectedError("Server disconnected")]

    with pytest.raises(opd.aiohttp.ServerDisconnectedError) as exc_info:
        await opd._post_json("http://teacher/generate", {"input_ids": [2]}, 10, persistent=True)

    assert len(sessions[0].post_calls) == 2
    assert exc_info.value.opd_transport_attempts == 1
    assert exc_info.value.opd_stale_connection_retry_count == 0

    await opd.close_scoring_transport()


@pytest.mark.asyncio
async def test_post_json_gives_up_after_exhausting_stale_connections(monkeypatch):
    sessions = _install_fake_http(monkeypatch)

    await opd._post_json("http://teacher/generate", {"input_ids": [1]}, 10, persistent=True)
    sessions[0].connection_reuse_events = [True] * opd._STALE_CONNECTION_ATTEMPTS
    sessions[0].post_exceptions = [
        opd.aiohttp.ServerDisconnectedError("Server disconnected") for _ in range(opd._STALE_CONNECTION_ATTEMPTS)
    ]

    with pytest.raises(opd.aiohttp.ServerDisconnectedError) as exc_info:
        await opd._post_json("http://teacher/generate", {"input_ids": [2]}, 10, persistent=True)

    assert len(sessions[0].post_calls) == 1 + opd._STALE_CONNECTION_ATTEMPTS
    assert exc_info.value.opd_transport_attempts == opd._STALE_CONNECTION_ATTEMPTS
    assert exc_info.value.opd_stale_connection_retry_count == opd._STALE_CONNECTION_ATTEMPTS - 1

    await opd.close_scoring_transport()


@pytest.mark.asyncio
async def test_persistent_connector_expires_before_sglang_keepalive(monkeypatch):
    captured = {}

    def fake_connector(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(opd.aiohttp, "TCPConnector", fake_connector)
    monkeypatch.setattr(opd.aiohttp, "ClientSession", lambda **kwargs: _FakeClientSession([], **kwargs))

    await opd._post_json("http://teacher/generate", {"input_ids": [1]}, 10, persistent=True)

    # sglang's uvicorn reaps idle connections at SGLANG_TIMEOUT_KEEP_ALIVE=5s;
    # the client pool must expire strictly earlier.
    assert captured["keepalive_timeout"] < 5

    await opd.close_scoring_transport()


@pytest.mark.asyncio
async def test_post_json_can_disable_persistent_session(monkeypatch):
    sessions = _install_fake_http(monkeypatch)

    first = await opd._post_json("http://teacher/generate", {"input_ids": [1]}, 10, persistent=False)
    second = await opd._post_json("http://teacher/generate", {"input_ids": [2]}, 10, persistent=False)

    assert len(sessions) == 2
    assert all(session.closed for session in sessions)
    assert first.client_session_reused is False
    assert second.client_session_reused is False


def test_persistent_session_is_isolated_and_closed_per_event_loop(monkeypatch):
    sessions = _install_fake_http(monkeypatch)

    async def post_once():
        return await opd._post_json("http://teacher/generate", {"input_ids": [1]}, 10, persistent=True)

    first = asyncio.run(post_once())
    second = asyncio.run(post_once())

    assert len(sessions) == 2
    assert all(session.closed for session in sessions)
    assert first.client_session_reused is False
    assert second.client_session_reused is False
    assert not opd._SCORING_LOOP_STATES


def test_dispose_rollout_function_closes_scoring_transport_when_opd_loaded(monkeypatch):
    """#24, re-wired after the 2026-08 sync removed the legacy fully-async
    worker: RolloutManager.dispose() must close the loop-local persistent
    scoring session on the shared rollout loop even though the class-based
    rollout fn defines no dispose of its own."""
    from miles.rollout.inference_rollout import compatibility

    closed = []

    async def fake_close():
        closed.append(asyncio.get_running_loop())

    monkeypatch.setattr(opd, "close_scoring_transport", fake_close)
    compatibility.dispose_rollout_function(SimpleNamespace())

    assert len(closed) == 1


def _versioned_sample(version):
    sample = Sample(tokens=[1, 2, 3], response_length=1)
    if version is not None:
        sample.weight_versions = [str(version)]
    return sample


def _fail_closed_buffer(max_weight_staleness=2):
    from examples.fully_async.fail_closed_data_buffer import FailClosedDataBuffer
    from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput

    args = Namespace(
        max_weight_staleness=max_weight_staleness,
        async_data_buffer_capacity_factor=1.0,
        rollout_batch_size=4,
        dynamic_sampling_filter_path=None,
    )
    return FailClosedDataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=lambda group: None))


def test_fail_closed_buffer_raises_when_any_sample_missing_weight_version():
    """Fork staleness contract (ported from the legacy collector): a group of
    unobservable staleness is an error, not a fail-open admit (upstream
    DefaultDataBuffer) nor an indefinite pend (the pre-sync collector)."""
    from miles.rollout.fully_async_data_buffer import DataBufferInput

    buffer = _fail_closed_buffer()
    group = [_versioned_sample(3), _versioned_sample(None)]

    async def scenario():
        await buffer.put(DataBufferInput(prompt_group=group, group=group))
        return await buffer.get(current_version=3)

    with pytest.raises(RuntimeError, match="no rollout weight version"):
        asyncio.run(scenario())


def test_fail_closed_buffer_requires_a_trainer_weight_version():
    buffer = _fail_closed_buffer()

    with pytest.raises(RuntimeError, match="no current weight"):
        asyncio.run(buffer.get(current_version=None))


def test_fail_closed_buffer_admits_fully_versioned_groups():
    from miles.rollout.fully_async_data_buffer import DataBufferInput

    buffer = _fail_closed_buffer()
    group = [_versioned_sample(3), _versioned_sample(4)]

    async def scenario():
        await buffer.put(DataBufferInput(prompt_group=group, group=group))
        return await buffer.get(current_version=4)

    entry = asyncio.run(scenario())
    assert entry.group is group



async def test_observed_task_reward_uses_builtin_rm_without_mutating_training_args(monkeypatch):
    training_args = Namespace(
        opd_log_task_reward=True,
        custom_rm_path="miles.rollout.on_policy_distillation.reward_func",
        rm_type="deepscaler",
    )
    sample = Sample(response="answer", label="42", metadata={"dataset": "math", "rm_type": "remote_rm"})
    call = {}

    async def fake_async_rm(args, received_sample):
        call.update(args=args, sample=received_sample)
        return 1

    monkeypatch.setattr("miles.rollout.rm_hub.async_rm", fake_async_rm)

    await opd._record_observed_task_reward(training_args, sample)

    assert call["args"] is not training_args
    assert call["args"].custom_rm_path is None
    assert call["args"].rm_type == "deepscaler"
    assert call["sample"] is not sample
    assert call["sample"].metadata == {"dataset": "math"}
    assert training_args.custom_rm_path == "miles.rollout.on_policy_distillation.reward_func"
    assert sample.metadata == {
        "dataset": "math",
        "rm_type": "remote_rm",
        opd.OPD_TASK_REWARD_METADATA_KEY: 1.0,
    }


@pytest.mark.asyncio
async def test_reward_func_records_scoring_telemetry(monkeypatch):
    response = {"meta_info": {"input_token_logprobs": [None, [-0.5, 11], [-0.25, 12]]}}
    telemetry = {
        "target": "teacher",
        "attempts": 1,
        "input_tokens": 3,
        "response_tokens": 2,
    }

    async def fake_scoring_post(*args, **kwargs):
        return opd._ScoringPostResult(response=response, telemetry=telemetry)

    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)
    sample = Sample(tokens=[10, 11, 12], response_length=2, metadata={"dataset": "math"})
    args = Namespace(opd_log_prob_top_k=0, rm_url="http://teacher/generate")

    result = await opd.reward_func(args, sample)

    assert result is response
    assert sample.metadata["dataset"] == "math"
    assert sample.metadata[opd.OPD_SCORING_TELEMETRY_KEY] == [telemetry]


@pytest.mark.asyncio
@pytest.mark.parametrize("dagger_top_k", [0, 2])
async def test_reward_func_attaches_ordered_images_to_sampled_and_dagger_teacher_requests(
    monkeypatch,
    dagger_top_k,
):
    calls = []
    encoded_images = []

    def fake_encode(image):
        encoded = f"encoded-{image}"
        encoded_images.append(encoded)
        return encoded

    async def fake_scoring_post(args, url, payload, *, target, response_length):
        calls.append((target, payload))
        return opd._ScoringPostResult(response={"meta_info": {}}, telemetry={"target": target})

    monkeypatch.setattr(opd, "encode_image_for_rollout_engine", fake_encode)
    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)
    sample = Sample(
        prompt="rendered multimodal prompt",
        tokens=[10, 11, 12],
        response_length=2,
        multimodal_inputs={"images": ["image-a", "image-b"]},
    )
    args = Namespace(
        opd_log_prob_top_k=0,
        opd_dagger_top_k=dagger_top_k,
        sglang_mm_exact_scoring_suffix=True,
        rm_url="http://teacher/generate",
    )

    await opd.reward_func(args, sample)

    assert encoded_images == ["encoded-image-a", "encoded-image-b"]
    assert len(calls) == 1
    assert calls[0][0] == "teacher"
    assert calls[0][1]["image_data"] == encoded_images
    assert calls[0][1]["text"] == sample.prompt
    assert calls[0][1]["scoring_suffix_ids"] == [11, 12]
    assert "input_ids" not in calls[0][1]
    if dagger_top_k:
        assert calls[0][1]["top_logprobs_num"] == dagger_top_k
    else:
        assert "top_logprobs_num" not in calls[0][1]


@pytest.mark.asyncio
async def test_multimodal_exact_suffix_dagger_round_trip_preserves_native_targets(monkeypatch):
    calls = []
    teacher_response = {
        "meta_info": {
            "input_token_logprobs": [None, [-0.3, 11], [-0.4, 12]],
            "input_top_logprobs": [
                None,
                [[math.log(0.6), 21], [math.log(0.3), 22]],
                [[math.log(0.5), 23], [math.log(0.2), 24]],
            ],
        }
    }

    async def fake_scoring_post(args, url, payload, *, target, response_length):
        calls.append((target, response_length, payload))
        return opd._ScoringPostResult(response=teacher_response, telemetry={"target": target})

    monkeypatch.setattr(opd, "encode_image_for_rollout_engine", lambda image: f"encoded-{image}")
    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)
    sample = Sample(
        prompt="rendered multimodal prompt",
        tokens=[7, 8, 9, 11, 12],
        response_length=2,
        multimodal_inputs={"images": ["image-a"]},
    )
    args = Namespace(
        opd_log_prob_top_k=0,
        opd_dagger_top_k=2,
        opd_log_task_reward=False,
        sglang_mm_exact_scoring_suffix=True,
        rm_url="http://teacher/generate",
        reward_key=None,
        vocab_size=64,
    )

    sample.reward = await opd.reward_func(args, sample)
    raw_rewards, rewards = opd.post_process_rewards(args, [sample])

    assert raw_rewards == rewards == [0.0]
    assert len(calls) == 1
    target, response_length, payload = calls[0]
    assert target == "teacher"
    assert response_length == 2
    assert payload["text"] == sample.prompt
    assert payload["image_data"] == ["encoded-image-a"]
    assert payload["scoring_suffix_ids"] == [11, 12]
    assert payload["top_logprobs_num"] == 2
    assert "input_ids" not in payload
    assert "token_ids_logprob" not in payload
    torch.testing.assert_close(sample.teacher_log_probs, torch.tensor([-0.3, -0.4]))
    assert sample.teacher_topk_token_ids.tolist() == [[21, 22], [23, 24]]
    torch.testing.assert_close(
        sample.teacher_topk_log_probs,
        torch.tensor(
            [[math.log(0.6), math.log(0.3)], [math.log(0.5), math.log(0.2)]],
            dtype=torch.float32,
        ),
    )
    assert sample.teacher_topk_valid_mask.tolist() == [[True, True], [True, True]]
    assert sample.metadata[opd.OPD_SCORING_TELEMETRY_KEY] == [{"target": "teacher"}]
    sample.validate()


@pytest.mark.asyncio
@pytest.mark.parametrize("dagger_top_k", [0, 2])
async def test_geo3k_multiturn_exact_suffix_keeps_actions_active_and_observations_inert(
    monkeypatch,
    dagger_top_k,
):
    calls = []
    teacher_response = {
        "meta_info": {
            "input_token_logprobs": [
                None,
                [-0.1, 11],
                [-0.2, 12],
                [-8.0, 31],
                [-8.0, 32],
                [-0.3, 13],
                [-0.4, 14],
            ],
            "input_top_logprobs": [
                None,
                [[math.log(0.6), 21], [math.log(0.3), 22]],
                [[math.log(0.5), 23], [math.log(0.2), 24]],
                [],
                [],
                [[math.log(0.7), 25], [math.log(0.2), 26]],
                [[math.log(0.4), 27], [math.log(0.3), 28]],
            ],
        }
    }

    async def fake_scoring_post(args, url, payload, *, target, response_length):
        calls.append((target, response_length, payload))
        return opd._ScoringPostResult(response=teacher_response, telemetry={"target": target})

    monkeypatch.setattr(opd, "encode_image_for_rollout_engine", lambda image: f"encoded-{image}")
    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)

    # Two sampled assistant spans surround a text-only Geo3K tool observation.
    # The complete suffix remains position-preserving, while loss_mask decides
    # which rows may contribute to RKLD or DAgger.
    sample = Sample(
        prompt="rendered Geo3K prompt",
        tokens=[7, 8, 9, 11, 12, 31, 32, 13, 14],
        response_length=6,
        loss_mask=[1, 1, 0, 0, 1, 1],
        multimodal_inputs={"images": ["geometry-image"]},
    )
    args = Namespace(
        opd_log_prob_top_k=0,
        opd_dagger_top_k=dagger_top_k,
        opd_log_task_reward=False,
        sglang_mm_exact_scoring_suffix=True,
        rm_url="http://teacher/generate",
        reward_key=None,
        vocab_size=128,
    )

    sample.reward = await opd.reward_func(args, sample)
    raw_rewards, rewards = opd.post_process_rewards(args, [sample])

    assert raw_rewards == rewards == [0.0]
    assert len(calls) == 1
    target, response_length, payload = calls[0]
    assert target == "teacher"
    assert response_length == 6
    assert payload["text"] == sample.prompt
    assert payload["image_data"] == ["encoded-geometry-image"]
    assert payload["scoring_suffix_ids"] == [11, 12, 31, 32, 13, 14]
    assert "input_ids" not in payload
    torch.testing.assert_close(sample.teacher_log_probs, torch.tensor([-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]))

    if dagger_top_k:
        assert payload["top_logprobs_num"] == 2
        assert sample.teacher_topk_token_ids.tolist() == [
            [21, 22],
            [23, 24],
            [0, 0],
            [0, 0],
            [25, 26],
            [27, 28],
        ]
        assert sample.teacher_topk_valid_mask.tolist() == [
            [True, True],
            [True, True],
            [False, False],
            [False, False],
            [True, True],
            [True, True],
        ]
        assert torch.isneginf(sample.teacher_topk_log_probs[2:4]).all()
    else:
        assert "top_logprobs_num" not in payload
        assert sample.teacher_topk_token_ids is None

    sample.validate()


@pytest.mark.parametrize(
    ("context_limit", "existing_response", "expected_budget"),
    [
        (None, [], 20),
        (10, [11, 12], 5),
    ],
)
def test_geo3k_multiturn_keeps_response_and_context_budgets_separate(
    context_limit,
    existing_response,
    expected_budget,
):
    prompt_ids = [7, 8, 9]
    tokenizer = SimpleNamespace(encode=lambda _prompt, add_special_tokens: list(prompt_ids))
    state = SimpleNamespace(tokenizer=tokenizer, processor=None)
    sample = Sample(
        prompt="rendered Geo3K prompt",
        tokens=[*prompt_ids, *existing_response],
    )
    args = Namespace(rollout_max_context_len=context_limit)

    _image_data, response_tokens, budget, _multimodal_inputs, _generation_tokens = geo3k_rollout._prepare_start_state(
        sample,
        state,
        args,
        {"max_new_tokens": 20},
    )

    assert response_tokens == existing_response
    assert budget == expected_budget


@pytest.mark.asyncio
async def test_geo3k_multiturn_rejects_observation_that_exceeds_remaining_budget(monkeypatch):
    class FakeEnv:
        closed = False

        def reset(self):
            return None

        def close(self):
            self.closed = True

    def decode(tokens, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "decoded:" + ",".join(str(token) for token in tokens)

    env = FakeEnv()
    state = SimpleNamespace(tokenizer=SimpleNamespace(decode=decode), processor=None)
    generation_inputs = []

    monkeypatch.setattr(
        geo3k_rollout,
        "_initialize_resources",
        lambda args, sample: (env, None, {"max_turns": 2}, state, "http://rollout/generate"),
    )
    monkeypatch.setattr(
        geo3k_rollout,
        "_prepare_start_state",
        lambda sample, state, args, sampling_params: (None, [], 3, [], [7, 8, 9]),
    )

    async def fake_run_inference_step(_url, tokens, *_args, **_kwargs):
        generation_inputs.append(list(tokens))
        return (
            "first action",
            [11],
            [-0.1],
            "stop",
            {"weight_version": "v1", "prompt_tokens": 3},
        )

    monkeypatch.setattr(geo3k_rollout, "_run_inference_step", fake_run_inference_step)
    monkeypatch.setattr(
        geo3k_rollout,
        "_process_env_step",
        lambda *args, **kwargs: ([31, 32, 33], [31, 32, 33], [], None, None, False),
    )

    sample = Sample(
        prompt="rendered Geo3K prompt",
        tokens=[7, 8, 9],
        loss_mask=[],
        rollout_log_probs=[],
        metadata={},
    )
    args = Namespace(partial_rollout=False)

    result = await geo3k_rollout.generate(args, sample, {"max_new_tokens": 3})

    assert result is sample
    assert sample.tokens == [7, 8, 9, 11]
    assert sample.response_length == 1
    assert sample.response_length <= 3
    assert sample.loss_mask == [1]
    assert sample.rollout_log_probs == [-0.1]
    assert sample.status is Sample.Status.TRUNCATED
    assert sample.response == "decoded:11"
    assert generation_inputs == [[7, 8, 9]]
    assert env.closed is True
    sample.validate()


@pytest.mark.asyncio
async def test_geo3k_multiturn_rollout_records_rounds_and_action_observation_masks(monkeypatch):
    class FakeEnv:
        closed = False
        reset_called = False

        def reset(self):
            self.reset_called = True

        def close(self):
            self.closed = True

    def decode(tokens, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "decoded:" + ",".join(str(token) for token in tokens)

    env = FakeEnv()
    state = SimpleNamespace(tokenizer=SimpleNamespace(decode=decode), processor=None)
    responses = iter(
        [
            (
                "first action",
                [11, 12],
                [-0.1, -0.2],
                "stop",
                {"weight_version": "v1", "prompt_tokens": 3},
            ),
            (
                "second action",
                [13, 14],
                [-0.3, -0.4],
                "stop",
                {"weight_version": "v2", "prompt_tokens": 7},
            ),
        ]
    )
    observations = iter(
        [
            ([31, 32], [31, 32], [], None, None, False),
            (None, None, None, None, None, True),
        ]
    )
    generation_inputs = []

    monkeypatch.setattr(
        geo3k_rollout,
        "_initialize_resources",
        lambda args, sample: (env, None, {"max_turns": 3}, state, "http://rollout/generate"),
    )
    monkeypatch.setattr(
        geo3k_rollout,
        "_prepare_start_state",
        lambda sample, state, args, sampling_params: (None, [], 20, [], [7, 8, 9]),
    )

    async def fake_run_inference_step(_url, tokens, *_args, **_kwargs):
        generation_inputs.append(list(tokens))
        return next(responses)

    monkeypatch.setattr(geo3k_rollout, "_run_inference_step", fake_run_inference_step)
    monkeypatch.setattr(geo3k_rollout, "_process_env_step", lambda *args, **kwargs: next(observations))

    sample = Sample(
        prompt="rendered Geo3K prompt",
        tokens=[7, 8, 9],
        loss_mask=[],
        rollout_log_probs=[],
    )
    args = Namespace(partial_rollout=False)

    result = await geo3k_rollout.generate(args, sample, {"max_new_tokens": 20})

    assert result is sample
    assert sample.tokens == [7, 8, 9, 11, 12, 31, 32, 13, 14]
    assert sample.response_length == 6
    assert sample.loss_mask == [1, 1, 0, 0, 1, 1]
    assert sample.rollout_log_probs == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]
    assert sample.metadata["round_number"] == 2
    assert sample.weight_versions == ["v1", "v2"]
    assert sample.status is Sample.Status.COMPLETED
    assert sample.response == "decoded:11,12,31,32,13,14"
    assert generation_inputs == [[7, 8, 9], [7, 8, 9, 11, 12, 31, 32]]
    assert env.closed is True
    sample.validate()


@pytest.mark.asyncio
async def test_geo3k_multiturn_generation_preserves_noncanonical_prior_action_ids(monkeypatch):
    image_token_id = 151655
    expanded_prompt_ids = [7, image_token_id, image_token_id, image_token_id, 9]
    compact_prompt_ids = [7, image_token_id, 9]
    noncanonical_action_ids = [41, 42]

    class FakeTokenizer:
        bos_token_id = None

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            if text == "rendered Geo3K prompt":
                return list(compact_prompt_ids)
            if text == "JK":
                return [34070]
            raise AssertionError(f"unexpected text passed to tokenizer: {text!r}")

        def decode(self, token_ids, *, skip_special_tokens=False):
            if list(token_ids) == noncanonical_action_ids:
                return "JK"
            assert skip_special_tokens is False
            return "decoded:" + ",".join(str(token_id) for token_id in token_ids)

    class FakeProcessor:
        def __call__(self, *, text, images):
            assert text == "rendered Geo3K prompt"
            assert images == ["image-a"]
            return {"input_ids": [list(expanded_prompt_ids)]}

    class FakeEnv:
        def reset(self):
            return None

        def close(self):
            return None

    tokenizer = FakeTokenizer()
    assert tokenizer.encode(
        tokenizer.decode(noncanonical_action_ids),
        add_special_tokens=False,
    ) == [34070]

    state = SimpleNamespace(tokenizer=tokenizer, processor=FakeProcessor())
    monkeypatch.setattr(
        geo3k_rollout,
        "_initialize_resources",
        lambda args, sample: (
            FakeEnv(),
            None,
            {"max_turns": 2},
            state,
            "http://rollout/generate",
        ),
    )
    monkeypatch.setattr(
        geo3k_rollout,
        "encode_image_for_rollout_engine",
        lambda image: f"encoded-{image}",
    )

    observations = iter(
        [
            ([31], [31], [], None, None, False),
            (None, None, None, None, None, True),
        ]
    )
    monkeypatch.setattr(
        geo3k_rollout,
        "_process_env_step",
        lambda *args, **kwargs: next(observations),
    )

    payloads = []
    responses = iter(
        [
            {
                "text": "JK",
                "meta_info": {
                    "output_token_logprobs": [[-0.1, 41], [-0.2, 42]],
                    "finish_reason": {"type": "stop"},
                    "prompt_tokens": len(expanded_prompt_ids),
                    "weight_version": "v1",
                },
            },
            {
                "text": "done",
                "meta_info": {
                    "output_token_logprobs": [[-0.3, 13]],
                    "finish_reason": {"type": "stop"},
                    "prompt_tokens": len(expanded_prompt_ids) + 3,
                    "weight_version": "v1",
                },
            },
        ]
    )

    async def fake_post(_url, payload):
        payloads.append(copy.deepcopy(payload))
        return next(responses)

    monkeypatch.setattr(geo3k_rollout, "post", fake_post)

    sample = Sample(
        prompt="rendered Geo3K prompt",
        multimodal_inputs={"images": ["image-a"]},
        metadata={},
    )
    args = Namespace(
        partial_rollout=False,
        rollout_max_context_len=None,
    )

    result = await geo3k_rollout.generate(args, sample, {"max_new_tokens": 20})

    assert result is sample
    assert payloads[0]["input_ids"] == compact_prompt_ids
    assert payloads[1]["input_ids"] == [
        *compact_prompt_ids,
        *noncanonical_action_ids,
        31,
    ]
    assert payloads[0]["image_data"] == payloads[1]["image_data"] == ["encoded-image-a"]
    assert sample.tokens == [
        *expanded_prompt_ids,
        *noncanonical_action_ids,
        31,
        13,
    ]
    assert sample.loss_mask == [1, 1, 0, 1]
    assert sample.rollout_log_probs == [-0.1, -0.2, 0.0, -0.3]
    sample.validate()


def test_geo3k_multiturn_generation_rejects_server_retokenization_drift():
    with pytest.raises(RuntimeError, match="token alignment mismatch"):
        geo3k_rollout._validate_multimodal_generation_alignment(
            {"prompt_tokens": 7},
            expected_prompt_tokens=8,
            image_data=["encoded-image"],
        )


@pytest.mark.asyncio
async def test_reward_func_rejects_video_before_scoring_request(monkeypatch):
    async def unexpected_scoring_post(*args, **kwargs):
        raise AssertionError("video input must fail before the scoring request")

    monkeypatch.setattr(opd, "_scoring_post", unexpected_scoring_post)
    sample = Sample(
        tokens=[10, 11, 12],
        response_length=2,
        multimodal_inputs={"videos": [object()]},
    )
    args = Namespace(opd_log_prob_top_k=0, rm_url="http://teacher/generate")

    with pytest.raises(NotImplementedError, match="does not yet support video"):
        await opd.reward_func(args, sample)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dagger_top_k", "top_k", "strategy", "expected_reward"),
    [
        (0, 0, "only-student", {}),
        (2, 0, "only-student", {"teacher": {}}),
        (0, 2, "only-teacher", {"teacher": {}, "student_on_teacher": {}}),
    ],
)
async def test_multimodal_empty_response_skips_scoring_request(
    monkeypatch,
    dagger_top_k,
    top_k,
    strategy,
    expected_reward,
):
    async def unexpected_scoring_post(*args, **kwargs):
        raise AssertionError("empty multimodal responses must not issue scoring requests")

    def unexpected_image_encode(*args, **kwargs):
        raise AssertionError("empty multimodal responses must not encode images")

    monkeypatch.setattr(opd, "_scoring_post", unexpected_scoring_post)
    monkeypatch.setattr(opd, "encode_image_for_rollout_engine", unexpected_image_encode)
    sample = Sample(
        prompt="rendered multimodal prompt",
        tokens=[10, 11, 12],
        response_length=0,
        multimodal_inputs={"images": ["image-a"]},
    )
    args = Namespace(
        opd_log_prob_top_k=top_k,
        opd_dagger_top_k=dagger_top_k,
        opd_top_k_strategy=strategy,
        rm_url="http://teacher/generate",
    )

    reward = await opd.reward_func(args, sample)

    assert reward == expected_reward
    assert opd.OPD_SCORING_TELEMETRY_KEY not in (sample.metadata or {})


@pytest.mark.asyncio
async def test_teacher_topk_reward_func_uses_response_window_for_both_scoring_calls(monkeypatch):
    calls = []
    encoded_images = []

    def fake_encode(image):
        encoded_images.append(image)
        return f"encoded-{image}"

    async def fake_scoring_post(args, url, payload, *, target, response_length):
        calls.append((target, response_length, payload))
        if target == "teacher":
            response = {
                "meta_info": {
                    "input_token_logprobs": [None, [-0.3, 11], [-0.4, 12]],
                    "input_top_logprobs": [None, [[-0.1, 21]], [[-0.2, 22]]],
                }
            }
        else:
            response = {
                "meta_info": {
                    "input_token_logprobs": [None, [-0.3, 11], [-0.4, 12]],
                    "input_token_ids_logprobs": [None, [[-0.5, 21]], [[-0.6, 22]]],
                }
            }
        return opd._ScoringPostResult(response=response, telemetry={"target": target})

    monkeypatch.setattr(opd, "encode_image_for_rollout_engine", fake_encode)
    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)
    sample = Sample(
        prompt="rendered multimodal prompt",
        tokens=[7, 8, 9, 11, 12],
        response_length=2,
        multimodal_inputs={"images": ["image-a", "image-b"]},
    )
    args = Namespace(
        opd_log_prob_top_k=2,
        opd_top_k_strategy="only-teacher",
        sglang_mm_exact_scoring_suffix=True,
        rm_url="http://teacher/generate",
        sglang_router_ip="student",
        sglang_router_port=30000,
    )

    reward = await opd.reward_func(args, sample)

    assert set(reward) == {"teacher", "student_on_teacher"}
    assert [target for target, _, _ in calls] == ["teacher", "student"]
    assert all(response_length == 2 for _, response_length, _ in calls)
    assert all(payload["text"] == sample.prompt for _, _, payload in calls)
    assert all(payload["scoring_suffix_ids"] == [11, 12] for _, _, payload in calls)
    assert all("input_ids" not in payload for _, _, payload in calls)
    assert all("logprob_start_len" not in payload for _, _, payload in calls)
    assert encoded_images == ["image-a", "image-b"]
    assert calls[0][2]["image_data"] is calls[1][2]["image_data"]
    assert calls[0][2]["image_data"] == ["encoded-image-a", "encoded-image-b"]
    assert calls[0][2]["top_logprobs_num"] == 2
    assert calls[1][2]["token_ids_logprob"] == [21, 22]


@pytest.mark.asyncio
async def test_dagger_teacher_topk_reward_func_uses_one_teacher_request(monkeypatch):
    calls = []
    teacher_response = {
        "meta_info": {
            "input_token_logprobs": [None, [-0.3, 11], [-0.4, 12]],
            "input_top_logprobs": [
                None,
                [[-0.1, 21], [-0.2, 22]],
                [[-0.3, 23], [-0.4, 24]],
            ],
        }
    }

    async def fake_scoring_post(args, url, payload, *, target, response_length):
        calls.append((target, response_length, payload))
        return opd._ScoringPostResult(response=teacher_response, telemetry={"target": target})

    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)
    sample = Sample(tokens=[7, 8, 9, 11, 12], response_length=2)
    args = Namespace(
        opd_log_prob_top_k=0,
        opd_dagger_top_k=2,
        rm_url="http://teacher/generate",
    )

    reward = await opd.reward_func(args, sample)

    assert reward == {"teacher": teacher_response}
    assert [target for target, _, _ in calls] == ["teacher"]
    assert calls[0][1] == sample.response_length
    assert calls[0][2]["top_logprobs_num"] == 2
    assert "token_ids_logprob" not in calls[0][2]
    assert sample.metadata[opd.OPD_SCORING_TELEMETRY_KEY] == [{"target": "teacher"}]


def _sampled_scoring_response(token_ids: list[int]) -> dict:
    return {
        "meta_info": {
            "input_token_logprobs": [None, *[[-0.1 * (i + 1), token_id] for i, token_id in enumerate(token_ids)]]
        }
    }


def _sampled_opd_args(**overrides) -> Namespace:
    values = {
        "opd_log_prob_top_k": 0,
        "opd_log_task_reward": False,
        "opd_optimize_task_reward": False,
        "opd_task_reward_coef": 1.0,
        "reward_key": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_sampled_token_post_process_extracts_same_values_from_full_and_response_windows():
    full_window_sample = Sample(tokens=[10, 11, 12], response_length=2)
    full_window_sample.reward = {"meta_info": {"input_token_logprobs": [None, [-0.1, 10], [-0.2, 11], [-0.3, 12]]}}
    response_window_sample = Sample(tokens=[10, 11, 12], response_length=2)
    response_window_sample.reward = {"meta_info": {"input_token_logprobs": [None, [-0.2, 11], [-0.3, 12]]}}

    raw_rewards, rewards = opd.post_process_rewards(
        _sampled_opd_args(),
        [full_window_sample, response_window_sample],
    )

    assert raw_rewards == [0.0, 0.0]
    assert rewards == [0.0, 0.0]
    assert full_window_sample.teacher_log_probs.tolist() == pytest.approx([-0.2, -0.3])
    assert response_window_sample.teacher_log_probs.tolist() == pytest.approx([-0.2, -0.3])


def test_observed_task_reward_is_raw_telemetry_but_optimization_reward_stays_zero():
    sample = Sample(tokens=[10, 11, 12], response_length=2)
    sample.reward = {"meta_info": {"input_token_logprobs": [None, [-0.2, 11], [-0.3, 12]]}}
    sample.metadata[opd.OPD_TASK_REWARD_METADATA_KEY] = 1.0

    raw_rewards, rewards = opd.post_process_rewards(
        _sampled_opd_args(opd_log_task_reward=True),
        [sample],
    )

    assert raw_rewards == [1.0]
    assert rewards == [0.0]
    assert sample.teacher_log_probs.tolist() == pytest.approx([-0.2, -0.3])


def test_observed_task_reward_can_drive_group_normalized_grpo_with_an_explicit_coefficient():
    task_rewards = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    samples = []
    for task_reward in task_rewards:
        sample = Sample(tokens=[10, 11], response_length=1)
        sample.reward = _sampled_scoring_response([11])
        sample.metadata[opd.OPD_TASK_REWARD_METADATA_KEY] = task_reward
        samples.append(sample)

    raw_rewards, rewards = opd.post_process_rewards(
        _sampled_opd_args(
            opd_log_task_reward=True,
            opd_optimize_task_reward=True,
            opd_task_reward_coef=0.5,
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=True,
            n_samples_per_prompt=4,
            rollout_batch_size=2,
        ),
        samples,
    )

    expected = torch.tensor(task_rewards, dtype=torch.float32).reshape(2, 4)
    expected = expected - expected.mean(dim=-1, keepdim=True)
    expected = 0.5 * expected / (expected.std(dim=-1, keepdim=True) + 1e-6)
    assert raw_rewards == task_rewards
    torch.testing.assert_close(torch.tensor(rewards), expected.flatten())
    for sample in samples:
        torch.testing.assert_close(sample.teacher_log_probs, torch.tensor([-0.1]))


def test_sampled_token_post_process_rejects_token_alignment_mismatch():
    sample = Sample(tokens=[10, 11, 12], response_length=2, index=7, group_index=3)
    sample.reward = _sampled_scoring_response([10, 99, 12])

    with pytest.raises(
        ValueError,
        match=r"teacher scoring token alignment mismatch at response position 0.*got token id 99, expected 11",
    ):
        opd.post_process_rewards(_sampled_opd_args(), [sample])


def test_sampled_token_post_process_rejects_short_scoring_response():
    sample = Sample(tokens=[10, 11, 12], response_length=2)
    sample.reward = _sampled_scoring_response([12])

    with pytest.raises(ValueError, match=r"teacher scoring token count mismatch: got 1.*expected 2"):
        opd.post_process_rewards(_sampled_opd_args(), [sample])


def test_sampled_token_post_process_rejects_nonfinite_teacher_logprob():
    sample = Sample(tokens=[10, 11, 12], response_length=2)
    sample.reward = {
        "meta_info": {
            "input_token_logprobs": [None, [-math.inf, 11], [-0.3, 12]],
        }
    }

    with pytest.raises(ValueError, match=r"sampled logprob at response position 0 is not finite"):
        opd.post_process_rewards(_sampled_opd_args(), [sample])


def test_sampled_token_post_process_maps_masked_observation_rows_to_zero():
    sample = Sample(tokens=[10, 11, 12], response_length=2, loss_mask=[1, 0])
    sample.reward = {
        "meta_info": {
            "input_token_logprobs": [None, [-0.2, 11], None],
        }
    }

    opd.post_process_rewards(_sampled_opd_args(), [sample])

    assert sample.teacher_log_probs.tolist() == pytest.approx([-0.2, 0.0])


def test_sampled_token_post_process_keeps_active_rows_strict_when_other_rows_are_masked():
    sample = Sample(tokens=[10, 11, 12], response_length=2, loss_mask=[1, 0])
    sample.reward = {
        "meta_info": {
            "input_token_logprobs": [None, None, [-0.3, 12]],
        }
    }

    with pytest.raises(ValueError, match=r"malformed input_token_logprobs entry at response position 0"):
        opd.post_process_rewards(_sampled_opd_args(), [sample])


def test_sampled_token_post_process_ignores_masked_token_id_mismatch():
    sample = Sample(tokens=[10, 11, 12], response_length=2, loss_mask=[1, 0])
    sample.reward = _sampled_scoring_response([10, 11, 99])

    opd.post_process_rewards(_sampled_opd_args(), [sample])

    assert sample.teacher_log_probs.tolist() == pytest.approx([-0.2, 0.0])


def test_sampled_token_post_process_rejects_loss_mask_length_mismatch():
    sample = Sample(tokens=[10, 11, 12], response_length=2, loss_mask=[1])
    sample.reward = _sampled_scoring_response([10, 11, 12])

    with pytest.raises(ValueError, match=r"loss-mask length mismatch: got 1, expected 2"):
        opd.post_process_rewards(_sampled_opd_args(), [sample])


def _opd_dagger_args() -> Namespace:
    return Namespace(
        opd_log_prob_top_k=0,
        opd_dagger_top_k=2,
        vocab_size=64,
        reward_key=None,
    )


def _opd_dagger_response(top_logprobs: list) -> dict:
    return {
        "teacher": {
            "meta_info": {
                "input_token_logprobs": [None, [-0.3, 11], [-0.4, 12]],
                "input_top_logprobs": [None, *top_logprobs],
            }
        }
    }


def test_opd_dagger_post_process_preserves_sampled_rkld_and_native_t_by_k_targets():
    top_log_probs = [
        [math.log(0.6), math.log(0.4)],
        [math.log(0.5), math.log(0.2)],
    ]
    sample = Sample(tokens=[7, 8, 9, 11, 12], response_length=2)
    sample.reward = _opd_dagger_response(
        [
            [[top_log_probs[0][0], 21], [top_log_probs[0][1], 22]],
            [[top_log_probs[1][0], 23], [top_log_probs[1][1], 24]],
        ]
    )

    raw_rewards, rewards = opd.post_process_rewards(_opd_dagger_args(), [sample])

    assert raw_rewards == rewards == [0.0]
    torch.testing.assert_close(sample.teacher_log_probs, torch.tensor([-0.3, -0.4], dtype=torch.float32))
    assert sample.teacher_topk_token_ids.tolist() == [[21, 22], [23, 24]]
    torch.testing.assert_close(
        sample.teacher_topk_log_probs,
        torch.tensor(top_log_probs, dtype=torch.float32),
    )
    assert sample.teacher_topk_valid_mask.tolist() == [[True, True], [True, True]]
    assert sample.opd_reverse_kl is None
    sample.validate()


def test_opd_dagger_post_process_can_return_task_optimization_reward_without_changing_sparse_targets():
    sample = Sample(tokens=[7, 8, 9, 11, 12], response_length=2)
    sample.reward = _opd_dagger_response(
        [
            [[math.log(0.6), 21], [math.log(0.4), 22]],
            [[math.log(0.5), 23], [math.log(0.2), 24]],
        ]
    )
    sample.metadata[opd.OPD_TASK_REWARD_METADATA_KEY] = 1.0
    args = _opd_dagger_args()
    args.opd_log_task_reward = True
    args.opd_optimize_task_reward = True
    args.opd_task_reward_coef = 0.25
    args.advantage_estimator = "grpo"
    args.rewards_normalization = False

    raw_rewards, rewards = opd.post_process_rewards(args, [sample])

    assert raw_rewards == [1.0]
    assert rewards == [0.25]
    assert sample.teacher_topk_token_ids.tolist() == [[21, 22], [23, 24]]
    assert sample.teacher_topk_valid_mask.all()


def test_opd_dagger_post_process_masks_rows_with_fewer_than_k_targets():
    sample = Sample(tokens=[7, 8, 9, 11, 12], response_length=2)
    sample.reward = _opd_dagger_response(
        [
            [[math.log(0.7), 21]],
            [[math.log(0.5), 23], [math.log(0.2), 24]],
        ]
    )

    opd.post_process_rewards(_opd_dagger_args(), [sample])

    assert sample.teacher_topk_token_ids.tolist() == [[21, 0], [23, 24]]
    assert sample.teacher_topk_valid_mask.tolist() == [[True, False], [True, True]]
    assert torch.isneginf(sample.teacher_topk_log_probs[0, 1])
    sample.validate()


def test_opd_dagger_post_process_maps_masked_observation_rows_to_inert_targets():
    sample = Sample(tokens=[7, 8, 9, 11, 12], response_length=2, loss_mask=[1, 0])
    sample.reward = _opd_dagger_response(
        [
            [[math.log(0.7), 21], [math.log(0.2), 22]],
            None,
        ]
    )
    sample.reward["teacher"]["meta_info"]["input_token_logprobs"][-1] = None

    opd.post_process_rewards(_opd_dagger_args(), [sample])

    assert sample.teacher_log_probs.tolist() == pytest.approx([-0.3, 0.0])
    assert sample.teacher_topk_token_ids.tolist() == [[21, 22], [0, 0]]
    assert sample.teacher_topk_valid_mask.tolist() == [[True, True], [False, False]]
    assert torch.isneginf(sample.teacher_topk_log_probs[1]).all()
    sample.validate()


def test_opd_dagger_post_process_rejects_duplicate_ids():
    sample = Sample(tokens=[7, 8, 9, 11, 12], response_length=2)
    sample.reward = _opd_dagger_response(
        [
            [[math.log(0.6), 21], [math.log(0.3), 21]],
            [[math.log(0.5), 23], [math.log(0.2), 24]],
        ]
    )

    with pytest.raises(ValueError, match=r"row 0 contains duplicate token ids"):
        opd.post_process_rewards(_opd_dagger_args(), [sample])


def test_opd_dagger_post_process_rejects_teacher_mass_above_one():
    sample = Sample(tokens=[7, 8, 9, 11, 12], response_length=2)
    sample.reward = _opd_dagger_response(
        [
            [[math.log(0.8), 21], [math.log(0.4), 22]],
            [[math.log(0.5), 23], [math.log(0.2), 24]],
        ]
    )

    with pytest.raises(ValueError, match=r"probability mass exceeds 1 at response position 0"):
        opd.post_process_rewards(_opd_dagger_args(), [sample])


def test_opd_dagger_post_process_rejects_teacher_id_outside_student_vocab():
    sample = Sample(tokens=[7, 8, 9, 11, 12], response_length=2)
    sample.reward = _opd_dagger_response(
        [
            [[math.log(0.6), 21], [math.log(0.3), 64]],
            [[math.log(0.5), 23], [math.log(0.2), 24]],
        ]
    )

    with pytest.raises(ValueError, match=r"outside \[0, 64\): 64"):
        opd.post_process_rewards(_opd_dagger_args(), [sample])


def _teacher_payload():
    return {
        "teacher": {
            "meta_info": {
                "input_top_logprobs": [
                    None,
                    [_entry(0.5, 2), _entry(0.5, 3)],
                    [_entry(0.8, 4), _entry(0.2, 6)],
                ],
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.3, 1), _entry(0.7, 2)],
                    [_entry(0.4, 4), _entry(0.6, 5)],
                ],
            }
        },
        "student_on_teacher": {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.4, 2), _entry(0.2, 3)],
                    [_entry(0.7, 4), _entry(0.1, 6)],
                ]
            }
        },
    }


def test_topk_only_student_uses_student_probability_weights():
    reverse_kl = _compute_topk_reverse_kl(_args("only-student"), _sample(), _teacher_payload())

    expected_0 = 0.6 * math.log(0.6 / 0.3) + 0.4 * math.log(0.4 / 0.7)
    expected_1 = 0.7 * math.log(0.7 / 0.4) + 0.3 * math.log(0.3 / 0.6)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_intersection_uses_overlap_only():
    reverse_kl = _compute_topk_reverse_kl(_args("intersection", "none"), _sample(), _teacher_payload())

    assert reverse_kl.tolist() == pytest.approx(
        [
            math.log(0.4 / 0.5),
            math.log(0.7 / 0.8),
        ]
    )


def test_topk_only_teacher_does_not_need_student_top_logprobs():
    sample = Sample(tokens=[10, 11, 12], response_length=2)

    reverse_kl = _compute_topk_reverse_kl(_args("only-teacher"), sample, _teacher_payload())

    expected_0 = (2 / 3) * math.log(0.4 / 0.5) + (1 / 3) * math.log(0.2 / 0.5)
    expected_1 = (7 / 8) * math.log(0.7 / 0.8) + (1 / 8) * math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_xor_uses_symmetric_difference_without_normalization():
    reverse_kl = _compute_topk_reverse_kl(_args("xor", "none"), _sample(), _teacher_payload())

    expected_0 = math.log(0.6 / 0.3) + math.log(0.2 / 0.5)
    expected_1 = math.log(0.3 / 0.6) + math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_per_position_ids_pads_prompt_and_keeps_response_order():
    # Two response positions, each with two top-k entries [logprob, token_id].
    student_top = [[_entry(0.6, 5), _entry(0.4, 7)], [_entry(0.7, 9), _entry(0.3, 11)]]
    per_pos = _per_position_ids(student_top, prompt_len=3)
    # 3 empty prompt slots, then response positions with their own token ids.
    assert per_pos == [[], [], [], [5, 7], [9, 11]]
    # Aligns with the existing _trim_input_field extraction values[1:][-R:]: for a
    # length-5 response, indices 3,4 are the response positions.
    values = list(range(5))
    assert values[1:][-2:] == [3, 4]
    assert per_pos[3] == [5, 7] and per_pos[4] == [9, 11]


def test_score_payload_routes_per_position_vs_flat():
    flat = _score_payload([1, 2, 3], response_length=2, token_ids=[5, 7])
    assert flat["token_ids_logprob"] == [5, 7]
    assert "token_ids_logprob_positions" not in flat

    per_pos = _score_payload([1, 2, 3], response_length=2, token_ids_positions=[[], [5, 7], [9, 11]])
    assert per_pos["token_ids_logprob_positions"] == [[], [5, 7], [9, 11]]
    assert "token_ids_logprob" not in per_pos


# ---------------------------------------------------------------------------
# Multi-teacher routing (--opd-teacher-urls)
# ---------------------------------------------------------------------------


def _routing_args(urls=None, key="opd_teacher", rm_url="http://single-teacher/generate"):
    return Namespace(opd_teacher_urls=urls, opd_teacher_key=key, rm_url=rm_url)


def _tagged_sample(metadata=None):
    return Sample(tokens=[1, 2, 3], response_length=2, metadata=metadata or {})


def test_parse_teacher_urls_parses_names_and_keeps_equals_in_url():
    url_map = parse_teacher_urls(["math=http://h1:30001/generate", "code=http://h2:30002/generate?tag=a=b"])
    assert url_map == {
        "math": "http://h1:30001/generate",
        "code": "http://h2:30002/generate?tag=a=b",
    }


def test_parse_teacher_urls_empty_or_none_gives_empty_map():
    assert parse_teacher_urls(None) == {}
    assert parse_teacher_urls([]) == {}


@pytest.mark.parametrize("bad", ["math", "=http://h1/generate", "math=", "  =  "])
def test_parse_teacher_urls_rejects_malformed_entries(bad):
    with pytest.raises(ValueError, match="expected NAME=URL"):
        parse_teacher_urls([bad])


def test_parse_teacher_urls_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate teacher name"):
        parse_teacher_urls(["math=http://h1/generate", "math=http://h2/generate"])


def test_routing_unset_map_falls_back_to_rm_url():
    args = _routing_args(urls=None)
    sample = _tagged_sample({"opd_teacher": "math"})
    assert _teacher_url_for_sample(args, sample) == "http://single-teacher/generate"


def test_routing_by_metadata_name():
    args = _routing_args(urls=["math=http://h1/generate", "code=http://h2/generate"])
    assert _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "math"})) == "http://h1/generate"
    assert _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "code"})) == "http://h2/generate"


def test_routing_respects_custom_metadata_key():
    args = _routing_args(urls=["math=http://h1/generate"], key="task")
    assert _teacher_url_for_sample(args, _tagged_sample({"task": "math"})) == "http://h1/generate"


def test_routing_missing_name_uses_default_entry():
    args = _routing_args(urls=["math=http://h1/generate", "default=http://h3/generate"])
    assert _teacher_url_for_sample(args, _tagged_sample({})) == "http://h3/generate"


def test_routing_unknown_name_uses_default_entry():
    args = _routing_args(urls=["math=http://h1/generate", "default=http://h3/generate"])
    assert _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "physics"})) == "http://h3/generate"


def test_routing_unknown_name_without_default_raises():
    args = _routing_args(urls=["math=http://h1/generate"])
    with pytest.raises(ValueError, match="matches no --opd-teacher-urls name"):
        _teacher_url_for_sample(args, _tagged_sample({"opd_teacher": "physics"}))


def test_routing_missing_name_without_default_raises():
    args = _routing_args(urls=["math=http://h1/generate"])
    with pytest.raises(ValueError, match="missing teacher key"):
        _teacher_url_for_sample(args, _tagged_sample({}))
