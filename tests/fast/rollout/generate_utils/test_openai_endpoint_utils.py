"""Tests for OpenAIEndpointTracer (session-server client side).

The sample-assembly and TITO multi-turn merge tests live in
tests/fast/rollout/session/test_samples.py (assembly) and
test_samples_codec.py (wire codec), next to the functions.
The collect_samples tests here lock the client's HTTP behavior deltas vs the
old collect_records path: single POST with no retries, non-2xx raises with the
body text, timeout raises (instead of silently ABORTing), and the session
DELETE is attempted on every path.
"""

import asyncio
from types import SimpleNamespace

import pytest
import torch

import miles.utils.http_utils as http_utils
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.samples.codec import COMPUTED_FIELDS, COMPUTED_FIELDS_V2, encode_samples
from miles.utils.http_utils import post_bytes_no_retry
from miles.utils.types import Sample


@pytest.mark.asyncio
async def test_create_reads_session_server_instance_id_from_args(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_post(url: str, payload: dict, action: str = "post"):
        calls.append((action, url))
        assert action == "post"
        assert url == "http://127.0.0.1:12345/sessions"
        return {"session_id": "session-123"}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

    args = SimpleNamespace(
        session_server_ip="127.0.0.1",
        session_server_ports=[12345],
        session_server_instance_ids={12345: "server-instance-123"},
    )
    tracer = await OpenAIEndpointTracer.create(args)

    assert tracer.base_url == "http://127.0.0.1:12345/sessions/session-123"
    assert tracer.session_server_id == "127.0.0.1:12345"
    assert tracer.session_server_instance_id == "server-instance-123"
    # No /health probe: the id is read locally, create() issues only the POST.
    assert calls == [("post", "http://127.0.0.1:12345/sessions")]


@pytest.mark.asyncio
async def test_create_without_instance_id_on_args(monkeypatch):
    async def fake_post(url: str, payload: dict, action: str = "post"):
        return {"session_id": "session-123"}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

    args = SimpleNamespace(session_server_ip="127.0.0.1", session_server_ports=[12345])
    tracer = await OpenAIEndpointTracer.create(args)

    assert tracer.session_server_instance_id is None


@pytest.mark.asyncio
async def test_create_distributes_sessions_across_port_range(monkeypatch):
    """With a multi-port range, sessions land on more than one instance, and every
    request of a session (create, samples POST, DELETE) hits the port chosen
    at create time — the URL is the router."""
    calls: list[tuple[str, str]] = []

    async def fake_post(url: str, payload: dict, action: str = "post"):
        calls.append((action, url))
        if action == "post" and url.endswith("/sessions"):
            return {"session_id": f"session-{len(calls)}"}
        return {}

    async def fake_post_bytes(url, payload, *, timeout):
        calls.append(("post_bytes", url))
        return encode_samples([], {}, "no_records")

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)
    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post_bytes_no_retry", fake_post_bytes)

    ports = [12345, 12346, 12347, 12348]
    args = SimpleNamespace(session_server_ip="127.0.0.1", session_server_ports=ports)

    chosen_ports = set()
    for _ in range(32):
        calls.clear()
        tracer = await OpenAIEndpointTracer.create(args)
        port = int(tracer.session_server_id.rsplit(":", 1)[1])
        assert port in ports
        chosen_ports.add(port)

        await tracer.collect_samples(Sample(), max_seq_len=None)
        prefix = f"http://127.0.0.1:{port}"
        assert [url for _, url in calls] == [
            f"{prefix}/sessions",
            f"{tracer.base_url}/samples",
            tracer.base_url,
        ]
        assert tracer.base_url.startswith(f"{prefix}/sessions/")

    # 32 uniform picks over 4 ports miss a given port with p = (3/4)^32 ≈ 1e-4.
    assert len(chosen_ports) > 1


# ── test: compute_samples_from_openai_records ────────────────────────


class TestComputeSamplesFromRecords:
    def test_single_record_builds_correct_sample(self):
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            output_log_probs=[-0.5, -0.6],
        )
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(_ARGS, input_sample, [record], tok)

        assert len(samples) == 1
        s = samples[0]
        assert s.tokens == [1, 2, 3, 10, 11]
        assert s.rollout_log_probs == [-0.5, -0.6]
        assert s.response_length == 2
        assert s.loss_mask == [1, 1]
        assert s.status == Sample.Status.COMPLETED

    def test_multiple_records_produce_multiple_samples(self):
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10]),
            _make_record(prompt_token_ids=[1, 2, 10, 20], output_token_ids=[30]),
        ]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(_ARGS, input_sample, records, tok)

        assert len(samples) == 2
        assert samples[0].tokens == [1, 2, 10]
        assert samples[1].tokens == [1, 2, 10, 20, 30]

    def test_last_sample_carries_raw_conversation(self):
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10]),
            _make_record(prompt_token_ids=[1, 2, 10, 20], output_token_ids=[30]),
        ]

        samples = compute_samples_from_openai_records(_ARGS_RECORDING, _make_input_sample(), records, tok)

        assert "messages" not in (samples[0].metadata or {})
        assert samples[1].metadata["messages"] == records[1].request["messages"] + [
            records[1].response["choices"][0]["message"]
        ]

    def test_merge_keeps_last_conversation_snapshot(self):
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 11, 20, 21], output_token_ids=[30, 31]),
        ]

        samples = compute_samples_from_openai_records(_ARGS_RECORDING, _make_input_sample(), records, tok)
        merged = merge_samples(samples, tok)

        assert merged.metadata["messages"] == samples[-1].metadata["messages"]

    def test_finish_reason_length_gives_truncated(self):
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2],
            output_token_ids=[10],
            finish_reason="length",
        )
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(_ARGS, input_sample, [record], tok)

        assert samples[0].status == Sample.Status.TRUNCATED


# ── test: multi-turn prefix chain (merge_samples integration) ────────


class TestMultiTurnPrefixChain:
    """Validate that session records from a well-behaved multi-turn
    conversation satisfy the prefix chain required by merge_samples.

    The contract: for consecutive records r[i] and r[i+1],
    r[i+1].prompt_token_ids must start with r[i].prompt_token_ids + r[i].output_token_ids.
    This is because the agent includes the previous response in the next prompt.
    """

    def test_two_turn_merge_succeeds(self):
        """Normal two-turn conversation: samples merge without error."""
        tok = _mock_tokenizer()

        # Turn 1: prompt=[1,2,3], model outputs [10,11]
        # Turn 2: prompt=[1,2,3, 10,11, 20,21], model outputs [30,31]
        #   (tokens 20,21 are the tool/observation tokens added by the environment)
        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=[10, 11],
                output_log_probs=[-0.1, -0.2],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
                output_log_probs=[-0.3, -0.4],
            ),
        ]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(_ARGS, input_sample, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.tokens == [1, 2, 3, 10, 11, 20, 21, 30, 31]
        assert merged.response_length == 2 + 2 + 2  # resp1 + obs + resp2
        assert merged.loss_mask == [1, 1, 0, 0, 1, 1]
        assert merged.status == Sample.Status.COMPLETED

    def test_three_turn_merge_succeeds(self):
        """Three-turn conversation: prefix chain holds across all turns."""
        tok = _mock_tokenizer()

        records = [
            _make_record(
                prompt_token_ids=[1, 2],
                output_token_ids=[10],
                output_log_probs=[-0.1],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 10, 20],
                output_token_ids=[30],
                output_log_probs=[-0.2],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 10, 20, 30, 40],
                output_token_ids=[50],
                output_log_probs=[-0.3],
            ),
        ]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(_ARGS, input_sample, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.tokens == [1, 2, 10, 20, 30, 40, 50]
        assert merged.response_length == 1 + 1 + 1 + 1 + 1  # 3 responses + 2 obs

    def test_prefix_mismatch_raises(self):
        """When the prefix chain is broken, merge_samples must fail."""
        tok = _mock_tokenizer()

        # Turn 2's prompt does NOT start with turn 1's full tokens
        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=[10, 11],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 3, 99, 99, 20, 21],  # 99,99 != 10,11
                output_token_ids=[30, 31],
            ),
        ]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(_ARGS, input_sample, records, tok)

        with pytest.raises(AssertionError, match="b.tokens must start with a.tokens"):
            merge_samples(samples, tok)

    def test_two_turn_merge_propagates_teacher_log_probs(self):
        """OPD teacher_log_probs merge like rollout_log_probs: per-turn values
        concatenated with zeros over the injected observation span."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11], output_log_probs=[-0.1, -0.2]),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
                output_log_probs=[-0.3, -0.4],
            ),
        ]
        input_sample = _make_input_sample()
        samples = compute_samples_from_openai_records(_ARGS, input_sample, records, tok)

        # OPD attaches per-response-token teacher log-probs to each turn's sample.
        samples[0].teacher_log_probs = [-1.0, -1.1]
        samples[1].teacher_log_probs = [-1.2, -1.3]

        merged = merge_samples(samples, tok)

        # resp1 (2) + obs (2 zeros) + resp2 (2)
        assert merged.teacher_log_probs == [-1.0, -1.1, 0.0, 0.0, -1.2, -1.3]
        assert len(merged.teacher_log_probs) == merged.response_length
        merged.validate()  # the new teacher_log_probs length assertion must hold

    def test_two_turn_merge_propagates_teacher_topk_targets(self):
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11]),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
            ),
        ]
        samples = compute_samples_from_openai_records(_ARGS, _make_input_sample(), records, tok)
        samples[0].teacher_topk_token_ids = torch.tensor([[101, 102], [103, 104]], dtype=torch.long)
        samples[0].teacher_topk_log_probs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])
        samples[0].teacher_topk_valid_mask = torch.ones((2, 2), dtype=torch.bool)
        samples[1].teacher_topk_token_ids = torch.tensor([[105, 106], [107, 108]], dtype=torch.long)
        samples[1].teacher_topk_log_probs = torch.tensor([[-0.5, -0.6], [-0.7, -0.8]])
        samples[1].teacher_topk_valid_mask = torch.ones((2, 2), dtype=torch.bool)

        merged = merge_samples(samples, tok)

        assert merged.teacher_topk_token_ids.tolist() == [
            [101, 102],
            [103, 104],
            [0, 0],
            [0, 0],
            [105, 106],
            [107, 108],
        ]
        torch.testing.assert_close(
            merged.teacher_topk_log_probs[[0, 1, 4, 5]],
            torch.tensor([[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6], [-0.7, -0.8]]),
        )
        assert torch.isneginf(merged.teacher_topk_log_probs[2:4]).all()
        assert merged.teacher_topk_valid_mask.tolist() == [
            [True, True],
            [True, True],
            [False, False],
            [False, False],
            [True, True],
            [True, True],
        ]
        assert merged.teacher_topk_token_ids.shape == (merged.response_length, 2)
        merged.validate()

    def test_two_turn_merge_rejects_partial_teacher_topk_targets(self):
        tok = _mock_tokenizer()
        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11]),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
            ),
        ]
        samples = compute_samples_from_openai_records(_ARGS, _make_input_sample(), records, tok)
        samples[0].teacher_topk_token_ids = torch.tensor([[101, 102], [103, 104]], dtype=torch.long)
        samples[0].teacher_topk_log_probs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])
        samples[0].teacher_topk_valid_mask = torch.ones((2, 2), dtype=torch.bool)

        with pytest.raises(AssertionError, match="must include ids/log-probs/valid-mask"):
            merge_samples(samples, tok)

    def test_two_turn_merge_propagates_opd_student_top_logprobs_metadata(self):
        """Top-k OPD student top-logprobs are per-token metadata, not equal metadata."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11], output_log_probs=[-0.1, -0.2]),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
                output_log_probs=[-0.3, -0.4],
            ),
        ]
        input_sample = _make_input_sample()
        samples = compute_samples_from_openai_records(_ARGS, input_sample, records, tok)

        turn_0_top_logprobs = [[[-0.1, 101]], [[-0.2, 102]]]
        turn_1_top_logprobs = [[[-0.3, 103]], [[-0.4, 104]]]
        samples[0].metadata = {
            "opd_student_top_logprobs": turn_0_top_logprobs,
            "shared_metadata": "same",
        }
        samples[1].metadata = {
            "opd_student_top_logprobs": turn_1_top_logprobs,
            "shared_metadata": "same",
        }

        merged = merge_samples(samples, tok)

        assert merged.metadata["shared_metadata"] == "same"
        assert merged.metadata["opd_student_top_logprobs"] == [
            *turn_0_top_logprobs,
            [],
            [],
            *turn_1_top_logprobs,
        ]
        assert len(merged.metadata["opd_student_top_logprobs"]) == merged.response_length

    def test_two_turn_merge_teacher_log_probs_none_stays_none(self):
        """Non-OPD runs leave teacher_log_probs unset; merge must keep it None."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 11, 20, 21], output_token_ids=[30, 31]),
        ]
        input_sample = _make_input_sample()
        samples = compute_samples_from_openai_records(_ARGS, input_sample, records, tok)

        merged = merge_samples(samples, tok)

        assert merged.teacher_log_probs is None

    def test_merge_raises_on_teacher_log_probs_length_mismatch(self):
        """validate() guards teacher_log_probs length (surfaced via merge_samples)."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 11, 20, 21], output_token_ids=[30, 31]),
        ]
        input_sample = _make_input_sample()
        samples = compute_samples_from_openai_records(_ARGS, input_sample, records, tok)

        samples[0].teacher_log_probs = [-1.0]  # length 1 != response_length 2

        with pytest.raises(AssertionError, match="teacher_log_probs length"):
            merge_samples(samples, tok)


# ── test: TITO trailing token trimming ────────────────────────────────

STOP = 99  # stands for <|observation|> stop token


class TestTITOTrailingTokenTrim:
    """Validate trailing-token trimming via ``accumulated_token_ids``.

    Worked example — agentic tool-call retries
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    An agent makes three turns.  The model's tool call fails to parse on
    turns 1 and 2, so the agent feeds back an error and retries.

    The session server sees three request/response pairs (records).  Each
    record's response is an independent inference re-stitched via
    pretokenized prefix reuse::

        record 0  prompt_token_ids:  [<|sys|>, aaa, <|user|>, bbb, <|asst|>]
                  output_token_ids:  [ccc, <|obs|>]      ← model stopped with <|obs|>

        record 1  prompt_token_ids:  [<|sys|>, aaa, <|user|>, bbb, <|asst|>, ccc, <|sys|>, ddd, <|asst|>]
                  output_token_ids:  [eee, <|obs|>]

        record 2  prompt_token_ids:  [..., eee, <|sys|>, fff, <|asst|>]
                  output_token_ids:  [ggg, <|obs|>]

    ``accumulated_token_ids`` = record 2's prompt + output::

        [<|sys|>, aaa, <|user|>, bbb, <|asst|>, ccc, <|sys|>, ddd,
         <|asst|>, eee, <|sys|>, fff, <|asst|>, ggg, <|obs|>]

    Note: there is NO ``<|obs|>`` between ``ccc`` and ``<|sys|>`` in the
    accumulated sequence — the stop token the model emitted at turn 1 was
    consumed by the chat template when rendering turn 2's prompt.

    The algorithm walks ``accumulated_token_ids`` with a cursor::

        Record 0:  cursor = len(prompt_0) → points to "ccc"
                   Match output [ccc, <|obs|>] against accumulated[cursor:]:
                     ccc OK, <|obs|> MISMATCH (accumulated has <|sys|> here)
                   → trim_count=1, strip <|obs|>; cursor advances past "ccc"

        Record 1:  cursor = len(prompt_1) → points to "eee"
                   Match [eee, <|obs|>]: eee OK, <|obs|> MISMATCH
                   → trim_count=1; cursor advances past "eee"

        Record 2:  cursor = len(prompt_2) → points to "ggg"
                   Match [ggg, <|obs|>]: ggg OK, <|obs|> OK (last turn)
                   → trim_count=0; cursor reaches end

    Result: three Samples with output tokens [ccc], [eee], [ggg, <|obs|>],
    each carrying original per-turn logprobs.

    The tests below encode this example (and variants) with concrete
    token IDs.  We use ``STOP = 99`` to represent ``<|observation|>``.
    """

    def test_three_turn_trim_trailing_stop_tokens(self):
        """Three-turn retry: non-final turns have 1 trailing stop token trimmed."""
        tok = _mock_tokenizer()

        #   prompt: [1, 2, 3]  output: [10, STOP]
        #   prompt: [1, 2, 3, 10, 4, 5, 6]  output: [20, STOP]
        #   prompt: [1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9]  output: [30, STOP]
        # accumulated (no intermediate STOPs):
        #   [1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30, STOP]
        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, STOP]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 4, 5, 6], output_token_ids=[20, STOP]),
            _make_record(prompt_token_ids=[1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9], output_token_ids=[30, STOP]),
        ]
        accumulated = [1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30, STOP]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(
            _ARGS,
            input_sample,
            records,
            tok,
            accumulated_token_ids=accumulated,
            max_trim_tokens=1,
        )

        assert len(samples) == 3
        # Turn 0: [10, STOP] → trim 1 → response_length=1
        assert samples[0].tokens == [1, 2, 3, 10]
        assert samples[0].response_length == 1
        # Turn 1: [20, STOP] → trim 1 → response_length=1
        assert samples[1].tokens == [1, 2, 3, 10, 4, 5, 6, 20]
        assert samples[1].response_length == 1
        # Turn 2 (last): [30, STOP] → trim 0 → response_length=2
        assert samples[2].tokens == [1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30, STOP]
        assert samples[2].response_length == 2

    def test_no_trim_when_no_trailing_stop(self):
        """When output tokens fully match accumulated, trim_count=0 for all turns."""
        tok = _mock_tokenizer()

        # Two turns, no trailing stop tokens — output aligns perfectly
        #   prompt: [1, 2]  output: [10, 11]
        #   prompt: [1, 2, 10, 11, 3, 4]  output: [20, 21]
        # accumulated: [1, 2, 10, 11, 3, 4, 20, 21]
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, 11]),
            _make_record(prompt_token_ids=[1, 2, 10, 11, 3, 4], output_token_ids=[20, 21]),
        ]
        accumulated = [1, 2, 10, 11, 3, 4, 20, 21]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(
            _ARGS,
            input_sample,
            records,
            tok,
            accumulated_token_ids=accumulated,
            max_trim_tokens=1,
        )

        assert len(samples) == 2
        assert samples[0].tokens == [1, 2, 10, 11]
        assert samples[0].response_length == 2
        assert samples[1].tokens == [1, 2, 10, 11, 3, 4, 20, 21]
        assert samples[1].response_length == 2

    def test_single_turn_no_trim(self):
        """Single turn: last turn never trims, even with accumulated_token_ids."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11, STOP]),
        ]
        accumulated = [1, 2, 3, 10, 11, STOP]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(
            _ARGS,
            input_sample,
            records,
            tok,
            accumulated_token_ids=accumulated,
            max_trim_tokens=1,
        )

        assert len(samples) == 1
        assert samples[0].tokens == [1, 2, 3, 10, 11, STOP]
        assert samples[0].response_length == 3

    def test_no_accumulated_skips_trimming(self):
        """Without accumulated_token_ids, no trimming is performed at all."""
        tok = _mock_tokenizer()

        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, STOP]),
            _make_record(prompt_token_ids=[1, 2, 10, STOP, 3, 4], output_token_ids=[20, STOP]),
        ]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(
            _ARGS,
            input_sample,
            records,
            tok,
            accumulated_token_ids=None,
        )

        assert len(samples) == 2
        # No trimming — STOP is kept for both turns
        assert samples[0].tokens == [1, 2, 10, STOP]
        assert samples[0].response_length == 2
        assert samples[1].tokens == [1, 2, 10, STOP, 3, 4, 20, STOP]
        assert samples[1].response_length == 2

    def test_trim_exceeding_max_raises(self):
        """If trailing tokens exceed max_trim_tokens, assert fires."""
        tok = _mock_tokenizer()

        # Output has 2 trailing tokens that don't match, but max_trim_tokens=1
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, STOP, STOP]),
            _make_record(prompt_token_ids=[1, 2, 10, 3, 4], output_token_ids=[20]),
        ]
        accumulated = [1, 2, 10, 3, 4, 20]
        input_sample = _make_input_sample()

        with pytest.raises(AssertionError, match="trim_count 2 exceeds allowed=1"):
            compute_samples_from_openai_records(
                _ARGS,
                input_sample,
                records,
                tok,
                accumulated_token_ids=accumulated,
                max_trim_tokens=1,
            )

    def test_cursor_covers_entire_accumulated(self):
        """After processing all records, cursor must equal len(accumulated)."""
        tok = _mock_tokenizer()

        # accumulated is shorter than what records imply — cursor won't reach end
        records = [
            _make_record(prompt_token_ids=[1, 2], output_token_ids=[10, STOP]),
            _make_record(prompt_token_ids=[1, 2, 10, 3], output_token_ids=[20]),
        ]
        # Missing the last token — accumulated should be [1,2,10,3,20] but we give [1,2,10,3,20,99]
        accumulated = [1, 2, 10, 3, 20, 99]
        input_sample = _make_input_sample()

        with pytest.raises(AssertionError, match="cursor .* != len\\(accumulated_token_ids\\)"):
            compute_samples_from_openai_records(
                _ARGS,
                input_sample,
                records,
                tok,
                accumulated_token_ids=accumulated,
                max_trim_tokens=1,
            )


# ── test: thinking token issue (documents known failure mode) ────────


class TestThinkingTokenPrefixBreak:
    """Documents the known issue where model-generated <think>...</think>
    tokens break the prefix chain.

    When a model (e.g. Qwen3) generates <think>reasoning</think> before
    the actual response, agents strip the thinking content from conversation
    history. This causes the next turn's prompt to not include the thinking
    tokens, breaking the prefix assumption in merge_samples.

    This is a MODEL-LEVEL issue — the fix should be at the model/serving
    config level (disable thinking mode), not in the merge logic.
    """

    THINK_TOKEN = 151667  # <think> in Qwen3
    END_THINK_TOKEN = 151668  # </think> in Qwen3
    NEWLINE_TOKEN = 198  # \n

    def test_thinking_tokens_break_prefix_chain(self):
        """Demonstrates the failure: model outputs <think>..., but the agent
        strips it from history, so the next prompt doesn't include those tokens."""
        tok = _mock_tokenizer()

        # Turn 1: model generates <think>\nreasoning\n</think>\n then actual response
        thinking_tokens = [
            self.THINK_TOKEN,
            self.NEWLINE_TOKEN,
            42,
            43,
            self.NEWLINE_TOKEN,
            self.END_THINK_TOKEN,
            self.NEWLINE_TOKEN,
        ]
        response_tokens = [10, 11]
        all_output = thinking_tokens + response_tokens

        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=all_output,
            ),
            # Turn 2: agent only included the actual response [10, 11] in history
            # (stripped thinking tokens), plus observation [20, 21]
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
            ),
        ]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(_ARGS, input_sample, records, tok)

        # sample[0].tokens = [1,2,3] + thinking + [10,11] = [1,2,3, <think>,\n,42,43,\n,</think>,\n, 10,11]
        # sample[1].tokens = [1,2,3, 10,11, 20,21, 30,31]
        # sample[1] does NOT start with sample[0] — prefix chain broken
        with pytest.raises(AssertionError, match="b.tokens must start with a.tokens"):
            merge_samples(samples, tok)

    def test_no_thinking_tokens_prefix_chain_holds(self):
        """When thinking is disabled, the same conversation merges fine."""
        tok = _mock_tokenizer()

        # Same conversation but model output has no thinking prefix
        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=[10, 11],
            ),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
            ),
        ]
        input_sample = _make_input_sample()

        samples = compute_samples_from_openai_records(_ARGS, input_sample, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.tokens == [1, 2, 3, 10, 11, 20, 21, 30, 31]


# ── test: prefix cache info population ────────────────────────────────


class TestPrefixCacheInfo:
    """Validate that prefix cache statistics from meta_info are collected."""

    def test_single_record_with_cache_stats(self):
        """cached_tokens and prompt_tokens from meta_info populate prefix_cache_info."""
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            cached_tokens=2,
            prompt_tokens=3,
        )
        input_sample = _make_input_sample()
        samples = compute_samples_from_openai_records(_ARGS, input_sample, [record], tok)

        assert samples[0].prefix_cache_info.cached_tokens == 2
        assert samples[0].prefix_cache_info.total_prompt_tokens == 3

    def test_multi_turn_cache_stats_accumulate_after_merge(self):
        """After merge_samples, prefix_cache_info sums across turns."""
        tok = _mock_tokenizer()
        records = [
            _make_record(
                prompt_token_ids=[1, 2, 3],
                output_token_ids=[10, 11],
                output_log_probs=[-0.1, -0.2],
                cached_tokens=0,
                prompt_tokens=3,
            ),
            _make_record(
                prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
                output_token_ids=[30, 31],
                output_log_probs=[-0.3, -0.4],
                cached_tokens=5,
                prompt_tokens=7,
            ),
        ]
        input_sample = _make_input_sample()
        samples = compute_samples_from_openai_records(_ARGS, input_sample, records, tok)
        merged = merge_samples(samples, tok)

        assert merged.prefix_cache_info.cached_tokens == 0 + 5
        assert merged.prefix_cache_info.total_prompt_tokens == 3 + 7
        assert merged.prefix_cache_info.prefix_cache_hit_rate == 5 / 10

    def test_missing_cache_fields_default_to_zero(self):
        """Records without cached_tokens/prompt_tokens give zero prefix_cache_info (regression)."""
        tok = _mock_tokenizer()
        record = _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
        )
        input_sample = _make_input_sample()
        samples = compute_samples_from_openai_records(_ARGS, input_sample, [record], tok)

        assert samples[0].prefix_cache_info.cached_tokens == 0
        assert samples[0].prefix_cache_info.total_prompt_tokens == 0

# ── collect_samples client behavior ──


def _tracer() -> OpenAIEndpointTracer:
    return OpenAIEndpointTracer(router_url="http://127.0.0.1:12345", session_id="sid-1")


def _computed_reply_payload() -> bytes:
    sample = Sample()
    sample.tokens = [1, 2, 10]
    sample.response = "r"
    sample.response_length = 1
    sample.loss_mask = [1]
    sample.rollout_log_probs = [-0.5]
    sample.status = Sample.Status.COMPLETED
    return encode_samples([sample], {"max_trim_tokens": 1}, None)


class _CollectCalls:
    """Patches the two HTTP primitives collect_samples uses and records order."""

    def __init__(self, monkeypatch, *, post_outcome, delete_outcome=None):
        self.calls: list[str] = []

        async def fake_post_bytes(url, payload, *, timeout):
            self.calls.append(f"POST {url}")
            assert payload == {"max_seq_len": 7}
            if isinstance(post_outcome, Exception):
                raise post_outcome
            return post_outcome

        async def fake_post(url, payload, action="post"):
            assert action == "delete"
            self.calls.append(f"DELETE {url}")
            if isinstance(delete_outcome, Exception):
                raise delete_outcome
            return {}

        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post_bytes_no_retry", fake_post_bytes)
        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)


@pytest.mark.asyncio
async def test_collect_samples_single_post_then_delete(monkeypatch):
    calls = _CollectCalls(monkeypatch, post_outcome=_computed_reply_payload())
    result = await _tracer().collect_samples(Sample(), max_seq_len=7)

    assert calls.calls == [
        "POST http://127.0.0.1:12345/sessions/sid-1/samples",
        "DELETE http://127.0.0.1:12345/sessions/sid-1",
    ]
    (sample,) = result.samples
    assert sample.tokens == [1, 2, 10] and sample.status == Sample.Status.COMPLETED
    assert result.session_metadata == {"max_trim_tokens": 1}


@pytest.mark.asyncio
async def test_collect_samples_non_2xx_raises_with_body_and_still_deletes(monkeypatch):
    calls = _CollectCalls(monkeypatch, post_outcome=RuntimeError("422: trim_count 2 exceeds allowed=1"))
    with pytest.raises(RuntimeError, match="trim_count 2 exceeds allowed=1"):
        await _tracer().collect_samples(Sample(), max_seq_len=7)
    assert calls.calls[-1] == "DELETE http://127.0.0.1:12345/sessions/sid-1"


@pytest.mark.asyncio
async def test_collect_samples_timeout_raises_and_still_deletes(monkeypatch):
    # The old collect_records swallowed the timeout and returned empty records
    # (silently ABORTing the sample); the samples path must raise it.
    calls = _CollectCalls(monkeypatch, post_outcome=asyncio.TimeoutError())
    with pytest.raises(asyncio.TimeoutError):
        await _tracer().collect_samples(Sample(), max_seq_len=7)
    assert calls.calls[-1] == "DELETE http://127.0.0.1:12345/sessions/sid-1"


@pytest.mark.asyncio
async def test_collect_samples_delete_failure_is_tolerated(monkeypatch):
    _CollectCalls(monkeypatch, post_outcome=_computed_reply_payload(), delete_outcome=RuntimeError("delete boom"))
    result = await _tracer().collect_samples(Sample(), max_seq_len=7)
    assert len(result.samples) == 1


# ── post_bytes_no_retry primitive ──


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self.content = content
        self.text = text


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.post_count = 0

    async def post(self, url, json=None):
        self.post_count += 1
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_post_bytes_no_retry_returns_raw_bytes(monkeypatch):
    client = _FakeClient([_FakeResponse(200, content=b"\x00\x01binary")])
    monkeypatch.setattr(http_utils, "_http_client", client)
    assert await post_bytes_no_retry("http://x/samples", {}, timeout=5) == b"\x00\x01binary"
    assert client.post_count == 1


@pytest.mark.asyncio
async def test_post_bytes_no_retry_does_not_retry_and_carries_body(monkeypatch):
    # Two queued outcomes; a retrying client would consume both. It must not.
    client = _FakeClient([_FakeResponse(422, text="cursor 3 != len(accumulated_token_ids) 4"), RuntimeError("late")])
    monkeypatch.setattr(http_utils, "_http_client", client)
    with pytest.raises(RuntimeError, match="422.*cursor 3"):
        await post_bytes_no_retry("http://x/samples", {}, timeout=5)
    assert client.post_count == 1


@pytest.mark.asyncio
async def test_post_bytes_no_retry_transport_error_propagates_once(monkeypatch):
    client = _FakeClient([ConnectionError("boom"), RuntimeError("late")])
    monkeypatch.setattr(http_utils, "_http_client", client)
    with pytest.raises(ConnectionError, match="boom"):
        await post_bytes_no_retry("http://x/samples", {}, timeout=5)
    assert client.post_count == 1


# ── v2 wire (--use-session-server v2): metadata channel + extended fields ──


@pytest.mark.asyncio
async def test_collect_samples_v2_payload_carries_metadata_and_decodes_extras(monkeypatch):
    """v2 pin: the collect body gains the "metadata" key only when the caller
    passes agent metadata, and the v2 field tuple overlays reward + merged
    metadata; the v1 pin above (`payload == {"max_seq_len": 7}`) stays."""
    from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2

    sample = Sample()
    sample.tokens = [1, 2, 10]
    sample.response = "r"
    sample.response_length = 1
    sample.loss_mask = [1]
    sample.rollout_log_probs = [-0.5]
    sample.status = Sample.Status.COMPLETED
    sample.reward = 0.75
    sample.metadata = {"leaf": {"node_id": 1}}
    payload = encode_samples([sample], {"max_trim_tokens": 1}, None, fields=COMPUTED_FIELDS_V2)

    seen = []

    async def fake_post_bytes(url, body, *, timeout):
        seen.append(body)
        return payload

    async def fake_post(url, body, action="post"):
        assert action == "delete"
        return {}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post_bytes_no_retry", fake_post_bytes)
    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

    tracer = OpenAIEndpointTracer(
        router_url="http://127.0.0.1:12345", session_id="sid-1", samples_wire_fields=COMPUTED_FIELDS_V2
    )
    input_sample = Sample()
    input_sample.metadata = {"env": "keep-me"}
    result = await tracer.collect_samples(input_sample, max_seq_len=7, agent_metadata={"reward": 0.75})

    assert seen == [{"max_seq_len": 7, "metadata": {"reward": 0.75}}]
    (decoded,) = result.samples
    assert decoded.reward == 0.75
    assert decoded.metadata == {"env": "keep-me", "leaf": {"node_id": 1}}


@pytest.mark.asyncio
async def test_create_selects_wire_fields_by_session_server_version(monkeypatch):
    async def fake_post(url: str, payload: dict, action: str = "post"):
        return {"session_id": "sid-x"}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

    def args(version):
        return SimpleNamespace(session_server_ip="127.0.0.1", session_server_ports=[7000], use_session_server=version)

    assert (await OpenAIEndpointTracer.create(args(True))).samples_wire_fields == COMPUTED_FIELDS
    assert (await OpenAIEndpointTracer.create(args("v2"))).samples_wire_fields == COMPUTED_FIELDS_V2
