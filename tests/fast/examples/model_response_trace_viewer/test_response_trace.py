import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.geo3k_vlm.multi_turn import rollout as geo3k_rollout
from examples.model_response_trace_viewer import response_trace as model_response_trace
from examples.model_response_trace_viewer.response_log import RESPONSE_TURNS_KEY, model_response_row
from examples.model_response_trace_viewer.response_trace import (
    model_response_trace_record,
    save_model_response_trace,
)
from PIL import Image
from tests.fast.examples.model_response_trace_viewer.conftest import make_sample

from orbit.utils.types import Sample


def _trace_args(path: Path | None, cap: int | None = 8) -> SimpleNamespace:
    return SimpleNamespace(
        save_model_response_trace_dir=None if path is None else str(path),
        model_response_trace_max_samples_per_step=cap,
    )


def test_model_response_trace_record_maps_compact_geo3k_row_exactly():
    sample = make_sample(
        group_index=48,
        index=384,
        response_length=6,
        response="accepted aggregate response",
        reward=1.0,
        weight_versions=[18, 19],
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
                "content": "tool feedback",
                "model_input_role": "user",
            },
            {
                "turn": 2,
                "role": "assistant",
                "content": "final answer",
                "finish_reason": "stop",
                "weight_version": 19,
            },
        ],
        prompt="rendered geometry prompt",
    )

    record = model_response_trace_record(
        sample,
        rollout_id=12,
        group_index=48,
        image_count=1,
    )

    assert record == {
        "trace_schema_version": 1,
        "ids": {"step": 12, "group_index": 48, "sample_index": 384},
        "env": {"name": None, "seed": None, "max_turns": None, "config": {}},
        "outcome": {
            "status": "completed",
            "reward": 1.0,
            "num_turns": 2,
            "remove_sample": False,
        },
        "counts": {
            "response_length": 6,
            "n_images": 1,
            "n_messages": 4,
            "n_tools": 0,
        },
        "trajectory": {
            "prompt": "rendered geometry prompt",
            "response": "accepted aggregate response",
        },
        "conversation": {
            "source_format": "rendered_prompt_plus_turns",
            "tools": [],
            "messages": [
                {
                    "role": "prompt",
                    "content": [{"type": "text", "text": "rendered geometry prompt"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "first answer"}],
                    "finish_reason": "stop",
                    "weight_version": 18,
                },
                {
                    "role": "environment",
                    "content": [{"type": "text", "text": "tool feedback"}],
                    "model_input_role": "user",
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "final answer"}],
                    "finish_reason": "stop",
                    "weight_version": 19,
                },
            ],
        },
        "media": [
            {
                "id": "image-0",
                "type": "image",
                "path": "turn0_obs.png",
                "message_index": None,
                "content_index": None,
            }
        ],
        "metadata": {"producer": "orbit"},
    }


def test_model_response_trace_record_uses_sanitized_deterministic_text():
    sample = make_sample(
        prompt=[{"type": "image", "image": "raw-image-sentinel"}],
        response="accepted",
        response_turns=[
            {
                "turn": 1,
                "role": "environment",
                "content": {"z": 2, "a": [1, True]},
                "tool_call_id": "call-1",
            }
        ],
    )

    record = model_response_trace_record(
        sample,
        rollout_id=0,
        group_index=0,
        image_count=0,
    )

    messages = record["conversation"]["messages"]
    assert messages[0]["content"] == [{"type": "text", "text": '[{"omitted":true,"type":"image"}]'}]
    assert messages[1] == {
        "role": "environment",
        "content": [{"type": "text", "text": '{"a":[1,true],"z":2}'}],
        "tool_call_id": "call-1",
    }
    assert messages[2] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "accepted"}],
    }
    assert "raw-image-sentinel" not in json.dumps(record, allow_nan=False)
    assert record["trajectory"]["prompt"] is None


def test_model_response_trace_record_does_not_duplicate_aggregate_assistant():
    sample = make_sample(
        response="aggregate",
        response_turns=[{"turn": 1, "role": "assistant", "content": "exact turn"}],
    )

    record = model_response_trace_record(
        sample,
        rollout_id=3,
        group_index=7,
        image_count=2,
    )

    messages = record["conversation"]["messages"]
    assert [message["role"] for message in messages] == ["prompt", "assistant"]
    assert messages[-1]["content"][0]["text"] == "exact turn"
    assert [row["path"] for row in record["media"]] == [
        "turn0_obs.png",
        "turn1_obs.png",
    ]


def test_model_response_trace_record_uses_null_for_invalid_sample_index():
    sample = make_sample(index=True, response="accepted", response_turns=[])

    record = model_response_trace_record(
        sample,
        rollout_id=1,
        group_index=0,
        image_count=0,
    )

    assert record["ids"]["sample_index"] is None
    assert record["conversation"]["messages"][-1]["role"] == "assistant"


def test_model_response_trace_record_does_not_mutate_compact_row():
    sample = make_sample(
        metadata={"dataset": "geo3k"},
        response="accepted",
        response_turns=[{"turn": 1, "role": "assistant", "content": "accepted"}],
    )
    before = model_response_row(sample, rollout_id=2)

    record = model_response_trace_record(
        sample,
        rollout_id=2,
        group_index=0,
        image_count=0,
    )

    assert model_response_row(sample, rollout_id=2) == before
    assert record["env"]["name"] is None


def test_save_model_response_trace_caps_samples_and_assigns_ordinals(tmp_path):
    samples = [
        make_sample(group_index=7 if index < 3 else 8, index=index, response=f"r{index}") for index in range(10)
    ]

    save_model_response_trace(_trace_args(tmp_path / "traces", cap=8), samples, rollout_id=12)

    step = tmp_path / "traces" / "train" / "step0012"
    names = sorted(path.name for path in step.iterdir())
    assert names == [
        "prompt00007_rollout00",
        "prompt00007_rollout01",
        "prompt00007_rollout02",
        "prompt00008_rollout00",
        "prompt00008_rollout01",
        "prompt00008_rollout02",
        "prompt00008_rollout03",
        "prompt00008_rollout04",
    ]
    indices = sorted(json.loads(path.read_text())["ids"]["sample_index"] for path in step.glob("*/record.json"))
    assert indices == list(range(8))
    assert not list(step.parent.glob(".step0012.*"))


def test_save_model_response_trace_exports_every_sample_when_cap_exceeds_batch(tmp_path):
    samples = [
        make_sample(group_index=7 if index < 3 else 8, index=index, response=f"r{index}") for index in range(10)
    ]

    save_model_response_trace(_trace_args(tmp_path / "traces", cap=20), samples, rollout_id=12)

    step = tmp_path / "traces" / "train" / "step0012"
    indices = sorted(json.loads(path.read_text())["ids"]["sample_index"] for path in step.glob("*/record.json"))
    assert indices == list(range(10))


def test_save_model_response_trace_fails_closed_without_a_cap(tmp_path, caplog):
    trace_dir = tmp_path / "traces"

    save_model_response_trace(_trace_args(trace_dir, cap=None), [make_sample()], rollout_id=12)

    assert not trace_dir.exists()
    assert "Failed to save model response trace" in caplog.text


def test_save_model_response_trace_writes_metadata_free_rgba_png(tmp_path):
    source = Image.new("RGB", (2, 1), (10, 20, 30))
    source.info["secret"] = "do-not-copy"
    sample = make_sample(multimodal_inputs={"images": [source]})

    save_model_response_trace(_trace_args(tmp_path / "traces"), [sample], rollout_id=0)

    image_path = tmp_path / "traces" / "train" / "step0000" / "prompt00000_rollout00" / "turn0_obs.png"
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(image_path) as saved:
        assert saved.mode == "RGBA"
        assert saved.size == (2, 1)
        assert list(saved.getdata()) == [(10, 20, 30, 255), (10, 20, 30, 255)]
        assert "secret" not in saved.info


def test_save_model_response_trace_never_acquires_media_from_content(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Image,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("source media must not be opened")),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("URLs must not be fetched")),
    )
    monkeypatch.setattr(
        "base64.b64decode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("base64 must not be decoded")),
    )
    sample = make_sample(
        prompt="https://example.invalid/private.png data:image/png;base64,SECRET",
        multimodal_inputs=None,
    )

    save_model_response_trace(_trace_args(tmp_path / "traces"), [sample], rollout_id=0)

    assert (tmp_path / "traces" / "train" / "step0000" / "prompt00000_rollout00" / "record.json").is_file()


def test_save_model_response_trace_record_failure_is_nonfatal_and_removes_owned_stage(tmp_path, monkeypatch, caplog):
    def fail_record(*_args, **_kwargs):
        raise TypeError("synthetic record failure")

    monkeypatch.setattr(model_response_trace, "model_response_trace_record", fail_record)

    save_model_response_trace(
        _trace_args(tmp_path / "traces"),
        [make_sample()],
        rollout_id=4,
    )

    train = tmp_path / "traces" / "train"
    assert not (train / "step0004").exists()
    assert not list(train.glob(".step0004.*"))
    assert "rollout 4" in caplog.text
    assert "step0004" in caplog.text


def test_save_model_response_trace_png_write_failure_is_nonfatal_and_removes_partial_stage(
    tmp_path, monkeypatch, caplog
):
    def fail_png(path, _image):
        path.write_bytes(b"partial-png")
        raise OSError("synthetic PNG failure")

    monkeypatch.setattr(model_response_trace, "_write_png", fail_png)
    sample = make_sample(multimodal_inputs={"images": [Image.new("RGB", (1, 1), "red")]})

    save_model_response_trace(
        _trace_args(tmp_path / "traces"),
        [sample],
        rollout_id=5,
    )

    train = tmp_path / "traces" / "train"
    assert not (train / "step0005").exists()
    assert not list(train.glob(".step0005.*"))
    assert "rollout 5" in caplog.text
    assert "step0005" in caplog.text


def test_save_model_response_trace_rejects_malformed_images_as_one_step(tmp_path):
    valid = make_sample(index=0)
    malformed = make_sample(index=1, multimodal_inputs={"images": ["/secret.png"]})

    save_model_response_trace(
        _trace_args(tmp_path / "traces"),
        [valid, malformed],
        rollout_id=6,
    )

    train = tmp_path / "traces" / "train"
    assert not (train / "step0006").exists()
    assert not list(train.glob(".step0006.*"))


def test_save_model_response_trace_preserves_existing_step(tmp_path):
    step = tmp_path / "traces" / "train" / "step0007"
    step.mkdir(parents=True)
    sentinel = step / "sentinel.txt"
    sentinel.write_text("original", encoding="utf-8")

    save_model_response_trace(
        _trace_args(tmp_path / "traces"),
        [make_sample(response="replacement")],
        rollout_id=7,
    )

    assert sentinel.read_text(encoding="utf-8") == "original"
    assert list(step.iterdir()) == [sentinel]


def test_save_model_response_trace_disabled_or_empty_does_no_filesystem_work(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_response_trace.tempfile,
        "mkdtemp",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("staging must not be created")),
    )

    save_model_response_trace(_trace_args(None), [make_sample()], rollout_id=0)
    save_model_response_trace(_trace_args(tmp_path / "traces"), [], rollout_id=0)

    assert not (tmp_path / "traces").exists()


def test_retry_trace_contains_prompt_and_fresh_images_but_not_stale_image(tmp_path):
    prompt_image = Image.new("RGB", (1, 1), "red")
    stale_image = Image.new("RGB", (1, 1), "green")
    fresh_image = Image.new("RGB", (1, 1), "blue")
    sample = make_sample(multimodal_inputs={"images": [prompt_image]})

    sample.capture_multimodal_inputs_for_retry()
    sample.multimodal_inputs["images"].append(stale_image)
    sample.reset_for_retry()
    sample.multimodal_inputs["images"].append(fresh_image)
    sample.status = Sample.Status.COMPLETED
    sample.tokens = [1]
    sample.response = "accepted retry"
    sample.response_length = 1
    sample.metadata["response_turns"] = [{"turn": 1, "role": "assistant", "content": "accepted retry"}]

    save_model_response_trace(_trace_args(tmp_path / "traces"), [sample], rollout_id=0)

    rollout = tmp_path / "traces" / "train" / "step0000" / "prompt00000_rollout00"
    with Image.open(rollout / "turn0_obs.png") as first:
        assert first.getpixel((0, 0)) == (255, 0, 0, 255)
    with Image.open(rollout / "turn1_obs.png") as second:
        assert second.getpixel((0, 0)) == (0, 0, 255, 255)
    assert not (rollout / "turn2_obs.png").exists()


@pytest.mark.asyncio
async def test_geo3k_abort_reset_retry_trace_uses_only_pristine_and_retry_state(tmp_path, monkeypatch):
    prompt_image = Image.new("RGB", (1, 1), "red")
    stale_image = Image.new("RGB", (1, 1), "green")
    fresh_image = Image.new("RGB", (1, 1), "blue")

    class FakeTokenizer:
        bos_token_id = None

        def encode(self, _text, *, add_special_tokens):
            assert add_special_tokens is False
            return [7, 8, 9]

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return "decoded:" + ",".join(str(token_id) for token_id in token_ids)

    class FakeEnv:
        def reset(self):
            return None

        def close(self):
            return None

    state = SimpleNamespace(tokenizer=FakeTokenizer(), processor=None)
    inference_steps = iter(
        [
            ("stale first", [11], [-0.1], "stop", "stale-v1"),
            ("stale aborted", [12], [-0.2], "abort", "stale-v2"),
            ("retry first", [21], [-0.3], "stop", "retry-v1"),
            ("retry final", [22], [-0.4], "stop", "retry-v2"),
        ]
    )
    environment_steps = iter(
        [
            (
                [31],
                [31],
                [stale_image],
                {"images": [stale_image]},
                None,
                {"role": "environment", "content": "stale observation"},
                False,
            ),
            (
                [41],
                [41],
                [fresh_image],
                {"images": [fresh_image]},
                None,
                {"role": "environment", "content": "fresh observation"},
                False,
            ),
            (None, None, None, None, None, None, True),
        ]
    )

    monkeypatch.setattr(
        geo3k_rollout,
        "_initialize_resources",
        lambda _args, _sample: (FakeEnv(), None, {"max_turns": 3}, state, "http://rollout/generate"),
    )
    monkeypatch.setattr(geo3k_rollout, "encode_image_for_rollout_engine", lambda image: image)

    async def fake_run_inference_step(_url, tokens, *_args, **_kwargs):
        response, token_ids, log_probs, finish_reason, weight_version = next(inference_steps)
        return (
            response,
            token_ids,
            log_probs,
            finish_reason,
            {
                "prompt_tokens": len(tokens),
                "weight_version": weight_version,
            },
        )

    monkeypatch.setattr(geo3k_rollout, "_run_inference_step", fake_run_inference_step)
    monkeypatch.setattr(
        geo3k_rollout,
        "_process_env_step",
        lambda *_args, **_kwargs: next(environment_steps),
    )

    trace_dir = tmp_path / "traces"
    args = SimpleNamespace(
        partial_rollout=False,
        rollout_max_context_len=None,
        save_model_response_log=None,
        save_model_response_trace_dir=str(trace_dir),
    )
    sample = Sample(
        prompt="rendered Geo3K prompt",
        multimodal_inputs={"images": [prompt_image]},
        metadata=None,
    )

    first_attempt = await geo3k_rollout.generate(args, sample, {"max_new_tokens": 20})
    assert first_attempt.status is Sample.Status.ABORTED
    assert first_attempt.multimodal_inputs["images"] == [prompt_image, stale_image]
    assert "stale aborted" in json.dumps(first_attempt.metadata[RESPONSE_TURNS_KEY])

    sample.reset_for_retry()
    retry = await geo3k_rollout.generate(args, sample, {"max_new_tokens": 20})
    assert retry.status is Sample.Status.COMPLETED
    assert retry.multimodal_inputs["images"] == [prompt_image, fresh_image]

    save_model_response_trace(_trace_args(trace_dir), [retry], rollout_id=0)

    rollout = trace_dir / "train" / "step0000" / "prompt00000_rollout00"
    with Image.open(rollout / "turn0_obs.png") as first:
        assert first.getpixel((0, 0)) == (255, 0, 0, 255)
    with Image.open(rollout / "turn1_obs.png") as second:
        assert second.getpixel((0, 0)) == (0, 0, 255, 255)
    assert not (rollout / "turn2_obs.png").exists()

    record = json.loads((rollout / "record.json").read_text(encoding="utf-8"))
    serialized = json.dumps(record)
    assert "retry first" in serialized
    assert "fresh observation" in serialized
    assert "retry final" in serialized
    assert "stale first" not in serialized
    assert "stale observation" not in serialized
    assert "stale aborted" not in serialized
