import json
from pathlib import Path
from types import SimpleNamespace

from examples.model_response_trace_viewer import response_trace as model_response_trace
from examples.model_response_trace_viewer.response_log import model_response_row
from examples.model_response_trace_viewer.response_trace import (
    model_response_trace_record,
    save_model_response_trace,
)
from PIL import Image
from tests.fast.examples.model_response_trace_viewer.conftest import make_sample


def _trace_args(path: Path | None, cap: int | None = None) -> SimpleNamespace:
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
        "metadata": {"producer": "miles"},
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


def test_save_model_response_trace_exports_every_sample_by_default(tmp_path):
    samples = [
        make_sample(group_index=7 if index < 3 else 8, index=index, response=f"r{index}") for index in range(10)
    ]

    save_model_response_trace(_trace_args(tmp_path / "traces"), samples, rollout_id=12)

    step = tmp_path / "traces" / "train" / "step0012"
    indices = sorted(json.loads(path.read_text())["ids"]["sample_index"] for path in step.glob("*/record.json"))
    assert indices == list(range(10))


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
