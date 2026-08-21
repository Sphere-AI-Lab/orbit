"""The viewer attaches through --custom-rollout-log-function-path, not core edits."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from tests.fast.examples.model_response_trace_viewer.conftest import make_sample

from miles.ray.rollout.debug_data import save_debug_rollout_data
from miles.ray.rollout.metrics import log_rollout_data

HOOK_PATH = "examples.model_response_trace_viewer.hook.log_rollout_data"


def _args(**overrides):
    base = dict(
        custom_rollout_log_function_path=HOOK_PATH,
        save_model_response_log=None,
        save_model_response_trace_dir=None,
        model_response_trace_max_samples_per_step=8,
        save_debug_rollout_data=None,
        save_debug_trajectory_data=None,
        load_debug_rollout_data=None,
        log_passrate=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _run(args, samples, rollout_id=3):
    """Drive the real hook dispatch in metrics.log_rollout_data."""
    default_logged = {"ran": False}

    def mark(*a, **k):
        default_logged["ran"] = True
        return {}

    with patch("miles.ray.rollout.metrics._compute_metrics_from_samples", mark), patch(
        "miles.ray.rollout.metrics._compute_perf_metrics_from_samples", lambda *a, **k: {}
    ), patch("miles.ray.rollout.metrics._compute_distillation_rpc_metrics", lambda *a, **k: {}), patch(
        "miles.ray.rollout.metrics.compute_rollout_step", lambda *a, **k: 0
    ), patch(
        "miles.ray.rollout.metrics.tracking"
    ):
        log_rollout_data(rollout_id, args, samples, {}, 1.0)
    return default_logged["ran"]


def test_hook_writes_trace_steps_through_the_public_dispatch(tmp_path):
    trace_dir = tmp_path / "traces"
    samples = [make_sample(index=0, response="first"), make_sample(index=1, response="second")]

    _run(_args(save_model_response_trace_dir=str(trace_dir)), samples, rollout_id=12)

    step = trace_dir / "train" / "step0012"
    assert step.is_dir()
    indices = sorted(json.loads(p.read_text())["ids"]["sample_index"] for p in step.glob("*/record.json"))
    assert indices == [0, 1]


def test_hook_writes_the_compact_log_through_the_public_dispatch(tmp_path):
    template = str(tmp_path / "{rollout_id}.jsonl")

    _run(_args(save_model_response_log=template), [make_sample(index=0, response="only")], rollout_id=5)

    rows = [json.loads(line) for line in (tmp_path / "5.jsonl").read_text().splitlines()]
    assert [row["sample_index"] for row in rows] == [0]


def test_debug_dump_preserves_messages_for_the_following_hook(tmp_path):
    response_log = tmp_path / "responses_{rollout_id}.jsonl"
    args = _args(
        save_debug_rollout_data=str(tmp_path / "rollout_{rollout_id}.pt"),
        save_debug_trajectory_data=str(tmp_path / "trajectory_{rollout_id}.jsonl"),
        save_model_response_log=str(response_log),
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "tool", "content": "tool result", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "final answer"},
    ]
    sample = make_sample(index=0, response="aggregate", metadata={"messages": messages})

    save_debug_rollout_data(args, [sample], rollout_id=9, evaluation=False)
    _run(args, [sample], rollout_id=9)

    assert sample.metadata["messages"] == messages
    row = json.loads((tmp_path / "responses_9.jsonl").read_text(encoding="utf-8"))
    assert [(turn["role"], turn["content"]) for turn in row["turns"]] == [
        ("assistant", "first answer"),
        ("environment", "tool result"),
        ("assistant", "final answer"),
    ]


def test_hook_layers_on_top_of_default_metrics_logging(tmp_path):
    """Returning False must not suppress Miles' own rollout metrics."""
    ran = _run(_args(save_model_response_trace_dir=str(tmp_path / "traces")), [make_sample()])

    assert ran


def test_hook_is_inert_when_no_trace_flag_is_set(tmp_path):
    ran = _run(_args(), [make_sample()])

    assert ran
    assert not list(tmp_path.iterdir())
