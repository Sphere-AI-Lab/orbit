from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts/experiments/OPD/optimize/summarize_03p_trace.py"
SPEC = importlib.util.spec_from_file_location("summarize_03p_trace", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
summarizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarizer)


def _event(name: str, ts: int, dur: int, *, category: str = "user_annotation") -> dict:
    return {"name": name, "ts": ts, "dur": dur, "ph": "X", "cat": category}


def _trace_events(*, include_c10d: bool = True) -> list[dict]:
    forward_name = "opd_dagger/stable_tp/operator_forward|rows=32|vlocal=64|k=2"
    annotations = [
        _event("ProfilerStep#2", 0, 10_000),
        _event(forward_name, 100, 4_000),
        _event("opd_dagger/stable_tp/full_local_lse", 150, 100),
        _event("opd_dagger/stable_tp/full_tp_lse", 300, 1_000),
        _event("opd_dagger/stable_tp/rest_local_lse", 1_400, 200),
        _event("opd_dagger/stable_tp/rest_tp_lse", 1_700, 1_000),
        _event("opd_dagger/stable_tp/selected_tp_sum", 3_000, 500),
        _event("opd_dagger/stable_tp/operator_backward", 5_000, 800),
    ]
    gpu_duplicates = [
        {**event, "ts": event["ts"] + 10, "dur": max(event["dur"] - 20, 1), "cat": "gpu_user_annotation"}
        for event in annotations
    ]
    events = annotations + gpu_duplicates
    if include_c10d:
        events.extend(
            [
                _event("c10d::allreduce_", 350, 40, category="cpu_op"),
                _event("c10d::allreduce_", 650, 40, category="cpu_op"),
                _event("c10d::allreduce_", 1_750, 40, category="cpu_op"),
                _event("c10d::allreduce_", 2_050, 40, category="cpu_op"),
                _event("c10d::allreduce_", 3_050, 40, category="cpu_op"),
            ]
        )
    events.extend(
        [
            # These nested/wrapper signals are diagnostic, not one-per-call c10d evidence.
            _event("torch.distributed.all_reduce", 400, 20, category="cpu_op"),
            _event("ncclKernel_AllReduce", 420, 30, category="kernel"),
            {
                "name": "[memory]",
                "ts": 50,
                "args": {"Total Allocated": 1_048_576, "Total Reserved": 2_097_152},
            },
            {
                "name": "[memory]",
                "ts": 200,
                "args": {"Total Allocated": 2_097_152, "Total Reserved": 3_145_728},
            },
        ]
    )
    return events


def test_cpu_annotations_define_logical_rows_and_observed_collectives() -> None:
    rows = summarizer.summarize_events(_trace_events(), Path("rank_0.pt.trace.json"))

    assert len(rows) == 1
    row = rows[0]
    assert row["profiler_step"] == "2"
    assert row["operator_forward_calls"] == 1
    assert row["operator_forward_ms"] == 4.0
    assert row["response_rows_total"] == 32
    assert row["opd_gpu_annotation_calls"] == 7
    assert row["expected_opd_tp_collectives"] == 5
    assert row["opd_collective_api_calls"] == 5
    assert row["full_tp_lse_collective_api_calls"] == 2
    assert row["rest_tp_lse_collective_api_calls"] == 2
    assert row["selected_tp_sum_collective_api_calls"] == 1
    assert summarizer._validate_rows(rows) == []


def test_strict_validation_rejects_phase_markers_without_observed_c10d_calls() -> None:
    rows = summarizer.summarize_events(
        _trace_events(include_c10d=False),
        Path("rank_0.pt.trace.json"),
    )

    assert rows[0]["expected_opd_tp_collectives"] == 5
    assert rows[0]["opd_collective_api_calls"] == 0
    errors = summarizer._validate_rows(rows)
    assert any("opd_collective_api_calls=0" in error for error in errors)
    assert any("full_tp_lse_collective_api_calls=0" in error for error in errors)
    assert any("rest_tp_lse_collective_api_calls=0" in error for error in errors)
    assert any("selected_tp_sum_collective_api_calls=0" in error for error in errors)
