import json

from examples.model_response_trace_viewer.verify_trace_run import _expected_from_args


def test_expected_record_count_respects_trace_cap(tmp_path):
    (tmp_path / "args.json").write_text(
        json.dumps(
            {
                "num-rollout": 3,
                "rollout-batch-size": 4,
                "n-samples-per-prompt": 4,
                "model-response-trace-max-samples-per-step": 8,
            }
        ),
        encoding="utf-8",
    )

    assert _expected_from_args(tmp_path) == (3, 8)
