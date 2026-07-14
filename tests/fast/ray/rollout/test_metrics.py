from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout import metrics as rollout_metrics
from miles.ray.rollout.metrics import (
    _compute_metrics_from_samples,
    _compute_opd_scoring_metrics,
    _compute_zero_std_metrics,
)
from miles.utils.types import Sample


class TestOpdScoringMetrics:
    def test_no_telemetry_emits_no_metrics(self):
        assert _compute_opd_scoring_metrics([Sample()]) == {}

    def test_aggregates_counts_bytes_and_latency_percentiles(self):
        samples = [
            Sample(
                metadata={
                    "opd_scoring_telemetry": [
                        {
                            "target": "teacher",
                            "attempts": 1,
                            "input_tokens": 100,
                            "response_tokens": 40,
                            "requested_token_ids": 0,
                            "request_body_bytes": 500,
                            "response_body_bytes": 1_000,
                            "returned_positions": 100,
                            "e2e_latency_s": 1.0,
                            "http_s": 0.8,
                            "semaphore_wait_s": 0.1,
                            "body_read_s": 0.2,
                            "json_decode_s": 0.05,
                        }
                    ]
                }
            ),
            Sample(
                metadata={
                    "opd_scoring_telemetry": [
                        {
                            "target": "student",
                            "attempts": 2,
                            "input_tokens": 200,
                            "response_tokens": 80,
                            "requested_token_ids": 3,
                            "request_body_bytes": None,
                            "response_body_bytes": 3_000,
                            "returned_positions": 200,
                            "e2e_latency_s": 3.0,
                            "http_s": 2.0,
                            "semaphore_wait_s": 0.5,
                            "body_read_s": 0.6,
                            "json_decode_s": 0.15,
                        }
                    ]
                }
            ),
        ]

        metrics = _compute_opd_scoring_metrics(samples)

        assert metrics["opd_scoring/sample_count"] == 2
        assert metrics["opd_scoring/request_count"] == 2
        assert metrics["opd_scoring/retry_count"] == 1
        assert metrics["opd_scoring/teacher_request_count"] == 1
        assert metrics["opd_scoring/student_request_count"] == 1
        assert metrics["opd_scoring/input_tokens_total"] == 300
        assert metrics["opd_scoring/response_tokens_total"] == 120
        assert metrics["opd_scoring/requested_token_ids_total"] == 3
        assert metrics["opd_scoring/request_body_bytes_total"] == 500
        assert metrics["opd_scoring/request_body_bytes_coverage"] == 0.5
        assert metrics["opd_scoring/response_body_bytes_total"] == 4_000
        assert metrics["opd_scoring/returned_positions_total"] == 300
        assert metrics["opd_scoring/response_bytes_per_returned_position"] == pytest.approx(4_000 / 300)
        assert metrics["opd_scoring/response_bytes_per_response_token"] == pytest.approx(4_000 / 120)
        assert metrics["opd_scoring/returned_positions_per_response_token"] == pytest.approx(300 / 120)
        assert metrics["opd_scoring/e2e_latency_s/mean"] == 2.0
        assert metrics["opd_scoring/e2e_latency_s/p50"] == 2.0
        assert metrics["opd_scoring/e2e_latency_s/p95"] == pytest.approx(2.9)
        assert metrics["opd_scoring/e2e_latency_s/max"] == 3.0

    def test_wandb_keys_use_a_separate_top_level_section(self, monkeypatch):
        sample = make_samples_grouped(1, 1)[0]
        sample.metadata["opd_scoring_telemetry"] = [{"target": "teacher", "attempts": 1}]
        logged = {}

        def capture_log(args, metrics, step_key):
            logged.update(metrics)

        monkeypatch.setattr(rollout_metrics.tracking_utils, "log", capture_log)
        args = make_args(wandb_always_use_train_step=False)

        rollout_metrics.log_rollout_data(0, args, [sample], rollout_extra_metrics=None, rollout_time=1.0)

        assert logged["opd_scoring/request_count"] == 1
        assert not any(key.startswith("rollout/opd_scoring/") for key in logged)
        assert "rollout/response_len/mean" in logged
        assert "perf/rollout_time" in logged


class TestComputeZeroStdMetrics:
    def test_returns_empty_for_ppo_regardless_of_reward_distribution(self):
        args = make_args(advantage_estimator="ppo")
        out = _compute_zero_std_metrics(args, make_samples_grouped(2, 4, rewards=[1.0] * 8))
        assert out == {}

    def test_grpo_mixed_rewards_yield_zero_percentages_and_no_buckets(self):
        """Happy path: every group has reward variation → no group is zero-std →
        no bucket counts; the all_zero/all_one percentages are 0."""
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[0.0, 0.5, 1.0, 0.7, 0.2, 0.8, 0.3, 0.6])
        out = _compute_zero_std_metrics(args, samples)
        assert out == {"zero_std/all_zero_percentage": 0.0, "zero_std/all_one_percentage": 0.0}

    def test_grpo_zero_std_groups_produce_bucket_counts_and_percentages(self):
        """1 group all-1, 1 group all-0, 1 group mixed → bucket counts plus the
        all_zero/all_one percentages over total groups."""
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(3, 4, rewards=[1.0] * 4 + [0.0] * 4 + [0.0, 1.0, 0.0, 1.0])
        out = _compute_zero_std_metrics(args, samples)
        assert out["zero_std/count_1"] == 1
        assert out["zero_std/count_0"] == 1
        assert out["zero_std/all_zero_percentage"] == pytest.approx(1 / 3)
        assert out["zero_std/all_one_percentage"] == pytest.approx(1 / 3)

    def test_grpo_uniform_non_binary_reward_gets_its_own_bucket(self):
        """Every group zero-std at reward=0.5 → bucket count_0.5=2, but
        all_zero/all_one percentages stay 0 because they only count 0.0 and 1.0."""
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[0.5] * 8)
        out = _compute_zero_std_metrics(args, samples)
        assert out["zero_std/count_0.5"] == 2
        assert out["zero_std/all_zero_percentage"] == 0.0
        assert out["zero_std/all_one_percentage"] == 0.0

    def test_empty_samples_does_not_crash(self):
        args = make_args(advantage_estimator="grpo", reward_key=None)
        out = _compute_zero_std_metrics(args, [])
        # No groups → no all_zero/all_one keys (the function guards on total_groups>0).
        assert "zero_std/all_zero_percentage" not in out
        assert "zero_std/all_one_percentage" not in out


class TestTitoMismatchMetrics:
    def test_no_tito_metadata_emits_no_tito_keys(self):
        args = make_args(advantage_estimator="ppo", ci_test=False, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        out = _compute_metrics_from_samples(args, samples)
        assert "tito_session_mismatch_rate" not in out

    def test_clean_tito_metadata_yields_zero_rates_per_mismatch_type(self):
        args = make_args(advantage_estimator="ppo", ci_test=False, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        for s in samples:
            s.metadata = {"tito_session_mismatch": []}
        out = _compute_metrics_from_samples(args, samples)
        assert out["tito_session_mismatch_rate"] == 0.0
        for mtype in ("special_token_count", "special_token_type", "non_assistant_text", "assistant_text"):
            assert out[f"tito_session_mismatch_rate/{mtype}"] == 0.0

    def test_strict_mismatch_raises_under_ci_test(self):
        """Under ci_test=True, a non-zero rate on the strict mismatch types
        (special_token_count / special_token_type / non_assistant_text) must
        hard-fail — these signal a TITO algorithm or chat-template bug."""
        args = make_args(advantage_estimator="ppo", ci_test=True, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {"tito_session_mismatch": [{"type": "special_token_count"}]}
        for s in samples[1:]:
            s.metadata = {"tito_session_mismatch": []}
        with pytest.raises(AssertionError, match="special_token_count"):
            _compute_metrics_from_samples(args, samples)

    def test_assistant_text_mismatch_does_not_raise_under_ci_test(self):
        """assistant_text mismatch is non-critical (tokens inherited from the
        pretokenized prefix) — even under ci_test, must not raise."""
        args = make_args(advantage_estimator="ppo", ci_test=True, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {"tito_session_mismatch": [{"type": "assistant_text"}]}
        for s in samples[1:]:
            s.metadata = {"tito_session_mismatch": []}
        out = _compute_metrics_from_samples(args, samples)
        assert out["tito_session_mismatch_rate/assistant_text"] > 0
