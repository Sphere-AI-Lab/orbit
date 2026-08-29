"""Unit tests for orbit.peft.utils.logprob_compare (teacher-equivalence harness leg).

The comparison utility is shared between these CPU tests and the future
GPU/SGLang leg of the teacher-logprob equivalence harness (see the runbook in
tests/fast/test_opd_teacher_equivalence.py), so its semantics are pinned here:
exactness, small diffs with index, explicit length-mismatch errors, empty
sequences, NaN poisoning, and dict/aggregate helpers.
"""

import math

import pytest
import torch

from orbit.peft.utils.logprob_compare import (
    LogprobCompareReport,
    compare_logprob_dicts,
    compare_logprobs,
    summarize_reports,
)


def test_exact_match_lists():
    report = compare_logprobs([-0.5, -1.25, -2.0], [-0.5, -1.25, -2.0])
    assert report == LogprobCompareReport(count=3, max_abs_diff=0.0, mean_abs_diff=0.0, max_abs_diff_index=0)
    assert report.within(0.0)


def test_exact_match_mixed_list_and_1d_tensor():
    # The GPU leg compares engine-side lists against trainer-side tensors.
    report = compare_logprobs([-0.5, -1.25], torch.tensor([-0.5, -1.25], dtype=torch.float32))
    assert report.count == 2
    assert report.max_abs_diff == 0.0
    assert report.mean_abs_diff == 0.0


def test_small_diff_reports_exact_stats_and_index():
    # Diffs 0.0, 0.25, 0.5 are exactly representable: stats are exact, not approximate.
    report = compare_logprobs([-1.0, -2.0, -3.0], [-1.0, -2.25, -3.5])
    assert report.count == 3
    assert report.max_abs_diff == 0.5
    assert report.mean_abs_diff == 0.25
    assert report.max_abs_diff_index == 2
    assert report.within(0.5)
    assert not report.within(0.49)


def test_length_mismatch_is_an_explicit_error():
    with pytest.raises(ValueError, match="length mismatch.*3.*2"):
        compare_logprobs([-1.0, -2.0, -3.0], [-1.0, -2.0])


def test_empty_inputs_compare_as_empty_report():
    report = compare_logprobs([], [])
    assert report == LogprobCompareReport(count=0, max_abs_diff=0.0, mean_abs_diff=0.0, max_abs_diff_index=None)
    assert report.within(0.0)


def test_non_scalar_elements_rejected():
    # A 2-D tensor iterates into multi-element rows: per-token means 1-D, so reject.
    with pytest.raises(TypeError, match="candidate\\[0\\]"):
        compare_logprobs([-1.0], torch.zeros(1, 2))


def test_nan_poisons_report_and_fails_every_tolerance():
    report = compare_logprobs([-1.0, float("nan"), -3.0], [-1.0, -2.0, -3.0])
    assert report.count == 3
    assert math.isnan(report.max_abs_diff)
    assert math.isnan(report.mean_abs_diff)
    assert report.max_abs_diff_index == 1
    assert not report.within(math.inf)


def test_dict_compare_reports_per_key():
    reports = compare_logprob_dicts(
        {"sample0": [-1.0, -2.0], "sample1": [-3.0]},
        {"sample0": [-1.0, -2.5], "sample1": [-3.0]},
    )
    assert set(reports) == {"sample0", "sample1"}
    assert reports["sample0"].max_abs_diff == 0.5
    assert reports["sample1"].max_abs_diff == 0.0


def test_dict_key_mismatch_is_an_explicit_error():
    with pytest.raises(ValueError, match="missing.*'a'.*unexpected.*'b'"):
        compare_logprob_dicts({"a": [-1.0]}, {"b": [-1.0]})


def test_dict_compare_empty_dicts():
    assert compare_logprob_dicts({}, {}) == {}


def test_summarize_pools_token_counts_and_stats():
    reports = compare_logprob_dicts(
        {"sample0": [-1.0, -2.0], "sample1": [-3.0, -4.0]},
        {"sample0": [-1.0, -2.5], "sample1": [-3.0, -4.25]},
    )
    summary = summarize_reports(reports.values())
    assert summary.count == 4
    assert summary.max_abs_diff == 0.5
    assert summary.mean_abs_diff == 0.1875  # (0 + 0.5 + 0 + 0.25) / 4, exactly representable
    assert summary.max_abs_diff_index is None


def test_summarize_empty_iterable():
    summary = summarize_reports([])
    assert summary == LogprobCompareReport(count=0, max_abs_diff=0.0, mean_abs_diff=0.0, max_abs_diff_index=None)
