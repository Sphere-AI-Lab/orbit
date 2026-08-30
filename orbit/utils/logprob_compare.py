"""Per-token logprob comparison for the teacher-equivalence harness.

Pure module (stdlib only) by design, mirroring opd_teacher_spec: inputs are
duck-typed float sequences (list[float], tuple, 1-D torch tensor, numpy
array), so the CPU equivalence tests, the trainer, and engine-side tooling
can all import it without torch/megatron/sglang.

This is the shared measurement leg of the teacher-logprob equivalence
harness. The CPU leg (tests/fast/test_opd_teacher_equivalence.py) uses it to
pin trainer-side teacher_forward_plan equivalences; the future GPU/SGLang leg
reuses the same functions to compare trainer-side teacher scoring against
engine-side teacher-forcing prefill (base weights vs the reserved
orbit_teacher adapter slot), gated on LogprobCompareReport.within(atol).

Semantics pinned by tests/fast/test_logprob_compare.py:
  * length mismatch and non-scalar elements are explicit errors, never
    silently truncated;
  * empty sequences are legal (a zero-length response) and yield count=0 —
    callers gate on count when emptiness would mask a broken harness;
  * any NaN poisons the report deterministically (NaN stats fail every
    within() tolerance).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class LogprobCompareReport:
    """Summary of an elementwise |reference - candidate| comparison.

    max_abs_diff_index is the position of the largest diff (the first NaN
    position when the report is NaN-poisoned), or None when count == 0 or the
    report pools multiple sequences.
    """

    count: int
    max_abs_diff: float
    mean_abs_diff: float
    max_abs_diff_index: int | None

    def within(self, atol: float) -> bool:
        """True when every compared token differs by at most atol (NaN never passes)."""
        return self.max_abs_diff <= atol


def _as_float_list(values: Iterable[object], label: str) -> list[float]:
    if isinstance(values, (str, bytes)) or isinstance(values, Mapping):
        raise TypeError(f"{label} must be a sequence of per-token logprobs, got {type(values).__name__}")
    out: list[float] = []
    for index, value in enumerate(values):
        try:
            out.append(float(value))  # accepts python floats, numpy scalars, 0-d tensor elements
        except (TypeError, ValueError, RuntimeError) as exc:  # RuntimeError: multi-element tensor
            raise TypeError(f"{label}[{index}] is not a scalar logprob: {value!r}") from exc
    return out


def compare_logprobs(reference: Iterable[object], candidate: Iterable[object]) -> LogprobCompareReport:
    """Compare two per-token logprob sequences elementwise.

    Raises ValueError on length mismatch (an explicit error: truncation would
    silently hide missing tokens) and TypeError on non-scalar elements.
    """
    ref = _as_float_list(reference, "reference")
    cand = _as_float_list(candidate, "candidate")
    if len(ref) != len(cand):
        raise ValueError(
            f"per-token logprob length mismatch: reference has {len(ref)} tokens, candidate has {len(cand)}"
        )
    if not ref:
        return LogprobCompareReport(count=0, max_abs_diff=0.0, mean_abs_diff=0.0, max_abs_diff_index=None)

    diffs = [abs(r - c) for r, c in zip(ref, cand, strict=True)]
    for index, diff in enumerate(diffs):
        if math.isnan(diff):
            # Deterministic poisoning: python max() is order-dependent with NaN.
            return LogprobCompareReport(
                count=len(diffs), max_abs_diff=math.nan, mean_abs_diff=math.nan, max_abs_diff_index=index
            )
    max_index = max(range(len(diffs)), key=diffs.__getitem__)
    return LogprobCompareReport(
        count=len(diffs),
        max_abs_diff=diffs[max_index],
        mean_abs_diff=math.fsum(diffs) / len(diffs),
        max_abs_diff_index=max_index,
    )


def compare_logprob_dicts(
    reference: Mapping[str, Iterable[object]], candidate: Mapping[str, Iterable[object]]
) -> dict[str, LogprobCompareReport]:
    """Compare two keyed collections of per-token logprob sequences.

    Keys typically identify samples (or named outputs like "teacher_log_probs").
    Key-set mismatch is an explicit error.
    """
    ref_keys = set(reference)
    cand_keys = set(candidate)
    if ref_keys != cand_keys:
        missing = sorted(ref_keys - cand_keys)
        extra = sorted(cand_keys - ref_keys)
        raise ValueError(f"logprob dict keys differ: missing from candidate {missing}, unexpected in candidate {extra}")
    return {key: compare_logprobs(reference[key], candidate[key]) for key in sorted(ref_keys)}


def summarize_reports(reports: Iterable[LogprobCompareReport]) -> LogprobCompareReport:
    """Pool per-sequence reports into one batch-level report (token-weighted mean)."""
    total = 0
    max_abs_diff = 0.0
    abs_diff_sum = 0.0
    for report in reports:
        total += report.count
        if math.isnan(report.max_abs_diff) or report.max_abs_diff > max_abs_diff:
            max_abs_diff = report.max_abs_diff
        abs_diff_sum += report.mean_abs_diff * report.count
    if total == 0:
        return LogprobCompareReport(count=0, max_abs_diff=0.0, mean_abs_diff=0.0, max_abs_diff_index=None)
    return LogprobCompareReport(
        count=total, max_abs_diff=max_abs_diff, mean_abs_diff=abs_diff_sum / total, max_abs_diff_index=None
    )
