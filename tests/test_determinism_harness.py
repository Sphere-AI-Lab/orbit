"""Unit tests for the rollout determinism harness (true-on-policy Phase 2).

The harness scores a fixed set of token sequences twice against an SGLang
server under different batch compositions and asserts the returned prefill
log-probs are byte-identical. These tests cover the pure logic: grouping
schemes, payload construction, and the comparison report.
"""

import pytest

from tools.rollout_determinism_harness import (
    build_scoring_payload,
    compare_logprob_sets,
    make_groupings,
)


def test_make_groupings_pass_schemes_cover_all_indices_exactly_once():
    for scheme in ("single-batch", "reversed-triples", "singletons"):
        groups = make_groupings(7, scheme)
        flat = [i for g in groups for i in g]
        assert sorted(flat) == list(range(7)), scheme


def test_make_groupings_schemes_differ_in_composition():
    a = make_groupings(7, "single-batch")
    b = make_groupings(7, "reversed-triples")
    c = make_groupings(7, "singletons")
    assert a != b and b != c and a != c


def test_make_groupings_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="Unknown grouping scheme"):
        make_groupings(4, "bogus")


def test_build_scoring_payload_scores_full_sequence():
    payload = build_scoring_payload([[1, 2, 3], [4, 5]])
    assert payload["input_ids"] == [[1, 2, 3], [4, 5]]
    assert payload["return_logprob"] is True
    assert payload["logprob_start_len"] == 0
    assert payload["sampling_params"]["max_new_tokens"] == 0
    assert payload["sampling_params"]["temperature"] == 0


def test_compare_identical_sets_pass():
    a = [[-0.5, -1.25], [-2.0]]
    identical, max_diff, n_mismatch = compare_logprob_sets(a, [list(x) for x in a])
    assert identical is True
    assert max_diff == 0.0
    assert n_mismatch == 0


def test_compare_detects_single_ulp_difference():
    a = [[-0.5, -1.25]]
    b = [[-0.5, -1.2500001]]
    identical, max_diff, n_mismatch = compare_logprob_sets(a, b)
    assert identical is False
    assert max_diff == pytest.approx(1e-7, rel=0.5)
    assert n_mismatch == 1


def test_compare_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_logprob_sets([[-0.5]], [[-0.5, -1.0]])
