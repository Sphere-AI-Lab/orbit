"""orbit/utils/metric_utils_patches.py: caller-chosen k, and a scale.

The delegation property has a sharp form here: every k upstream CAN compute must
come back with upstream's own number, so `k_values=[1, 4]` is a filter over
upstream's result rather than a recomputation. Only a k upstream has no spelling
for (anything that is not a power of two) is orbit's, and even that goes through
upstream's estimator.
"""

import numpy as np
import pytest

import orbit  # noqa: F401  -- importing orbit installs the patches
from miles.utils import metric_utils

# Two groups of four; one group solves once, the other twice.
_REWARDS = [1, 0, 0, 0, 1, 1, 0, 0]
_GROUP_SIZE = 4


def test_the_patch_is_actually_installed():
    assert metric_utils.compute_pass_rate.__module__ == "orbit.utils.metric_utils_patches"
    assert hasattr(metric_utils, "_orbit_unpatched_compute_pass_rate"), (
        "the pristine upstream function must be kept so the patch can delegate"
    )


def test_the_default_call_is_exactly_upstreams_result():
    """The delegation property: no k_values, no scale, nothing but upstream."""
    patched = metric_utils.compute_pass_rate(_REWARDS, _GROUP_SIZE)
    upstream = metric_utils._orbit_unpatched_compute_pass_rate(_REWARDS, _GROUP_SIZE)
    assert patched == upstream
    assert sorted(patched) == ["pass@1", "pass@2", "pass@4"]


def test_requested_powers_of_two_are_upstreams_own_numbers():
    """Not merely equal-valued: k_values narrows upstream's dict, it does not
    recompute it. If orbit ever grew its own copy of the estimator loop this is
    where the numbers would start drifting."""
    upstream = metric_utils._orbit_unpatched_compute_pass_rate(_REWARDS, _GROUP_SIZE)

    out = metric_utils.compute_pass_rate(_REWARDS, _GROUP_SIZE, k_values=[1, 4])

    assert sorted(out) == ["pass@1", "pass@4"]
    assert out["pass@1"] == upstream["pass@1"]
    assert out["pass@4"] == upstream["pass@4"]


def test_a_k_upstream_cannot_produce_is_computed_with_upstreams_estimator():
    out = metric_utils.compute_pass_rate(_REWARDS, _GROUP_SIZE, k_values=[3])

    assert sorted(out) == ["pass@3"]
    # ...and prove the patch is what did it: upstream's k set has no 3 in it.
    assert "pass@3" not in metric_utils._orbit_unpatched_compute_pass_rate(
        _REWARDS, _GROUP_SIZE
    )
    # The estimate itself is still upstream's function, not a reimplementation.
    expected = np.mean(
        metric_utils._estimate_pass_at_k(np.full(2, 4), np.array([1, 2]), 3)
    )
    assert out["pass@3"] == pytest.approx(expected)


def test_a_mixed_k_list_keeps_both_halves():
    out = metric_utils.compute_pass_rate(_REWARDS, _GROUP_SIZE, k_values=[2, 3])
    upstream = metric_utils._orbit_unpatched_compute_pass_rate(_REWARDS, _GROUP_SIZE)
    assert sorted(out) == ["pass@2", "pass@3"]
    assert out["pass@2"] == upstream["pass@2"]


def test_k_outside_the_group_is_dropped_and_an_empty_request_reports_nothing():
    assert metric_utils.compute_pass_rate(_REWARDS, _GROUP_SIZE, k_values=[9, 0]) == {}
    assert metric_utils.compute_pass_rate(_REWARDS, _GROUP_SIZE, k_values=[]) == {}
    # Duplicates and unsorted input collapse to one key each.
    assert sorted(
        metric_utils.compute_pass_rate(_REWARDS, _GROUP_SIZE, k_values=[4, 1, 4])
    ) == ["pass@1", "pass@4"]


def test_scale_multiplies_upstreams_values():
    upstream = metric_utils._orbit_unpatched_compute_pass_rate(_REWARDS, _GROUP_SIZE)

    out = metric_utils.compute_pass_rate(_REWARDS, _GROUP_SIZE, scale=100.0)

    assert out == pytest.approx({k: v * 100.0 for k, v in upstream.items()})


def test_scale_applies_to_a_requested_k_too():
    plain = metric_utils.compute_pass_rate(_REWARDS, _GROUP_SIZE, k_values=[3])
    scaled = metric_utils.compute_pass_rate(_REWARDS, _GROUP_SIZE, k_values=[3], scale=100.0)
    assert scaled["pass@3"] == pytest.approx(plain["pass@3"] * 100.0)


def test_a_group_of_one_reports_nothing_however_it_is_asked():
    """Upstream declines to estimate over a group of one; asking for explicit k
    does not create a group to estimate over."""
    assert metric_utils.compute_pass_rate([1, 0], 1) == {}
    assert metric_utils.compute_pass_rate([1, 0], 1, k_values=[1], scale=100.0) == {}


def test_an_explicit_num_groups_is_honoured_on_both_paths():
    half = _REWARDS[:4]
    assert metric_utils.compute_pass_rate(half, _GROUP_SIZE, num_groups=1, k_values=[3])[
        "pass@3"
    ] == pytest.approx(
        np.mean(metric_utils._estimate_pass_at_k(np.full(1, 4), np.array([1]), 3))
    )
    # A wrong num_groups still hits upstream's own assertion, not orbit's code.
    with pytest.raises(AssertionError):
        metric_utils.compute_pass_rate(half, _GROUP_SIZE, num_groups=5, k_values=[3])
