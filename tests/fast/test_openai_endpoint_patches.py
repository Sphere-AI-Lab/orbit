"""The OpenAI-endpoint path records orbit's NORMALIZED policy version.

Upstream appends the raw ``meta_info["weight_version"]``. Orbit's true-on-policy
contract needs ``_extract_policy_version``: adapter-only rollouts report an
``adapter_version`` and no weight version at all, the two must agree when both
are present, and the value is compared as a string.

Orbit owns two lines of a forty-line function, so the replacement delegates and
then corrects that one field. These tests pin both halves: orbit's version
recording happens, and every other field on the returned sample is still the one
UPSTREAM's body computed.
"""

import argparse

import pytest

import orbit  # noqa: F401  -- importing orbit installs the patches
from miles.rollout.generate_utils import openai_endpoint_utils as oe
from miles.utils.types import Sample


class _Tokenizer:
    def decode(self, token_ids):
        return "".join(str(t) for t in token_ids)


class _Record:
    """The two fields ``_compute_sample_from_openai_record`` reads off a record."""

    def __init__(self, meta_info, finish_reason="stop"):
        self.request = {}
        self.response = {
            "choices": [
                {
                    "prompt_token_ids": [1, 2],
                    "finish_reason": finish_reason,
                    "meta_info": {
                        "output_token_logprobs": [[-0.1, 5], [-0.2, 6]],
                        **meta_info,
                    },
                }
            ]
        }


def _call(fn, meta_info, input_sample=None):
    return fn(argparse.Namespace(), input_sample or Sample(), _Record(meta_info), _Tokenizer())


def test_the_patch_is_actually_installed():
    assert oe._compute_sample_from_openai_record.__module__ == "orbit.rollout.openai_endpoint_patches"
    assert hasattr(oe, "_orbit_unpatched__compute_sample_from_openai_record"), (
        "the pristine upstream function must be kept so the patch can delegate"
    )


def test_an_adapter_only_rollout_records_a_version():
    """The case upstream has no spelling for: no weight_version in meta_info."""
    meta_info = {"adapter_version": 3}
    assert _call(oe._compute_sample_from_openai_record, meta_info).weight_versions == ["3"]

    # ...and prove the patch is what did it: upstream alone records nothing.
    upstream = _call(oe._orbit_unpatched__compute_sample_from_openai_record, meta_info)
    assert upstream.weight_versions == []


def test_a_weight_version_is_normalized_to_a_string():
    meta_info = {"weight_version": 7}
    assert _call(oe._compute_sample_from_openai_record, meta_info).weight_versions == ["7"]

    # Upstream appends it raw, which is exactly the entry orbit replaces.
    upstream = _call(oe._orbit_unpatched__compute_sample_from_openai_record, meta_info)
    assert upstream.weight_versions == [7]


def test_disagreeing_adapter_and_weight_versions_raise():
    with pytest.raises(ValueError, match="disagree in meta_info"):
        _call(oe._compute_sample_from_openai_record, {"adapter_version": 3, "weight_version": 4})


def test_versions_the_input_sample_already_carried_survive():
    """The correction is length-based, so it removes upstream's entry only."""
    carried = Sample()
    carried.weight_versions = ["1", "2"]
    sample = _call(oe._compute_sample_from_openai_record, {"weight_version": 7}, carried)
    assert sample.weight_versions == ["1", "2", "7"]


def test_everything_else_on_the_sample_still_comes_from_upstreams_body():
    """The delegation property. If this fails because orbit copied the body
    instead, upstream's fixes to that function stop reaching us."""
    meta_info = {"weight_version": 7}
    patched = _call(oe._compute_sample_from_openai_record, meta_info)
    upstream = _call(oe._orbit_unpatched__compute_sample_from_openai_record, meta_info)

    for field in ("tokens", "response", "response_length", "loss_mask", "rollout_log_probs", "status"):
        assert getattr(patched, field) == getattr(upstream, field), field
    assert patched.tokens == [1, 2, 5, 6]
    assert patched.status is Sample.Status.COMPLETED
