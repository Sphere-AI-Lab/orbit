"""A tool observation must not fabricate logprobs for a sample tracking none.

Upstream extends five parallel sample fields by the observation length, the last
being ``rollout_log_probs += [0.0] * n``. Orbit leaves that field ``None`` for
phases where it never asked the engine for logprobs, so upstream's unconditional
``+=`` raises mid-rollout. The fix is a delegating patch that lends upstream a
list and takes it back (orbit/rollout/tool_call_patches.py), so all five updates
stay upstream's.
"""

import pytest

pytest.importorskip("torch")

import orbit  # noqa: F401,E402  -- importing orbit installs the patch

from miles.rollout.generate_utils import tool_call_utils  # noqa: E402
from miles.utils.types import Sample  # noqa: E402

PATCH_MODULE = "orbit.rollout.tool_call_patches"


class _Tokenizer:
    def decode(self, ids):
        return "obs" * len(ids)


def _sample(rollout_log_probs):
    return Sample(
        prompt="p",
        response="r",
        response_length=1,
        tokens=[1],
        loss_mask=[1],
        rollout_log_probs=rollout_log_probs,
    )


def _messages():
    return [{"role": "tool", "content": "x", "tool_call_id": "1", "name": "t"}]


@pytest.fixture(autouse=True)
def _fixed_observation(monkeypatch):
    monkeypatch.setattr(tool_call_utils, "tokenize_tool_responses", lambda *a, **k: [7, 8])


def test_the_patch_is_installed():
    assert tool_call_utils.update_sample_with_tool_responses.__module__ == PATCH_MODULE
    assert hasattr(tool_call_utils, "_orbit_unpatched_update_sample_with_tool_responses")


def test_a_sample_tracking_no_logprobs_keeps_tracking_none():
    sample = _sample(None)
    tool_call_utils.update_sample_with_tool_responses(sample, _messages(), _Tokenizer())
    assert sample.rollout_log_probs is None
    # ...and upstream still did the other four updates.
    assert sample.tokens == [1, 7, 8]
    assert sample.loss_mask == [1, 0, 0]
    assert sample.response_length == 3


def test_a_sample_that_tracks_logprobs_gets_upstreams_filler():
    sample = _sample([0.5])
    tool_call_utils.update_sample_with_tool_responses(sample, _messages(), _Tokenizer())
    assert sample.rollout_log_probs == [0.5, 0.0, 0.0]


def test_upstream_alone_raises_on_the_none_sample():
    """Proves the patch is what fixed it, not a coincidence of the fixture."""
    sample = _sample(None)
    with pytest.raises(TypeError):
        tool_call_utils._orbit_unpatched_update_sample_with_tool_responses(
            sample, _messages(), _Tokenizer()
        )


def test_the_borrowed_list_is_returned_even_when_upstream_raises(monkeypatch):
    """The restore is in a finally: a mid-update failure must not leave the
    sample holding a list it never tracked."""
    def boom(*args, **kwargs):
        raise RuntimeError("tokenise failed")

    monkeypatch.setattr(tool_call_utils, "tokenize_tool_responses", boom)
    sample = _sample(None)
    with pytest.raises(RuntimeError):
        tool_call_utils.update_sample_with_tool_responses(sample, _messages(), _Tokenizer())
    assert sample.rollout_log_probs is None


def test_the_vendored_file_no_longer_carries_orbit_code():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "miles/rollout/generate_utils/tool_call_utils.py").read_text()
    assert "ORBIT" not in src
