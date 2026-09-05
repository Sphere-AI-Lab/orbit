"""Sample builder for the trace-viewer example tests.

Delegates to the shared ``make_sample`` so defaults cannot drift, and only adds
what is specific here: ``response_turns`` is routed onto ``Sample.metadata``.
Turn capture belongs to the customization layer, so the viewer reads it from
metadata rather than from a core ``Sample`` field.
"""

from typing import Any

from examples.model_response_trace_viewer.response_log import RESPONSE_TURNS_KEY
from tests.fast.ray.rollout.conftest import make_sample as _make_sample

from orbit.utils.types import Sample


def make_sample(
    *,
    response_turns: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> Sample:
    """Build a Sample with the shared defaults, recording turns on metadata."""
    merged = dict(metadata or {})
    if response_turns is not None:
        merged[RESPONSE_TURNS_KEY] = response_turns
    return _make_sample(metadata=merged, **overrides)
