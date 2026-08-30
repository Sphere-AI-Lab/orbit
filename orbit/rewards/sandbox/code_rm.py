"""Code-execution reward: run the rollout's program against stdin/stdout tests.

The Nemotron-RL-Ultra ``code_gen_simple_agent`` contract: each row carries
``metadata["unit_tests"] = {"inputs": [...], "outputs": [...]}`` (competitive-
programming style). Reward is binary — 1.0 iff every executed test passes —
matching the all-or-nothing judging of the source datasets; execution
short-circuits on the first failing test, so wrong programs are cheap.

Wire-up::

    --custom-rm-path orbit.rewards.sandbox.code_rm.reward_func
    [--code-rm-timeout-secs 6] [--code-rm-memory-mb 512] [--code-rm-max-tests 0]
"""

from __future__ import annotations

import logging
import re
from argparse import Namespace

from orbit.rewards.sandbox.executor import run_python
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_PYTHON_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def _extract_python_code(text: str) -> str | None:
    """The last ```python fenced block (falling back to the last plain fence)."""
    matches = _PYTHON_FENCE_RE.findall(text or "") or _ANY_FENCE_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].strip()


def _normalize(output: str) -> list[str]:
    lines = [line.rstrip() for line in (output or "").splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _outputs_match(expected: str, actual: str) -> bool:
    return _normalize(expected) == _normalize(actual)


async def reward_func(args: Namespace, sample: Sample, **kwargs) -> float:
    """``--custom-rm-path`` hook: 1.0 iff the extracted program passes all tests."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    unit_tests = metadata.get("unit_tests") or {}
    inputs = unit_tests.get("inputs") or []
    outputs = unit_tests.get("outputs") or []
    if not inputs or len(inputs) != len(outputs):
        logger.warning(
            "code_rm: sample %s has no usable unit_tests (%d inputs / %d outputs); reward 0.",
            sample.index,
            len(inputs),
            len(outputs),
        )
        return 0.0

    code = _extract_python_code(sample.response)
    if code is None:
        return 0.0

    max_tests = int(getattr(args, "code_rm_max_tests", 0) or 0)
    if max_tests > 0:
        inputs, outputs = inputs[:max_tests], outputs[:max_tests]

    timeout_secs = float(getattr(args, "code_rm_timeout_secs", 6) or 6)
    memory_mb = int(getattr(args, "code_rm_memory_mb", 512) or 512)

    for stdin_text, expected in zip(inputs, outputs, strict=True):
        result = await run_python(code, stdin_text, timeout_secs=timeout_secs, memory_mb=memory_mb)
        if result.timed_out or result.returncode != 0 or not _outputs_match(expected, result.stdout):
            return 0.0
    return 1.0
