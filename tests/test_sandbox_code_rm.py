"""Unit tests for the code-execution reward (orbit/peft/rewards/sandbox/code_rm.py).

Judges a rollout by running its extracted Python program against
stdin/stdout unit tests (the Nemotron-RL-Ultra ``code_gen_simple_agent``
contract: ``metadata["unit_tests"] = {"inputs": [...], "outputs": [...]}``).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import orbit.peft.rewards.sandbox.code_rm as code_rm
from orbit.utils.types import Sample


def _args(**overrides):
    values = {
        "code_rm_timeout_secs": 5,
        "code_rm_memory_mb": 256,
        "code_rm_max_tests": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample(response: str, inputs: list[str], outputs: list[str]) -> Sample:
    return Sample(
        prompt=[{"role": "user", "content": "solve it"}],
        response=response,
        metadata={"unit_tests": {"inputs": inputs, "outputs": outputs}},
    )


def _run(coro):
    return asyncio.run(coro)


ECHO_SOLUTION = "```python\nprint(int(input()) * 2)\n```"


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def test_extracts_last_python_fenced_block():
    text = "first\n```python\nprint(1)\n```\nthen\n```python\nprint(2)\n```\ndone"
    assert code_rm._extract_python_code(text) == "print(2)"


def test_extracts_plain_fenced_block_as_fallback():
    assert code_rm._extract_python_code("```\nprint(3)\n```") == "print(3)"


def test_no_code_block_returns_none():
    assert code_rm._extract_python_code("no code here") is None


# ---------------------------------------------------------------------------
# Output comparison
# ---------------------------------------------------------------------------


def test_output_match_ignores_trailing_whitespace_and_blank_lines():
    assert code_rm._outputs_match("2\n1\n5\n", "2 \n1\n5\n\n")
    assert code_rm._outputs_match("a\nb", "a\nb\n")
    assert not code_rm._outputs_match("2\n1\n5\n", "2\n1\n4\n")
    assert not code_rm._outputs_match("2\n1\n", "2\n1\n5\n")


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def test_correct_program_earns_full_reward():
    sample = _sample(ECHO_SOLUTION, inputs=["3\n", "10\n"], outputs=["6\n", "20\n"])
    assert _run(code_rm.reward_func(_args(), sample)) == 1.0


def test_wrong_program_earns_zero():
    sample = _sample(ECHO_SOLUTION, inputs=["3\n"], outputs=["7\n"])
    assert _run(code_rm.reward_func(_args(), sample)) == 0.0


def test_missing_code_block_earns_zero_without_execution(monkeypatch):
    async def never_called(*a, **k):
        raise AssertionError("executor must not run without a code block")

    monkeypatch.setattr(code_rm, "run_python", never_called)
    sample = _sample("I cannot solve this.", inputs=["1\n"], outputs=["1\n"])
    assert _run(code_rm.reward_func(_args(), sample)) == 0.0


def test_short_circuits_on_first_failing_test(monkeypatch):
    calls = []
    real_run_python = code_rm.run_python

    async def counting(code, stdin_text, **kwargs):
        calls.append(stdin_text)
        return await real_run_python(code, stdin_text, **kwargs)

    monkeypatch.setattr(code_rm, "run_python", counting)
    # doubling program vs. expectations that fail on the FIRST test
    sample = _sample(ECHO_SOLUTION, inputs=["1\n", "2\n", "3\n"], outputs=["9\n", "9\n", "9\n"])
    assert _run(code_rm.reward_func(_args(), sample)) == 0.0
    assert len(calls) == 1


def test_max_tests_caps_execution(monkeypatch):
    calls = []
    real_run_python = code_rm.run_python

    async def counting(code, stdin_text, **kwargs):
        calls.append(stdin_text)
        return await real_run_python(code, stdin_text, **kwargs)

    monkeypatch.setattr(code_rm, "run_python", counting)
    sample = _sample(
        ECHO_SOLUTION,
        inputs=[f"{i}\n" for i in range(10)],
        outputs=[f"{2 * i}\n" for i in range(10)],
    )
    assert _run(code_rm.reward_func(_args(code_rm_max_tests=3), sample)) == 1.0
    assert len(calls) == 3


def test_missing_unit_tests_metadata_earns_zero():
    sample = Sample(prompt="q", response=ECHO_SOLUTION, metadata={})
    assert _run(code_rm.reward_func(_args(), sample)) == 0.0


def test_crashing_program_earns_zero():
    sample = _sample("```python\nraise RuntimeError('nope')\n```", inputs=["1\n"], outputs=["1\n"])
    assert _run(code_rm.reward_func(_args(), sample)) == 0.0
