"""Unit tests for the sandboxed Python executor (orbit/rollout/sandbox/).

Pure CPU tests — the executor runs real subprocesses with tiny programs.
"""

from __future__ import annotations

import asyncio
import shutil

import pytest

from orbit.rollout.sandbox.executor import ExecResult, network_isolation_available, run_python


def _run(coro):
    return asyncio.run(coro)


def test_echo_program_captures_stdout():
    result = _run(run_python("print(input())", stdin_text="hello\n", timeout_secs=5))
    assert isinstance(result, ExecResult)
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"
    assert not result.timed_out


def test_stderr_and_nonzero_exit_are_captured():
    result = _run(run_python("import sys; sys.exit('boom')", stdin_text="", timeout_secs=5))
    assert result.returncode != 0
    assert "boom" in result.stderr


def test_infinite_loop_times_out():
    result = _run(run_python("while True: pass", stdin_text="", timeout_secs=1))
    assert result.timed_out
    assert result.returncode != 0


def test_memory_hog_is_killed():
    code = "x = []\nwhile True:\n    x.append(' ' * 10_000_000)"
    result = _run(run_python(code, stdin_text="", timeout_secs=10, memory_mb=128))
    assert result.returncode != 0 or result.timed_out


@pytest.mark.skipif(not network_isolation_available(), reason="unshare -rn unavailable")
def test_network_is_unreachable_inside_sandbox():
    code = (
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(2)\n"
        "try:\n"
        "    s.connect(('1.1.1.1', 80))\n"
        "    print('CONNECTED')\n"
        "except OSError:\n"
        "    print('BLOCKED')\n"
    )
    result = _run(run_python(code, stdin_text="", timeout_secs=10))
    assert "BLOCKED" in result.stdout


def test_program_that_never_reads_large_stdin_still_succeeds():
    # Regression: with pipe-fed stdin, a fast-exiting program broke the writer
    # (uvloop raises where CPython suppresses). File-fed stdin has no writer.
    result = _run(run_python("print('ok')", stdin_text="x" * 1_000_000, timeout_secs=5))
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_crash_before_reading_stdin_reports_failure_not_writer_error():
    result = _run(run_python("import sys; sys.exit(3)", stdin_text="y" * 500_000, timeout_secs=5))
    assert result.returncode == 3
    assert not result.timed_out


def test_multiple_runs_are_independent():
    async def both():
        return await asyncio.gather(
            run_python("print(1+1)", stdin_text="", timeout_secs=5),
            run_python("print(2+2)", stdin_text="", timeout_secs=5),
        )

    r1, r2 = _run(both())
    assert r1.stdout.strip() == "2"
    assert r2.stdout.strip() == "4"
