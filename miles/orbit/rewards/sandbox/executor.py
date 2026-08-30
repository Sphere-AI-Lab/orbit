from __future__ import annotations

import asyncio
import functools
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass

_STDOUT_LIMIT_BYTES = 4 * 1024 * 1024  # judges compare short outputs; cap runaway prints


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


@functools.cache
def network_isolation_available() -> bool:
    """Whether ``unshare -rn`` (user + empty network namespace) works here."""
    if shutil.which("unshare") is None:
        return False
    try:
        probe = subprocess.run(
            ["unshare", "-rn", "true"], capture_output=True, timeout=10
        )
    except Exception:
        return False
    return probe.returncode == 0


def _make_preexec(memory_mb: int, timeout_secs: float):
    def preexec() -> None:
        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        cpu = max(1, int(timeout_secs) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024,) * 2)
        os.setsid()

    return preexec


async def run_python(
    code: str,
    stdin_text: str = "",
    *,
    timeout_secs: float = 6.0,
    memory_mb: int = 512,
    isolate_network: bool = True,
) -> ExecResult:
    """Run an untrusted Python program in a rlimited scratch subprocess."""
    with tempfile.TemporaryDirectory(prefix="orbit_sandbox_") as workdir:
        program = os.path.join(workdir, "main.py")
        with open(program, "w") as f:
            f.write(code)
        # stdin via a real file, not a pipe: a program that exits before
        # consuming its input (crash, or it never reads stdin) must not fail
        # the *writer* — under uvloop a broken stdin pipe raises RuntimeError
        # out of communicate(), where CPython's loop silently suppresses it.
        stdin_path = os.path.join(workdir, "stdin.txt")
        with open(stdin_path, "w") as f:
            f.write(stdin_text)

        cmd = [sys.executable, "-I", program]
        if isolate_network and network_isolation_available():
            cmd = ["unshare", "-rn", *cmd]

        env = {"PATH": "/usr/bin:/bin", "HOME": workdir, "LANG": "C.UTF-8"}
        with open(stdin_path, "rb") as stdin_file:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=stdin_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=env,
                preexec_fn=_make_preexec(memory_mb, timeout_secs),
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_secs)
                timed_out = False
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
                stdout, stderr, timed_out = b"", b"timeout", True

        return ExecResult(
            stdout=stdout[:_STDOUT_LIMIT_BYTES].decode(errors="replace"),
            stderr=stderr[:_STDOUT_LIMIT_BYTES].decode(errors="replace"),
            returncode=proc.returncode if proc.returncode is not None else -1,
            timed_out=timed_out,
        )
