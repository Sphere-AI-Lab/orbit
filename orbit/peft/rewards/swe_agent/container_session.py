"""Per-episode Apptainer container session for agentic SWE rollouts.

One session = one SWE instance container held open for one episode:

- start: copy the SIF's read-only repo to host scratch (same trick as
  ``sandbox/swe_rm``: binding the writable copy back over the original path
  preserves installed-package paths). No ``apptainer instance`` is used —
  ``instance start --contain`` cannot mount /proc on this cluster — episode
  state lives in the HOST-side writable repo, so each command is a fresh
  ``exec`` against the same binds (persistence via filesystem, ~1-2 s
  startup per command; environment/processes do not persist across turns).
- run: ``apptainer exec`` per shell command, with timeout and output-tail
  truncation before injection into the model's context.
- verify: apply the row's ``test_patch`` in the session repo (a conflict
  with model-edited test files fails verification — anti-cheat by
  construction) and run FAIL_TO_PASS + PASS_TO_PASS under pytest.
- stop: scratch cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import tempfile
import uuid

from orbit.peft.rewards.sandbox.swe_rm import _apptainer, _find_repo_dir, _sif_path

logger = logging.getLogger(__name__)

_OUTPUT_TAIL_BYTES = 4096


class ContainerSession:
    def __init__(self, sif: str, *, cmd_timeout_secs: float = 30.0):
        self.sif = sif
        self.cmd_timeout_secs = cmd_timeout_secs
        self.name = f"orbit_swe_{uuid.uuid4().hex[:12]}"
        self.repo_dir: str | None = None
        self._scratch: str | None = None
        self._started = False

    async def start(self, setup_timeout_secs: float = 180.0) -> bool:
        self.repo_dir = await _find_repo_dir(self.sif, min(60.0, setup_timeout_secs))
        if not self.repo_dir:
            logger.warning("swe_agent: no repo dir found inside %s", self.sif)
            return False

        self._scratch = tempfile.mkdtemp(prefix="orbit_swe_agent_")
        rc, out = await _apptainer(
            [
                "apptainer",
                "exec",
                "--no-home",
                "--contain",
                "--bind",
                f"{self._scratch}:/orbit_scratch",
                self.sif,
                "cp",
                "-a",
                self.repo_dir,
                "/orbit_scratch/repo",
            ],
            setup_timeout_secs,
        )
        if rc != 0:
            logger.warning("swe_agent: repo copy failed rc=%s: %s", rc, out[-300:])
            self._cleanup_scratch()
            return False

        self._started = True
        return True

    def _exec_cmd(self, script: str) -> list[str]:
        return [
            "apptainer",
            "exec",
            "--no-home",
            "--contain",
            "--bind",
            f"{self._scratch}/repo:{self.repo_dir}",
            "--bind",
            f"{self._scratch}:/orbit_scratch",
            self.sif,
            "sh",
            "-c",
            script,
        ]

    async def run(self, command: str, timeout_secs: float | None = None) -> tuple[int, str]:
        """Run one shell command in the session's repo dir; tail-truncated output."""
        assert self._started, "session not started"
        timeout = timeout_secs if timeout_secs is not None else self.cmd_timeout_secs
        rc, out = await _apptainer(
            self._exec_cmd(f"cd {shlex.quote(self.repo_dir)} && {command}"),
            timeout,
        )
        if len(out.encode()) > _OUTPUT_TAIL_BYTES:
            out = "...(truncated)...\n" + out.encode()[-_OUTPUT_TAIL_BYTES:].decode(errors="replace")
        return rc, out

    async def verify(self, swe: dict, timeout_secs: float = 300.0) -> bool:
        """SWE-bench verification of the CURRENT session repo state."""
        assert self._started, "session not started"
        tests = list(swe.get("fail_to_pass") or []) + list(swe.get("pass_to_pass") or [])
        if not tests:
            return False
        with open(os.path.join(self._scratch, "test.patch"), "w") as f:
            f.write(swe.get("test_patch") or "")
        script = (
            f"cd {shlex.quote(self.repo_dir)} && "
            "if [ -s /orbit_scratch/test.patch ]; then "
            "(git apply --whitespace=nowarn /orbit_scratch/test.patch || "
            " patch -p1 --forward --silent < /orbit_scratch/test.patch) || exit 42; fi; "
            "export PYTHONDONTWRITEBYTECODE=1; "
            f"python -m pytest -q -p no:cacheprovider {' '.join(shlex.quote(t) for t in tests)}"
        )
        rc, out = await _apptainer(self._exec_cmd(script), timeout_secs)
        if rc == 42:
            logger.debug("swe_agent: test_patch failed to apply (model edited test files?)")
        return rc == 0

    async def stop(self) -> None:
        self._started = False
        self._cleanup_scratch()

    def _cleanup_scratch(self) -> None:
        if self._scratch and os.path.isdir(self._scratch):
            shutil.rmtree(self._scratch, ignore_errors=True)
        self._scratch = None


def sif_for_instance(cache_dir: str, image_name: str) -> str:
    return _sif_path(cache_dir, image_name)
