"""SWE patch reward: apply the rollout's diff in the instance container, run tests.

Rung 2a of the SWE harness (design doc
docs/plans/2026-07-07-swe-harness-scoping.md): single-turn patch RL over the
Nemotron-RL-Ultra ``swe`` blend. Each row ships a prebuilt per-instance
Docker image (repo at ``base_commit`` with deps installed) plus the SWE-bench
verification contract. The reward:

1. extract the last unified diff from the response (```diff fence, falling
   back to a raw ``diff --git`` tail);
2. copy the container's repo to a host scratch dir (SIF images are
   read-only; binding the writable copy back over the repo path preserves
   installed-package paths);
3. ``git apply`` the model patch (fallback ``patch -p1``), apply the row's
   ``test_patch``;
4. run FAIL_TO_PASS + PASS_TO_PASS under pytest with a wall-clock timeout;
5. reward 1.0 iff everything passes — the SWE-bench standard, binary.

Containers run ``apptainer exec --no-home --contain`` (the default config
binds $HOME — the host environment must not leak in). SIFs are looked up in
``--swe-rm-sif-cache`` by sanitized image name; use
``tools/prepare_swe_subset.py`` to build the subset + cache.

Wire-up (standalone or via the reward router's ``swe_agents_train`` route)::

    --custom-rm-path orbit.rewards.sandbox.swe_rm.reward_func
    --swe-rm-sif-cache /path/to/sif_cache
    [--swe-rm-timeout-secs 300]
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import tempfile
from argparse import Namespace

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_RAW_DIFF_RE = re.compile(r"^diff --git .*", re.MULTILINE)

# One repo-copy + test-suite per sample is heavy; keep a global cap.
_CONTAINER_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("ORBIT_SWE_RM_MAX_CONCURRENCY", "4")))


def _extract_patch(text: str) -> str | None:
    """The last fenced diff block, else everything from the last raw ``diff --git``."""
    matches = _DIFF_FENCE_RE.findall(text or "")
    candidate = None
    if matches:
        candidate = matches[-1]
    else:
        raw = list(_RAW_DIFF_RE.finditer(text or ""))
        if raw:
            candidate = (text or "")[raw[-1].start() :]
    if candidate is None:
        return None
    candidate = candidate.strip()
    if not candidate.startswith("diff --git") and "--- " not in candidate:
        return None
    return candidate + "\n"


def _sif_path(cache_dir: str, image_name: str) -> str:
    stem = image_name.split("://")[-1]
    stem = stem.removeprefix("docker.io/")
    stem = stem.replace("/", "__").replace(":", "__")
    return os.path.join(cache_dir, f"{stem}.sif")


async def _apptainer(cmd: list[str], timeout_secs: float) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_secs)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "timeout"
    return proc.returncode or 0, out.decode(errors="replace")[-4000:]


async def _find_repo_dir(sif: str, timeout_secs: float) -> str | None:
    """The container's repo = the top-level directory holding a .git."""
    rc, out = await _apptainer(
        [
            "apptainer",
            "exec",
            "--no-home",
            "--contain",
            sif,
            "sh",
            "-c",
            'for d in /*/; do [ -d "${d}.git" ] && echo "${d%/}" && exit 0; done; exit 3',
        ],
        timeout_secs,
    )
    if rc != 0:
        return None
    return out.strip().splitlines()[-1]


async def _run_verification(sif: str, swe: dict, patch: str, timeout_secs: float) -> bool:
    repo_dir = await _find_repo_dir(sif, min(60.0, timeout_secs))
    if not repo_dir:
        logger.warning("swe_rm: no repo dir found inside %s", sif)
        return False

    tests = list(swe.get("fail_to_pass") or []) + list(swe.get("pass_to_pass") or [])
    if not tests:
        logger.warning("swe_rm: instance has no tests; reward 0.")
        return False

    with tempfile.TemporaryDirectory(prefix="orbit_swe_") as scratch:
        with open(os.path.join(scratch, "model.patch"), "w") as f:
            f.write(patch)
        with open(os.path.join(scratch, "test.patch"), "w") as f:
            f.write(swe.get("test_patch") or "")

        # 1. copy the (read-only) repo out of the SIF into host scratch
        rc, out = await _apptainer(
            [
                "apptainer",
                "exec",
                "--no-home",
                "--contain",
                "--bind",
                f"{scratch}:/orbit_scratch",
                sif,
                "cp",
                "-a",
                repo_dir,
                "/orbit_scratch/repo",
            ],
            min(120.0, timeout_secs),
        )
        if rc != 0:
            logger.warning("swe_rm: repo copy failed rc=%s: %s", rc, out[-300:])
            return False

        # 2. bind the writable copy over the repo path; apply patches; run tests
        script = (
            f"cd {shlex.quote(repo_dir)} && "
            "(git apply --whitespace=nowarn /orbit_scratch/model.patch || "
            " patch -p1 --forward --silent < /orbit_scratch/model.patch) || exit 41; "
            "if [ -s /orbit_scratch/test.patch ]; then "
            "(git apply --whitespace=nowarn /orbit_scratch/test.patch || "
            " patch -p1 --forward --silent < /orbit_scratch/test.patch) || exit 42; fi; "
            "export PYTHONDONTWRITEBYTECODE=1; "
            f"python -m pytest -q -p no:cacheprovider {' '.join(shlex.quote(t) for t in tests)}"
        )
        rc, out = await _apptainer(
            [
                "apptainer",
                "exec",
                "--no-home",
                "--contain",
                "--bind",
                f"{scratch}/repo:{repo_dir}",
                "--bind",
                f"{scratch}:/orbit_scratch",
                sif,
                "sh",
                "-c",
                script,
            ],
            timeout_secs,
        )
        if rc == 41:
            logger.debug("swe_rm: model patch failed to apply")
        elif rc == 42:
            logger.warning("swe_rm: test_patch failed to apply (data issue?)")
        return rc == 0


async def reward_func(args: Namespace, sample: Sample, **kwargs) -> float:
    """``--custom-rm-path`` hook: 1.0 iff the patch makes the instance's tests pass."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    swe = metadata.get("swe") or {}
    image_name = swe.get("image_name")
    if not image_name:
        logger.warning("swe_rm: sample %s has no metadata['swe']['image_name']; reward 0.", sample.index)
        return 0.0

    patch = _extract_patch(sample.response)
    if patch is None:
        return 0.0

    cache_dir = getattr(args, "swe_rm_sif_cache", None)
    if not cache_dir:
        raise ValueError("swe_rm requires --swe-rm-sif-cache <dir> (see tools/prepare_swe_subset.py).")
    sif = _sif_path(cache_dir, image_name)
    if not os.path.exists(sif):
        logger.warning("swe_rm: SIF missing for %s (expected %s); reward 0.", image_name, sif)
        return 0.0

    timeout_secs = float(getattr(args, "swe_rm_timeout_secs", 300) or 300)
    try:
        async with _CONTAINER_SEMAPHORE:
            passed = await _run_verification(sif, swe, patch, timeout_secs)
    except Exception:
        logger.exception("swe_rm: verification crashed for %s; reward 0 (fail-soft).", image_name)
        return 0.0
    return 1.0 if passed else 0.0
