"""Unit tests for the SWE patch reward (miles/orbit/rewards/sandbox/swe_rm.py).

Rung 2a of the SWE harness (design doc 2026-07-07-swe-harness-scoping.md):
single-turn patch RL. The model emits a unified diff; the reward applies it
plus the row's test_patch inside the instance's Apptainer image and runs the
FAIL_TO_PASS + PASS_TO_PASS suites. Pure logic tested here; the container
path is verified by the golden-patch oracle (tools/swe_rm_oracle.py).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import miles.orbit.rewards.sandbox.swe_rm as swe_rm
from miles.utils.types import Sample

DIFF = """diff --git a/foo.py b/foo.py
index 111..222 100644
--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-x = 1
+x = 2
"""


def _args(**overrides):
    values = {
        "swe_rm_sif_cache": "/cache",
        "swe_rm_timeout_secs": 300,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample(response: str, **swe_overrides) -> Sample:
    swe = {
        "image_name": "docker.io/swerebenchv2/python-markdown-markdown:1529-f2b9fd1",
        "test_patch": "diff --git a/tests/t.py b/tests/t.py\n...",
        "fail_to_pass": ["tests/t.py::test_a"],
        "pass_to_pass": ["tests/t.py::test_b"],
    }
    swe.update(swe_overrides)
    return Sample(prompt=[{"role": "user", "content": "fix it"}], response=response, metadata={"swe": swe})


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Diff extraction
# ---------------------------------------------------------------------------


def test_extracts_diff_fenced_block():
    text = f"Here is my fix:\n```diff\n{DIFF}```\ndone"
    assert swe_rm._extract_patch(text) == DIFF.strip() + "\n"


def test_extracts_raw_diff_without_fence():
    text = f"Explanation...\n{DIFF}"
    patch = swe_rm._extract_patch(text)
    assert patch is not None
    assert patch.startswith("diff --git")


def test_last_fenced_block_wins():
    text = f"```diff\nWRONG\n```\n```diff\n{DIFF}```"
    assert "x = 2" in swe_rm._extract_patch(text)


def test_no_patch_returns_none():
    assert swe_rm._extract_patch("I cannot fix this.") is None
    assert swe_rm._extract_patch("") is None


# ---------------------------------------------------------------------------
# SIF path resolution
# ---------------------------------------------------------------------------


def test_sif_path_is_sanitized_and_cached_by_image_name():
    p = swe_rm._sif_path("/cache", "docker.io/swerebenchv2/python-markdown-markdown:1529-f2b9fd1")
    assert p == "/cache/swerebenchv2__python-markdown-markdown__1529-f2b9fd1.sif"


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def test_no_patch_zero_without_container(monkeypatch):
    async def never(*a, **k):
        raise AssertionError("container must not run without a patch")

    monkeypatch.setattr(swe_rm, "_run_verification", never)
    assert _run(swe_rm.reward_func(_args(), _sample("no diff here"))) == 0.0


def test_missing_swe_metadata_zero():
    sample = Sample(prompt="q", response=f"```diff\n{DIFF}```", metadata={})
    assert _run(swe_rm.reward_func(_args(), sample)) == 0.0


def test_missing_sif_zero(monkeypatch, tmp_path):
    # cache dir exists but the sif does not
    sample = _sample(f"```diff\n{DIFF}```")
    assert _run(swe_rm.reward_func(_args(swe_rm_sif_cache=str(tmp_path)), sample)) == 0.0


def test_verification_verdict_maps_to_reward(monkeypatch, tmp_path):
    sif = tmp_path / "swerebenchv2__python-markdown-markdown__1529-f2b9fd1.sif"
    sif.write_bytes(b"fake")
    calls = {}

    async def fake_verify(sif_path, swe, patch, timeout_secs):
        calls["sif"] = sif_path
        calls["patch"] = patch
        return calls["verdict"]

    monkeypatch.setattr(swe_rm, "_run_verification", fake_verify)

    sample = _sample(f"```diff\n{DIFF}```")
    calls["verdict"] = True
    assert _run(swe_rm.reward_func(_args(swe_rm_sif_cache=str(tmp_path)), sample)) == 1.0
    calls["verdict"] = False
    assert _run(swe_rm.reward_func(_args(swe_rm_sif_cache=str(tmp_path)), sample)) == 0.0
    assert calls["sif"].endswith(".sif")
    assert calls["patch"].startswith("diff --git")


def test_verification_exception_fails_soft(monkeypatch, tmp_path):
    sif = tmp_path / "swerebenchv2__python-markdown-markdown__1529-f2b9fd1.sif"
    sif.write_bytes(b"fake")

    async def boom(*a, **k):
        raise RuntimeError("apptainer exploded")

    monkeypatch.setattr(swe_rm, "_run_verification", boom)
    assert _run(swe_rm.reward_func(_args(swe_rm_sif_cache=str(tmp_path)), _sample(f"```diff\n{DIFF}```"))) == 0.0


# ---------------------------------------------------------------------------
# Router integration
# ---------------------------------------------------------------------------


def test_router_routes_swe_agent(monkeypatch):
    import miles.orbit.rewards.reward_router as router

    assert router._route_for_agent("swe_agents_train") == "swe"

    async def fake_swe(args, sample, **kwargs):
        return 1.0

    monkeypatch.setattr(router, "_swe_reward", fake_swe)
    sample = _sample(f"```diff\n{DIFF}```")
    sample.metadata["agent"] = "swe_agents_train"
    rewards = asyncio.run(
        router.reward_func(SimpleNamespace(judge_base_url=None, reward_router_unmapped="zero"), [sample])
    )
    assert rewards == [1.0]
