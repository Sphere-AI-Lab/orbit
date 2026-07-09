"""Unit tests for the Lean proof grader (rm_hub/lean_rm.py); server mocked.

The real-toolchain path (kimina-lean-server + Mathlib) is covered by
tools/lean_rm_oracle.py.
"""

from __future__ import annotations

import asyncio

import orbit.rollout.rm_hub.lean_rm as lr

HEADER = "import Mathlib\nopen Nat\n"
STATEMENT = "theorem two : 1 + 1 = 2 := by\n"

FULL_CODE = "```lean4\nimport Mathlib\ntheorem two : 1 + 1 = 2 := by norm_num\n```"


class _Args:
    lean_server_url = "http://scripted:8000"
    lean_timeout_secs = 60


class _FakeResp:
    def __init__(self, payload, raise_exc=None):
        self._payload = payload
        self._raise_exc = raise_exc

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    """Stand-in for httpx.AsyncClient used as an async context manager."""

    def __init__(self, tracker, result, raise_exc):
        self._tracker = tracker
        self._result = result
        self._raise_exc = raise_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        self._tracker["called"] = True
        self._tracker["payload"] = json
        if self._raise_exc is not None:
            raise self._raise_exc
        return _FakeResp({"results": [self._result]})


def _mock_server(monkeypatch, result, raise_exc=None):
    tracker = {"called": False, "payload": None}
    monkeypatch.setattr(lr.httpx, "AsyncClient", lambda *a, **k: _FakeClient(tracker, result, raise_exc))
    return tracker


# ---------------------------------------------------------------------------
# Extraction / composition
# ---------------------------------------------------------------------------


def test_extract_full_code_used_verbatim():
    code = lr.extract_lean_code(FULL_CODE, HEADER, STATEMENT)
    assert code.startswith("import Mathlib")
    assert "norm_num" in code


def test_extract_theorem_without_imports_gets_header():
    resp = "```lean4\ntheorem two : 1 + 1 = 2 := by norm_num\n```"
    code = lr.extract_lean_code(resp, HEADER, STATEMENT)
    assert code.startswith(HEADER)


def test_extract_bare_tactics_completes_statement():
    resp = "plan...\n```lean4\n  norm_num\n```"
    code = lr.extract_lean_code(resp, HEADER, STATEMENT)
    assert code.startswith(HEADER)
    assert STATEMENT in code
    assert code.rstrip().endswith("norm_num")


def test_extract_no_fence_or_empty():
    assert lr.extract_lean_code("no code here", HEADER, STATEMENT) is None
    assert lr.extract_lean_code("```lean4\n\n```", HEADER, STATEMENT) is None


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def test_clean_compile_passes(monkeypatch):
    _mock_server(monkeypatch, {"error": None, "response": {"messages": [], "sorries": []}})
    assert asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT)) == 1.0


def test_error_message_fails(monkeypatch):
    _mock_server(
        monkeypatch,
        {"error": None, "response": {"messages": [{"severity": "error", "data": "unknown identifier"}]}},
    )
    assert asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT)) == 0.0


def test_sorry_warning_and_sorries_field_fail(monkeypatch):
    _mock_server(
        monkeypatch,
        {"error": None, "response": {"messages": [{"severity": "warning", "data": "declaration uses 'sorry'"}]}},
    )
    assert asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT)) == 0.0
    _mock_server(monkeypatch, {"error": None, "response": {"messages": [], "sorries": [{"pos": 1}]}})
    assert asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT)) == 0.0


def test_sorry_in_code_rejected_before_server(monkeypatch):
    tracker = _mock_server(monkeypatch, {"error": None, "response": {"messages": []}})
    resp = "```lean4\nimport Mathlib\ntheorem two : 1 + 1 = 2 := by sorry\n```"
    assert asyncio.run(lr.grade_lean_proof(_Args(), resp, HEADER, STATEMENT)) == 0.0
    assert tracker["called"] is False  # server never called


def test_transport_error_and_no_url_fail_soft(monkeypatch):
    _mock_server(monkeypatch, {}, raise_exc=RuntimeError("down"))
    assert asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT)) == 0.0

    class NoUrl:
        lean_server_url = None

    assert asyncio.run(lr.grade_lean_proof(NoUrl(), FULL_CODE, HEADER, STATEMENT)) == 0.0
