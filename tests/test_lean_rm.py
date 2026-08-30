"""Unit tests for the Lean proof grader (rm_hub/lean_rm.py); server mocked.

The real-toolchain path (kimina-lean-server + Mathlib) is covered by
tools/lean_rm_oracle.py.
"""

from __future__ import annotations

import asyncio

import pytest

import miles.orbit.rewards.lean_rm as lr
from miles.orbit.rewards.grader_errors import GraderInfrastructureError, InfrastructureErrorCode

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
        if self._raise_exc is not None:
            raise self._raise_exc

    def json(self):
        if isinstance(self._payload, BaseException):
            raise self._payload
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


class _RawClient:
    def __init__(self, payload, raise_status=None):
        self.payload = payload
        self.raise_status = raise_status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json=None):
        return _FakeResp(self.payload, self.raise_status)


def _mock_server(monkeypatch, result, raise_exc=None):
    tracker = {"called": False, "payload": None}
    monkeypatch.setattr(lr.httpx, "AsyncClient", lambda *a, **k: _FakeClient(tracker, result, raise_exc))
    return tracker


def _mock_raw_payload(monkeypatch, payload, raise_status=None):
    monkeypatch.setattr(
        lr.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _RawClient(payload, raise_status),
    )


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


def test_top_level_compile_error_without_response_remains_semantic(monkeypatch):
    _mock_server(monkeypatch, {"error": "Lean compilation failed", "response": None})
    assert asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT)) == 0.0


def test_sorry_warning_and_sorries_field_fail(monkeypatch):
    _mock_server(
        monkeypatch,
        {"error": None, "response": {"messages": [{"severity": "warning", "data": "declaration uses 'sorry'"}]}},
    )
    assert asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT)) == 0.0
    _mock_server(monkeypatch, {"error": None, "response": {"messages": [], "sorries": [{"pos": 1}]}})
    assert asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT)) == 0.0


@pytest.mark.parametrize("placeholder", ["sorry", "admit"])
def test_placeholder_in_code_rejected_before_server(monkeypatch, placeholder):
    tracker = _mock_server(monkeypatch, {"error": None, "response": {"messages": []}})
    resp = f"```lean4\nimport Mathlib\ntheorem two : 1 + 1 = 2 := by {placeholder}\n```"
    assert asyncio.run(lr.grade_lean_proof(_Args(), resp, HEADER, STATEMENT)) == 0.0
    assert tracker["called"] is False  # server never called


def test_lean_transport_and_configuration_errors_propagate(monkeypatch):
    _mock_server(monkeypatch, {}, raise_exc=lr.httpx.ConnectError("down"))
    with pytest.raises(GraderInfrastructureError) as transport:
        asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT))
    assert transport.value.code is InfrastructureErrorCode.TRANSPORT_ERROR
    assert transport.value.retryable is True

    class NoUrl:
        lean_server_url = None

    with pytest.raises(GraderInfrastructureError) as configuration:
        asyncio.run(lr.grade_lean_proof(NoUrl(), FULL_CODE, HEADER, STATEMENT))
    assert configuration.value.code is InfrastructureErrorCode.CONFIGURATION
    assert configuration.value.retryable is False


@pytest.mark.parametrize(
    "payload",
    [None, [], {}, {"results": {}}, {"results": []}, {"results": ["bad"]}],
)
def test_lean_invalid_service_schema_is_infrastructure(monkeypatch, payload):
    _mock_raw_payload(monkeypatch, payload)
    with pytest.raises(GraderInfrastructureError) as caught:
        asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT))
    assert caught.value.code is InfrastructureErrorCode.PROTOCOL_ERROR
    assert caught.value.retryable is False


@pytest.mark.parametrize(
    "result",
    [
        {"error": None, "response": None},
        {"error": None, "response": []},
        {"error": None, "response": {"messages": {}}},
        {"error": None, "response": {"messages": ["bad"]}},
        {"error": None, "response": {"messages": [{}]}},
        {"error": None, "response": {"messages": [{"severity": None}]}},
        {"error": None, "response": {"messages": [{"severity": ""}]}},
        {"error": None, "response": {"messages": [{"severity": "   "}]}},
        {"error": None, "response": {"messages": [{"severity": 1}]}},
        {"error": None, "response": {"messages": [], "sorries": {}}},
        {"error": None, "response": {"messages": [], "sorries": "bad"}},
    ],
)
def test_lean_invalid_accessed_nested_schema_is_infrastructure(monkeypatch, result):
    _mock_server(monkeypatch, result)
    with pytest.raises(GraderInfrastructureError) as caught:
        asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT))
    assert caught.value.code is InfrastructureErrorCode.PROTOCOL_ERROR
    assert caught.value.retryable is False


def test_lean_invalid_json_is_protocol_error(monkeypatch):
    _mock_raw_payload(monkeypatch, ValueError("invalid JSON"))
    with pytest.raises(GraderInfrastructureError) as caught:
        asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT))
    assert caught.value.code is InfrastructureErrorCode.PROTOCOL_ERROR
    assert caught.value.retryable is False


@pytest.mark.parametrize(("status", "retryable"), [(400, False), (503, True)])
def test_lean_http_errors_are_transport_with_status_retryability(monkeypatch, status, retryable):
    request = lr.httpx.Request("POST", "http://scripted:8000/verify")
    response = lr.httpx.Response(status, request=request)
    failure = lr.httpx.HTTPStatusError(
        "verifier rejected request",
        request=request,
        response=response,
    )
    _mock_raw_payload(monkeypatch, {}, raise_status=failure)

    with pytest.raises(GraderInfrastructureError) as caught:
        asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT))

    assert caught.value.code is InfrastructureErrorCode.TRANSPORT_ERROR
    assert caught.value.retryable is retryable


@pytest.mark.parametrize(
    ("failure", "expected_type"),
    [
        (
            GraderInfrastructureError(
                InfrastructureErrorCode.CONFIGURATION,
                grader="upstream",
                stage="setup",
                retryable=False,
                safe_detail="upstream configuration failed",
            ),
            GraderInfrastructureError,
        ),
        (asyncio.CancelledError("stop"), asyncio.CancelledError),
    ],
)
def test_lean_preserves_infrastructure_and_cancellation_identity(monkeypatch, failure, expected_type):
    _mock_server(monkeypatch, {}, raise_exc=failure)
    with pytest.raises(expected_type) as caught:
        asyncio.run(lr.grade_lean_proof(_Args(), FULL_CODE, HEADER, STATEMENT))
    assert caught.value is failure
