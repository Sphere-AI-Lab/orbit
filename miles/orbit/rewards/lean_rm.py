"""Lean 4 proof grader for ``math_formal_lean_refinement_agent``.

The row asks the model to "Complete the following Lean 4 code" (header +
theorem statement ending in ``by\\n  sorry``); the model replies with a plan
plus a complete ```lean4 fence. Grading = the completed code must compile
against Lean 4 + Mathlib with no errors and no ``sorry``/``admit``.

Verification backend: a Kimina Lean server (project-numina/kimina-lean-server
— Lean REPL + precompiled Mathlib behind HTTP). Boot it from the pulled SIF::

    apptainer exec --bind <workdir> lean_server.sif ... # or docker/native
    # POST {base}/verify {"codes": [{"custom_id": ..., "proof": <code>}]}

Wire-up::

    --lean-server-url http://host:8000   (--lean-timeout-secs 180)

Pass criteria (defensive across kimina/REPL response shapes): the result has
no transport/compile ``error``, no message with severity ``error``, and no
reported ``sorries``. A ``sorry``/``admit`` token in the submitted code is
rejected before ever reaching the server.
"""

from __future__ import annotations

import logging
import re

import httpx

from miles.orbit.rewards.grader_errors import GraderInfrastructureError, InfrastructureErrorCode

logger = logging.getLogger(__name__)

_LEAN_FENCE_RE = re.compile(r"```lean4?\s*\n(.*?)```", re.DOTALL)
_SORRY_RE = re.compile(r"\b(sorry|admit)\b")


def extract_lean_code(response: str, header: str, formal_statement: str) -> str | None:
    """Last lean fence, composed with the row's header/statement if partial."""
    fences = _LEAN_FENCE_RE.findall(response or "")
    if not fences:
        return None
    code = fences[-1].strip()
    if not code:
        return None
    if "theorem" not in code and "lemma" not in code and "example" not in code:
        # bare tactic block: complete the row's statement with it
        return f"{header}{formal_statement}{code}\n"
    if "import" not in code:
        return f"{header}{code}\n"
    return code


def _result_passes(result: dict) -> bool:
    if result.get("error"):
        return False
    response = result.get("response")
    if not isinstance(response, dict):
        raise TypeError("Lean result response must be an object")
    messages = response.get("messages")
    if messages is None:
        messages = []
    if not isinstance(messages, list):
        raise TypeError("Lean result messages must be a list")
    for msg in messages:
        if not isinstance(msg, dict):
            raise TypeError("Lean result message must be an object")
        severity = msg.get("severity")
        if not isinstance(severity, str) or not severity.strip():
            raise TypeError("Lean result message severity must be a nonblank string")
        severity = severity.strip().lower()
        data = str(msg.get("data") or "")
        if severity == "error":
            return False
        if "sorry" in data or "admit" in data:  # "declaration uses 'sorry'"
            return False
    sorries = response.get("sorries")
    if sorries is not None and not isinstance(sorries, list):
        raise TypeError("Lean result sorries must be a list")
    if sorries:
        return False
    return True


async def grade_lean_proof(args, response: str, header: str, formal_statement: str) -> float:
    base_url = getattr(args, "lean_server_url", None)
    if not base_url:
        raise GraderInfrastructureError(
            InfrastructureErrorCode.CONFIGURATION,
            grader="lean",
            stage="configuration",
            retryable=False,
            safe_detail="Lean verifier URL is not configured",
        )
    code = extract_lean_code(response, header or "", formal_statement or "")
    if code is None:
        return 0.0
    if _SORRY_RE.search(code):
        return 0.0

    timeout = float(getattr(args, "lean_timeout_secs", 180) or 180)
    payload = {"codes": [{"custom_id": "orbit", "proof": code}], "timeout": timeout}
    # Self-contained httpx call: the grader must not depend on orbit's global
    # rollout http client being initialized (so it also runs from the oracle).
    # The first import-Mathlib verify loads Mathlib into a REPL and is slow, so
    # allow generous connect/read time beyond the per-proof timeout.
    try:
        async with httpx.AsyncClient(timeout=timeout + 120) as client:
            http_response = await client.post(f"{base_url.rstrip('/')}/verify", json=payload)
        http_response.raise_for_status()
        output = http_response.json()
    except httpx.HTTPStatusError as exc:
        retryable = exc.response.status_code >= 500
        raise GraderInfrastructureError(
            InfrastructureErrorCode.TRANSPORT_ERROR,
            grader="lean",
            stage="verify_request",
            retryable=retryable,
            safe_detail="Lean verifier returned an HTTP error",
        ) from exc
    except (httpx.TransportError, TimeoutError) as exc:
        raise GraderInfrastructureError(
            InfrastructureErrorCode.TRANSPORT_ERROR,
            grader="lean",
            stage="verify_request",
            retryable=True,
            safe_detail="Lean verifier request failed",
        ) from exc
    except (ValueError, TypeError) as exc:
        raise GraderInfrastructureError(
            InfrastructureErrorCode.PROTOCOL_ERROR,
            grader="lean",
            stage="verify_response",
            retryable=False,
            safe_detail="Lean verifier returned invalid JSON",
        ) from exc

    if not isinstance(output, dict):
        raise GraderInfrastructureError(
            InfrastructureErrorCode.PROTOCOL_ERROR,
            grader="lean",
            stage="verify_response",
            retryable=False,
            safe_detail="Lean verifier returned an invalid response schema",
        )
    results = output.get("results")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise GraderInfrastructureError(
            InfrastructureErrorCode.PROTOCOL_ERROR,
            grader="lean",
            stage="verify_response",
            retryable=False,
            safe_detail="Lean verifier returned an invalid response schema",
        )
    try:
        passed = _result_passes(results[0])
    except (AttributeError, TypeError) as exc:
        raise GraderInfrastructureError(
            InfrastructureErrorCode.PROTOCOL_ERROR,
            grader="lean",
            stage="verify_response",
            retryable=False,
            safe_detail="Lean verifier returned an invalid response schema",
        ) from exc
    return 1.0 if passed else 0.0
