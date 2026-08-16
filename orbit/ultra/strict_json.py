from __future__ import annotations

import json
import math
from typing import Any


MAX_JSON_INTEGER_DIGITS = 4096


class _StrictJSONError(ValueError):
    pass


def _reject_non_finite_constant(value: str) -> None:
    raise _StrictJSONError("non-finite JSON constants are not allowed")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _StrictJSONError("non-finite JSON numbers are not allowed")
    return parsed


def _parse_bounded_int(value: str) -> int:
    digits = value.removeprefix("-")
    if len(digits) > MAX_JSON_INTEGER_DIGITS:
        raise _StrictJSONError("JSON integer exceeds its digit limit")
    try:
        return int(value)
    except ValueError:
        raise _StrictJSONError("JSON integer is not representable") from None


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJSONError("duplicate JSON keys are not allowed")
        result[key] = value
    return result


def _validate_tree(payload: Any, *, max_depth: int) -> None:
    pending = [(payload, 1)]
    while pending:
        value, depth = pending.pop()
        if depth > max_depth:
            raise _StrictJSONError("JSON value exceeds its depth limit")
        if type(value) is dict:
            for key in value:
                _validate_string(key)
            pending.extend((item, depth + 1) for item in value.values())
        elif type(value) is list:
            pending.extend((item, depth + 1) for item in value)
        elif type(value) is str:
            _validate_string(value)


def _validate_string(value: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _StrictJSONError("JSON strings may not contain unpaired surrogates")


def loads_strict(
    raw: bytes,
    *,
    max_bytes: int,
    max_depth: int,
) -> Any:
    """Load one bounded UTF-8 JSON value with strict numeric and key semantics."""

    if type(raw) is not bytes:
        raise TypeError("strict JSON input must be exact bytes")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive exact integer")
    if type(max_depth) is not int or max_depth <= 0:
        raise ValueError("max_depth must be a positive exact integer")
    if len(raw) > max_bytes:
        raise ValueError("strict JSON input exceeds its byte limit")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid UTF-8 JSON") from exc

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_non_finite_constant,
            parse_float=_parse_finite_float,
            parse_int=_parse_bounded_int,
        )
        _validate_tree(parsed, max_depth=max_depth)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON syntax") from exc
    except _StrictJSONError as exc:
        raise ValueError(str(exc)) from exc
    except (RecursionError, OverflowError) as exc:
        raise ValueError("JSON value exceeds parser limits") from exc
    return parsed
