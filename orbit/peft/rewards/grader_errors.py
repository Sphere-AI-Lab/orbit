from __future__ import annotations

from enum import StrEnum

# Bound exception-note rendering to portable signed 64-bit sample indices.
_MIN_REWARD_CONTEXT_INDEX = -(2**63)
_MAX_REWARD_CONTEXT_INDEX = 2**63 - 1


class InfrastructureErrorCode(StrEnum):
    CONFIGURATION = "configuration"
    INVALID_SOURCE = "invalid_source"
    IMAGE_MISSING = "image_missing"
    IMAGE_MISMATCH = "image_mismatch"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    LAUNCH_FAILED = "launch_failed"
    TRANSPORT_ERROR = "transport_error"
    PROTOCOL_ERROR = "protocol_error"
    CLEANUP_FAILED = "cleanup_failed"
    UNEXPECTED = "unexpected"


class GraderInfrastructureError(RuntimeError):
    def __init__(
        self,
        code: InfrastructureErrorCode,
        *,
        grader: str,
        stage: str,
        retryable: bool,
        safe_detail: str,
    ) -> None:
        if not isinstance(code, InfrastructureErrorCode):
            raise TypeError("code must be an InfrastructureErrorCode")
        for name, value in (
            ("grader", grader),
            ("stage", stage),
            ("safe_detail", safe_detail),
        ):
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError(f"{name} must be a nonblank NUL-free string")
        if type(retryable) is not bool:
            raise TypeError("retryable must be bool")
        self.code = code
        self.grader = grader
        self.stage = stage
        self.retryable = retryable
        self.safe_detail = safe_detail
        super().__init__(f"{grader} infrastructure error [{code.value}] during {stage}: {safe_detail}")

    def add_reward_context(self, *, agent: str, sample_index: object) -> None:
        safe_agent = agent if type(agent) is str and agent.isprintable() else "<invalid>"
        safe_sample_index = (
            sample_index
            if sample_index is None
            or (type(sample_index) is int and _MIN_REWARD_CONTEXT_INDEX <= sample_index <= _MAX_REWARD_CONTEXT_INDEX)
            else "<invalid>"
        )
        self.add_note(f"reward context: agent={safe_agent!r}, sample_index={safe_sample_index!r}")

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            _rebuild_grader_infrastructure_error,
            (
                self.code,
                self.grader,
                self.stage,
                self.retryable,
                self.safe_detail,
                tuple(getattr(self, "__notes__", ())),
            ),
        )


def _rebuild_grader_infrastructure_error(
    code: InfrastructureErrorCode,
    grader: str,
    stage: str,
    retryable: bool,
    safe_detail: str,
    notes: tuple[str, ...],
) -> GraderInfrastructureError:
    error = GraderInfrastructureError(
        code,
        grader=grader,
        stage=stage,
        retryable=retryable,
        safe_detail=safe_detail,
    )
    for note in notes:
        error.add_note(note)
    return error
