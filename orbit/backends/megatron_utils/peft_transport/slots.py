from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


_STUDENT_NAMES = {
    "lora": "orbit_lora",
    "oft": "orbit_oft",
}
_TEACHER_NAME = "orbit_teacher"
_MAX_ADAPTER_NAME_BYTES = 256


class MutationPurpose(StrEnum):
    STUDENT_SYNC = "student_sync"
    LEGACY_SELF_TEACHER_PROMOTION = "legacy_self_teacher_promotion"


@dataclass(frozen=True)
class StudentSlotPlan:
    method: Literal["lora", "oft"]
    public_name: str
    physical_slot_count: Literal[1, 2]

    def __post_init__(self) -> None:
        if (
            type(self.method) is not str
            or self.method not in _STUDENT_NAMES
            or type(self.public_name) is not str
            or self.public_name != _STUDENT_NAMES[self.method]
            or type(self.physical_slot_count) is not int
            or self.physical_slot_count not in {1, 2}
        ):
            raise ValueError("student slot plan is invalid")


def student_slot_plan(args: Namespace) -> StudentSlotPlan:
    if type(args) is not Namespace:
        raise TypeError("student slot plan args must be an exact Namespace")
    method = getattr(args, "peft_method", "none")
    if type(method) is not str or method not in _STUDENT_NAMES:
        raise ValueError("student slot plan requires LoRA or OFT")
    double_buffer = getattr(args, "adapter_double_buffer", False)
    if type(double_buffer) is not bool:
        raise ValueError("student slot buffer mode must be an exact boolean")
    count = 2 if double_buffer else 1
    return StudentSlotPlan(method, _STUDENT_NAMES[method], count)


def _validate_requested_name(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ValueError("adapter mutation destination is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("adapter mutation destination is invalid") from None
    if len(encoded) > _MAX_ADAPTER_NAME_BYTES or any(
        ord(character) < 0x20 or ord(character) == 0x7F
        for character in value
    ):
        raise ValueError("adapter mutation destination is invalid")
    return value


def authorize_adapter_destination(
    args: Namespace,
    *,
    requested_name: str | None,
    purpose: MutationPurpose,
) -> str:
    if type(purpose) is not MutationPurpose:
        raise ValueError("adapter mutation purpose is invalid")
    plan = student_slot_plan(args)
    requested_name = _validate_requested_name(requested_name)
    if purpose is MutationPurpose.STUDENT_SYNC:
        if requested_name not in (None, plan.public_name):
            raise ValueError("student synchronization destination is forbidden")
        return plan.public_name
    if purpose is not MutationPurpose.LEGACY_SELF_TEACHER_PROMOTION:
        raise ValueError("adapter mutation purpose is invalid")
    spec = getattr(args, "opd_teacher_spec", None)
    if (
        getattr(args, "ultra_teacher_pool_plan", None) is not None
        or requested_name != _TEACHER_NAME
        or getattr(spec, "source", None) not in {"self_ema", "self_lag"}
    ):
        raise ValueError("legacy self-teacher destination is forbidden")
    return _TEACHER_NAME
