"""TeacherSpec: what an OPD teacher *is*, decoupled from where it scores.

Pure module (stdlib only) so argument validation, rollout code, and CPU unit
tests can all import it without pulling in torch/megatron/sglang.
"""

from dataclasses import dataclass

# Reserved rollout-engine adapter slot for teacher scoring. Student weight
# sync must never write this name; only explicit promotion does.
OPD_TEACHER_ADAPTER_NAME = "orbit_teacher"

_SAME_BASE_SOURCES = ("base", "adapter", "self_ema", "self_lag")
_SELF_SOURCES = ("self_ema", "self_lag")


@dataclass(frozen=True)
class TeacherSpec:
    """source: "base" | "adapter" | "self_ema" | "self_lag" | "load".

    path is the adapter checkpoint dir for "adapter", the Megatron checkpoint
    dir for "load", None otherwise.
    """

    source: str
    path: str | None = None


def parse_teacher_spec(opd_teacher: str | None, opd_teacher_load: str | None) -> TeacherSpec | None:
    if opd_teacher and opd_teacher_load:
        raise ValueError(
            "--opd-teacher and --opd-teacher-load are mutually exclusive: "
            "--opd-teacher-load X is legacy sugar for --opd-teacher load:X. Pick one."
        )
    if opd_teacher_load:
        return TeacherSpec("load", opd_teacher_load)
    if not opd_teacher:
        return None
    if opd_teacher == "base":
        return TeacherSpec("base")
    if opd_teacher == "self:ema":
        return TeacherSpec("self_ema")
    if opd_teacher == "self:lag":
        return TeacherSpec("self_lag")
    for prefix, source in (("adapter:", "adapter"), ("load:", "load")):
        if opd_teacher.startswith(prefix):
            path = opd_teacher[len(prefix):]
            if not path:
                raise ValueError(f"--opd-teacher {opd_teacher!r} has an empty path.")
            return TeacherSpec(source, path)
    raise ValueError(
        f"Unknown --opd-teacher spec {opd_teacher!r}: expected "
        "base, adapter:<path>, self:ema, self:lag, or load:<megatron-ckpt>."
    )


def is_same_base(spec: TeacherSpec | None) -> bool:
    return spec is not None and spec.source in _SAME_BASE_SOURCES


def is_self_teacher(spec: TeacherSpec | None) -> bool:
    return spec is not None and spec.source in _SELF_SOURCES


def needs_engine_teacher_slot(spec: TeacherSpec | None) -> bool:
    """True when rollout-side scoring needs the reserved orbit_teacher slot.

    "base" scores against the engine's base weights (a request with no
    lora_path), so it needs no slot.
    """
    return spec is not None and spec.source in ("adapter", "self_ema", "self_lag")
