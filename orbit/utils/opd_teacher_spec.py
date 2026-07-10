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


def should_promote_teacher(spec_source: str, promote_interval: int | None, rollout_id: int) -> bool:
    """Promotion cadence for self:* teachers scored by the rollout engine.

    rollout_id 0 always promotes (the engine slot starts empty; scoring an
    unfilled slot would 404), then every promote_interval rollouts.
    """
    if spec_source not in ("self_ema", "self_lag") or not promote_interval:
        return False
    return rollout_id % promote_interval == 0


def teacher_forward_plan(
    spec: TeacherSpec | None, peft_enabled: bool, ref_available: bool, *, opd_type: str | None
) -> str:
    """Decide how the trainer produces teacher_log_probs this cycle.

    Returns "none" (no teacher), "alias_ref" (teacher==base and the ref
    forward already ran: reuse it), "adapter_off" (base teacher, run one
    forward with the adapter disabled), "adapter_swap" (swap frozen/self
    teacher adapter tensors in for the forward), or "switch_model" (legacy
    full second model).

    opd_type is the teacher producer ("megatron", "sglang", or None). Under
    "sglang" the teacher is scored on the rollout engine and its log-probs are
    authoritative, so the trainer produces nothing ("none") for every source.
    """
    if opd_type == "sglang":
        return "none"
    if spec is None:
        return "none"
    if spec.source == "load":
        return "switch_model"
    if not peft_enabled:
        # Validated at arg-parse time; defensive for direct callers.
        raise ValueError(f"--opd-teacher {spec.source} requires PEFT to toggle adapters.")
    if spec.source == "base":
        return "alias_ref" if ref_available else "adapter_off"
    return "adapter_swap"
