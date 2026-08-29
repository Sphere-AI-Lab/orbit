"""First-class OPD teacher scoring stage (same-engine, adapter-slot teachers).

Replaces the custom-rm hijack for same-base teachers: scoring happens as a
post-generation rollout step against the LOCAL rollout engine (per-request
lora_path selects the teacher), leaving --custom-rm-path free for real task
rewards (blend) and keeping eval metrics meaningful.
"""

from argparse import Namespace

from orbit.opd.opd_teacher_spec import (
    OPD_TEACHER_ADAPTER_NAME,
    is_same_base,
    needs_engine_teacher_slot,
    parse_teacher_spec,
)


def _spec(args: Namespace):
    return parse_teacher_spec(getattr(args, "opd_teacher", None), getattr(args, "opd_teacher_load", None))


def local_scoring_enabled(args: Namespace) -> bool:
    if getattr(args, "opd_type", None) != "sglang":
        return False
    if getattr(args, "opd_teacher_url", None) or getattr(args, "opd_teacher_urls", None):
        return False  # external teachers keep the existing transport
    return is_same_base(_spec(args))


def teacher_lora_path(args: Namespace) -> str | None:
    return OPD_TEACHER_ADAPTER_NAME if needs_engine_teacher_slot(_spec(args)) else None


async def opd_score_sample(args: Namespace, sample) -> None:
    """Score one generated sample against the local engine's teacher.

    Sets sample.teacher_log_probs (sampled-token path) or
    sample.opd_reverse_kl (top-k path) in place.
    """
    from orbit.opd.opd_sglang import (
        STUDENT_TOP_LOGPROBS_METADATA_KEY,
        TeacherTarget,
        _compute_topk_reverse_kl,
        _get_opd_top_k,
        _sampled_teacher_log_probs,
        _score_top_k,
        _score_with_teacher,
        _student_score_url,
    )

    # TeacherTarget is `tuple[str, float]` (url, weight) -- a plain tuple
    # alias, not a NamedTuple/dataclass, so it is constructed positionally.
    target: TeacherTarget = (_student_score_url(args), 1.0)
    lora = teacher_lora_path(args)
    if _get_opd_top_k(args) > 0:
        payload = await _score_top_k(args, sample, [target], lora_path=lora)
        sample.opd_reverse_kl = _compute_topk_reverse_kl(args, sample, payload).tolist()
        sample.metadata.pop(STUDENT_TOP_LOGPROBS_METADATA_KEY, None)
    else:
        payload = await _score_with_teacher(args, sample, [target], lora_path=lora)
        sample.teacher_log_probs = _sampled_teacher_log_probs(payload, sample.response_length)
