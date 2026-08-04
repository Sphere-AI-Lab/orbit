from copy import deepcopy
from dataclasses import fields

import numpy as np

from orbit.utils.types import Sample

_OPD_STUDENT_TOP_LOGPROBS_KEY = "opd_student_top_logprobs"


def merge_samples(samples: list[Sample], tokenizer) -> Sample:
    acc = samples[0]
    for sample in samples[1:]:
        acc = _merge_sample_pair(acc, sample, tokenizer=tokenizer)
    return acc


def _merge_sample_pair(a: Sample, b: Sample, tokenizer) -> Sample:
    """Merge two samples generated from sibling inference engine calls."""
    from orbit.rollout.opd_sglang import _TOPK_PAD_LOGPROB, _TOPK_PAD_TOKEN_ID

    a, b = deepcopy(a), deepcopy(b)

    def _merge_equal_value(field):
        x = getattr(a, field)
        y = getattr(b, field)
        assert x == y, f"{field} mismatch: a.{field}={x}, b.{field}={y}"
        return x

    def _fill_defaults(sample: Sample):
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = [0.0] * sample.response_length

    def _merge_optional_per_token(field):
        # Optional OPD per-token lists (teacher_log_probs, opd_reverse_kl): merge like
        # rollout_log_probs when present (zeros over the injected observation span),
        # else keep None — zero-filling for non-OPD runs would poison batches that
        # mix merged and unmerged samples.
        av, bv = getattr(a, field), getattr(b, field)
        if av is None and bv is None:
            return None
        av = av if av is not None else [0.0] * a.response_length
        bv = bv if bv is not None else [0.0] * b.response_length
        return av + [0.0] * obs_len + bv

    def _merge_optional_hidden_states():
        # Full-vocab OPD teacher hidden states are produced by the custom-rm hooks on
        # the *merged* sample, so both sides are normally still None here. Mirror the
        # per-token list merge anyway (zero rows over the injected observation span,
        # which the loss mask zeroes out) so a scored segment survives a late merge
        # instead of silently vanishing.
        av, bv = a.teacher_hidden_states, b.teacher_hidden_states
        if av is None and bv is None:
            return None
        hidden_size = av.shape[1] if av is not None else bv.shape[1]
        if av is not None and bv is not None:
            assert av.shape[1] == bv.shape[1], f"teacher hidden size mismatch: {av.shape} vs {bv.shape}"
        av = av if av is not None else np.zeros((a.response_length, hidden_size), dtype=np.float32)
        bv = bv if bv is not None else np.zeros((b.response_length, hidden_size), dtype=np.float32)
        return np.concatenate([av, np.zeros((obs_len, hidden_size), dtype=av.dtype), bv], axis=0)

    def _merge_optional_topk_rows(field, pad_value):
        # Optional [R][K] OPD top-k rows (teacher_topk_ids/teacher_topk_logprobs,
        # --loss-type opd_topk_loss): mirror _merge_optional_per_token, but pad the
        # injected observation span with K-wide all-pad-sentinel rows (zero valid
        # mass) instead of scalar zeros, matching _extract_teacher_topk's fixed
        # row width.
        av, bv = getattr(a, field), getattr(b, field)
        if av is None and bv is None:
            return None
        top_k = len(av[0]) if av else (len(bv[0]) if bv else 0)
        pad_row = [pad_value] * top_k
        av = av if av is not None else [pad_row] * a.response_length
        bv = bv if bv is not None else [pad_row] * b.response_length
        return av + [pad_row] * obs_len + bv

    def _pop_opd_student_top_logprobs(metadata):
        if metadata is None:
            return None, None
        metadata = deepcopy(metadata)
        top_logprobs = metadata.pop(_OPD_STUDENT_TOP_LOGPROBS_KEY, None)
        return metadata, top_logprobs

    def _merge_opd_student_top_logprobs(av, bv):
        if av is None and bv is None:
            return None
        assert av is not None and bv is not None, (
            f"{_OPD_STUDENT_TOP_LOGPROBS_KEY} must be present on both samples when merging top-k OPD metadata: "
            f"a has {av is not None}, b has {bv is not None}"
        )
        assert len(av) == a.response_length, (
            f"{_OPD_STUDENT_TOP_LOGPROBS_KEY} length mismatch: "
            f"a.{_OPD_STUDENT_TOP_LOGPROBS_KEY} has length {len(av)}, "
            f"a.response_length={a.response_length}"
        )
        assert len(bv) == b.response_length, (
            f"{_OPD_STUDENT_TOP_LOGPROBS_KEY} length mismatch: "
            f"b.{_OPD_STUDENT_TOP_LOGPROBS_KEY} has length {len(bv)}, "
            f"b.response_length={b.response_length}"
        )
        return av + [[] for _ in range(obs_len)] + bv

    def _merge_metadata():
        a_metadata, a_top_logprobs = _pop_opd_student_top_logprobs(a.metadata)
        b_metadata, b_top_logprobs = _pop_opd_student_top_logprobs(b.metadata)
        assert a_metadata == b_metadata, f"metadata mismatch: a.metadata={a.metadata}, b.metadata={b.metadata}"

        merged_metadata = deepcopy(a_metadata)
        merged_top_logprobs = _merge_opd_student_top_logprobs(a_top_logprobs, b_top_logprobs)
        if merged_top_logprobs is not None:
            if merged_metadata is None:
                merged_metadata = {}
            merged_metadata[_OPD_STUDENT_TOP_LOGPROBS_KEY] = merged_top_logprobs
        return merged_metadata

    _fill_defaults(a)
    _fill_defaults(b)

    obs_len = len(b.tokens) - len(a.tokens) - b.response_length
    obs_tokens = b.tokens[len(a.tokens) : len(a.tokens) + obs_len]
    # Follow-up: is this acceptable?
    obs_text = tokenizer.decode(obs_tokens)

    try:
        a.validate()
        b.validate()
        assert _startswith(short=a.prompt, long=b.prompt), "b.prompt must start with a.prompt"
        assert _startswith(short=a.tokens, long=b.tokens), "b.tokens must start with a.tokens"
        assert obs_len > 0, f"obs_len must be > 0, got {obs_len}"
        if a.rollout_routed_experts is not None:
            assert a.rollout_routed_experts.shape[0] <= b.rollout_routed_experts.shape[0]
        assert a.status == Sample.Status.COMPLETED, f"a.status must be COMPLETED, got {a.status}"

        return _create_with_all_fields(
            Sample,
            group_index=_merge_equal_value("group_index"),
            index=_merge_equal_value("index"),
            prompt=b.prompt,
            tokens=b.tokens,
            multimodal_inputs=_merge_equal_value("multimodal_inputs"),
            multimodal_train_inputs=_merge_equal_value("multimodal_train_inputs"),
            response=a.response + obs_text + b.response,
            response_length=a.response_length + obs_len + b.response_length,
            label=_merge_equal_value("label"),
            reward=_merge_equal_value("reward"),
            loss_mask=a.loss_mask + [0] * obs_len + b.loss_mask,
            weight_versions=a.weight_versions + b.weight_versions,
            rollout_log_probs=a.rollout_log_probs + [0.0] * obs_len + b.rollout_log_probs,
            teacher_log_probs=_merge_optional_per_token("teacher_log_probs"),
            teacher_hidden_states=_merge_optional_hidden_states(),
            opd_reverse_kl=_merge_optional_per_token("opd_reverse_kl"),
            teacher_topk_ids=_merge_optional_topk_rows("teacher_topk_ids", _TOPK_PAD_TOKEN_ID),
            teacher_topk_logprobs=_merge_optional_topk_rows("teacher_topk_logprobs", _TOPK_PAD_LOGPROB),
            rollout_routed_experts=b.rollout_routed_experts,
            remove_sample=_merge_equal_value("remove_sample"),
            status=b.status,
            metadata=_merge_metadata(),
            generate_function_path=_merge_equal_value("generate_function_path"),
            train_metadata=_merge_equal_value("train_metadata"),
            session_id=_merge_equal_value("session_id"),
            non_generation_time=_merge_equal_value("non_generation_time"),
            spec_info=_merge_spec_info(a.spec_info, b.spec_info),
            prefix_cache_info=_merge_prefix_cache_info(a.prefix_cache_info, b.prefix_cache_info),
        )
    except AssertionError as e:
        e.add_note(f"{a=} {b=}")
        raise


def _merge_spec_info(a: Sample.SpecInfo, b: Sample.SpecInfo) -> Sample.SpecInfo:
    def _merge_plus_value(field):
        return getattr(a, field) + getattr(b, field)

    return _create_with_all_fields(
        Sample.SpecInfo,
        spec_accept_token_num=_merge_plus_value("spec_accept_token_num"),
        spec_draft_token_num=_merge_plus_value("spec_draft_token_num"),
        spec_verify_ct=_merge_plus_value("spec_verify_ct"),
        completion_token_num=_merge_plus_value("completion_token_num"),
    )


def _merge_prefix_cache_info(a: Sample.PrefixCacheInfo, b: Sample.PrefixCacheInfo) -> Sample.PrefixCacheInfo:
    def _merge_plus_value(field):
        return getattr(a, field) + getattr(b, field)

    return _create_with_all_fields(
        Sample.PrefixCacheInfo,
        cached_tokens=_merge_plus_value("cached_tokens"),
        total_prompt_tokens=_merge_plus_value("total_prompt_tokens"),
    )


def _create_with_all_fields(cls, **kwargs):
    expected = {f.name for f in fields(cls)}
    actual = set(kwargs.keys())
    assert (
        expected == actual
    ), f"{cls.__name__} field mismatch. Missing: {expected - actual}, Extra: {actual - expected}"
    return cls(**kwargs)


def _startswith(*, short, long) -> bool:
    if isinstance(short, str) and isinstance(long, str):
        return long.startswith(short)
    if isinstance(short, list) and isinstance(long, list):
        return (len(long) >= len(short)) and (long[: len(short)] == short)
    raise NotImplementedError
