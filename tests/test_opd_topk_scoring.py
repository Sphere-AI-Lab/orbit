"""Top-k OPD reverse-KL scoring (port of miles [2/N] af28a061d tests).

_compute_topk_reverse_kl forms a token set per response position (strategy:
only-student / only-teacher / intersection / union / xor over the student's and
teacher's top-k), looks up both models' logprobs for every selected token
(cross-scored via token_ids_logprob where a side's own top-k lacks the token),
weights them (student_p / teacher_p softmax-normalized, or none), and returns
the per-position weighted reverse KL  sum_i w_i * (student_i - teacher_i).
"""

import math
from argparse import Namespace

import pytest

from miles.orbit.opd.opd_sglang import _compute_topk_reverse_kl
from miles.utils.types import Sample


def _entry(prob: float, token_id: int):
    return [math.log(prob), token_id]


def _args(strategy: str, weight_mode: str = "student_p"):
    return Namespace(
        opd_top_k_strategy=strategy,
        opd_reward_weight_mode=weight_mode,
    )


def _sample():
    return Sample(
        tokens=[10, 11, 12],
        response_length=2,
        metadata={
            "opd_student_top_logprobs": [
                [_entry(0.6, 1), _entry(0.4, 2)],
                [_entry(0.7, 4), _entry(0.3, 5)],
            ]
        },
    )


def _teacher_payload():
    return {
        "teacher": {
            "meta_info": {
                "input_top_logprobs": [
                    None,
                    [_entry(0.5, 2), _entry(0.5, 3)],
                    [_entry(0.8, 4), _entry(0.2, 6)],
                ],
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.3, 1), _entry(0.7, 2)],
                    [_entry(0.4, 4), _entry(0.6, 5)],
                ],
            }
        },
        "student_on_teacher": {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.4, 2), _entry(0.2, 3)],
                    [_entry(0.7, 4), _entry(0.1, 6)],
                ]
            }
        },
    }


def test_topk_only_student_uses_student_probability_weights():
    reverse_kl = _compute_topk_reverse_kl(_args("only-student"), _sample(), _teacher_payload())

    expected_0 = 0.6 * math.log(0.6 / 0.3) + 0.4 * math.log(0.4 / 0.7)
    expected_1 = 0.7 * math.log(0.7 / 0.4) + 0.3 * math.log(0.3 / 0.6)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_intersection_uses_overlap_only():
    reverse_kl = _compute_topk_reverse_kl(_args("intersection", "none"), _sample(), _teacher_payload())

    assert reverse_kl.tolist() == pytest.approx(
        [
            math.log(0.4 / 0.5),
            math.log(0.7 / 0.8),
        ]
    )


def test_topk_only_teacher_does_not_need_student_top_logprobs():
    sample = Sample(tokens=[10, 11, 12], response_length=2)

    reverse_kl = _compute_topk_reverse_kl(_args("only-teacher"), sample, _teacher_payload())

    expected_0 = (2 / 3) * math.log(0.4 / 0.5) + (1 / 3) * math.log(0.2 / 0.5)
    expected_1 = (7 / 8) * math.log(0.7 / 0.8) + (1 / 8) * math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_xor_uses_symmetric_difference_without_normalization():
    reverse_kl = _compute_topk_reverse_kl(_args("xor", "none"), _sample(), _teacher_payload())

    expected_0 = math.log(0.6 / 0.3) + math.log(0.2 / 0.5)
    expected_1 = math.log(0.3 / 0.6) + math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


# ---------------------------------------------------------------------------
# Multi-teacher routing + ensembles (--opd-teacher-urls), port of miles [3/N]
# 41a06ffd9 + [4/N] 3f4858ca1
# ---------------------------------------------------------------------------

import torch  # noqa: E402

from miles.orbit.opd.opd_sglang import (  # noqa: E402
    _mixture_log_probs,
    _mixture_logprob_maps,
    _post_teacher_group,
    _tail_bucket_reverse_kl,
    _teacher_targets_for_sample,
    parse_teacher_urls,
)


def _routing_args(urls=None, key="opd_teacher", teacher_url="http://single-teacher/generate"):
    return Namespace(opd_teacher_urls=urls, opd_teacher_key=key, opd_teacher_url=teacher_url)


def _tagged_sample(metadata=None):
    return Sample(tokens=[1, 2, 3], response_length=2, metadata=metadata or {})


def test_parse_teacher_urls_parses_names_and_keeps_equals_in_url():
    url_map = parse_teacher_urls(["math=http://h1:30001/generate", "code=http://h2:30002/generate?tag=a=b"])
    assert url_map == {
        "math": [("http://h1:30001/generate", 1.0)],
        "code": [("http://h2:30002/generate?tag=a=b", 1.0)],
    }


def test_parse_teacher_urls_empty_or_none_gives_empty_map():
    assert parse_teacher_urls(None) == {}
    assert parse_teacher_urls([]) == {}


@pytest.mark.parametrize("bad", ["math", "=http://h1/generate", "math=", "  =  "])
def test_parse_teacher_urls_rejects_malformed_entries(bad):
    with pytest.raises(ValueError, match="expected NAME=URL"):
        parse_teacher_urls([bad])


def test_parse_teacher_urls_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate teacher name"):
        parse_teacher_urls(["math=http://h1/generate", "math=http://h2/generate"])


def test_parse_teacher_urls_ensemble_groups_with_weights():
    url_map = parse_teacher_urls(["ens=http://h1/generate@2,http://h2/generate"])
    assert url_map == {"ens": [("http://h1/generate", 2.0), ("http://h2/generate", 1.0)]}


def test_parse_teacher_urls_at_suffix_not_a_float_is_part_of_url():
    url_map = parse_teacher_urls(["a=http://h1/generate@latest"])
    assert url_map == {"a": [("http://h1/generate@latest", 1.0)]}


@pytest.mark.parametrize("bad_weight", ["@0", "@-1", "@inf", "@nan"])
def test_parse_teacher_urls_rejects_nonpositive_or_nonfinite_weights(bad_weight):
    with pytest.raises(ValueError, match="positive finite"):
        parse_teacher_urls([f"a=http://h1/generate{bad_weight}"])


def test_parse_teacher_urls_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="http"):
        parse_teacher_urls(["a=ftp://h1/generate"])


def test_parse_teacher_urls_rejects_duplicate_url_within_group():
    with pytest.raises(ValueError, match="Duplicate URL"):
        parse_teacher_urls(["a=http://h1/generate,http://h1/generate"])


def test_routing_unset_map_falls_back_to_single_teacher_url():
    args = _routing_args(urls=None)
    sample = _tagged_sample({"opd_teacher": "math"})
    assert _teacher_targets_for_sample(args, sample) == [("http://single-teacher/generate", 1.0)]


def test_routing_by_metadata_name():
    args = _routing_args(urls=["math=http://h1/generate", "code=http://h2/generate"])
    assert _teacher_targets_for_sample(args, _tagged_sample({"opd_teacher": "math"})) == [("http://h1/generate", 1.0)]
    assert _teacher_targets_for_sample(args, _tagged_sample({"opd_teacher": "code"})) == [("http://h2/generate", 1.0)]


def test_routing_respects_custom_metadata_key():
    args = _routing_args(urls=["math=http://h1/generate"], key="task")
    assert _teacher_targets_for_sample(args, _tagged_sample({"task": "math"})) == [("http://h1/generate", 1.0)]


def test_routing_missing_name_uses_default_entry():
    args = _routing_args(urls=["math=http://h1/generate", "default=http://h3/generate"])
    assert _teacher_targets_for_sample(args, _tagged_sample({})) == [("http://h3/generate", 1.0)]


def test_routing_unknown_name_uses_default_entry():
    args = _routing_args(urls=["math=http://h1/generate", "default=http://h3/generate"])
    assert _teacher_targets_for_sample(args, _tagged_sample({"opd_teacher": "physics"})) == [("http://h3/generate", 1.0)]


def test_routing_unknown_name_without_default_raises():
    args = _routing_args(urls=["math=http://h1/generate"])
    with pytest.raises(ValueError, match="no 'default' entry"):
        _teacher_targets_for_sample(args, _tagged_sample({"opd_teacher": "physics"}))


def test_routing_missing_name_without_default_raises():
    args = _routing_args(urls=["math=http://h1/generate"])
    with pytest.raises(ValueError, match="no 'default' entry"):
        _teacher_targets_for_sample(args, _tagged_sample({}))


# ---------------------------------------------------------------------------
# Ensemble mixture math + exact tail-bucket KL (miles [4/N])
# ---------------------------------------------------------------------------


def test_mixture_log_probs_is_probability_space_mixture():
    t1 = torch.tensor([math.log(0.2), math.log(0.8)])
    t2 = torch.tensor([math.log(0.4), math.log(0.6)])
    mixed = _mixture_log_probs([t1, t2], [1.0, 1.0])
    assert mixed.tolist() == pytest.approx([math.log(0.3), math.log(0.7)], rel=1e-6)


def test_mixture_log_probs_respects_weights():
    t1 = torch.tensor([math.log(0.2)])
    t2 = torch.tensor([math.log(0.8)])
    mixed = _mixture_log_probs([t1, t2], [3.0, 1.0])
    assert mixed.tolist() == pytest.approx([math.log((3 * 0.2 + 0.8) / 4)], rel=1e-6)


def test_mixture_logprob_maps_mixes_per_token_id():
    m1 = [{5: math.log(0.2), 7: math.log(0.6)}]
    m2 = [{5: math.log(0.4), 7: math.log(0.2)}]
    mixed = _mixture_logprob_maps([m1, m2], [1.0, 1.0])
    assert mixed[0][5] == pytest.approx(math.log(0.3), rel=1e-6)
    assert mixed[0][7] == pytest.approx(math.log(0.4), rel=1e-6)


def test_mixture_logprob_maps_missing_id_raises():
    m1 = [{5: math.log(0.2)}]
    m2 = [{7: math.log(0.4)}]
    with pytest.raises(ValueError, match="missing logprob"):
        _mixture_logprob_maps([m1, m2], [1.0, 1.0])


def test_tail_bucket_reverse_kl_adds_exact_tail_term():
    student = [math.log(0.6), math.log(0.3)]
    teacher = [math.log(0.5), math.log(0.2)]
    expected = (
        0.6 * math.log(0.6 / 0.5)
        + 0.3 * math.log(0.3 / 0.2)
        + 0.1 * (math.log(0.1) - math.log(0.3))
    )
    assert _tail_bucket_reverse_kl(student, teacher) == pytest.approx(expected, rel=1e-9)


def test_tail_bucket_reverse_kl_full_mass_has_no_tail_term():
    student = [math.log(0.6), math.log(0.4)]
    teacher = [math.log(0.5), math.log(0.5)]
    expected = 0.6 * math.log(0.6 / 0.5) + 0.4 * math.log(0.4 / 0.5)
    assert _tail_bucket_reverse_kl(student, teacher) == pytest.approx(expected, rel=1e-9)


def test_topk_tail_bucket_requires_single_softmax_strategy():
    args = _args("union")
    args.opd_topk_tail_bucket = True
    with pytest.raises(ValueError, match="only-student"):
        _compute_topk_reverse_kl(args, _sample(), _teacher_payload())


def test_topk_ensemble_mixes_teachers_on_student_ids():
    # Two uniform-weight teachers scored at the student's top-k ids; the
    # per-token teacher logprob must be the probability-space mixture.
    args = _args("only-student")
    t1 = {
        "meta_info": {
            "input_token_ids_logprobs": [
                None,
                [_entry(0.3, 1), _entry(0.7, 2)],
                [_entry(0.4, 4), _entry(0.6, 5)],
            ]
        }
    }
    t2 = {
        "meta_info": {
            "input_token_ids_logprobs": [
                None,
                [_entry(0.5, 1), _entry(0.3, 2)],
                [_entry(0.2, 4), _entry(0.4, 5)],
            ]
        }
    }
    payload = {"teachers": [t1, t2], "teacher_weights": [1.0, 1.0]}

    reverse_kl = _compute_topk_reverse_kl(args, _sample(), payload)

    expected_0 = 0.6 * math.log(0.6 / 0.4) + 0.4 * math.log(0.4 / 0.5)
    expected_1 = 0.7 * math.log(0.7 / 0.3) + 0.3 * math.log(0.3 / 0.5)
    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1], rel=1e-6)


def test_topk_ensemble_rejects_non_student_strategy():
    args = _args("only-teacher")
    payload = {"teachers": [{}, {}], "teacher_weights": [1.0, 1.0]}
    with pytest.raises(ValueError, match="only-student"):
        _compute_topk_reverse_kl(args, Sample(tokens=[1, 2, 3], response_length=2), payload)


async def test_post_teacher_group_singleton_returns_raw_response(monkeypatch):
    from miles.orbit.opd import opd_sglang

    async def fake_post(url, payload, timeout_secs=None, max_response_bytes=None):
        # Assert the response cap is forwarded (payload-sized limit for scoring)
        assert max_response_bytes is None
        return {"meta_info": {"url": url}}

    monkeypatch.setattr(opd_sglang, "_post_json", fake_post)
    out = await _post_teacher_group([("http://h1/generate", 1.0)], {"p": 1}, None)
    assert out == {"meta_info": {"url": "http://h1/generate"}}


async def test_post_teacher_group_ensemble_returns_responses_and_weights(monkeypatch):
    from miles.orbit.opd import opd_sglang

    async def fake_post(url, payload, timeout_secs=None, max_response_bytes=None):
        # Assert the response cap is forwarded (payload-sized limit for scoring)
        assert max_response_bytes is None
        return {"meta_info": {"url": url}}

    monkeypatch.setattr(opd_sglang, "_post_json", fake_post)
    out = await _post_teacher_group([("http://h1/g", 2.0), ("http://h2/g", 1.0)], {"p": 1}, None)
    assert out["teachers"] == [{"meta_info": {"url": "http://h1/g"}}, {"meta_info": {"url": "http://h2/g"}}]
    assert out["teacher_weights"] == [2.0, 1.0]


# ---------------------------------------------------------------------------
# KL direction (--opd-kl-type): reverse (default) / forward / mixed
# (NeMo-RL DistillationLossFn parity, adapted to rollout-side scoring)
# ---------------------------------------------------------------------------

from miles.orbit.opd.opd_sglang import _tail_bucket_forward_kl  # noqa: E402


def _kl_args(kl_type, mixed_weight=0.5, strategy="only-student"):
    args = _args(strategy)
    args.opd_kl_type = kl_type
    args.opd_mixed_kl_weight = mixed_weight
    return args


def _expected_reverse():
    e0 = 0.6 * math.log(0.6 / 0.3) + 0.4 * math.log(0.4 / 0.7)
    e1 = 0.7 * math.log(0.7 / 0.4) + 0.3 * math.log(0.3 / 0.6)
    return e0, e1


def _expected_forward():
    # Forward KL over the renormalized set: teacher-probability weights.
    e0 = 0.3 * math.log(0.3 / 0.6) + 0.7 * math.log(0.7 / 0.4)
    e1 = 0.4 * math.log(0.4 / 0.7) + 0.6 * math.log(0.6 / 0.3)
    return e0, e1


def test_topk_kl_type_forward_uses_teacher_weights_and_direction():
    reverse_kl = _compute_topk_reverse_kl(_kl_args("forward"), _sample(), _teacher_payload())
    assert reverse_kl.tolist() == pytest.approx(list(_expected_forward()))


def test_topk_kl_type_mixed_combines_both_directions():
    r0, r1 = _expected_reverse()
    f0, f1 = _expected_forward()
    mixed = _compute_topk_reverse_kl(_kl_args("mixed", mixed_weight=0.25), _sample(), _teacher_payload())
    assert mixed.tolist() == pytest.approx([0.25 * f0 + 0.75 * r0, 0.25 * f1 + 0.75 * r1])


def test_topk_kl_type_default_is_reverse():
    r0, r1 = _expected_reverse()
    out = _compute_topk_reverse_kl(_args("only-student"), _sample(), _teacher_payload())
    assert out.tolist() == pytest.approx([r0, r1])


def test_tail_bucket_forward_kl_adds_exact_tail_term():
    student = [math.log(0.6), math.log(0.3)]
    teacher = [math.log(0.5), math.log(0.2)]
    expected = (
        0.5 * math.log(0.5 / 0.6)
        + 0.2 * math.log(0.2 / 0.3)
        + 0.3 * (math.log(0.3) - math.log(0.1))
    )
    assert _tail_bucket_forward_kl(student, teacher) == pytest.approx(expected, rel=1e-9)


def test_tail_bucket_forward_kl_full_teacher_mass_has_no_tail_term():
    student = [math.log(0.6), math.log(0.3)]
    teacher = [math.log(0.4), math.log(0.6)]
    expected = 0.4 * math.log(0.4 / 0.6) + 0.6 * math.log(0.6 / 0.3)
    assert _tail_bucket_forward_kl(student, teacher) == pytest.approx(expected, rel=1e-9)


def test_topk_tail_bucket_mixed_combines_both_tails():
    args = _kl_args("mixed", mixed_weight=0.5)
    args.opd_topk_tail_bucket = True
    out = _compute_topk_reverse_kl(args, _sample(), _teacher_payload())
    # Position 0: student probs (.6,.4) mass 1.0 -> reverse tail 0; teacher
    # (on student ids) probs (.3,.7) mass 1.0 -> forward tail 0.
    rev0 = 0.6 * math.log(0.6 / 0.3) + 0.4 * math.log(0.4 / 0.7)
    fwd0 = 0.3 * math.log(0.3 / 0.6) + 0.7 * math.log(0.7 / 0.4)
    assert out.tolist()[0] == pytest.approx(0.5 * fwd0 + 0.5 * rev0, rel=1e-6)
