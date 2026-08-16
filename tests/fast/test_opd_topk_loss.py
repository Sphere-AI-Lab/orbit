"""Closed-form + integration tests for the direct top-k OPD loss (Task 3 of the
direct top-k OPD plan): ``_topk_kl_terms``, the ``get_log_probs_and_entropy``
top-k extension, and ``opd_topk_loss_function``.

Step 1 builds every reference by hand -- explicit softmax via
``torch.log_softmax`` plus a plain Python loop over the K dimension -- so these
tests validate the implementation's math, not just its self-consistency with
itself. They are the authority on every sign convention in the loss (forward
vs. reverse weighting, the out-of-support correction, the mixed blend).
"""

import math
import warnings
from argparse import Namespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.fast.dist_utils import find_free_port, init_gloo

from orbit.backends.training_utils import teacher_lm_head as teacher_lm_head_module
from orbit.backends.training_utils.cp_utils import get_sum_of_sample_mean
from orbit.backends.training_utils.loss import (
    _TOPK_LOG_INF,
    _response_masked_min,
    _topk_kl_terms,
    _topk_overlap_membership,
    get_log_probs_and_entropy,
    opd_jsd_loss_function,
    opd_topk_loss_function,
)
from orbit.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _single_state() -> None:
    # tp.group carries a real (single-member) process group, not None: Megatron's
    # fused_vocab_parallel_cross_entropy (unconditionally called for get_log_probs_and_
    # entropy's log_probs half) calls tp_group.rank()/.size() directly and does not
    # accept None. `get_log_probs_and_entropy` still resolves the *topk gather*'s own
    # tp_group to None here (gated on parallel_state.tp.size > 1), exercising the
    # process_group=None path the brief asks for.
    single = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    set_parallel_state(ParallelState(intra_dp=single, intra_dp_cp=single, cp=single, tp=single))


def _row(teacher_logits: list[float], student_logits: list[float], k: int):
    """One response row: gather both sides at the *teacher's own* top-k ids,
    exactly like ``compute_vocab_parallel_topk_log_probs`` gathers the student
    at externally supplied ids. Returns (teacher_topk_lp, student_topk_lp,
    entropy) each shaped ``[1, ...]`` (single response position)."""
    t_logits = torch.tensor([teacher_logits], dtype=torch.float32)
    s_logits = torch.tensor([student_logits], dtype=torch.float32)
    t_lp_full = F.log_softmax(t_logits, dim=-1)
    s_lp_full = F.log_softmax(s_logits, dim=-1)
    ids = torch.topk(t_lp_full, k=k, dim=-1).indices
    t_topk_lp = t_lp_full.gather(-1, ids)
    s_topk_lp = s_lp_full.gather(-1, ids)
    entropy = -(s_lp_full.exp() * s_lp_full).sum(dim=-1)  # standard +H convention
    return ids, t_topk_lp, s_topk_lp, entropy


def _stack_rows(rows):
    """Stack per-row (ids, t_lp, s_lp, entropy) 4-tuples into batched [R, ...] tensors."""
    ids = torch.cat([r[0] for r in rows], dim=0)
    t_lp = torch.cat([r[1] for r in rows], dim=0)
    s_lp = torch.cat([r[2] for r in rows], dim=0)
    entropy = torch.cat([r[3] for r in rows], dim=0)
    return ids, t_lp, s_lp, entropy


# ---------------------------------------------------------------------------
# Step 1: closed-form _topk_kl_terms tests (pure, CPU)
# ---------------------------------------------------------------------------


def test_forward_matches_hand_sum():
    # V=5, K=2, R=2.
    rows = [
        _row([2.0, -1.0, 0.5, 3.0, -2.0], [1.0, 0.5, -0.5, 2.0, 1.5], k=2),
        _row([0.0, 1.0, 2.0, -1.0, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0], k=2),
    ]
    _, t_lp, s_lp, _ = _stack_rows(rows)

    result = _topk_kl_terms(t_lp, s_lp, None, "forward", 0.5, zero_outside=False)

    expected = []
    for r in range(2):
        total = 0.0
        for kk in range(2):
            w = math.exp(t_lp[r, kk].item())
            total += w * (t_lp[r, kk].item() - s_lp[r, kk].item())
        expected.append(total)
    assert torch.allclose(result, torch.tensor(expected), atol=1e-6)


def test_reverse_without_correction_matches_hand_sum():
    rows = [
        _row([2.0, -1.0, 0.5, 3.0, -2.0], [1.0, 0.5, -0.5, 2.0, 1.5], k=2),
        _row([0.0, 1.0, 2.0, -1.0, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0], k=2),
    ]
    _, t_lp, s_lp, _ = _stack_rows(rows)

    result = _topk_kl_terms(t_lp, s_lp, None, "reverse", 0.5, zero_outside=False)

    expected = []
    for r in range(2):
        total = 0.0
        for kk in range(2):
            w = math.exp(s_lp[r, kk].item())
            total += w * (s_lp[r, kk].item() - t_lp[r, kk].item())
        expected.append(total)
    assert torch.allclose(result, torch.tensor(expected), atol=1e-6)


def test_reverse_with_correction_matches_full_vocab_reference():
    """The correction's defining property: reverse + correction must equal the
    exact full-vocab reverse KL against a teacher whose out-of-(reported-top-k)
    slots are replaced by log_inf, built explicitly here (not via the impl)."""
    k = 3
    teacher_logits = torch.tensor(
        [
            [2.0, -1.0, 0.5, 3.0, -2.0, 1.0],
            [0.0, 1.0, 2.0, -1.0, 0.5, -0.5],
        ],
        dtype=torch.float32,
    )
    student_logits = torch.tensor(
        [
            [1.0, 0.5, -0.5, 2.0, 1.5, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    t_lp_full = F.log_softmax(teacher_logits, dim=-1)
    s_lp_full = F.log_softmax(student_logits, dim=-1)
    s_p_full = s_lp_full.exp()

    ids = torch.topk(t_lp_full, k=k, dim=-1).indices
    t_topk_lp = t_lp_full.gather(-1, ids)
    s_topk_lp = s_lp_full.gather(-1, ids)
    entropy = -(s_p_full * s_lp_full).sum(dim=-1)

    # Explicit full-vocab reference: teacher extended with log_inf outside its
    # reported top-k support.
    in_support = torch.zeros_like(t_lp_full, dtype=torch.bool)
    in_support.scatter_(-1, ids, True)
    t_lp_ext = torch.where(in_support, t_lp_full, torch.full_like(t_lp_full, _TOPK_LOG_INF))
    reference = (s_p_full * (s_lp_full - t_lp_ext)).sum(dim=-1)

    result = _topk_kl_terms(t_topk_lp, s_topk_lp, entropy, "reverse", 0.5, zero_outside=True)

    assert torch.allclose(result, reference, atol=1e-5), f"{result} vs {reference}"


@pytest.mark.parametrize("w", [0.0, 0.3, 1.0])
def test_mixed_is_affine_combination(w):
    rows = [
        _row([2.0, -1.0, 0.5, 3.0, -2.0], [1.0, 0.5, -0.5, 2.0, 1.5], k=2),
        _row([0.0, 1.0, 2.0, -1.0, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0], k=2),
    ]
    _, t_lp, s_lp, entropy = _stack_rows(rows)

    forward = _topk_kl_terms(t_lp, s_lp, None, "forward", 0.5, zero_outside=False)
    reverse = _topk_kl_terms(t_lp, s_lp, entropy, "reverse", 0.5, zero_outside=True)
    mixed = _topk_kl_terms(t_lp, s_lp, entropy, "mixed", w, zero_outside=True)

    expected = w * forward + (1 - w) * reverse
    assert torch.allclose(mixed, expected, atol=1e-6)


def test_pad_slot_excluded_changes_nothing():
    """A padded (id=0, logprob=_TOPK_PAD_LOGPROB) column must not change the result
    versus the same row with that slot simply absent."""
    pad_lp = -1e4
    # K=3 row: 2 real slots + 1 pad.
    t_lp_padded = torch.tensor([[-0.5, -1.5, pad_lp]])
    s_lp_padded = torch.tensor([[-0.7, -1.2, -3.0]])  # padded student value must not matter either
    # Same row with only the 2 real slots (K=2, no padding).
    t_lp_bare = torch.tensor([[-0.5, -1.5]])
    s_lp_bare = torch.tensor([[-0.7, -1.2]])

    for kl_type, zero_outside in [("forward", False), ("reverse", False)]:
        padded = _topk_kl_terms(t_lp_padded, s_lp_padded, None, kl_type, 0.5, zero_outside)
        bare = _topk_kl_terms(t_lp_bare, s_lp_bare, None, kl_type, 0.5, zero_outside)
        assert torch.allclose(padded, bare, atol=1e-6), kl_type


def test_all_pad_row_forward_and_reverse_zero_finite_grad():
    """A row where every K slot is a pad sentinel (e.g. an injected merge-observation
    filler position that a downstream loss_mask=0 will exclude): the *uncorrected*
    forward/reverse terms must be exactly 0 (nothing valid to sum), and gradients
    into the student side must be finite (no NaN/Inf despite exp(-1e4))."""
    pad_lp = -1e4
    t_lp = torch.tensor([[pad_lp, pad_lp]])
    s_lp = torch.tensor([[-1.0, -2.0]], requires_grad=True)

    forward = _topk_kl_terms(t_lp, s_lp, None, "forward", 0.5, zero_outside=False)
    assert torch.equal(forward, torch.zeros_like(forward))
    forward.sum().backward()
    assert torch.isfinite(s_lp.grad).all()

    s_lp2 = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    reverse = _topk_kl_terms(t_lp, s_lp2, None, "reverse", 0.5, zero_outside=False)
    assert torch.equal(reverse, torch.zeros_like(reverse))
    reverse.sum().backward()
    assert torch.isfinite(s_lp2.grad).all()

    # With the correction, an all-pad row is no longer trivially 0 (it represents
    # "the teacher reported zero real top-k entries here" -- maximal divergence
    # under the log_inf floor) but must stay finite.
    s_lp3 = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    entropy = -(s_lp3.exp() * s_lp3).sum(dim=-1).detach()
    corrected = _topk_kl_terms(t_lp, s_lp3, entropy, "reverse", 0.5, zero_outside=True)
    assert torch.isfinite(corrected).all()
    corrected.sum().backward()
    assert torch.isfinite(s_lp3.grad).all()


def test_k_ge_v_forward_equals_full_vocab_forward_kl():
    """CPU endpoint pin: with K covering the whole vocabulary (no padding), the
    top-k forward KL must equal the exact full-vocab forward KL."""
    teacher_logits = torch.tensor([[2.0, -1.0, 0.5, 3.0, -2.0]])
    student_logits = torch.tensor([[1.0, 0.5, -0.5, 2.0, 1.5]])
    t_lp_full = F.log_softmax(teacher_logits, dim=-1)
    s_lp_full = F.log_softmax(student_logits, dim=-1)

    ids = torch.topk(t_lp_full, k=5, dim=-1).indices
    t_topk_lp = t_lp_full.gather(-1, ids)
    s_topk_lp = s_lp_full.gather(-1, ids)

    result = _topk_kl_terms(t_topk_lp, s_topk_lp, None, "forward", 0.5, zero_outside=False)
    reference = (t_lp_full.exp() * (t_lp_full - s_lp_full)).sum(dim=-1)
    assert torch.allclose(result, reference, atol=1e-6)


def test_zero_outside_with_forward_warns_and_is_inert():
    t_lp = torch.tensor([[-0.5, -1.5]])
    s_lp = torch.tensor([[-0.7, -1.2]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with_warn = _topk_kl_terms(t_lp, s_lp, None, "forward", 0.5, zero_outside=True)
    assert any("zero_outside" in str(w.message) or "zero-outside" in str(w.message) for w in caught)

    without_warn = _topk_kl_terms(t_lp, s_lp, None, "forward", 0.5, zero_outside=False)
    assert torch.equal(with_warn, without_warn)


# ---------------------------------------------------------------------------
# Regression (code review round 1): _response_masked_min's empty/all-masked-sample
# identity. `-_response_masked_max(-x, ...)` silently reported `-0.` for any batch
# containing an empty or fully-masked sample, because `_response_masked_max`'s `0`
# fallback (a safe identity for maxing a non-negative quantity) becomes the *supremum*
# once negated into a min -- it then wins over every real value in [0, 1] and dominates
# the reported minimum. Reviewer repro: real per-sample masses [0.66, 0.70, 0.58] with
# one empty sample -> old code reported -0. instead of 0.58.
# ---------------------------------------------------------------------------


def test_response_masked_min_ignores_empty_sample_not_zero():
    _single_state()
    real_masses = torch.tensor([0.66, 0.70, 0.58])  # sample 0's 3 real per-position masses
    x = torch.cat([real_masses, torch.zeros(0)])  # sample 1: empty response, contributes 0 rows

    result = _response_masked_min(
        x,
        total_lengths=[5, 2],
        response_lengths=[3, 0],
        loss_masks=[torch.ones(3, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)],
    )

    assert torch.allclose(result, torch.tensor(0.58), atol=1e-6), result


def test_response_masked_min_all_masked_sample_excluded_too():
    """A non-empty but fully-masked sample (loss_mask all 0) must also be excluded from
    the min, not just a genuinely empty (R=0) one."""
    _single_state()
    real_masses = torch.tensor([0.9])
    masked_out_masses = torch.tensor([0.01, 0.02])  # would wrongly win an unguarded min
    x = torch.cat([real_masses, masked_out_masses])

    result = _response_masked_min(
        x,
        total_lengths=[1, 2],
        response_lengths=[1, 2],
        loss_masks=[torch.ones(1, dtype=torch.int64), torch.zeros(2, dtype=torch.int64)],
    )

    assert torch.allclose(result, torch.tensor(0.9), atol=1e-6), result


def test_response_masked_min_no_valid_sample_falls_back_to_one():
    _single_state()
    x = torch.zeros(0)

    result = _response_masked_min(
        x,
        total_lengths=[2],
        response_lengths=[0],
        loss_masks=[torch.zeros(0, dtype=torch.int64)],
    )

    assert torch.allclose(result, torch.tensor(1.0)), result


# ---------------------------------------------------------------------------
# Step 3: opd_topk_loss_function integration test
# ---------------------------------------------------------------------------


# Keep a real single-member default group to mirror the normal trainer state. The
# direct top-k fast path no longer invokes the sampled-token fused CE kernel, but
# adjacent full-vocab equivalence cases still exercise distributed helpers.
@pytest.fixture(scope="module", autouse=True)
def _single_process_gloo_group():
    init_gloo(0, 1, port=find_free_port())
    yield
    dist.destroy_process_group()


VOCAB = 6
K = 2


def _build_args(kl_type: str = "reverse", mixed_weight: float = 0.5, zero_outside=None) -> Namespace:
    return Namespace(
        qkv_format="thd",
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        allgather_cp=False,
        vocab_size=None,
        opd_kl_type=kl_type,
        opd_mixed_kl_weight=mixed_weight,
        opd_topk_zero_outside=zero_outside,
    )


def _build_batch(generator: torch.Generator):
    # sample 0: response_length=3, prompt_length=2; sample 1: empty response (R=0).
    response_lengths = [3, 0]
    prompt_lengths = [2, 2]
    total_lengths = [p + r for p, r in zip(prompt_lengths, response_lengths, strict=True)]

    unconcat_tokens = [torch.randint(0, VOCAB, (total,), generator=generator) for total in total_lengths]
    loss_masks = [torch.ones(r, dtype=torch.int64) for r in response_lengths]

    teacher_topk_ids = []
    teacher_topk_logprobs = []
    for r in response_lengths:
        if r == 0:
            # Real transport shape for an empty response: the raw per-sample payload
            # is a plain Python `[]` (not `[[], ...]`), so torch.tensor([]) tensorizes
            # to a 1-D `[0]` shape, not `[0, K]`.
            teacher_topk_ids.append(torch.zeros(0, dtype=torch.long))
            teacher_topk_logprobs.append(torch.zeros(0, dtype=torch.float32))
            continue
        teacher_logits = torch.randn(r, VOCAB, generator=generator)
        t_lp = F.log_softmax(teacher_logits, dim=-1)
        ids = torch.topk(t_lp, k=K, dim=-1).indices
        teacher_topk_ids.append(ids)
        teacher_topk_logprobs.append(t_lp.gather(-1, ids))

    logits = torch.randn(1, sum(total_lengths), VOCAB, generator=generator, dtype=torch.float32)

    batch = {
        "unconcat_tokens": unconcat_tokens,
        "response_lengths": response_lengths,
        "total_lengths": total_lengths,
        "loss_masks": loss_masks,
        "teacher_topk_ids": teacher_topk_ids,
        "teacher_topk_logprobs": teacher_topk_logprobs,
    }
    return logits, batch


@pytest.mark.parametrize("kl_type", ["forward", "reverse", "mixed"])
def test_opd_topk_loss_function_end_to_end(kl_type):
    _single_state()
    generator = torch.Generator().manual_seed(0)
    logits, batch = _build_batch(generator)
    logits = logits.detach().clone().requires_grad_(True)
    args = _build_args(kl_type=kl_type)

    loss, metrics = opd_topk_loss_function(args, batch, logits, sum_of_sample_mean=lambda x: x.sum())

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()

    for key in (
        "loss",
        "opd_topk/teacher_mass",
        "opd_topk/teacher_mass_min",
        "opd_topk/student_mass",
        "opd_topk/overlap_ratio",
    ):
        assert key in metrics, key
        assert torch.isfinite(metrics[key]), key


def test_opd_topk_loss_excludes_megatron_padding_from_distribution_and_grad():
    """Identical real-vocab teacher/student distributions have zero forward KL.

    Very large padded logits make the historical bug decisive: normalizing over
    all four model columns instead of the real two-token vocabulary creates a
    nonzero loss and gradients in the padded columns.
    """
    _single_state()
    real_student_logits = torch.tensor([[2.0, 1.0]])
    teacher_log_probs = F.log_softmax(real_student_logits, dim=-1)
    teacher_ids = torch.tensor([[0, 1]], dtype=torch.long)

    full_logits = torch.zeros(1, 3, 4)
    full_logits[0, 1] = torch.tensor([2.0, 1.0, 10.0, 11.0])
    logits = full_logits.requires_grad_(True)
    batch = {
        "unconcat_tokens": [torch.tensor([0, 0, 1])],
        "response_lengths": [1],
        "total_lengths": [3],
        "loss_masks": [torch.ones(1, dtype=torch.int64)],
        "teacher_topk_ids": [teacher_ids],
        "teacher_topk_logprobs": [teacher_log_probs],
    }
    args = _build_args(kl_type="forward")
    args.vocab_size = 2

    loss, metrics = opd_topk_loss_function(args, batch, logits, sum_of_sample_mean=lambda x: x.sum())
    torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=0)
    torch.testing.assert_close(metrics["opd_topk/student_mass"], torch.tensor(1.0), rtol=0, atol=1e-7)
    torch.testing.assert_close(metrics["opd_topk/overlap_ratio"], torch.tensor(1.0), rtol=0, atol=0)

    loss.backward()
    torch.testing.assert_close(logits.grad[..., 2:], torch.zeros_like(logits.grad[..., 2:]), rtol=0, atol=0)


def test_true_on_policy_topk_scores_and_entropy_share_native_bf16_real_vocab():
    _single_state()
    full_logits = torch.zeros(1, 3, 4, dtype=torch.bfloat16)
    full_logits[0, 1] = torch.tensor([2.0, 1.0, 10.0, 11.0], dtype=torch.bfloat16)
    logits = full_logits.requires_grad_(True)
    ids = torch.tensor([[0, 1]], dtype=torch.long)
    args = _build_args(kl_type="reverse")
    args.true_on_policy_mode = True
    args.vocab_size = 2

    result = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=[torch.tensor([0, 0, 1])],
        total_lengths=[3],
        response_lengths=[1],
        with_entropy=True,
        teacher_topk_ids=[ids],
        with_log_probs=False,
    )

    expected_log_probs = torch.log_softmax(logits[0, 1, :2], dim=-1)
    expected_entropy = -(expected_log_probs.exp() * expected_log_probs).sum()
    assert result["log_probs"] == []  # sampled-token CE fast path is skipped
    assert result["student_topk_log_probs"][0].dtype == torch.bfloat16
    torch.testing.assert_close(result["student_topk_log_probs"][0], expected_log_probs.unsqueeze(0), rtol=0, atol=0)
    torch.testing.assert_close(result["entropy"][0], expected_entropy.unsqueeze(0), rtol=0, atol=0)

    (result["student_topk_log_probs"][0].sum() + result["entropy"][0].sum()).backward()
    torch.testing.assert_close(logits.grad[..., 2:], torch.zeros_like(logits.grad[..., 2:]), rtol=0, atol=0)


@pytest.mark.parametrize("with_teacher_topk", [False, True])
def test_true_on_policy_single_token_log_probs_keep_vector_shape(with_teacher_topk):
    _single_state()
    logits = torch.zeros(1, 3, 4, dtype=torch.bfloat16)
    logits[0, 1] = torch.tensor([2.0, 1.0, 10.0, 11.0], dtype=torch.bfloat16)
    args = _build_args()
    args.true_on_policy_mode = True
    args.vocab_size = 2

    result = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=[torch.tensor([0, 0, 1])],
        total_lengths=[3],
        response_lengths=[1],
        teacher_topk_ids=[torch.tensor([[0, 1]])] if with_teacher_topk else None,
    )

    assert result["log_probs"][0].shape == (1,)


def test_opd_topk_loss_function_diagnostics_match_hand_computed_references():
    """Regression for finding 3 (code review round 1): the diagnostics were only ever
    checked for finiteness, which is exactly why finding 1's `teacher_mass_min` sign bug
    slipped through. This pins all four `opd_topk/*` metrics against references computed
    independently in the test (fixed, non-random logits; the real `get_sum_of_sample_mean`
    reducer, not a toy `.sum()`), including a genuinely empty-response sample so
    `teacher_mass_min` must equal the min over the *real* sample's positions only."""
    _single_state()

    # sample 0: 3 real response positions; sample 1: empty response (R=0).
    response_lengths = [3, 0]
    total_lengths = [5, 2]  # prompt_length=2 for both
    unconcat_tokens = [torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1])]
    loss_masks = [torch.ones(3, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)]

    teacher_logits_0 = torch.tensor(
        [
            [2.0, -1.0, 0.5, 3.0, -2.0, 1.0],
            [0.0, 1.0, 2.0, -1.0, 0.5, -0.5],
            [1.0, 1.0, -1.0, 0.5, 2.0, 0.0],
        ]
    )
    student_logits_0 = torch.tensor(
        [
            [1.0, 0.5, -0.5, 2.0, 1.5, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, -0.5, 1.5, 0.0, -1.0, 2.0],
        ]
    )

    t_lp_0 = F.log_softmax(teacher_logits_0, dim=-1)
    ids_0 = torch.topk(t_lp_0, k=K, dim=-1).indices
    t_topk_lp_0 = t_lp_0.gather(-1, ids_0)

    # Real transport shape for an empty response: 1-D `[0]`, not `[0, K]` (see
    # _build_batch above).
    teacher_topk_ids = [ids_0, torch.zeros(0, dtype=torch.long)]
    teacher_topk_logprobs = [t_topk_lp_0, torch.zeros(0, dtype=torch.float32)]

    # Embed student_logits_0 at the exact rows get_responses' thd slicing picks out for
    # sample 0: end=5, start=end-response_length=2 -> logits[start-1:end-1] = logits[1:4].
    full_logits = torch.zeros(1, sum(total_lengths), VOCAB)
    full_logits[0, 1:4] = student_logits_0
    logits = full_logits.detach().clone().requires_grad_(True)

    batch = {
        "unconcat_tokens": unconcat_tokens,
        "response_lengths": response_lengths,
        "total_lengths": total_lengths,
        "loss_masks": loss_masks,
        "teacher_topk_ids": teacher_topk_ids,
        "teacher_topk_logprobs": teacher_topk_logprobs,
    }
    args = _build_args(kl_type="reverse")
    sum_of_sample_mean = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, False, "thd", None)

    loss, metrics = opd_topk_loss_function(args, batch, logits, sum_of_sample_mean)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()

    # Independent references built from the same fixed logits, not by reusing the
    # implementation's own vectorized ops.
    s_lp_0 = F.log_softmax(student_logits_0, dim=-1)
    teacher_mass_ref = t_topk_lp_0.exp().sum(dim=-1)  # [3], no padding (V=6 >= K=2)
    student_mass_ref = s_lp_0.exp().gather(-1, ids_0).sum(dim=-1)  # [3]
    overlap_ref = torch.tensor(
        [
            len(set(torch.topk(student_logits_0[r], k=K).indices.tolist()) & set(ids_0[r].tolist())) / K
            for r in range(3)
        ]
    )

    # sample 1 is empty: get_sum_of_sample_mean's per-sample term for it is 0/clamp_min(0,1)
    # = 0, so the aggregate below reduces to sample 0's own mean over its 3 positions.
    torch.testing.assert_close(metrics["opd_topk/teacher_mass"], teacher_mass_ref.mean())
    torch.testing.assert_close(metrics["opd_topk/student_mass"], student_mass_ref.mean())
    torch.testing.assert_close(metrics["opd_topk/overlap_ratio"], overlap_ref.mean())
    # The regression itself: min over the *real* sample's positions only, not -0./0.
    # from the empty sample 1.
    torch.testing.assert_close(metrics["opd_topk/teacher_mass_min"], teacher_mass_ref.min())


# ---------------------------------------------------------------------------
# Regression (final-review finding 2): teacher-vocab overhang ids must be masked to
# a pad slot before the student gather, not corrupt it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kl_type", ["forward", "reverse"])
def test_teacher_vocab_overhang_id_masked_like_pad_slot(kl_type):
    """A bigger-config-vocab teacher (e.g. Qwen2.5-7B pads to 152064 vs a <3B
    student's 151936) can report top-k ids past the student's own vocabulary. Left
    unmasked these break compute_vocab_parallel_topk_log_probs's gather: at TP=1 they
    index-error (this test's VOCAB=6 student would IndexError on id=11 without the
    fix); at TP>1 every rank's ownership mask is False for them so the gather
    silently returns a fake `0 - log_normalizer` value instead. Pins the fix's
    behavior: a slot with an overhang id must produce exactly the same loss and
    diagnostics as the same slot manually replaced by a pad slot (id=0,
    logprob=-1e4, matching orbit.rollout.opd_sglang's own padding convention)."""
    _single_state()

    response_lengths = [2]
    total_lengths = [4]  # prompt_length=2
    unconcat_tokens = [torch.tensor([0, 1, 2, 3])]
    loss_masks = [torch.ones(2, dtype=torch.int64)]

    student_logits = torch.tensor(
        [
            [1.0, 0.5, -0.5, 2.0, 1.5, 0.0],
            [0.2, -0.3, 1.1, 0.0, -0.7, 0.9],
        ]
    )
    # get_responses' thd slicing for response_length=2, total_length=4: end=4,
    # start=end-2=2 -> logits[start-1:end-1] = logits[1:3].
    full_logits = torch.zeros(1, sum(total_lengths), VOCAB)
    full_logits[0, 1:3] = student_logits

    overhang_id = VOCAB + 5  # past the student's vocabulary (VOCAB=6)
    teacher_topk_ids_overhang = torch.tensor([[0, overhang_id], [1, overhang_id]], dtype=torch.long)
    teacher_topk_logprobs = torch.tensor([[-0.5, -0.2], [-0.9, -0.1]], dtype=torch.float32)

    # Reference: the overhang slot manually replaced by a pad slot, exactly like the
    # transport's own padding (_TOPK_PAD_TOKEN_ID=0, _TOPK_PAD_LOGPROB=-1e4).
    teacher_topk_ids_padded = torch.tensor([[0, 0], [1, 0]], dtype=torch.long)
    teacher_topk_logprobs_padded = torch.tensor([[-0.5, -1e4], [-0.9, -1e4]], dtype=torch.float32)

    args = _build_args(kl_type=kl_type)
    sum_of_sample_mean = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, False, "thd", None)

    def _run(ids, logprobs):
        logits = full_logits.detach().clone().requires_grad_(True)
        batch = {
            "unconcat_tokens": unconcat_tokens,
            "response_lengths": response_lengths,
            "total_lengths": total_lengths,
            "loss_masks": loss_masks,
            "teacher_topk_ids": [ids.clone()],
            "teacher_topk_logprobs": [logprobs.clone()],
        }
        return opd_topk_loss_function(args, batch, logits, sum_of_sample_mean)

    loss_overhang, metrics_overhang = _run(teacher_topk_ids_overhang, teacher_topk_logprobs)
    loss_padded, metrics_padded = _run(teacher_topk_ids_padded, teacher_topk_logprobs_padded)

    torch.testing.assert_close(loss_overhang, loss_padded)
    for key in (
        "opd_topk/teacher_mass",
        "opd_topk/teacher_mass_min",
        "opd_topk/student_mass",
        "opd_topk/overlap_ratio",
    ):
        torch.testing.assert_close(metrics_overhang[key], metrics_padded[key])


# ---------------------------------------------------------------------------
# Regression (gate-discovered defect 4): the overlap diagnostic's `[R, K, K]` broadcast
# OOMs at k >= vocab (688 GiB at k=vocab_size=151936, R=32, confirmed in the gate log).
# `_topk_overlap_membership` replaces it with a sort + `torch.searchsorted` (O(R*K)
# memory) -- these pin exact equivalence to the old broadcast and a scale smoke test
# that would OOM the old code even on CPU at this size.
# ---------------------------------------------------------------------------


def test_topk_overlap_membership_matches_kxk_broadcast_reference():
    """Random ids with duplicates of the pad sentinel (-1) and some -1-masked teacher
    slots, small k -- `_topk_overlap_membership`'s per-row overlap must exactly equal
    the old `[R, K, K]` broadcast-equality it replaces, computed explicitly here (not
    by re-deriving the same vectorized ops as the implementation)."""
    generator = torch.Generator().manual_seed(0)
    r, k_student, k_teacher, vocab = 6, 5, 7, 20

    student_topk_ids = torch.randint(0, vocab, (r, k_student), generator=generator, dtype=torch.long)
    teacher_ids_for_match = torch.randint(0, vocab, (r, k_teacher), generator=generator, dtype=torch.long)
    # Mask ~40% of teacher slots to the -1 sentinel (mirrors invalid/pad slots) -- some
    # rows end up with several -1 duplicates, exercising the sentinel-block sort case.
    mask = torch.rand(r, k_teacher, generator=generator) < 0.4
    teacher_ids_for_match = torch.where(mask, torch.full_like(teacher_ids_for_match, -1), teacher_ids_for_match)

    # Reference: the exact old implementation, an explicit [R, K, K] broadcast.
    reference = (student_topk_ids.unsqueeze(-1) == teacher_ids_for_match.unsqueeze(-2)).any(dim=-1)

    result = _topk_overlap_membership(student_topk_ids, teacher_ids_for_match)

    torch.testing.assert_close(result, reference)


def test_topk_overlap_membership_scale_smoke_no_kxk_materialization():
    """R=8, k=20000: the old `[R, K, K]` broadcast would allocate 8*20000^2 = 3.2e9
    bools (3.2 GB) on CPU alone at this size -- at the gate's real k=vocab_size=151936,
    R=32 it was 688 GiB and OOM'd. Must complete without materializing a K*K tensor and
    produce a valid ratio in [0, 1]. Do NOT run the old K*K-broadcast code at this size."""
    generator = torch.Generator().manual_seed(1)
    r, k = 8, 20000

    student_topk_ids = torch.randint(0, k, (r, k), generator=generator, dtype=torch.long)
    teacher_ids_for_match = torch.randint(0, k, (r, k), generator=generator, dtype=torch.long)

    match = _topk_overlap_membership(student_topk_ids, teacher_ids_for_match)
    overlap_ratio = match.sum(dim=-1).float() / k

    assert match.shape == (r, k)
    assert torch.isfinite(overlap_ratio).all()
    assert (overlap_ratio >= 0).all() and (overlap_ratio <= 1).all()


def test_topk_overlap_membership_all_sentinel_row_never_matches():
    """A row where every teacher slot is the -1 sentinel: no student id (always >= 0)
    can match, regardless of duplicate -1s. Verifies the sort-then-searchsorted
    reasoning that -1 sentinels sort to the front of the row and are therefore inert
    against real (>= 0) ids -- a lower-bound search for a non-negative value can never
    land inside the leading -1 block."""
    student_topk_ids = torch.tensor([[0, 3, 3, 19]], dtype=torch.long)
    teacher_ids_for_match = torch.full((1, 4), -1, dtype=torch.long)

    result = _topk_overlap_membership(student_topk_ids, teacher_ids_for_match)

    assert not result.any()


# ---------------------------------------------------------------------------
# Decisive direct-loss equivalence: opd_topk_loss_function vs opd_jsd_loss_function
# on IDENTICAL inputs in one process. The end-to-end GPU gate (train each loss under
# a separate launcher, compare curves) is confounded -- different launchers produce
# different sampled bf16 rollouts, so a curve comparison can never prove numerical
# equivalence. Running both production loss functions on the same tensors in the
# same process is the only way to actually settle it, and it's a permanent
# regression test besides.
#
# beta<->direction mapping, derived from opd_jsd_loss_function's code (not assumed
# from its docstring): the beta==0.0 and beta==1.0 branches bypass the mixture
# entirely and hard-code
#   beta=0.0: teacher_probs * (teacher_logp - student_logp) = KL(teacher||student)  (teacher-weighted)
#   beta=1.0: student_probs * (student_logp - teacher_logp) = KL(student||teacher)  (student-weighted)
# which is exactly _topk_kl_terms's own forward (teacher-weighted) / reverse
# (student-weighted) split -- so beta=0.0 pairs with --opd-kl-type forward and
# beta=1.0 with --opd-kl-type reverse. This is the mapping the docstring already
# claimed, but it does not fall out of the mixture *formula* shown there (plugging
# b=0 or b=1 into `jsd(b) = b*KL(teacher||M) + (1-b)*KL(student||M)` degenerates to
# `KL(Q||Q)=0`, not the stated endpoint value); only the special-cased branches
# produce it, which is why the docstring is corrected in this same commit to say so.
# ---------------------------------------------------------------------------

_EQUIV_VOCAB_SIZE = 64
_EQUIV_CHECKPOINT_KEY = "<test-opd-topk-direct-equivalence>"


def _build_direct_equivalence_inputs(seed: int = 12345):
    """One student logits tensor plus one teacher distribution per response
    position, expressed in both loss functions' native input formats: jsd's
    `teacher_hidden_states` (reconstructed through an identity LM head, so the
    reconstruction is exact) and topk's `teacher_topk_ids` = `arange(V)` /
    `teacher_topk_logprobs` = `log_softmax(teacher_logits)` -- i.e. k >= vocab,
    no padding, no truncation."""
    generator = torch.Generator().manual_seed(seed)
    response_lengths = [3, 2]
    prompt_lengths = [2, 3]
    total_lengths = [p + r for p, r in zip(prompt_lengths, response_lengths, strict=True)]

    unconcat_tokens = [torch.randint(0, _EQUIV_VOCAB_SIZE, (total,), generator=generator) for total in total_lengths]
    loss_masks = [torch.ones(r, dtype=torch.int64) for r in response_lengths]
    logits = torch.randn(1, sum(total_lengths), _EQUIV_VOCAB_SIZE, generator=generator, dtype=torch.float32)

    teacher_logits_per_sample = [
        torch.randn(r, _EQUIV_VOCAB_SIZE, generator=generator, dtype=torch.float32) for r in response_lengths
    ]
    teacher_logprobs_per_sample = [F.log_softmax(tl, dim=-1) for tl in teacher_logits_per_sample]

    common = {
        "unconcat_tokens": unconcat_tokens,
        "response_lengths": response_lengths,
        "total_lengths": total_lengths,
        "loss_masks": loss_masks,
    }
    batch_jsd = {**common, "teacher_hidden_states": teacher_logits_per_sample}
    teacher_topk_ids = [
        torch.arange(_EQUIV_VOCAB_SIZE, dtype=torch.long).unsqueeze(0).expand(r, _EQUIV_VOCAB_SIZE).clone()
        for r in response_lengths
    ]
    batch_topk = {
        **common,
        "teacher_topk_ids": teacher_topk_ids,
        "teacher_topk_logprobs": teacher_logprobs_per_sample,
    }

    sum_of_sample_mean = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, False, "thd", None)
    return logits, batch_jsd, batch_topk, sum_of_sample_mean


def _build_jsd_args(beta: float) -> Namespace:
    return Namespace(
        opd_jsd_beta=beta,
        rollout_temperature=1.0,
        # Inert: real log-probs / summed KL at V=64 never approach these bounds, so
        # both losses are compared on their unclamped math, not a clamp artifact.
        opd_log_prob_min_clamp=-1e30,
        opd_loss_max_clamp=1e30,
        opd_jsd_pointwise_clip=None,
        opd_log_topk_overlap=False,
        use_kl_loss=False,
        teacher_hf_checkpoint=_EQUIV_CHECKPOINT_KEY,
        qkv_format="thd",
        allgather_cp=False,
        log_probs_chunk_size=-1,
        true_on_policy_mode=False,
        vocab_size=_EQUIV_VOCAB_SIZE,
    )


def _build_topk_args(kl_type: str, zero_outside: bool | None) -> Namespace:
    return Namespace(
        qkv_format="thd",
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        allgather_cp=False,
        vocab_size=None,
        opd_kl_type=kl_type,
        opd_mixed_kl_weight=0.5,
        opd_topk_zero_outside=zero_outside,
    )


def _run_both_losses(beta: float, kl_type: str, zero_outside: bool | None):
    _single_state()
    logits, batch_jsd, batch_topk, sum_of_sample_mean = _build_direct_equivalence_inputs()

    # Identity LM head: teacher_hidden_states @ I.T == teacher_hidden_states exactly
    # (every output element sums exact zeros plus one exact *1.0 term -- no rounding),
    # so opd_jsd_loss_function reconstructs precisely the logits placed into
    # teacher_hidden_states, with zero reconstruction error to worry about.
    teacher_lm_head_module._TEACHER_LM_HEAD_CACHE[_EQUIV_CHECKPOINT_KEY] = torch.eye(
        _EQUIV_VOCAB_SIZE, dtype=torch.float32
    )
    teacher_lm_head_module._SHARDED.add(_EQUIV_CHECKPOINT_KEY)

    args_jsd = _build_jsd_args(beta)
    args_topk = _build_topk_args(kl_type, zero_outside)

    logits_jsd = logits.detach().clone().requires_grad_(True)
    loss_jsd, _ = opd_jsd_loss_function(args_jsd, batch_jsd, logits_jsd, sum_of_sample_mean)
    loss_jsd.backward()

    logits_topk = logits.detach().clone().requires_grad_(True)
    loss_topk, _ = opd_topk_loss_function(args_topk, batch_topk, logits_topk, sum_of_sample_mean)
    loss_topk.backward()

    return loss_jsd.detach(), logits_jsd.grad.detach(), loss_topk.detach(), logits_topk.grad.detach()


def test_opd_topk_forward_matches_opd_jsd_beta0_at_k_ge_vocab():
    """k >= vocab (teacher_topk_ids = arange(V): no padding, no truncation): the
    top-k forward KL(teacher||student) must equal opd_jsd_loss's beta=0.0 branch,
    which sums the identical (teacher, student) distributions the identical way.
    Both losses read the same student logits through the same `get_responses`
    slicing, so this also pins gradient equivalence w.r.t. those logits -- the
    part that actually matters for training, not just the scalar loss value.

    Observed: bit-exact (0.0 diff, both loss and grad) -- forward never touches the
    entropy kernel, so there is no second code path to disagree with the first."""
    loss_jsd, grad_jsd, loss_topk, grad_topk = _run_both_losses(beta=0.0, kl_type="forward", zero_outside=None)

    torch.testing.assert_close(loss_topk, loss_jsd, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_topk, grad_jsd, atol=1e-5, rtol=1e-5)


def test_opd_topk_reverse_matches_opd_jsd_beta1_at_k_ge_vocab():
    """k >= vocab: the top-k reverse KL(student||teacher) with the zero-outside
    correction (a no-op here -- there is no student mass outside a full-support
    top-k) must equal opd_jsd_loss's beta=1.0 branch. Unlike the forward case, the
    correction recomputes `sum_v student_prob(v) * student_logprob(v)` (and relies
    on `sum_v student_prob(v) == 1`) through the entropy kernel
    (`_VocabParallelEntropy`), a second, independently-implemented code path from
    the plain log_softmax+gather the rest of the loss uses -- so the two losses
    only agree up to float32 cross-path rounding here, not bit-exactly.

    Observed max over a 20-seed x {16, 64, 256}-vocab sweep: loss diff < 2e-5, grad
    diff < 3e-6 -- both several orders of magnitude below the tolerance here, and
    consistent with float32 cross-path rounding rather than a real sign/semantic
    mismatch (which would show up at O(0.1-1.0), the scale of the quantities
    themselves, not O(1e-5))."""
    loss_jsd, grad_jsd, loss_topk, grad_topk = _run_both_losses(beta=1.0, kl_type="reverse", zero_outside=True)

    torch.testing.assert_close(loss_topk, loss_jsd, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(grad_topk, grad_jsd, atol=1e-4, rtol=1e-4)
