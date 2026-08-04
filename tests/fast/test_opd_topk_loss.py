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

from orbit.backends.training_utils.cp_utils import get_sum_of_sample_mean
from orbit.backends.training_utils.loss import (
    _TOPK_LOG_INF,
    _response_masked_min,
    _topk_kl_terms,
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

# get_log_probs_and_entropy's log_probs half always runs through Megatron's fused
# vocab-parallel cross-entropy kernel (independent of this task's top-k gather),
# which calls torch.distributed collectives even for process_group=None -- it still
# needs *a* default process group to exist. World size 1 is a single process, no
# spawn needed, mirroring test_ppo_cp_advantages.py / test_vocab_parallel_topk.py's
# single-process-group convention.
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
            teacher_topk_ids.append(torch.zeros((0, K), dtype=torch.long))
            teacher_topk_logprobs.append(torch.zeros((0, K), dtype=torch.float32))
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

    teacher_topk_ids = [ids_0, torch.zeros((0, K), dtype=torch.long)]
    teacher_topk_logprobs = [t_topk_lp_0, torch.zeros((0, K), dtype=torch.float32)]

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
