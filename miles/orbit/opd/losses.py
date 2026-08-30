"""Orbit's on-policy-distillation (OPD) training losses.

Home for the loss code orbit added to ``miles/backends/training_utils/loss.py``:
the exact full-vocabulary generalized-JSD loss against a frozen teacher
(``opd_jsd_loss_function``), the direct top-k KL loss
(``opd_topk_loss_function``) with its closed-form term builder
(``_topk_kl_terms``) and diagnostics helpers, and the two masked
response-reduction helpers (``_response_masked_max`` / ``_response_masked_min``)
that orbit added alongside them.

The miles module keeps the upstream ``loss_function`` dispatcher and its
``match args.loss_type`` arms, re-exports the names below so existing importers
(and tests that rebind them on the miles module) keep working, and calls in here
through stamped ``# ORBIT-SEAM`` hooks.

Import direction is miles -> orbit: nothing here imports ``miles.*`` at module
level. The five call-time imports of ``miles.backends.training_utils.loss``
below are deliberate and do double duty: they break the import cycle (the miles
module imports this one at module level), and they resolve every miles
collaborator through the *same* miles module attribute the code read before the
move, so ``monkeypatch.setattr(loss, ...)`` in the existing tests still steers
this layer exactly as it did in place.
"""

from __future__ import annotations

import math
import warnings
from argparse import Namespace
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from miles.orbit.opd.teacher_lm_head import load_teacher_lm_head
from miles.orbit.opd.vocab_parallel import (
    compute_vocab_parallel_topk_log_probs,
    compute_vocab_parallel_topk_log_probs_and_entropy,
    vocab_parallel_log_softmax,
    vocab_parallel_sum,
    vocab_parallel_topk_indices,
    vocab_shard_start,
)

if TYPE_CHECKING:
    from miles.utils.types import RolloutBatch


def _response_masked_max(
    x: torch.Tensor,
    *,
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> torch.Tensor:
    # Call-time import: keeps this module free of module-level miles imports and
    # preserves the pre-move lookup (these were loss.py module globals).
    from miles.backends.training_utils.loss import get_logits_and_tokens_offset_with_cp, get_parallel_state

    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size

    if cp_size == 1:
        chunk_lengths = response_lengths
        chunked_loss_masks = loss_masks
    else:
        chunk_lengths = []
        chunked_loss_masks = []
        for i, (total_length, response_length, loss_mask) in enumerate(
            zip(total_lengths, response_lengths, loss_masks, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            prompt_length = total_length - response_length
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            chunked_loss_mask = torch.cat([loss_mask_0, loss_mask_1], dim=0)
            chunked_loss_masks.append(chunked_loss_mask)
            chunk_lengths.append(chunked_loss_mask.size(0))

    max_values = []
    for x_i, loss_mask_i in zip(x.split(chunk_lengths, dim=0), chunked_loss_masks, strict=False):
        valid_mask = loss_mask_i.to(device=x_i.device, dtype=torch.bool)
        if x_i.numel() == 0:
            max_values.append(torch.zeros((), dtype=x.dtype, device=x.device))
        else:
            max_value = x_i.masked_fill(~valid_mask, -torch.inf).max()
            max_values.append(torch.where(torch.isneginf(max_value), torch.zeros_like(max_value), max_value))

    if not max_values:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.stack(max_values).max()


def _response_masked_min(
    x: torch.Tensor,
    *,
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> torch.Tensor:
    """Minimum of `x` over loss-mask-valid response positions -- the `_response_masked_max`
    sibling for diagnostics that want a worst-case floor (e.g. `opd_topk/teacher_mass_min`).

    Not implemented as `-_response_masked_max(-x, ...)`: `_response_masked_max`'s fallback
    of `0` for an empty/all-masked sample is a safe *identity* only for a max of a
    non-negative quantity (0 is a lower bound, so it can never win a real max). Negated
    into a min, that same `0` becomes the *supremum* of `-x` for `x` in `[0, 1]` (like
    `teacher_mass`) and would silently dominate every real value -- reported min ends up
    `-0.` regardless of the real data (caught by review; see the regression test). Samples
    with nothing valid are therefore skipped entirely here rather than injected as a fake
    reading; if literally no sample in the microbatch has a valid position, there is no
    worst case to report, so this returns `1.0` (this metric's natural upper bound, i.e.
    "no evidence of a problem") rather than fabricate one.
    """
    # Call-time import: keeps this module free of module-level miles imports and
    # preserves the pre-move lookup (these were loss.py module globals).
    from miles.backends.training_utils.loss import get_logits_and_tokens_offset_with_cp, get_parallel_state

    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size

    if cp_size == 1:
        chunk_lengths = response_lengths
        chunked_loss_masks = loss_masks
    else:
        chunk_lengths = []
        chunked_loss_masks = []
        for i, (total_length, response_length, loss_mask) in enumerate(
            zip(total_lengths, response_lengths, loss_masks, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            prompt_length = total_length - response_length
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            chunked_loss_mask = torch.cat([loss_mask_0, loss_mask_1], dim=0)
            chunked_loss_masks.append(chunked_loss_mask)
            chunk_lengths.append(chunked_loss_mask.size(0))

    min_values = []
    for x_i, loss_mask_i in zip(x.split(chunk_lengths, dim=0), chunked_loss_masks, strict=False):
        valid_mask = loss_mask_i.to(device=x_i.device, dtype=torch.bool)
        if x_i.numel() == 0 or not bool(valid_mask.any()):
            continue
        min_values.append(x_i.masked_fill(~valid_mask, torch.inf).min())

    if not min_values:
        return torch.ones((), dtype=x.dtype, device=x.device)
    return torch.stack(min_values).min()


def opd_topk_sample_log_probs(
    logits_chunk: torch.Tensor,
    tokens_chunk: torch.Tensor,
    sample_topk_ids: torch.Tensor,
    *,
    args: Namespace,
    parallel_state,
    tp_group,
    with_entropy: bool,
    entropy_no_grad: bool,
    with_log_probs: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Score one sample's response positions at a supplied set of top-k token ids.

    The body of ``get_log_probs_and_entropy``'s orbit-added
    ``sample_topk_ids is not None`` branch, lifted verbatim. Returns
    ``(topk_log_prob, log_prob, entropy)``; ``log_prob`` is ``None`` when
    ``with_log_probs`` is false and ``entropy`` is ``None`` when ``with_entropy``
    is false -- exactly the cases in which the caller never reads them (in place
    those names were simply left unbound for the iteration).
    """
    # Call-time import: keeps this module free of module-level miles imports and
    # preserves the pre-move lookup (these were loss.py module globals).
    from miles.backends.training_utils.loss import _gather_true_on_policy_full_logits, calculate_log_probs_and_entropy

    log_prob = None
    vocab_size = getattr(args, "vocab_size", None)
    if args.true_on_policy_mode:
        # Match the sampled-token true-on-policy contract exactly: gather
        # and truncate the real vocabulary, then run native-dtype
        # log_softmax once for sampled ids, teacher ids, and entropy.
        full_logits = _gather_true_on_policy_full_logits(
            logits_chunk,
            tp_group,
            vocab_size=vocab_size,
        )
        full_log_probs = torch.log_softmax(full_logits, dim=-1)
        sample_topk_ids = sample_topk_ids.to(device=full_logits.device)
        topk_log_prob = full_log_probs.gather(-1, sample_topk_ids)
        if with_log_probs:
            log_prob = full_log_probs.gather(-1, tokens_chunk.unsqueeze(-1)).squeeze(-1)
        entropy_log_probs = full_log_probs.detach() if entropy_no_grad else full_log_probs
        entropy = -(entropy_log_probs.exp() * entropy_log_probs).sum(dim=-1) if with_entropy else None
    else:
        # Deliberately avoid fused_vocab_parallel_cross_entropy here: it
        # recompiles/re-autotunes per shape. This eager helper scores all K
        # ids at once and, when needed, derives entropy from the same
        # real-vocabulary normalizer so padding cannot skew the correction.
        if with_entropy:
            topk_log_prob, entropy = compute_vocab_parallel_topk_log_probs_and_entropy(
                logits_chunk,
                sample_topk_ids,
                tp_group,
                vocab_size=vocab_size,
            )
            if entropy_no_grad:
                entropy = entropy.detach()
        else:
            topk_log_prob = compute_vocab_parallel_topk_log_probs(
                logits_chunk,
                sample_topk_ids,
                tp_group,
                vocab_size=vocab_size,
            )
            entropy = None

        if with_log_probs:
            log_prob, _ = calculate_log_probs_and_entropy(
                logits_chunk,
                tokens_chunk,
                parallel_state.tp.group,
                with_entropy=False,
                chunk_size=args.log_probs_chunk_size,
                true_on_policy=False,
                vocab_size=vocab_size,
            )

    return topk_log_prob, log_prob, entropy


def _clip_pointwise_kl(kl_elem: torch.Tensor, clip: float | None) -> torch.Tensor:
    """Cap each individual (response-position, vocab-token) divergence summand before it is
    summed over the vocabulary dimension.

    Borrowed from OPSD's (github.com/siyan-zhao/OPSD) `--jsd_token_clip`: they found stylistic
    tokens (e.g. "wait", "think") can carry 6-15x higher per-vocab-entry divergence than
    content/math tokens and dominate the training signal if left unclipped.
    """
    if clip is None:
        return kl_elem
    return kl_elem.clamp(max=clip)


def opd_jsd_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute exact full-vocabulary generalized JSD against a frozen teacher.

    Follows Eq. (1) of the GKD paper. The teacher distribution is reconstructed locally from
    `batch["teacher_hidden_states"]` and the teacher's LM head rather than shipped over the
    wire. `--opd-jsd-beta` (`b`) interpolates between forward `KL(teacher||student)` at `b=0`
    and reverse `KL(student||teacher)` at `b=1`, over the mixture `M = (1-b)*student + b*teacher`:

        jsd(b) = b * KL(teacher || M) + (1-b) * KL(student || M)

    `b=0`/`b=1` are their own branch in the loop below, not literal evaluations of the formula
    above: plugging either endpoint into it degenerates to `KL(Q||Q) = 0` (and, approached from
    `0 < b < 1`, the `else` branch's `math.log(b)`/`math.log1p(-b)` would hit a domain error
    exactly there), so the two endpoints are hard-coded to the non-degenerate KL directions
    stated above instead of falling out of the mixture formula.

    `batch["teacher_hidden_states"]` holds one CPU fp32 tensor per sample, already CP-sliced
    row-for-row with this rank's response logits by `get_rollout_data` (the same
    `slice_log_prob_with_cp` treatment `teacher_log_probs` gets), so each chunk only needs a
    device move here.

    Returns `(loss, metrics)`; `metrics` holds detached "loss" and "entropy", plus "kl_loss"
    under --use-kl-loss and "topk_overlap_k{k}" under --opd-log-topk-overlap.
    """
    # Call-time import: keeps this module free of module-level miles imports and
    # preserves the pre-move lookup (these were loss.py module globals, so a test
    # that rebinds them on the miles module still steers this loss).
    from miles.backends.training_utils.loss import (
        calculate_log_probs_and_entropy,
        compute_approx_kl,
        get_parallel_state,
        get_responses,
    )

    parallel_state = get_parallel_state()
    assert not args.allgather_cp, (
        "opd_jsd_loss does not support --allgather-cp: teacher_hidden_states are CP-sliced by "
        "slice_log_prob_with_cp (get_logits_and_tokens_offset_with_cp chunks), not by the DSA "
        "split get_responses takes under that flag."
    )
    beta = args.opd_jsd_beta
    assert 0.0 <= beta <= 1.0, f"--opd-jsd-beta must be in [0, 1], got {beta}"
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    tp_group = parallel_state.tp.group if parallel_state.tp.size > 1 else None
    # The student's logits are the authority on how the vocabulary is split -- they carry
    # exactly this rank's shard, whichever global vocab size the model was actually built with.
    local_vocab_size = logits.size(-1)
    vocab_start = vocab_shard_start(local_vocab_size) if tp_group is not None else 0
    padded_student_vocab = local_vocab_size * parallel_state.tp.size
    configured_vocab_size = getattr(args, "vocab_size", None)
    real_student_vocab = padded_student_vocab if configured_vocab_size is None else int(configured_vocab_size)
    if not 0 < real_student_vocab <= padded_student_vocab:
        raise ValueError(f"Student vocab_size must be in [1, {padded_student_vocab}], got {real_student_vocab}.")
    valid_local_width = min(max(real_student_vocab - vocab_start, 0), local_vocab_size)
    teacher_lm_head = load_teacher_lm_head(args, local_vocab_size=local_vocab_size).to(logits.device, torch.float32)
    # How many of this rank's vocab columns are real rather than divisibility padding.
    # Clamped from above too: a bigger same-tokenizer teacher can carry MORE padded rows
    # than the student (Qwen2.5-7B pads to 152064 vs 151936 below 3B); rows past the
    # student's width are padding the student cannot emit, so dropping them conditions
    # the teacher on the shared vocabulary. The TP shard path already slices this way.
    teacher_vocab_size = min(teacher_lm_head.size(0), valid_local_width)

    kl_per_sample = []
    entropy_per_sample = []

    ref_kl_sampled_log_probs = [] if args.use_kl_loss else None
    topk_ks = tuple(args.opd_topk_overlap_ks) if args.opd_log_topk_overlap else ()
    if any(type(k) is not int or k <= 0 for k in topk_ks):
        raise ValueError(f"--opd-topk-overlap-ks values must be positive integers, got {topk_ks}.")
    topk_overlap_per_sample: dict[int, list[torch.Tensor]] = {k: [] for k in topk_ks}
    max_seq_lens = batch.get("max_seq_lens", None)
    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )
    for i, (logits_chunk, tokens_chunk) in enumerate(responses):
        # Keep full-width containers for diagnostics and elementwise KL, but
        # normalize only the real student vocabulary.  The -1e4 padding has
        # exactly zero mass in fp32 and, because no padded logit participates in
        # the normalizer, receives exactly zero gradient.
        vocab_size = logits_chunk.size(-1)
        student_log_probs_full = logits_chunk.float().new_full((logits_chunk.size(0), vocab_size), -1e4)
        # Columns past the teacher's real vocab stay at this fill. A large finite negative
        # rather than -inf, which would go NaN (0 * -inf) on stray student mass.
        teacher_log_probs_full = logits_chunk.float().new_full((logits_chunk.size(0), vocab_size), -1e4)
        if logits_chunk.size(0) > 0:
            student_log_probs_full[:, :valid_local_width] = vocab_parallel_log_softmax(
                logits_chunk[:, :valid_local_width].float(), tp_group
            ).clamp(min=args.opd_log_prob_min_clamp)
            with torch.no_grad():
                teacher_hidden_states = batch["teacher_hidden_states"][i].to(
                    dtype=torch.float32, device=logits_chunk.device
                )
                assert teacher_hidden_states.size(0) == logits_chunk.size(0), (
                    f"sample {i}: {teacher_hidden_states.size(0)} teacher hidden-state rows vs "
                    f"{logits_chunk.size(0)} response logits -- get_rollout_data's CP slicing "
                    "has drifted from get_responses()."
                )
                teacher_logits = teacher_hidden_states @ teacher_lm_head[:teacher_vocab_size].T

                rollout_temperature = float(args.rollout_temperature)
                if rollout_temperature != 1.0:
                    teacher_logits.div_(rollout_temperature)
                # The clamp bounds forward KL, which weights by the fixed teacher probs.
                teacher_log_probs_full[:, :teacher_vocab_size] = vocab_parallel_log_softmax(
                    teacher_logits, tp_group
                ).clamp_(min=args.opd_log_prob_min_clamp)
        student_probs_full = student_log_probs_full.exp()
        teacher_probs_full = teacher_log_probs_full.exp()

        if topk_ks:
            max_k = max(topk_ks)
            student_topk_idx = vocab_parallel_topk_indices(
                student_log_probs_full,
                max_k,
                vocab_start,
                tp_group,
                vocab_size=real_student_vocab,
            )
            teacher_topk_idx = vocab_parallel_topk_indices(
                teacher_log_probs_full,
                max_k,
                vocab_start,
                tp_group,
                vocab_size=real_student_vocab,
            )
            topk_match = student_topk_idx.unsqueeze(-1) == teacher_topk_idx.unsqueeze(-2)  # [R, max_k, max_k]
            for k in topk_ks:
                effective_k = min(k, real_student_vocab)
                overlap_count = topk_match[:, :effective_k, :effective_k].any(dim=-1).sum(dim=-1)  # [R]
                topk_overlap_per_sample[k].append(overlap_count.float() / effective_k)

        if beta == 0.0:
            kl_elem = teacher_probs_full * (teacher_log_probs_full - student_log_probs_full)
        elif beta == 1.0:
            kl_elem = student_probs_full * (student_log_probs_full - teacher_log_probs_full)
        else:
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs_full + math.log1p(-beta), teacher_log_probs_full + math.log(beta)]),
                dim=0,
            )
            kl_teacher_elem = teacher_probs_full * (teacher_log_probs_full - mixture_log_probs)
            kl_student_elem = student_probs_full * (student_log_probs_full - mixture_log_probs)

            kl_elem = beta * kl_teacher_elem + (1 - beta) * kl_student_elem

        kl_elem = _clip_pointwise_kl(kl_elem, args.opd_jsd_pointwise_clip)
        # The vocab sum crosses TP shards, so it must complete before the per-position clamp.
        kl = vocab_parallel_sum(kl_elem, tp_group).clamp(max=args.opd_loss_max_clamp)
        kl_per_sample.append(kl)
        entropy_per_sample.append(vocab_parallel_sum(-(student_probs_full * student_log_probs_full), tp_group))
        if ref_kl_sampled_log_probs is not None:
            student_log_prob, _ = calculate_log_probs_and_entropy(
                logits_chunk,
                tokens_chunk,
                parallel_state.tp.group,
                chunk_size=args.log_probs_chunk_size,
                true_on_policy=args.true_on_policy_mode,
                vocab_size=getattr(args, "vocab_size", None),
            )
            ref_kl_sampled_log_probs.append(student_log_prob.squeeze(-1))

    kl_per_sample = torch.cat(kl_per_sample, dim=0)
    loss = sum_of_sample_mean(kl_per_sample)

    # compute_ref_log_probs() populates batch["ref_log_probs"] for any loss_type.
    ref_kl_loss = None
    if args.use_kl_loss:
        student_sampled_log_probs = torch.cat(ref_kl_sampled_log_probs, dim=0)
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        ref_kl = compute_approx_kl(student_sampled_log_probs, ref_log_probs, kl_loss_type=args.kl_loss_type)
        ref_kl_loss = sum_of_sample_mean(ref_kl)
        loss = loss + args.kl_loss_coef * ref_kl_loss

    # make sure the gradient could backprop correctly.
    if kl_per_sample.numel() == 0:
        loss = loss + 0 * logits.sum()

    # Per-token quantities, so the same reduction as loss keeps them on a comparable scale.
    entropy_concat = torch.cat(entropy_per_sample, dim=0)
    entropy_metric = sum_of_sample_mean(entropy_concat)

    topk_overlap_concat = {k: torch.cat(topk_overlap_per_sample[k], dim=0) for k in topk_ks}
    topk_overlap_metric = {k: sum_of_sample_mean(topk_overlap_concat[k]) for k in topk_ks}

    metrics = {
        "loss": loss.clone().detach(),
        "entropy": entropy_metric.clone().detach(),
    }
    if ref_kl_loss is not None:
        metrics["kl_loss"] = ref_kl_loss.clone().detach()
    for k in topk_ks:
        metrics[f"topk_overlap_k{k}"] = topk_overlap_metric[k].clone().detach()

    return (loss, metrics)


_TOPK_LOG_INF = -100.0
_TOPK_KL_TYPES = ("forward", "reverse", "mixed")


def _topk_kl_terms(
    teacher_topk_logprobs: torch.Tensor,
    student_topk_logprobs: torch.Tensor,
    entropy: torch.Tensor | None,
    kl_type: str,
    mixed_weight: float,
    zero_outside: bool,
) -> torch.Tensor:
    """Per-token top-k KL between the frozen teacher and the student, truncated to the
    teacher's own top-k support (plus, for the reverse direction, an optional correction
    for the student mass that falls outside that support).

    Padded slots (see `miles.orbit.opd.opd_sglang._TOPK_PAD_LOGPROB`) carry a teacher
    log-prob of -1e4, so `teacher_topk_logprobs.exp()` underflows to exactly 0.0 in
    float32 -- used below as an exact (not approximate) validity mask over the K
    dimension. Both the forward and the uncorrected-reverse sums only ever touch valid
    slots, so a padded column changes nothing (czy's `_topk_forward_kl`, generalized to
    all three directions; their `renormalize` branch is dropped per spec).

    The same float32 underflow floor sits at log-prob ~-103.97 (ln(2**-149), the
    smallest denormal): a genuine (non-pad) teacher entry that far below the peak also
    reads as invalid and is dropped from the support the same way a pad slot is, with
    the reverse-direction correction re-flooring it at `_TOPK_LOG_INF` (-100.0) instead
    of its true value -- unreachable at any realistic `k` (the teacher's own top-k
    entries are never that improbable), but reachable once `k` approaches the full
    vocabulary.

    Forward (teacher-weighted, `--opd-kl-type forward`):
        `sum_K valid * teacher_prob * (teacher_log_prob - student_log_prob)`
    `zero_outside` is structurally inert here -- the sum never leaves the teacher's own
    reported support -- so passing it true is a caller mistake we warn about once
    (Python's default warning filter already dedupes by message+location) rather than
    silently ignore.

    Reverse (student-weighted, `--opd-kl-type reverse`):
        `sum_K valid * student_prob * (student_log_prob - teacher_log_prob)`
    truncated the same way, but `student_prob`/`student_log_prob` keep gradients (the
    teacher side is always detached -- it is frozen). Without `zero_outside`, this
    silently drops all of the student's probability mass that falls *outside* the
    teacher's reported top-k, which lets the optimizer push probability there for free.
    `zero_outside=True` adds a correction that makes the result exactly equal to the
    full-vocabulary reverse KL against a teacher extended with a `log_inf=-100.0`
    log-prob at every out-of-support token id (see the closed-form test for the
    from-scratch full-vocab derivation this mirrors):

        correction = (H_all - sum_K valid * student_prob * student_log_prob)
                     - log_inf * (1 - sum_K valid * student_prob)

    where `H_all = sum_v student_prob(v) * student_log_prob(v)` is the student's own
    full-vocabulary self-term (note: negative). `calculate_log_probs_and_entropy`'s
    "entropy" output was verified (see the closed-form correction test, which pins this
    sign) to already be the *standard* positive entropy `-sum_v p_v log p_v`, so
    `H_all = -entropy` here, not `entropy` directly.

    Mixed (`--opd-kl-type mixed`, `--opd-mixed-kl-weight` on the forward term, NeMo's
    convention): `w * forward + (1 - w) * reverse`, where `reverse` already includes its
    own correction when requested -- so the correction is implicitly scaled by `(1 - w)`
    too, matching NeMo's DistillationLossFn.

    Args:
        teacher_topk_logprobs: `[R, K]` teacher log-probs at its own top-k token ids.
            Treated as a constant; detached here regardless of what the caller passes.
        student_topk_logprobs: `[R, K]` student log-probs at those same ids,
            differentiable w.r.t. the student's parameters.
        entropy: `[R]` student full-vocabulary entropy (standard positive convention),
            or `None`. Required only when `zero_outside` and `kl_type != "forward"`.
        kl_type: One of "forward", "reverse", "mixed".
        mixed_weight: Weight on the forward term when `kl_type == "mixed"`, in `[0, 1]`.
        zero_outside: Whether to add the reverse-direction out-of-support correction.

    Returns:
        `[R]` tensor of per-token KL values (the loss to minimize).
    """
    if kl_type not in _TOPK_KL_TYPES:
        raise ValueError(f"Unknown top-k KL type: {kl_type!r}")

    # Teacher is frozen: its log-probs arrive as plain (non-autograd) tensors from Ray
    # anyway, but detach explicitly so the intent -- no gradient into the teacher side --
    # is unambiguous regardless of caller.
    teacher_topk_logprobs = teacher_topk_logprobs.detach()
    teacher_weights = teacher_topk_logprobs.exp()
    valid = teacher_weights > 0  # exact float32 underflow at padded slots, see above
    masked_teacher_weights = torch.where(valid, teacher_weights, torch.zeros_like(teacher_weights))

    if kl_type == "forward":
        if zero_outside:
            warnings.warn(
                "--opd-topk-zero-outside has no effect with --opd-kl-type forward: the "
                "forward top-k KL only ever sums over the teacher's own reported support.",
                stacklevel=2,
            )
        return (masked_teacher_weights * (teacher_topk_logprobs - student_topk_logprobs)).sum(dim=-1)

    student_weights = student_topk_logprobs.exp()
    masked_student_weights = torch.where(valid, student_weights, torch.zeros_like(student_weights))
    reverse = (masked_student_weights * (student_topk_logprobs - teacher_topk_logprobs)).sum(dim=-1)

    if zero_outside:
        if entropy is None:
            raise ValueError("`entropy` is required when `zero_outside` is set for the reverse-direction term.")
        h_all = -entropy  # see docstring: the machinery's "entropy" is the standard +H convention
        sum_k_student_weight = masked_student_weights.sum(dim=-1)
        sum_k_student_weighted_logprob = (masked_student_weights * student_topk_logprobs).sum(dim=-1)
        correction = (h_all - sum_k_student_weighted_logprob) - _TOPK_LOG_INF * (1 - sum_k_student_weight)
        reverse = reverse + correction

    if kl_type == "reverse":
        return reverse

    forward = (masked_teacher_weights * (teacher_topk_logprobs - student_topk_logprobs)).sum(dim=-1)
    return mixed_weight * forward + (1 - mixed_weight) * reverse


def _topk_overlap_membership(
    student_topk_ids: torch.Tensor,
    teacher_ids_for_match: torch.Tensor,
) -> torch.Tensor:
    """Row-wise membership of each `student_topk_ids` entry in that row's
    `teacher_ids_for_match` -- an O(R*K log K) replacement for the naive `[R, K, K]`
    broadcast-equality (`student.unsqueeze(-1) == teacher.unsqueeze(-2)`), which OOMs at
    large k (688 GiB at k=vocab_size=151936, R=32; the gate-discovered defect this fixes).

    Sorts each row of `teacher_ids_for_match` once (`O(K log K)`) and binary-searches
    each student id into it (`torch.searchsorted`, batched 2-D x 2-D), instead of
    comparing every student id against every teacher id.

    `teacher_ids_for_match` may hold `-1` sentinels for invalid/masked slots (see the
    caller). Real ids are always >= 0, so `-1` sorts to the front of every row and a
    lower-bound search for a non-negative value can never land inside that block --
    the sentinels are therefore inert without any separate exclusion.

    Returns a `[R, student_K]` bool tensor.
    """
    k_teacher = teacher_ids_for_match.size(-1)
    if k_teacher == 0:
        # No teacher columns to match against (e.g. the reshaped-(0,0) empty-response
        # sample) -- searchsorted's clamp below would need a nonexistent index 0..-1.
        return torch.zeros_like(student_topk_ids, dtype=torch.bool)
    sorted_teacher, _ = torch.sort(teacher_ids_for_match, dim=-1)
    insert_pos = torch.searchsorted(sorted_teacher, student_topk_ids).clamp(max=k_teacher - 1)
    return sorted_teacher.gather(-1, insert_pos) == student_topk_ids


def _resolve_opd_topk_kl_type(args: Namespace) -> tuple[str, float]:
    """Local counterpart to `miles.orbit.opd.opd_sglang._get_kl_type` -- kept independent
    (not imported) so this training-side loss module doesn't reach into rollout code for
    a two-line resolution. Mirrors NeMo-RL's DistillationLossFn `kl_type`/`mixed_kl_weight`
    convention: `reverse` (default), `forward`, or `mixed` with `--opd-mixed-kl-weight` on
    the forward term.
    """
    kl_type = getattr(args, "opd_kl_type", "reverse") or "reverse"
    if kl_type not in _TOPK_KL_TYPES:
        raise ValueError(f"Unknown OPD KL type: {kl_type!r}")
    mixed_weight = float(getattr(args, "opd_mixed_kl_weight", 0.5))
    if not (0.0 <= mixed_weight <= 1.0):
        raise ValueError(f"--opd-mixed-kl-weight must be in [0, 1], got {mixed_weight}.")
    return kl_type, mixed_weight


def opd_topk_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Direct (non-policy-gradient) top-k KL on_policy_distillation loss.

    Unlike `--opd-loss-type sampled_token` (which treats `teacher_log_prob(a_t) -
    student_log_prob(a_t)` as a REINFORCE advantage on the token the student happened to
    sample, routed through `compute_policy_loss`'s PPO ratio/clip), this backpropagates
    directly through the student's log-probs at all `--opd-topk-k` of the teacher's top-k
    token ids for every response position -- mirroring verl's `forward_kl_topk` (see
    https://verl.readthedocs.io/en/latest/algo/opd.html, "PG OPD" section). There is no
    importance-sampling ratio here (and hence no PPO clip, no old/rollout log-probs
    needed): the loss is computed directly against the current parameters in the same
    forward pass, so there is no train/rollout policy mismatch to correct for.

    `--opd-kl-type` (`reverse` default, `forward`, or `mixed`) and `--opd-mixed-kl-weight`
    select the KL direction via `_topk_kl_terms`; `--opd-topk-zero-outside` (Task 4 wires
    the arg) controls the reverse-direction out-of-support correction -- until then this
    resolves `None` to `kl_type != "forward"` (correct the reverse direction's blind spot
    by default; inert for forward either way).

    Returns `(loss, metrics)`. In addition to "loss", `metrics` carries diagnostics that
    do not affect the loss itself (Task 4 wires the args gating whether these get
    logged): "opd_topk/teacher_mass" (+"_min") -- how much of the teacher's own
    distribution its reported top-k actually covers, "opd_topk/student_mass" -- how much
    of the *student's* distribution currently sits on the teacher's top-k ids, and
    "opd_topk/overlap_ratio" -- the fraction of the student's own local top-k ids that
    coincide with the teacher's. Every reduction here is a masked sum over a
    clamped->=1 denominator (mirroring `sum_of_sample_mean`'s own convention), never a
    bare `.mean()` over a selection that can be empty.

    Args:
        args: Configuration; uses `opd_kl_type`, `opd_mixed_kl_weight`, and (once Task 4
            lands) `opd_topk_zero_outside`.
        batch: Mini-batch with "teacher_topk_ids" (list of `[R, K]` token ids per sample),
            "teacher_topk_logprobs" (list of `[R, K]` teacher log-probs per sample),
            "unconcat_tokens", "total_lengths", "response_lengths", "loss_masks".
        logits: Policy logits with shape `[1, T, V]`, from the current (grad-enabled)
            forward pass.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)`.
    """
    # Call-time import: keeps this module free of module-level miles imports and
    # preserves the pre-move lookup (these were loss.py module globals, so a test
    # that rebinds them on the miles module still steers this loss).
    from miles.backends.training_utils.loss import get_log_probs_and_entropy, get_parallel_state, get_responses

    parallel_state = get_parallel_state()
    device = logits.device
    teacher_topk_ids = [t.to(device=device) for t in batch["teacher_topk_ids"]]
    teacher_topk_logprobs = [t.to(device=device) for t in batch["teacher_topk_logprobs"]]
    # The real transport (get_rollout_data's torch.tensor(...) over the raw per-sample
    # list[list[int]] payload) collapses an empty response's row list (`[]`, not
    # `[[], ...]`) to a 1-D `[0]` tensor rather than `[0, K]`. Normalize to 2-D here --
    # R=0 either way, K is unknowable from an empty sample and irrelevant since there
    # are no rows -- so the per-sample `.sum(dim=-1)` diagnostics below reduce the K
    # axis, not the (already-empty) R axis, and concatenate cleanly with real samples'
    # `[R]`-shaped output instead of collapsing to a 0-d scalar.
    teacher_topk_ids = [t if t.dim() > 1 else t.reshape(0, 0) for t in teacher_topk_ids]
    teacher_topk_logprobs = [t if t.dim() > 1 else t.reshape(0, 0) for t in teacher_topk_logprobs]

    # For the overlap_ratio diagnostic's *student* top-k: dev's opd_jsd pattern (mirrors
    # `vocab_parallel_topk_indices`'s two call sites in opd_jsd_loss_function above) --
    # the student's own local shard is the authority on the vocab split, and vocab_start
    # is only meaningful once TP is actually on.
    tp_group = parallel_state.tp.group if parallel_state.tp.size > 1 else None
    local_vocab_size = logits.size(-1)
    vocab_start = vocab_shard_start(local_vocab_size) if tp_group is not None else 0

    # A bigger-config-vocab teacher (e.g. Qwen2.5-7B pads to 152064 vs a <3B student's
    # 151936) can report top-k ids past the student's own vocabulary. Left alone these
    # break compute_vocab_parallel_topk_log_probs's gather: at TP=1 they index-error; at
    # TP>1 every rank's ownership mask is False for them, so the gather silently returns
    # a fake `0 - log_normalizer` log-prob instead. Mask them to a pad slot before the
    # gather -- id -> 0, logprob -> -1e4 -- exactly like the transport's own padding
    # (miles.orbit.opd.opd_sglang._TOPK_PAD_TOKEN_ID/_TOPK_PAD_LOGPROB): the -1e4 underflows
    # to exact 0 mass under _topk_kl_terms's `valid` mask.
    padded_student_vocab = local_vocab_size * parallel_state.tp.size
    configured_vocab_size = getattr(args, "vocab_size", None)
    global_student_vocab = padded_student_vocab if configured_vocab_size is None else int(configured_vocab_size)
    if not 0 < global_student_vocab <= padded_student_vocab:
        raise ValueError(f"Student vocab_size must be in [1, {padded_student_vocab}], got {global_student_vocab}.")
    for i, (t_ids, t_lp) in enumerate(zip(teacher_topk_ids, teacher_topk_logprobs, strict=True)):
        if t_ids.size(-1) > global_student_vocab:
            raise ValueError(
                f"Teacher top-k width K={t_ids.size(-1)} exceeds the student's real vocabulary "
                f"size {global_student_vocab}. Reduce --opd-log-prob-top-k."
            )
        overhang = (t_ids < 0) | (t_ids >= global_student_vocab)
        teacher_topk_ids[i] = torch.where(overhang, torch.zeros_like(t_ids), t_ids)
        teacher_topk_logprobs[i] = torch.where(overhang, torch.full_like(t_lp, -1e4), t_lp)

    kl_type, mixed_weight = _resolve_opd_topk_kl_type(args)
    zero_outside = getattr(args, "opd_topk_zero_outside", None)
    if zero_outside is None:
        # Task 4 moves this default into arg validation; until then, correct the reverse
        # direction's out-of-support blind spot by default (inert for forward either way).
        zero_outside = kl_type != "forward"
    needs_correction = zero_outside and kl_type != "forward"

    total_lengths = batch["total_lengths"]
    response_lengths = batch["response_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=needs_correction,
        max_seq_lens=max_seq_lens,
        teacher_topk_ids=teacher_topk_ids,
        with_log_probs=False,
    )
    student_topk_log_probs = log_probs_and_entropy["student_topk_log_probs"]
    entropy_per_sample = log_probs_and_entropy["entropy"] if needs_correction else [None] * len(teacher_topk_ids)

    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )

    topk_kl_per_sample = []
    teacher_mass_per_sample = []
    student_mass_per_sample = []
    overlap_ratio_per_sample = []
    for (logits_chunk, _), t_ids, t_lp, s_lp, entropy_i in zip(
        responses, teacher_topk_ids, teacher_topk_logprobs, student_topk_log_probs, entropy_per_sample, strict=True
    ):
        topk_kl_per_sample.append(_topk_kl_terms(t_lp, s_lp, entropy_i, kl_type, mixed_weight, zero_outside))

        # Diagnostics only -- detached, no gradient needed.
        valid = t_lp.exp() > 0
        masked_teacher_weight = torch.where(valid, t_lp.exp(), torch.zeros_like(t_lp))
        masked_student_weight = torch.where(valid, s_lp.exp().detach(), torch.zeros_like(s_lp))
        teacher_mass_per_sample.append(masked_teacher_weight.sum(dim=-1))
        student_mass_per_sample.append(masked_student_weight.sum(dim=-1))

        k = t_ids.size(-1)
        # vocab_parallel_topk_indices returns *global* ids (shard-local candidates offset
        # by vocab_start, then all-gathered/re-ranked across TP -- see its docstring):
        # a plain local torch.topk on logits_chunk would instead be shard-local ids in
        # [0, V_local), only coincidentally comparable to the teacher's global ids at
        # tp.size == 1. Raw logits (not log-probs) are fine here: log_softmax only
        # shifts each row by a per-row constant, so it never changes the top-k ordering,
        # and this is diagnostic-only (no_grad inside the helper).
        student_topk_ids = vocab_parallel_topk_indices(
            logits_chunk,
            k,
            vocab_start,
            tp_group,
            vocab_size=global_student_vocab,
        )
        teacher_ids_for_match = torch.where(valid, t_ids, torch.full_like(t_ids, -1))
        overlap_match = _topk_overlap_membership(student_topk_ids, teacher_ids_for_match)
        overlap_ratio_per_sample.append(overlap_match.sum(dim=-1).float() / max(k, 1))

    topk_kl = torch.cat(topk_kl_per_sample, dim=0)
    loss = sum_of_sample_mean(topk_kl)

    # make sure the gradient could backprop correctly.
    if topk_kl.numel() == 0:
        loss = loss + 0 * logits.sum()

    teacher_mass = torch.cat(teacher_mass_per_sample, dim=0)
    student_mass = torch.cat(student_mass_per_sample, dim=0)
    overlap_ratio = torch.cat(overlap_ratio_per_sample, dim=0)

    teacher_mass_min = _response_masked_min(
        teacher_mass,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        loss_masks=batch["loss_masks"],
        qkv_format=getattr(args, "qkv_format", "thd"),
        max_seq_lens=max_seq_lens,
    )

    metrics = {
        "loss": loss.clone().detach(),
        "opd_topk/teacher_mass": sum_of_sample_mean(teacher_mass).clone().detach(),
        "opd_topk/teacher_mass_min": teacher_mass_min.clone().detach(),
        "opd_topk/student_mass": sum_of_sample_mean(student_mass).clone().detach(),
        "opd_topk/overlap_ratio": sum_of_sample_mean(overlap_ratio).clone().detach(),
    }

    return loss, metrics
