# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py

import math
from argparse import Namespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

_LOG_RATIO_EXP_CLAMP = 20.0

# Critic explained-variance metric plumbing.
#
# EV = 1 - Var(returns - values) / Var(returns), over trainable (unmasked)
# tokens of the whole optimizer step. Per-micro-batch EV values cannot simply
# be averaged (micro-batches differ in token count and mean), so
# value_loss_function emits the masked token-level sufficient statistics below
# per micro-batch; aggregate_train_losses SUM-reduces them across micro-batches
# and DP/CP ranks (applying one normalization constant shared by all metrics,
# which cancels in the ratios) and folds them into VALUE_EV_METRIC_KEY via
# compute_value_explained_var.
VALUE_EV_STAT_KEYS: tuple[str, ...] = (
    "value_ev/token_count",
    "value_ev/return_sum",
    "value_ev/return_sumsq",
    "value_ev/err_sum",
    "value_ev/err_sumsq",
)
VALUE_EV_METRIC_KEY = "value_explained_var"
_VALUE_EV_MIN_RETURN_VAR = 1e-8


def compute_value_explained_var(
    token_count: float,
    return_sum: float,
    return_sumsq: float,
    err_sum: float,
    err_sumsq: float,
) -> float:
    """Compute EV = 1 - Var(returns - values) / Var(returns) from token sums.

    The five inputs are masked token-level sums (population statistics). They
    may all carry one common positive scale factor — e.g. the
    `cp_size / num_samples_or_tokens` normalization that aggregate_train_losses
    applies to every metric — since it cancels in every ratio below.

    Degenerate cases return 0.0 by convention so logs never carry NaN/inf:
    no trainable tokens, (near-)constant returns, or non-finite statistics.
    """
    if not all(map(math.isfinite, (token_count, return_sum, return_sumsq, err_sum, err_sumsq))):
        return 0.0
    if token_count <= 0.0:
        return 0.0
    return_mean = return_sum / token_count
    return_var = return_sumsq / token_count - return_mean**2
    if return_var <= _VALUE_EV_MIN_RETURN_VAR:
        return 0.0
    err_mean = err_sum / token_count
    err_var = max(err_sumsq / token_count - err_mean**2, 0.0)
    return 1.0 - err_var / return_var


def _safe_clamp_log_ratio(log_ratio: torch.Tensor) -> torch.Tensor:
    log_ratio = torch.nan_to_num(
        log_ratio.float(),
        nan=0.0,
        posinf=_LOG_RATIO_EXP_CLAMP,
        neginf=-_LOG_RATIO_EXP_CLAMP,
    )
    return torch.clamp(log_ratio, min=-_LOG_RATIO_EXP_CLAMP, max=_LOG_RATIO_EXP_CLAMP)


def _safe_exp_neg_ppo_kl(ppo_kl: torch.Tensor) -> torch.Tensor:
    return _safe_clamp_log_ratio(-ppo_kl).exp()


@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        kl_loss_type: Type of KL estimator (k1, k2, k3, low_var_kl).
        importance_ratio: Optional IS ratio (π_θ/π_old) for unbiased KL estimation.
    """
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        # The non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # Besides non negative, it is also unbiased and have lower variance.
        log_ratio = -log_ratio
        if kl_loss_type == "low_var_kl":
            log_ratio = _safe_clamp_log_ratio(log_ratio)
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    # Apply IS ratio for unbiased KL estimation (DeepSeek-V3.2)
    if importance_ratio is not None:
        kl = importance_ratio * kl

    # Clamp only for low_var_kl for numerical stability
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl


def compute_opsm_mask(
    args: Namespace,
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Off-Policy Sequence Masking (OPSM) mask.

    Args:
        args: Configuration containing `opsm_delta` threshold.
        full_log_probs: Current policy log-probs per sample.
        full_old_log_probs: Old policy log-probs per sample.
        advantages: Advantage values per sample.
        loss_masks: Loss masks per sample.

    Returns:
        Tuple of `(opsm_mask, opsm_clipfrac)` where `opsm_mask` is a
        concatenated tensor of per-token masks and
        `opsm_clipfrac` is the count of masked sequences.
    """
    opsm_mask_list = []
    device = advantages[0].device
    opsm_clipfrac = torch.tensor(0.0, device=device)

    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(
        full_log_probs, full_old_log_probs, advantages, loss_masks, strict=False
    ):
        # Calculate sequence-level KL
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)

        # Create mask: 0 if (advantage < 0 and seq_kl > delta), else 1
        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        opsm_clipfrac += mask.sum() / torch.clamp_min(loss_mask.sum(), 1)

        opsm_mask_list.append(1 - mask)

    opsm_mask = torch.cat(opsm_mask_list, dim=0)
    return opsm_mask, opsm_clipfrac


def compute_gspo_kl(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Compute GSPO-style per-sequence KL divergence.

    Args:
        full_log_probs: Current policy log-probs per sample (full or CP-local).
        full_old_log_probs: Old policy log-probs per sample (full or CP-local).
        local_log_probs: Local (CP-local) log-probs for expansion shape reference.
        loss_masks: Loss masks per sample.

    Returns:
        Concatenated tensor of per-token KL values where each token in a
        sequence has the same KL value (the sequence-level KL).
    """
    # Compute sequence-level KL and expand to per-token
    ppo_kl = [
        ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
        for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
    ]
    ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
    ppo_kl = torch.cat(ppo_kl, dim=0)

    return ppo_kl


@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    ratio = _safe_exp_neg_ppo_kl(ppo_kl)
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac


def compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor, process_group: dist.ProcessGroup | None):
    # Follow-up: when megatron is not installed, fall back to naive implementation
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    # fused_vocab_parallel_cross_entropy upcasts via .float(), which creates a
    # fresh tensor for bf16/fp16/fp64 inputs, then does in-place sub_/exp_ on the
    # upcast result. For fp32 input, .float() returns self, so the in-place ops
    # would corrupt the caller's storage — clone defensively in that case.
    if logits.dtype == torch.float32:
        logits = logits.clone()
    # convert to [seq_len, batch_size, vocab_size] as expected by fused_vocab_parallel_cross_entropy
    logits = logits.unsqueeze(1)
    tokens = tokens.unsqueeze(1)
    return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)


# from https://github.com/volcengine/verl/blob/0bdf7f469854815177e73dcfe9e420836c952e6e/verl/utils/megatron/tensor_parallel.py#L99
class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:

        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits, None


def compute_entropy_from_logits(logits: torch.Tensor, process_group) -> torch.Tensor:
    return _VocabParallelEntropy.apply(logits, process_group)


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns


def get_reinforce_plus_plus_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    kl_coef: float,
    gamma: float,
) -> list[torch.Tensor]:
    """
    Calculates discounted returns for REINFORCE++ (https://arxiv.org/pdf/2501.03262)

    Args:
        rewards (Tensor): A tensor of scalar rewards for each sequence.
        kl (List[Tensor]): List of per-token KL divergence tensors for sequence chunks.
        loss_masks (List[Tensor]): List of response-only loss masks for each full sequence.
        response_lengths (List[int]): The full length of each response sequence.
        total_lengths (List[int]): The full length of each sequence (prompt + response).
        kl_coef (float): Coefficient for the KL penalty.
        gamma (float): The discount factor.

    Returns:
        List[torch.Tensor]: A list of return (G_t) tensors for the
                            local sequence chunks owned by the current GPU rank.
    """
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()

    final_returns_chunks = []
    for i in range(len(rewards)):
        local_kl_chunk = kl[i]
        total_len, response_len = total_lengths[i], response_lengths[i]

        if cp_size > 1:
            # Step 1,2:Gather all chunks and token_offsets from all ranks and reconstruct the full response tensor by splitting and placing each part
            from orbit.backends.training_utils.cp_utils import all_gather_with_cp

            full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
        else:
            full_kl_response = local_kl_chunk

        # Step 3: Compute returns on full response kl tensor.
        token_level_rewards = -kl_coef * full_kl_response
        full_mask = loss_masks[i]
        assert full_mask.sum().item() > 0, f"Sequence at index {i} is fully masked."
        last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
        token_level_rewards[last_idx] += rewards[i]

        returns_for_seq = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        for t in reversed(range(token_level_rewards.size(0))):
            # G_t = r_t + gamma * G_{t+1}
            running_return = token_level_rewards[t] + gamma * running_return
            returns_for_seq[t] = running_return

        # Step 4: Pick up the results corresponding to our local chunk's parts.
        if cp_size > 1:
            from orbit.backends.training_utils.cp_utils import slice_log_prob_with_cp

            local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
        else:
            local_returns_chunk = returns_for_seq

        final_returns_chunks.append(local_returns_chunk)

    return final_returns_chunks


def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    kl_coef: float,
) -> list[torch.Tensor]:
    """
    Calculates the unwhitened advantages for the REINFORCE++-baseline algorithm.
    Broadcasting the scalar (reward - group_baseline) to each token.

    Args:
        rewards (Tensor): A tensor of scalar rewards, where the group-wise
                                baseline has already been subtracted.
        kl (list[Tensor]): A list of per-token KL divergence tensors. Used to
                                 get the shape for broadcasting.
        loss_masks (list[Tensor]): A list of per-token loss masks.
        kl_coef (float): Coefficient for the KL penalty.

    Returns:
        list[Tensor]: A list of tensors containing the unwhitened advantages.
    """
    # Broadcast to get unwhitened advantages
    unwhitened_advantages = [
        torch.ones_like(kl_tensor) * reward_val - kl_coef * kl_tensor
        for kl_tensor, reward_val in zip(kl, rewards, strict=False)
    ]

    return unwhitened_advantages


def opd_mopd_advantages(
    rollout_data: dict,
    student_log_probs: list[torch.Tensor],
    response_lengths: list[int],
) -> list[torch.Tensor]:
    """Pure on-policy distillation (MOPD) advantage: teacher_logp - student_logp.

    Args:
        rollout_data: Rollout batch dict; must contain `teacher_log_probs`
            (list[torch.Tensor], one per sample, aligned to `response_lengths`).
        student_log_probs: Current policy log-probs per sample.
        response_lengths: Response length per sample.

    Returns:
        list[torch.Tensor]: `teacher_i[-L_i:] - student_i` per sample.
    """
    precomputed_reverse_kls = rollout_data.get("opd_reverse_kl")
    if precomputed_reverse_kls is not None:
        # Top-k OPD: rollout-side scoring already computed the per-token
        # weighted reverse KL; the MOPD advantage is simply its negation.
        if len(precomputed_reverse_kls) != len(student_log_probs):
            raise ValueError(
                f"OPD length mismatch: opd_reverse_kl={len(precomputed_reverse_kls)} "
                f"student={len(student_log_probs)}"
            )
        device = student_log_probs[0].device
        out = []
        for i, reverse_kl in enumerate(precomputed_reverse_kls):
            if not torch.is_tensor(reverse_kl):
                reverse_kl = torch.tensor(reverse_kl, dtype=torch.float32)
            reverse_kl = reverse_kl.to(device=device)
            if reverse_kl.shape != student_log_probs[i].shape:
                raise ValueError(
                    f"OPD shape mismatch at {i}: opd_reverse_kl={tuple(reverse_kl.shape)} "
                    f"student={tuple(student_log_probs[i].shape)}"
                )
            out.append(-reverse_kl)
        return out

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(
            "advantage_estimator='on_policy_distillation' needs teacher_log_probs. "
            "Enable a teacher producer with --opd-type {megatron,sglang}."
        )
    if len(teacher_log_probs) != len(student_log_probs):
        raise ValueError(f"OPD length mismatch: teacher={len(teacher_log_probs)} student={len(student_log_probs)}")
    device = student_log_probs[0].device
    out = []
    for i, (t, s, response_length) in enumerate(
        zip(teacher_log_probs, student_log_probs, response_lengths, strict=False)
    ):
        t = t.to(device=device)[-response_length:]
        if t.shape != s.shape:
            raise ValueError(f"OPD shape mismatch at {i}: teacher={tuple(t.shape)} student={tuple(s.shape)}")
        out.append(t - s)
    return out


def apply_opd_kl_to_advantages(
    opd_kl_coef: float,
    rollout_data: dict,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor],
) -> None:
    """Blend an on-policy-distillation reverse-KL penalty into base advantages, in place.

    `adv_i -= opd_kl_coef * (student_i - teacher_i)`, i.e. the blend form of OPD
    (slime/miles): any base estimator's advantage combined with distillation
    toward the teacher. Also stores the per-sample reverse KL in
    `rollout_data["opd_reverse_kl"]`.

    Args:
        opd_kl_coef: Coefficient for the reverse-KL penalty.
        rollout_data: Rollout batch dict; must contain `teacher_log_probs`
            (list[torch.Tensor], one per sample, aligned to `student_log_probs`).
        advantages: Base-estimator advantages per sample; mutated in place.
        student_log_probs: Current policy log-probs per sample, or None (e.g. on
            the critic with KL off) — the blend is then a silent no-op.
    """
    if student_log_probs is None:
        return

    precomputed_reverse_kls = rollout_data.get("opd_reverse_kl")
    if precomputed_reverse_kls is not None:
        # Top-k OPD: consume the rollout-side precomputed per-token reverse KL.
        if len(advantages) != len(precomputed_reverse_kls):
            raise ValueError(
                f"OPD length mismatch: advantages={len(advantages)}, "
                f"opd_reverse_kl={len(precomputed_reverse_kls)}."
            )
        reverse_kls = []
        for i, adv in enumerate(advantages):
            reverse_kl = precomputed_reverse_kls[i]
            if not torch.is_tensor(reverse_kl):
                reverse_kl = torch.tensor(reverse_kl, dtype=torch.float32)
            reverse_kl = reverse_kl.to(device=adv.device)
            if adv.shape != reverse_kl.shape:
                raise ValueError(
                    f"OPD shape mismatch at sample {i}: advantages={tuple(adv.shape)}, "
                    f"opd_reverse_kl={tuple(reverse_kl.shape)}."
                )
            advantages[i] = adv - opd_kl_coef * reverse_kl
            reverse_kls.append(reverse_kl)
        rollout_data["opd_reverse_kl"] = reverse_kls
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError("--use-opd requires teacher_log_probs; enable a teacher producer (--opd-type).")
    if not (len(advantages) == len(student_log_probs) == len(teacher_log_probs)):
        raise ValueError(
            f"OPD length mismatch: advantages={len(advantages)}, "
            f"student_log_probs={len(student_log_probs)}, teacher_log_probs={len(teacher_log_probs)}."
        )
    device = student_log_probs[0].device
    reverse_kls = []
    for i, adv in enumerate(advantages):
        t = teacher_log_probs[i].to(device=device)
        s = student_log_probs[i]
        if t.shape != s.shape:
            raise ValueError(f"OPD shape mismatch at {i}: teacher={tuple(t.shape)} student={tuple(s.shape)}")
        reverse_kl = s - t
        advantages[i] = adv - opd_kl_coef * reverse_kl
        reverse_kls.append(reverse_kl)
    rollout_data["opd_reverse_kl"] = reverse_kls


def icepop_gate(ratio: torch.Tensor, clip_low: float, clip_high: float) -> torch.Tensor:
    """ICE-POP hard gate: pass the importance ratio through inside the band, zero outside.

    Shared masking core used by both the policy-gradient ICE-POP path
    (``icepop_function`` in loss.py) and the OPD advantage gate
    (``apply_opd_icepop_gate``). Tokens whose ``ratio`` lies in
    ``[clip_low, clip_high]`` keep ``ratio`` (importance weight); tokens outside
    the band are zeroed (hard-gated).

    Args:
        ratio: Per-token importance ratio ``exp(train_logp - rollout_logp)``.
        clip_low: Lower band edge (``args.tis_clip_low``).
        clip_high: Upper band edge (``args.tis_clip``).

    Returns:
        Per-token gate weight, same shape as ``ratio``.
    """
    return torch.where(
        (ratio >= clip_low) & (ratio <= clip_high), ratio, torch.zeros_like(ratio)
    )


def apply_opd_icepop_gate(
    rollout_data: dict,
    advantages: list[torch.Tensor],
    clip_low: float,
    clip_high: float,
) -> None:
    """Hard-gate the OPD advantage by the per-token train/rollout ratio, in place.

    For async / off-policy rollouts the acting (rollout) policy drifts from the
    current student, biasing the OPD advantage. Following NeMo-RL MOPD, gate each
    token by the ICE-POP importance ratio ``exp(train_logp - rollout_logp)``:
    in-band tokens are reweighted by the ratio, out-of-band tokens are zeroed —
    reusing the exact ratio and gate as orbit's policy-gradient path
    (``icepop_function``).

    Args:
        rollout_data: Rollout batch dict; must contain per-sample ``log_probs``
            (train-recomputed student log-probs) and ``rollout_log_probs``
            (acting-policy log-probs), aligned to ``advantages``.
        advantages: OPD advantages per sample; mutated in place.
        clip_low: Lower band edge (``args.tis_clip_low``).
        clip_high: Upper band edge (``args.tis_clip``).
    """
    train_log_probs = rollout_data.get("log_probs")
    rollout_log_probs = rollout_data.get("rollout_log_probs")
    if train_log_probs is None or rollout_log_probs is None:
        raise ValueError(
            "--opd-icepop needs train log_probs and rollout_log_probs to form the "
            "train/rollout importance ratio, but one is missing from rollout_data. "
            "Ensure --use-rollout-logprobs is off (so student log-probs are recomputed) "
            "and rollout log-probs are collected."
        )
    for i in range(len(advantages)):
        ratio = torch.exp(train_log_probs[i] - rollout_log_probs[i])
        advantages[i] = advantages[i] * icepop_gate(ratio, clip_low, clip_high)


def get_advantages_and_returns(
    total_len: int,
    response_len: int,
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.

    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Input:
    - values: Tensor of shape (response_size,)
    - rewards: Tensor of shape (response_size,)

    Output:
    - advantages: Tensor of shape (response_size,)
    - returns: Tensor of shape (response_size,)
    """
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        from orbit.backends.training_utils.cp_utils import all_gather_with_cp

        full_rewards = all_gather_with_cp(rewards, total_len, response_len)
        full_values = all_gather_with_cp(values, total_len, response_len)
    else:
        full_rewards = rewards
        full_values = values

    lastgaelam = 0
    advantages_reversed = []

    for t in reversed(range(response_len)):
        nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
        delta = full_rewards[t] + gamma * nextvalues - full_values[t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    full_advantages = torch.tensor(advantages_reversed[::-1], dtype=full_values.dtype, device=full_values.device)
    full_returns = full_advantages + full_values

    if cp_size > 1:
        from orbit.backends.training_utils.cp_utils import slice_log_prob_with_cp

        advantages = slice_log_prob_with_cp(full_advantages, total_len, response_len)
        returns = slice_log_prob_with_cp(full_returns, total_len, response_len)
    else:
        advantages = full_advantages
        returns = full_returns

    return advantages.detach(), returns


def get_advantages_and_returns_batch(
    total_lengths,
    response_lengths,
    values_list,
    rewards_list,
    terminal_rewards,
    qkv_format,
    max_seq_lens,
    loss_masks,
    gamma,
    lambd,
    chunked: bool = True,
):
    """
    Batched GAE with CP support, computed over trainable tokens only.

    Semantics:
      - Masked tokens (`loss_mask == 0`, e.g. tool/env observations in
        multi-turn rollouts) are not MDP transitions. GAE runs on the
        subsequence of trainable tokens, so masked tokens carry no reward
        (including KL shaping), contribute no value delta, and the GAE carry
        crosses them without extra `gamma * lambd` decay.
      - The terminal reward is added at the last trainable token, not the last
        response token.
      - Fully masked samples get zero advantages and returns; their terminal
        reward is dropped.
      - Truncated sequences use the same zero bootstrap as terminated ones:
        the value after the last trainable token is taken as 0 and the
        observed terminal reward is still applied.
      - This function outputs zero advantages and returns at masked positions.
        Downstream transforms may still shift these entries to nonzero values
        (advantage whitening applies its affine transform to every position,
        and the on-policy distillation KL penalty is added per token), but the
        whitening statistics themselves are mask-weighted, so the injected
        zeros do not bias them. Correctness relies on the policy and value
        losses masking these positions out (the policy loss re-zeros
        advantages at inactive tokens and all loss reducers weight by
        `loss_mask`), so masked positions never receive gradient.

    C_i is the length of values_list[i] and rewards_list[i] on the current CP rank.
    Input:
        total_lengths:     list[int], each sample's total_len
        response_lengths:  list[int], each sample's response_len
        values_list:       list[Tensor], each current-CP-rank tensor has shape [C_i]
        rewards_list:      list[Tensor], same shape as values_list
        terminal_rewards:  list[float], one scalar sequence reward per sample
        qkv_format:        str, sequence layout used to split tensors across CP ranks
        max_seq_lens:      list[int] of padded lengths (BSHD, or padded THD e.g. DSV4), or None
        loss_masks:        list[Tensor], full-response masks, each has shape [R_i]
    Output:
        advantages_list:   list[Tensor], each current-CP-rank tensor has shape [C_i]
        returns_list:      list[Tensor], same shape
    """

    with torch.no_grad():
        B = len(response_lengths)
        assert B == len(values_list)
        assert B == len(rewards_list)
        assert B == len(terminal_rewards)
        assert B == len(loss_masks)

        from orbit.backends.training_utils.parallel import get_parallel_state

        cp_size = get_parallel_state().cp.size
        if cp_size > 1 and qkv_format == "bshd":
            assert max_seq_lens is not None, "max_seq_lens is required for BSHD with CP"
            assert B == len(max_seq_lens)
            max_seq_lens_per_sample = max_seq_lens
        elif cp_size > 1 and max_seq_lens is not None:  # padded THD (e.g. DSV4)
            assert B == len(max_seq_lens)
            max_seq_lens_per_sample = max_seq_lens
        else:
            max_seq_lens_per_sample = [None] * B

        device = values_list[0].device
        dtype = values_list[0].dtype

        if cp_size > 1:
            from orbit.backends.training_utils.cp_utils import all_gather_with_cp

            full_values_list = []
            full_rewards_list = []

            for total_len, resp_len, v, r, max_seq_len in zip(
                total_lengths, response_lengths, values_list, rewards_list,
                max_seq_lens_per_sample, strict=False,
            ):
                full_v = all_gather_with_cp(v, total_len, resp_len, qkv_format=qkv_format, max_seq_len=max_seq_len)
                full_r = all_gather_with_cp(r, total_len, resp_len, qkv_format=qkv_format, max_seq_len=max_seq_len)
                full_values_list.append(full_v)
                full_rewards_list.append(full_r)

            # full_values_list[i].shape = [resp_len_i]
        else:
            full_values_list = values_list
            full_rewards_list = rewards_list

        # Compress each sample to its trainable positions so that masked
        # tokens do not act as MDP transitions in the GAE recursion.
        trainable_indices = [
            loss_masks[i][: response_lengths[i]].to(device).nonzero(as_tuple=True)[0] for i in range(B)
        ]
        trainable_lengths = [idx.numel() for idx in trainable_indices]

        # pad to max_len for batched GAE
        max_len = max(trainable_lengths)

        packed_values = torch.zeros(B, max_len, device=device, dtype=dtype)
        packed_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)

        for i in range(B):
            K = trainable_lengths[i]
            if K > 0:
                idx = trainable_indices[i]
                packed_values[i, :K] = full_values_list[i][idx]
                packed_rewards[i, :K] = full_rewards_list[i][idx]
                packed_rewards[i, K - 1] += terminal_rewards[i]

        if max_len == 0:
            packed_advantages = torch.zeros(B, 0, device=device, dtype=dtype)
            packed_returns = torch.zeros(B, 0, device=device, dtype=dtype)
        elif not chunked:
            packed_advantages, packed_returns = vanilla_gae(
                rewards=packed_rewards, values=packed_values, gamma=gamma, lambd=lambd,
            )
        else:
            packed_advantages, packed_returns = chunked_gae(
                rewards=packed_rewards, values=packed_values, gamma=gamma, lambd=lambd,
            )

        advantages_list = []
        returns_list = []

        if cp_size > 1:
            from orbit.backends.training_utils.cp_utils import slice_log_prob_with_cp

        for i in range(B):
            resp_len = response_lengths[i]
            K = trainable_lengths[i]

            adv_full = torch.zeros(resp_len, device=device, dtype=dtype)
            ret_full = torch.zeros(resp_len, device=device, dtype=dtype)
            if K > 0:
                idx = trainable_indices[i]
                adv_full[idx] = packed_advantages[i, :K]
                ret_full[idx] = packed_returns[i, :K]

            if cp_size > 1:
                max_seq_len = max_seq_lens_per_sample[i]
                adv_full = slice_log_prob_with_cp(
                    adv_full, total_lengths[i], resp_len,
                    qkv_format=qkv_format, max_token_len=max_seq_len,
                )
                ret_full = slice_log_prob_with_cp(
                    ret_full, total_lengths[i], resp_len,
                    qkv_format=qkv_format, max_token_len=max_seq_len,
                )

            advantages_list.append(adv_full)
            returns_list.append(ret_full)

    return advantages_list, returns_list


def vanilla_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
):
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)

    full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
    full_returns = full_advantages + values  # [B, max_len]
    return full_advantages, full_returns


def chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int = 128,
):
    """
    Compute Generalized Advantage Estimation (GAE) using a FlashLinearAttention-
    inspired algorithm: parallel prefix scan within chunks and recurrent state
    propagation across chunks.

    This reduces the sequential dependency length from O(T) to O(T / chunk_size),
    while keeping chunk computations fully parallelizable (O(C^2) per chunk).

    Args:
        rewards (Tensor): [B, T] reward sequence.
        values (Tensor):  [B, T] value predictions. The next-value of the final
                          step is assumed to be zero (standard PPO convention).
        gamma (float): discount factor.
        lam (float): GAE lambda.
        chunk_size (int): sequence chunk length for parallel scan.

    Returns:
        advantages (Tensor): [B, T] computed advantages.
        returns (Tensor):    [B, T] advantages + values.
    """

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)

    device = rewards.device
    dtype = rewards.dtype

    # -------------------------------------------------------------------------
    # Build δ_t = r_t + γ * V_{t+1} - V_t   with V_{T} = 0
    # -------------------------------------------------------------------------
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values

    # Reformulate backward GAE as a forward scan on the reversed sequence:
    #   S[i] = Δ[i] + w * S[i - 1],   w = γλ
    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])  # [B, T]

    # -------------------------------------------------------------------------
    # Pad to a multiple of chunk_size
    # -------------------------------------------------------------------------
    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0

    B, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size

    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    # -------------------------------------------------------------------------
    # Construct the intra-chunk parallel scan kernel M
    #
    # For a chunk Δ[0..C-1], we want:
    #   S_local[t] = sum_{k=0..t} w^(t-k) * Δ[k]
    #
    # This is implemented as:
    #   S_local = Δ @ M
    #
    # where:
    #   M[i, j] = w^(j - i)    if j >= i
    #             0            otherwise
    # -------------------------------------------------------------------------
    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row

    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    mask = diff >= 0

    if w == 0.0:
        M[mask & (diff == 0)] = 1.0
    else:
        M[mask] = w ** diff[mask].to(dtype)

    # pow_vec[t] = w^(t+1), used to inject the recurrent state s_prev
    if w == 0.0:
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Parallel compute local chunk results (assuming initial state = 0)
    # -------------------------------------------------------------------------
    deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
    S_local_flat = deltas_flat @ M
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    # Effective length of each chunk (the last chunk may be padded)
    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    # -------------------------------------------------------------------------
    # Recurrent propagation between chunks
    #
    # Each chunk contributes:
    #   S_global[t] = S_local[t] + w^(t+1) * s_prev
    #
    # And updates:
    #   s_prev = S_global[last_t]
    # -------------------------------------------------------------------------
    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]  # state for next chunk

    # Remove padding and flip back to original time order
    if pad > 0:
        S_rev = S_rev[:, :T]

    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values

    return advantages, returns


def calculate_log_probs_and_entropy(
    logits,
    tokens,
    tp_group,
    with_entropy: bool = False,
    entropy_no_grad: bool = False,
    chunk_size: int = -1,
    true_on_policy: bool = False,
    vocab_size: int | None = None,
):
    if true_on_policy:
        return _calculate_log_probs_and_entropy_true_on_policy(
            logits,
            tokens,
            tp_group,
            with_entropy=with_entropy,
            entropy_no_grad=entropy_no_grad,
            vocab_size=vocab_size,
        )

    logits = logits.contiguous()
    # Follow-up: not sure why we need to clone the logits here.
    # Without the clone, the backward will trigger inplace edit error.
    # It seems that the function with tp will modify the logits inplace.
    # When entropy_no_grad is True, the entropy backward never runs, so the
    # in-place mutation in _VocabParallelEntropy.backward is moot and the
    # clone can be dropped.
    entropy = None

    def _entropy(logits_in):
        if entropy_no_grad:
            with torch.no_grad():
                return compute_entropy_from_logits(logits_in, tp_group)
        return compute_entropy_from_logits(logits_in.clone(), tp_group)

    if logits.size(0) != 0:
        if chunk_size > 0:
            num_chunks = (logits.size(0) - 1) // chunk_size + 1
            tokens_chunks = tokens.chunk(num_chunks, dim=0)
            logits_chunks = logits.chunk(num_chunks, dim=0)
            log_probs = []
            for tokens_chunk, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                log_prob = compute_log_probs(logits_chunk, tokens_chunk, tp_group)
                log_probs.append(log_prob)
            log_prob = torch.cat(log_probs, dim=0)
            if with_entropy:
                entropys = [_entropy(logits_chunk) for _, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True)]
                entropy = torch.cat(entropys, dim=0)
        else:
            log_prob = compute_log_probs(logits, tokens, tp_group)
            if with_entropy:
                entropy = _entropy(logits)
    else:
        log_prob = logits.new_zeros((0,))
        if with_entropy:
            entropy = logits.new_zeros((0,))

    return log_prob, entropy


def _prepare_true_on_policy_full_logits(
    logits_or_shards: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    vocab_size: int | None = None,
) -> torch.Tensor:
    if isinstance(logits_or_shards, (list, tuple)):
        full_logits = torch.cat([shard.contiguous() for shard in logits_or_shards], dim=-1)
    else:
        full_logits = logits_or_shards.contiguous()

    # Truncate Megatron's padded vocab back to the real tokenizer vocab before
    # log_softmax, matching what SGLang normalizes over.
    if vocab_size is not None and full_logits.size(-1) > vocab_size:
        full_logits = full_logits[..., :vocab_size]

    return full_logits


def _split_replicated_loss_gather_grad(
    grad_output: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    local_last_dim: int,
) -> torch.Tensor:
    if world_size <= 1:
        return grad_output.contiguous()

    expected_last_dim = local_last_dim * world_size
    if grad_output.size(-1) != expected_last_dim:
        raise RuntimeError(
            "True-on-policy replicated-loss gather backward expected the full padded "
            f"vocab dimension to be {expected_last_dim}, got {grad_output.size(-1)}."
        )
    start = rank * local_last_dim
    return grad_output[..., start : start + local_last_dim].contiguous()


class _ReplicatedLossAllGatherLastDim(torch.autograd.Function):
    """All-gather vocab shards for a loss replicated on every TP rank.

    Megatron's standard all-gather autograd uses reduce-scatter in backward,
    which is correct when each rank contributes a distinct output gradient. In
    the true-on-policy logprob path every TP rank computes the same scalar loss
    from the gathered full vocabulary, so reduce-scatter would sum identical
    gradients and scale the local logits gradient by TP size.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        world_size = group.size()
        ctx.group = group
        ctx.local_last_dim = input_.shape[-1]
        ctx.world_size = world_size

        if world_size == 1:
            return input_.contiguous()

        from megatron.core.tensor_parallel.mappings import dist_all_gather_func

        gather_shape = list(input_.shape)
        gather_shape[0] *= world_size
        gathered = torch.empty(gather_shape, dtype=input_.dtype, device=input_.device)
        dist_all_gather_func(gathered, input_.contiguous(), group=group)
        return torch.cat(gathered.chunk(world_size, dim=0), dim=-1).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            _split_replicated_loss_gather_grad(
                grad_output,
                rank=ctx.group.rank(),
                world_size=ctx.world_size,
                local_last_dim=ctx.local_last_dim,
            ),
            None,
        )


def _gather_true_on_policy_full_logits(
    logits: torch.Tensor,
    process_group: dist.ProcessGroup | None,
    *,
    vocab_size: int | None = None,
) -> torch.Tensor:
    if process_group is None or process_group.size() <= 1:
        return _prepare_true_on_policy_full_logits(logits, vocab_size=vocab_size)

    full_logits = _ReplicatedLossAllGatherLastDim.apply(logits.contiguous(), process_group)
    return _prepare_true_on_policy_full_logits(full_logits, vocab_size=vocab_size)


def _calculate_log_probs_and_entropy_true_on_policy(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    tp_group: dist.ProcessGroup | None,
    with_entropy: bool = False,
    entropy_no_grad: bool = False,
    vocab_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Log-prob and entropy computation matching SGLang's scoring contract.

    Args:
        logits: Aligned local logits of shape ``[R, V_local]`` (already
            response-sliced and temperature-scaled by ``get_responses``;
            a vocab shard under TP>1).
        tokens: Target tokens of shape ``[R]``.
        tp_group: Tensor-parallel process group for the full-vocab gather.
        with_entropy: If True, also compute entropy.
        vocab_size: Real tokenizer vocab size. If provided, padded logits are
            truncated after the full-vocab gather and before ``log_softmax``.

    Returns:
        Tuple of ``(log_probs, entropy)`` where *log_probs* has shape ``[R]``
        and *entropy* has shape ``[R]`` or is ``None``.
    """
    if logits.size(0) == 0:
        log_prob = logits.new_zeros((0,))
        entropy = logits.new_zeros((0,)) if with_entropy else None
        return log_prob, entropy

    full_logits = _gather_true_on_policy_full_logits(logits, tp_group, vocab_size=vocab_size)
    log_probs_full = torch.log_softmax(full_logits, dim=-1)
    log_prob = torch.gather(log_probs_full, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)

    entropy = None
    if with_entropy:
        entropy_log_probs = log_probs_full.detach() if entropy_no_grad else log_probs_full
        probs = entropy_log_probs.exp()
        entropy = -(probs * entropy_log_probs).sum(dim=-1)

    return log_prob, entropy
