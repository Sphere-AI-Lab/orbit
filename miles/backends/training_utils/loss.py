from argparse import Namespace

import torch
from torch.utils.checkpoint import checkpoint

from miles.backends.training_utils.cp_utils import get_local_response_loss_masks, get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub.advantages import compute_advantages, normalize_advantages
from miles.backends.training_utils.loss_hub.losses import get_loss_function
from miles.backends.training_utils.loss_hub.math_utils import compute_approx_kl
from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import TrainAdvantageComputationEvent
from miles.utils.multi_lora import is_multi_lora_enabled
from miles.utils.types import RolloutBatch

# ORBIT-SEAM: orbit's ICE-POP hard gate for the OPD advantage, applied by
# compute_advantages_and_returns below. Home: miles/orbit/opd/advantages.py, re-exported by
# loss_hub.math_utils.
from miles.backends.training_utils.loss_hub.math_utils import apply_opd_icepop_gate

# ORBIT-SEAM: back-compatible re-export surface. Upstream decomposed this module into
# loss_hub/{advantages,corrections,logit_processors,losses,opd,math_utils}.py and left loss.py
# as the thin advantage/loss dispatcher; every name orbit's own code and tests used to import
# from `miles.backends.training_utils.loss` is re-bound here so those import sites (the two
# actors, megatron_utils.model, miles.orbit.opd.losses' call-time imports and the OPD/true-on-policy
# tests) keep resolving exactly as they did before the decomposition. Each name has exactly one
# implementation, in the loss_hub module or the orbit home named beside it.
from miles.backends.training_utils.cp_utils import (  # noqa: F401
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
)
from miles.backends.training_utils.loss_hub.corrections import icepop_function, vanilla_tis_function  # noqa: F401
from miles.backends.training_utils.loss_hub.logit_processors import (  # noqa: F401
    get_log_probs_and_entropy,
    get_responses,
    get_values,
)
from miles.backends.training_utils.loss_hub.losses import (  # noqa: F401
    policy_loss_function,
    sft_loss_function,
    value_loss_function,
)
from miles.backends.training_utils.loss_hub.math_utils import (  # noqa: F401
    VALUE_EV_STAT_KEYS,
    _gather_true_on_policy_full_logits,
    _safe_clamp_log_ratio,
    _safe_exp_neg_ppo_kl,
    calculate_log_probs_and_entropy,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
    icepop_gate,
    opd_mopd_advantages,
)
from miles.orbit.opd.losses import (  # noqa: F401
    _TOPK_LOG_INF,
    _response_masked_max,
    _response_masked_min,
    _topk_kl_terms,
    _topk_overlap_membership,
    opd_jsd_loss_function,
    opd_topk_loss_function,
    opd_topk_sample_log_probs,
)


def _detach_rollout_tensor_list(rollout_data: RolloutBatch, key: str) -> list[torch.Tensor] | None:
    tensors = rollout_data.get(key)
    if tensors is None:
        return None

    detached_tensors = [tensor.detach() for tensor in tensors]
    rollout_data[key] = detached_tensors
    return detached_tensors


# ORBIT-SEAM: orbit adds the `role` argument (defaulted, so base callers are unaffected) so
# the critic can skip the actor-only OPD advantage adjustments below; documented in the
# docstring's Args block
def compute_advantages_and_returns(
    args: Namespace,
    rollout_data: RolloutBatch,
    role: str = "actor",
) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    "reinforce_plus_plus_baseline" and "on_policy_distillation". On-policy
    distillation (OPD) is also applied orthogonally on top of any estimator via
    `args.use_opd`. When `args.normalize_advantages` is True, advantages are
    whitened across the data-parallel group using masked statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages), unless the last stage is explicitly allowed to derive
    zero-KL shapes without the standalone actor pass.

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
        role: "actor" or "critic". The critic never receives teacher_log_probs
            (sync_actor_critic_data does not broadcast them) and its value loss
            consumes `returns`, which the OPD blend does not touch — so OPD
            advantage adjustments are skipped for role="critic".
    """
    allow_missing_log_probs = args.skip_actor_forward_only and not args.use_rollout_logprobs
    log_probs_key = "rollout_log_probs" if args.use_rollout_logprobs else "log_probs"
    log_probs: list[torch.Tensor] = rollout_data.get(log_probs_key)
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if log_probs is None and values is None:
        if not (allow_missing_log_probs and get_parallel_state().is_pp_last_stage):
            return

    # This is the authoritative persistence boundary: scores produced before
    # the policy update are fixed training data and must not retain a graph.
    _detach_rollout_tensor_list(rollout_data, "log_probs")
    _detach_rollout_tensor_list(rollout_data, "rollout_log_probs")
    _detach_rollout_tensor_list(rollout_data, "ref_log_probs")
    _detach_rollout_tensor_list(rollout_data, "teacher_log_probs")
    log_probs = rollout_data.get(log_probs_key)
    ref_log_probs = rollout_data.get("ref_log_probs")

    if log_probs is None and values is None:
        local_masks = get_local_response_loss_masks(
            total_lengths, response_lengths, loss_masks, args.qkv_format, max_seq_lens
        )
        kl = [torch.zeros_like(mask, dtype=torch.float32) for mask in local_masks]
    elif args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    advantages, returns = compute_advantages(
        args=args,
        kl=kl,
        rewards=rewards,
        log_probs=log_probs,
        loss_masks=loss_masks,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        values=values,
        # ORBIT-SEAM: only the orbit-added "on_policy_distillation" estimator arm reads this
        rollout_data=rollout_data,
    )

    # Apply on-policy distillation KL penalty to advantages (orthogonal to advantage estimator)
    # ORBIT-SEAM: base applies the blend for every role; orbit restricts it to the actor
    if role == "actor" and args.use_opd:
        apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )

    # ORBIT-SEAM: optional async/off-policy ICE-POP correction for the OPD advantage (pure-MOPD
    # or blend): hard-gate tokens whose train/rollout importance ratio leaves the band.
    if role == "actor" and getattr(args, "opd_icepop", False):
        apply_opd_icepop_gate(rollout_data, advantages, args.tis_clip_low, args.tis_clip)

    if args.normalize_advantages:
        advantages = normalize_advantages(args, advantages, loss_masks, total_lengths, response_lengths, max_seq_lens)

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
    apply_megatron_loss_scaling: bool = False,
    num_rollouts: int | None = None,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", "opd_jsd_loss", "opd_topk_loss",
    or a custom loss function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).
        num_rollouts: This step's rollout count (total across DP), used as
            the loss normalizer; None falls back to the legacy batch/args value.

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    parallel_state = get_parallel_state()
    # ORBIT-SEAM: base clamps each sample's token count to a minimum of 1; orbit keeps the
    # count exact (rationale inline below)
    # Megatron sums this normalizer across micro-batches and DP/CP ranks before
    # scaling gradients, and already leaves gradients unscaled when that global
    # count is zero. Keep the local count exact: a rejected/all-masked sample has
    # a zero loss numerator and must not add a phantom token to the denominator.
    num_tokens = sum(loss_mask.sum() for loss_mask in batch["loss_masks"])
    num_samples = len(batch["response_lengths"])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
        denominators=batch.get("rollout_mask_sums", None),
    )

    func = get_loss_function(args)

    if args.recompute_loss_function:
        loss, log = checkpoint(
            func,
            args,
            batch,
            logits,
            sum_of_sample_mean,
        )
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # Forces autograd to traverse the full graph on every rank to avoid hang.
    if parallel_state.cp.size > 1 and args.allgather_cp:
        loss = loss + 0 * logits.sum()

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    if num_rollouts is not None:
        global_batch_size = num_rollouts
    else:
        assert args.use_dynamic_global_batch_size == ("dynamic_global_batch_size" in batch)
        global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    # Multi-LoRA: samples enter the gradient buffers with weight 1; per-adapter
    # normalization (1/adapter_global_batch_size, a constant known in advance)
    # is applied to the accumulated slot gradient at optimizer-step time.
    if is_multi_lora_enabled(args):
        global_batch_size = 1
    if not args.calculate_per_token_loss:
        if apply_megatron_loss_scaling:
            loss_parallel_size = (
                parallel_state.intra_dp.size
                if args.true_on_policy_mode and parallel_state.is_ulysses_cp
                else parallel_state.intra_dp_cp.size
            )
            loss = loss * num_microbatches / global_batch_size * loss_parallel_size
        else:
            loss = loss / global_batch_size * parallel_state.intra_dp.size
    else:
        if apply_megatron_loss_scaling:
            loss = loss * parallel_state.cp.size

    # ORBIT-SEAM: base builds the normalizer with torch.tensor(num_tokens, ...), which would
    # now drag the (unclamped, still-attached) token-count graph into Megatron; detach and
    # move the existing tensor instead
    normalizer = (
        num_tokens.detach().to(device=logits.device)
        if args.calculate_per_token_loss
        else torch.tensor(1, device=logits.device)
    )
    return (
        loss,
        normalizer,
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [num_samples if not args.calculate_per_token_loss else num_tokens] + list(log.values()),
                device=logits.device,
            ),
        },
    )


def log_train_advantage_computation_event(rollout_data: RolloutBatch) -> None:
    if not is_event_logger_initialized():
        return

    advantages = rollout_data.get("advantages")
    witness_ids = rollout_data.get("witness_ids")
    if advantages is None or witness_ids is None:
        return

    get_event_logger().log(
        TrainAdvantageComputationEvent,
        dict(
            advantages=[x.tolist() for x in advantages],
            witness_ids=[x.tolist() for x in witness_ids],
        ),
        print_log=False,
    )
