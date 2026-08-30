"""Orbit's argument home: every orbit-added CLI argument, predicate and validator.

Phase 2 of the miles-isolation work moved this code out of the vendored
``miles/utils/arguments.py`` (see
``docs/superpowers/plans/2026-08-29-phase2-arguments-registration.md``), which now
carries upstream's bytes plus stamped ``# ORBIT-SEAM`` hooks into this module.

Two entry points are called across the seam:

* :func:`add_orbit_arguments` -- registers every orbit-added argument, and
  re-defaults, re-``choices``-es and re-``help``-s the handful of miles arguments orbit
  overrides. It runs at the end of miles' ``add_miles_arguments``, so every
  argument it overrides is already registered.
* :func:`orbit_validate_args` -- the reward-side validator bundle.

The other validators are called individually from ``miles_validate_args`` because
their position in the base validation order is load-bearing.

Import direction across the seam is strictly miles -> orbit: this module must never
import ``miles.utils.arguments`` at module level (circular import). The few
``orbit.*`` imports below are function-local for the same reason the originals were:
they pull in heavy or optional subsystems.
"""

import argparse
import json
import logging
import math
import os

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Moved verbatim out of miles/utils/arguments.py (top-level orbit definitions).
# ---------------------------------------------------------------------------


_PEFT_LORA_DEFAULTS = {
    "lora_rank": 0,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "lora_type": "lora",
    "lora_adapter_path": None,
    "lora_sync_from_tensor": False,
    "lora_a_init_method": "xavier",
}
_PEFT_OFT_DEFAULTS = {
    "oft_type": "canonical_oft",
    "oft_block_size": 0,
    "oft_coft": False,
    "oft_eps": 1e-5,
    "oft_block_share": False,
    "oft_adapter_path": None,
}
_PEFT_METHODS = {"none", "oft", "lora"}
SFT_ROLLOUT_FUNCTION_PATH = "miles.rollout.sft_rollout.generate_rollout"
DEFAULT_ROLLOUT_FUNCTION_PATHS = {
    "miles.rollout.sglang_rollout.generate_rollout",
    "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn",
}


def uses_rollout_engines(args) -> bool:
    """Whether this run needs SGLang rollout engines and weight sync."""
    return bool(getattr(args, "use_rollout_engines", True))


def needs_opd_teacher(args) -> bool:
    """Whether a teacher log-prob producer is needed for on-policy distillation.

    Both OPD objective forms consume the same ``rollout_data["teacher_log_probs"]``:
    pure MOPD (``--advantage-estimator on_policy_distillation``) and the blend
    (``--use-opd``). Either one requires teacher production.
    """
    return args.advantage_estimator == "on_policy_distillation" or getattr(args, "use_opd", False)


def uses_separate_critic(args) -> bool:
    """True when PPO runs the legacy separate full-model critic workers."""
    return getattr(args, "use_critic", False) and getattr(args, "critic_mode", "full") == "full"


def uses_adapter_critic(args) -> bool:
    """True when PPO runs the one-trunk adapter critic inside the actor workers."""
    return getattr(args, "use_critic", False) and getattr(args, "critic_mode", "full") == "adapter"


def uses_head_critic(args) -> bool:
    """True when PPO runs the one-trunk value-head-only critic (detached trunk)."""
    return getattr(args, "use_critic", False) and getattr(args, "critic_mode", "full") == "head"


def uses_one_trunk_critic(args) -> bool:
    """True for either one-trunk critic (adapter or head): colocated on the actor
    workers, trunk storage aliased to the actor's, no separate critic worker."""
    return uses_adapter_critic(args) or uses_head_critic(args)


def validate_async_off_policy_correction(args) -> None:
    """Require an explicit behavior-policy choice for async PPO training.

    In the async train loop the next rollout is generated before the current
    weight update is published, so samples can come from a stale policy. With
    the default flags the PPO ratio denominator (``log_probs``) is recomputed
    by the *current* actor, silently anchoring clipping (and KL-shaped
    advantages) to a policy that never generated the trajectory; the recorded
    ``weight_versions`` are a metric, not an enforcement mechanism.

    Called from ``train_async.py`` only — synchronous training recomputes log
    probs against the same weights that generated the rollout. Mirrors miles
    bc232eb88 with the ``use_critic`` gate adapted to dev's estimator arg
    (PPO implies a critic in both adapter and separate modes).
    """
    update_weights_interval = args.update_weights_interval
    if type(update_weights_interval) is not int or update_weights_interval <= 0:
        raise ValueError(
            "--update-weights-interval must be a positive integer for async training, "
            f"got {update_weights_interval!r}."
        )

    if args.advantage_estimator != "ppo":
        return
    keep_old_actor_matches_behavior = args.keep_old_actor and update_weights_interval == 1
    assert args.use_rollout_logprobs or args.use_tis or keep_old_actor_matches_behavior, (
        "Async PPO training requires an explicit behavior-policy correction, because rollouts are "
        "generated before the current weight update while log probs are recomputed by the current "
        "actor by default. Pass one of: --use-rollout-logprobs (use the rollout engine's log probs "
        "as the ratio denominator), --use-tis (truncated importance sampling correction), or "
        "--keep-old-actor with --update-weights-interval 1 (recompute the denominator with the "
        "weights the rollout engines used)."
    )


def validate_rollout_temperature(args) -> None:
    """Reject non-finite or non-positive training rollout temperatures (spec Phase S).

    ``get_responses`` divides logits by this value; 0 would produce infs and
    a negative value silently flips the distribution. Greedy evaluation is
    configured via the eval args, not by zeroing the training temperature.
    """
    rollout_temperature = float(args.rollout_temperature)
    if not math.isfinite(rollout_temperature) or rollout_temperature <= 0:
        raise ValueError(
            "--rollout-temperature must be finite and > 0 for training rollouts, " f"got {args.rollout_temperature}."
        )


def validate_opd_topk_reference_kl_args(args) -> None:
    """Reject ref-policy KL knobs before generic ref-checkpoint validation."""
    if getattr(args, "loss_type", None) != "opd_topk_loss":
        return
    if getattr(args, "use_kl_loss", False) or float(getattr(args, "kl_coef", 0) or 0) != 0:
        raise ValueError(
            "--loss-type opd_topk_loss is incompatible with reference-policy KL settings "
            "(--use-kl-loss/--kl-coef): this direct distillation loss does not consume "
            "reference log-probs. Disable those settings or use policy_loss."
        )


def validate_opd_topk_vocab_size(args) -> None:
    """Ensure direct-OPD K fits the real student vocabulary once it is known."""
    if getattr(args, "loss_type", None) != "opd_topk_loss":
        return
    top_k = getattr(args, "opd_log_prob_top_k", 0) or 0
    vocab_size = getattr(args, "vocab_size", None)
    if vocab_size is not None and top_k > vocab_size:
        raise ValueError(
            f"--opd-log-prob-top-k ({top_k}) cannot exceed the student's real vocabulary " f"size ({vocab_size})."
        )


def add_on_policy_distillation_arguments(parser):
    """On-policy distillation (OPD) teacher config. Mirrors slime arguments.py:1084-1125."""
    parser.add_argument(
        "--use-opd",
        action="store_true",
        default=False,
        help=(
            "Enable blend-mode on-policy distillation: subtract opd_kl_coef * (student - teacher) "
            "from a reward-based estimator's advantage. Requires a teacher producer (--opd-type). "
            "Mutually exclusive with --advantage-estimator on_policy_distillation (pure MOPD)."
        ),
    )
    parser.add_argument(
        "--opd-type",
        type=str,
        choices=["megatron", "sglang"],
        default=None,
        help=(
            "Teacher log-prob producer: 'megatron' loads a second in-process Megatron model "
            "scored by a forward pass; 'sglang' scores on the rollout engine, either against "
            "an external SGLang teacher server (--opd-teacher-url) or a local same-base teacher "
            "in the reserved orbit_teacher adapter slot."
        ),
    )
    parser.add_argument(
        "--opd-kl-coef",
        type=float,
        default=1.0,
        help="Blend coefficient lambda for the distillation KL term applied to advantages under --use-opd.",
    )
    parser.add_argument(
        "--opd-teacher-load",
        type=str,
        default=None,
        help="Megatron checkpoint directory for the in-process OPD teacher; legacy sugar for --opd-teacher load:<ckpt>.",
    )
    parser.add_argument(
        "--opd-teacher",
        type=str,
        default=None,
        help=(
            "What the OPD teacher IS: base (frozen base, adapter off), adapter:<path> "
            "(base + frozen adapter checkpoint), self:ema / self:lag (EMA or lagged "
            "snapshot of the student adapter), or load:<megatron-ckpt> (full second "
            "model; same as legacy --opd-teacher-load). Same-base specs require PEFT."
        ),
    )
    parser.add_argument(
        "--opd-ema-decay",
        type=float,
        default=0.999,
        help="EMA decay beta for --opd-teacher self:ema (per training step).",
    )
    parser.add_argument(
        "--opd-self-teacher-interval",
        type=int,
        default=1,
        help="Snapshot refresh cadence (training steps) for --opd-teacher self:lag.",
    )
    parser.add_argument(
        "--opd-promote-interval",
        type=int,
        default=None,
        help=(
            "Promote the self-teacher (EMA/lag) adapter to the rollout engine's "
            "orbit_teacher slot every N training steps. Required for self:* teachers "
            "with --opd-type sglang."
        ),
    )
    parser.add_argument(
        "--opd-teacher-ckpt-step",
        type=int,
        default=None,
        help="Checkpoint step (iteration) to load for the OPD teacher. If None, use the latest iteration.",
    )
    parser.add_argument(
        "--opd-teacher-url",
        type=str,
        default=None,
        help=(
            "URL of the external SGLang teacher server's /generate endpoint, e.g. http://host:port/generate "
            "(required only for external-teacher sglang mode, not for a local same-base teacher)."
        ),
    )
    parser.add_argument(
        "--opd-teacher-urls",
        type=str,
        nargs="+",
        default=None,
        metavar="NAME=URL[@W][,URL[@W]...]",
        help=(
            "Multi-teacher routing/ensemble map for --opd-type=sglang, e.g. "
            "--opd-teacher-urls math=http://h1:30001/generate code=http://h2:30002/generate. "
            "Each sample is routed to the teacher group named by "
            "sample.metadata[--opd-teacher-key]; the reserved name 'default' is the "
            "fallback for samples with a missing or unknown name. A name mapping to "
            "several comma-separated URLs is an ensemble: every member scores the "
            "sample in parallel and the targets are combined as a weighted mixture "
            "in probability space (logsumexp of weighted logprobs); per-URL weights "
            "default to 1.0 (uniform). With --opd-log-prob-top-k > 0, ensembles "
            "require --opd-top-k-strategy only-student. When unset, all samples are "
            "scored by the single teacher at --opd-teacher-url (original behavior)."
        ),
    )
    parser.add_argument(
        "--opd-topk-tail-bucket",
        action="store_true",
        default=False,
        help=(
            "Compute the top-k OPD reward as the exact reverse KL over the selected "
            "token ids plus one tail bucket (k+1 buckets summing to 1), instead of "
            "the softmax-renormalized truncated estimate. Keeps the estimate "
            "sensitive to probability mass the student moves outside the top-k. "
            "Requires --opd-log-prob-top-k > 0 and --opd-reward-weight-mode "
            "student_p (the bucket weights are the raw student probabilities)."
        ),
    )
    parser.add_argument(
        "--opd-scoring-timeout-secs",
        type=float,
        default=None,
        help=(
            "Per-request timeout for OPD teacher/student scoring calls. Set this to "
            "give (typically larger, slower) teacher servers a different bound than "
            "generation requests."
        ),
    )
    parser.add_argument(
        "--opd-defer-full-vocab-scoring",
        action="store_true",
        default=False,
        help=(
            "Score full-vocab teacher hidden states only after the complete student rollout batch "
            "has finished. This matches the original train_opd.py ordering and prevents colocated "
            "teacher prefills from perturbing stochastic student-generation scheduling."
        ),
    )
    parser.add_argument(
        "--force-on-policy-ratio",
        action="store_true",
        default=False,
        help=(
            "Force the PPO update ratio to exactly one while preserving gradients. "
            "Independent actor/behaviour correction may still be applied with TIS."
        ),
    )
    parser.add_argument(
        "--opd-teacher-pool",
        type=str,
        default=None,
        help=(
            "Path to a teacher pool manifest (yaml/json): several named frozen teachers, "
            "kind url (external endpoint) or served (this job serves the HF checkpoint on "
            "extra GPUs, like --opd-serve-teacher). Resolves to the --opd-teacher-urls "
            "router: per-sample routing via sample.metadata[--opd-teacher-key], weighted "
            "ensembles per name, 'default' as fallback. Sampled-token scoring only."
        ),
    )
    parser.add_argument(
        "--opd-serve-teacher",
        action="store_true",
        default=False,
        help=(
            "Serve the frozen OPD teacher inside this job: --teacher-hf-checkpoint is "
            "launched as an extra sglang model entry (own router, update_weights=False, "
            "scoring-safe server flags baked in) and its endpoint is published as "
            "--opd-teacher-url automatically. Under --colocate the teacher time-shares "
            "the actor/rollout GPUs; otherwise it gets --opd-teacher-num-gpus extra GPUs "
            "after the rollout bucket. Mutually exclusive with --opd-teacher-url(s)."
        ),
    )
    parser.add_argument(
        "--opd-teacher-num-gpus",
        type=int,
        default=1,
        help="GPUs for the managed OPD teacher (--opd-serve-teacher); one engine with TP across them.",
    )
    parser.add_argument(
        "--opd-teacher-mem-fraction",
        type=float,
        default=None,
        help=(
            "mem_fraction_static override for the managed OPD teacher's engine; set a small "
            "value (e.g. 0.25) under --colocate so the teacher fits beside the student engine."
        ),
    )
    parser.add_argument(
        "--opd-teacher-max-running-requests",
        type=int,
        default=None,
        help="Managed OPD teacher-only max_running_requests override.",
    )
    parser.add_argument(
        "--opd-teacher-max-prefill-tokens",
        type=int,
        default=None,
        help="Managed OPD teacher-only max_prefill_tokens override.",
    )
    parser.add_argument(
        "--teacher-score-mode",
        type=str,
        choices=["sampled_token", "full_vocab"],
        default="sampled_token",
        help=(
            "How the external sglang OPD teacher is scored: 'sampled_token' (default) scores "
            "only the response tokens the student already sampled. 'full_vocab' requests the "
            "teacher's last-layer hidden state at every response position "
            "(return_hidden_states=True) for --loss-type opd_jsd_loss's exact divergence; the "
            "trainer reconstructs the full teacher distribution via --teacher-hf-checkpoint's "
            "LM head. The teacher server must run with --enable-return-hidden-states, "
            "--disable-radix-cache and --chunked-prefill-size -1."
        ),
    )
    parser.add_argument(
        "--teacher-hf-checkpoint",
        type=str,
        default=None,
        help=(
            "HF checkpoint directory of the frozen full-vocab OPD teacher; the trainer loads "
            "its LM head (model.embed_tokens.weight when tie_word_embeddings) to reconstruct "
            "full-vocab teacher logits from the hidden states. Must be the same checkpoint the "
            "teacher server at --opd-teacher-url serves."
        ),
    )
    parser.add_argument(
        "--opd-jsd-beta",
        type=float,
        default=0.5,
        help=(
            "Generalized-JSD interpolation for --loss-type opd_jsd_loss: 0 = forward "
            "KL(teacher||student), 1 = reverse KL(student||teacher), in between the "
            "GKD Eq.(1) mixture over M = (1-b)*student + b*teacher."
        ),
    )
    parser.add_argument(
        "--opd-log-prob-min-clamp",
        type=float,
        default=-30.0,
        help="Lower clamp on student/teacher log-probs inside opd_jsd_loss (bounds forward-KL summands).",
    )
    parser.add_argument(
        "--opd-loss-max-clamp",
        type=float,
        default=10.0,
        help="Upper clamp on the per-position vocab-summed divergence in opd_jsd_loss.",
    )
    parser.add_argument(
        "--opd-jsd-pointwise-clip",
        type=float,
        default=None,
        help=(
            "Cap each (position, vocab-token) divergence summand before the vocab sum "
            "(OPSD's --jsd_token_clip); unset disables."
        ),
    )
    parser.add_argument(
        "--opd-log-topk-overlap",
        action="store_true",
        default=False,
        help="Log student/teacher top-k overlap metrics from opd_jsd_loss.",
    )
    parser.add_argument(
        "--opd-topk-overlap-ks",
        type=int,
        nargs="+",
        default=[1, 5, 20],
        help="k values for --opd-log-topk-overlap.",
    )
    parser.add_argument(
        "--opd-teacher-key",
        type=str,
        default="opd_teacher",
        help=(
            "Sample metadata key holding the teacher name used for --opd-teacher-urls "
            "routing. Populated from the dataset's metadata column."
        ),
    )
    parser.add_argument(
        "--opd-icepop",
        action="store_true",
        default=False,
        help=(
            "Apply the ICE-POP async/off-policy correction to the OPD advantage: hard-gate (zero) tokens "
            "whose train/rollout importance ratio leaves [--tis-clip-low, --tis-clip]. Reuses the same gate "
            "as the policy-gradient path. Requires the student log-probs to be recomputed by the trainer, so "
            "it is incompatible with --use-rollout-logprobs."
        ),
    )
    parser.add_argument(
        "--opd-log-prob-top-k",
        type=int,
        default=0,
        help=(
            "Number of top-k tokens to use for the re-think OPD token-level reward. "
            "Set to 0 to use sampled-token OPD."
        ),
    )
    parser.add_argument(
        "--opd-top-k-strategy",
        type=str,
        choices=["only-student", "only-teacher", "intersection", "union", "xor"],
        default="only-student",
        help="Token set strategy for top-k OPD.",
    )
    parser.add_argument(
        "--opd-reward-weight-mode",
        type=str,
        choices=["student_p", "teacher_p", "none"],
        default="student_p",
        help="Weighting scheme for top-k OPD token rewards (applies to the reverse-KL term only).",
    )
    parser.add_argument(
        "--opd-kl-type",
        type=str,
        choices=["reverse", "forward", "mixed"],
        default="reverse",
        help=(
            "KL direction for the top-k OPD estimate (mirrors NeMo-RL's distillation "
            "kl_type): 'reverse' (default) weights by the student distribution, "
            "'forward' by the teacher distribution, 'mixed' is the convex combination "
            "with --opd-mixed-kl-weight on the forward term. Requires "
            "--opd-log-prob-top-k > 0 (the sampled-token path is reverse-only)."
        ),
    )
    parser.add_argument(
        "--opd-mixed-kl-weight",
        type=float,
        default=0.5,
        help=(
            "Weight on the forward-KL term for --opd-kl-type mixed, in [0, 1] "
            "(NeMo-RL's mixed_kl_weight; 0.5 matches their default recipe)."
        ),
    )
    parser.add_argument(
        "--opd-topk-zero-outside",
        action=argparse.BooleanOptionalAction,
        help=(
            "For --loss-type opd_topk_loss's reverse/mixed KL: add the out-of-support "
            "correction for student mass that falls outside the teacher's reported top-k "
            "(see opd_topk_loss_function). Unset resolves at validation time to on for "
            "--opd-kl-type reverse/mixed, off (inert; a warning is logged) for forward, "
            "where the top-k KL never leaves the teacher's own support."
        ),
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=None,
        help=(
            "Base URL of an OpenAI-compatible judge server (e.g. an sglang server: "
            "http://host:port) used by orbit.rewards.llm_judge.reward_func. "
            "Required when --custom-rm-path points at the LLM-judge hook."
        ),
    )
    parser.add_argument(
        "--judge-mode",
        type=str,
        choices=["equivalence", "score"],
        default="equivalence",
        help=(
            "LLM-judge grading mode: 'equivalence' compares the response's final "
            "answer to sample.label (reward 1/0); 'score' is a pointwise 0-10 "
            "quality grade normalized to [0, 1]."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="default",
        help="Model name passed to the judge's chat-completions endpoint.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=1024,
        help="Max tokens for the judge's reply (reasoning + final verdict line).",
    )
    parser.add_argument(
        "--judge-timeout-secs",
        type=float,
        default=None,
        help="Per-request timeout for judge calls (one automatic retry on transient failures).",
    )
    parser.add_argument(
        "--code-rm-timeout-secs",
        type=float,
        default=6.0,
        help="Sandbox code-execution reward: wall-clock timeout per unit test.",
    )
    parser.add_argument(
        "--code-rm-memory-mb",
        type=int,
        default=512,
        help="Sandbox code-execution reward: address-space limit per test process.",
    )
    parser.add_argument(
        "--code-rm-max-tests",
        type=int,
        default=0,
        help="Sandbox code-execution reward: cap on unit tests executed per sample (0 = all).",
    )
    parser.add_argument(
        "--swe-rm-sif-cache",
        type=str,
        default=None,
        help=(
            "SWE patch reward: directory of pre-pulled Apptainer SIFs keyed by sanitized "
            "image name (build with tools/prepare_swe_subset.py)."
        ),
    )
    parser.add_argument(
        "--swe-rm-timeout-secs",
        type=float,
        default=300.0,
        help="SWE patch reward: wall-clock timeout per verification (copy + patch + tests).",
    )
    parser.add_argument(
        "--swe-agent-max-turns",
        type=int,
        default=12,
        help="Agentic SWE episodes: maximum model turns per episode.",
    )
    parser.add_argument(
        "--swe-agent-cmd-timeout-secs",
        type=float,
        default=30.0,
        help="Agentic SWE episodes: wall-clock timeout per shell command in the container session.",
    )
    parser.add_argument(
        "--lean-server-url",
        type=str,
        default=None,
        help="Base URL of a kimina-lean-server for math_formal_lean verification.",
    )
    parser.add_argument(
        "--lean-timeout-secs",
        type=float,
        default=180.0,
        help="Per-proof Lean verification timeout.",
    )
    parser.add_argument(
        "--reward-router-unmapped",
        type=str,
        choices=["zero", "error"],
        default="zero",
        help=(
            "Blend reward router: what to do with rows whose agent has no orbit grader — "
            "'zero' rewards them 0.0 with a warning (train on the covered subset), 'error' aborts."
        ),
    )
    return parser


def _validate_judge_args(args) -> None:
    """Validate LLM-judge reward args when the judge hook is wired."""
    custom_rm = getattr(args, "custom_rm_path", None) or ""
    if not custom_rm.endswith("llm_judge.reward_func"):
        return
    if not getattr(args, "judge_base_url", None):
        raise ValueError(
            "--custom-rm-path orbit.rewards.llm_judge.reward_func requires --judge-base-url "
            "<http://judge-host:port> (an OpenAI-compatible chat-completions server)."
        )
    if getattr(args, "judge_mode", "equivalence") not in ("equivalence", "score"):
        raise ValueError(f"Unknown --judge-mode: {args.judge_mode!r}.")


def _validate_reward_router_args(args) -> None:
    """Validate blend reward-router args when the router hook is wired."""
    custom_rm = getattr(args, "custom_rm_path", None) or ""
    if not custom_rm.endswith("reward_router.reward_func"):
        return
    if not getattr(args, "group_rm", False):
        raise ValueError(
            "--custom-rm-path orbit.rewards.reward_router.reward_func is a batch-mode hook: "
            "it must be combined with --group-rm."
        )
    if not getattr(args, "judge_base_url", None):
        logger.warning(
            "reward_router is wired without --judge-base-url: judge/genrm-routed rows will "
            "fail soft to reward 0.0. Fine for pure-code blends, wrong otherwise."
        )


def _validate_genrm_args(args) -> None:
    """Validate group-wise GenRM args when the genrm hook is wired."""
    custom_rm = getattr(args, "custom_rm_path", None) or ""
    if not custom_rm.endswith("genrm_judge.reward_func"):
        return
    if not getattr(args, "group_rm", False):
        raise ValueError(
            "--custom-rm-path orbit.rewards.genrm_judge.reward_func is a batch-mode hook: "
            "it must be combined with --group-rm (otherwise it would receive single samples)."
        )
    if not getattr(args, "judge_base_url", None):
        raise ValueError(
            "--custom-rm-path orbit.rewards.genrm_judge.reward_func requires --judge-base-url "
            "<http://judge-host:port> (an OpenAI-compatible chat-completions server)."
        )


def validate_opd_topk_loss_args(args) -> None:
    """Validate --loss-type opd_topk_loss's structural requirements ("raw-mass v1",
    spec Phase D). No-op unless opd_topk_loss is selected.

    --opd-log-prob-top-k > 0 and --opd-type sglang are already enforced by the
    top-k block above regardless of loss type; this adds opd_topk_loss-specific
    requirements: --opd-top-k-strategy only-teacher (the raw-mass semantics
    truncate to the teacher's own reported support), no teacher ensembles, the
    external single-URL teacher transport (not the managed/same-engine path --
    see below), an untempered rollout, CP == 1, --opd-topk-tail-bucket off, and
    the OPD custom-reward hooks (--custom-rm-path/--custom-reward-post-process-path),
    since opd_topk_loss bypasses needs_opd_teacher()'s own hook check the same way
    --teacher-score-mode full_vocab does. It also resolves --opd-topk-zero-outside's
    default and couples compute_advantages_and_returns=False, exactly like
    opd_jsd_loss's --teacher-score-mode full_vocab block above.
    """
    if getattr(args, "loss_type", None) != "opd_topk_loss":
        return

    top_k = getattr(args, "opd_log_prob_top_k", 0) or 0
    if top_k <= 0:
        raise ValueError("--loss-type opd_topk_loss requires --opd-log-prob-top-k > 0.")
    validate_opd_topk_vocab_size(args)
    validate_opd_topk_reference_kl_args(args)

    strategy = getattr(args, "opd_top_k_strategy", "only-student")
    if strategy != "only-teacher":
        raise ValueError(
            "--loss-type opd_topk_loss requires --opd-top-k-strategy only-teacher: the raw-mass "
            f"semantics truncate to the teacher's own reported top-k support, got {strategy!r}."
        )

    if getattr(args, "opd_teacher_urls", None):
        # Local import to keep miles.utils free of rollout imports at module load
        # (matches the existing --opd-teacher-urls parse above).
        from orbit.opd.opd_sglang import parse_teacher_urls

        url_map = parse_teacher_urls(args.opd_teacher_urls)
        if any(len(targets) > 1 for targets in url_map.values()):
            raise ValueError(
                "--loss-type opd_topk_loss does not support teacher ensembles (--opd-teacher-urls "
                "groups with more than one URL): the retained transport (teacher_topk_ids/"
                "teacher_topk_logprobs) is single-teacher only in v1."
            )

    # Mirrors --teacher-score-mode full_vocab's own presence check above: without this,
    # a config with no teacher at all (no --opd-teacher-url(s), no --opd-serve-teacher,
    # and a not-same-base or unset --opd-teacher) sails through local_scoring_enabled
    # below (False, since is_same_base is False too) and the hooks check further down
    # (which only checks hook *names*, not that a teacher exists), only surfacing as a
    # KeyError deep into a rollout once training reads the missing transport keys.
    if not (
        getattr(args, "opd_teacher_url", None)
        or getattr(args, "opd_teacher_urls", None)
        or getattr(args, "opd_serve_teacher", False)
    ):
        raise ValueError(
            "--loss-type opd_topk_loss requires an external teacher: --opd-teacher-url, "
            "--opd-teacher-urls, or --opd-serve-teacher (managed in-job serving that publishes "
            "its endpoint as --opd-teacher-url once its engines are up)."
        )

    # The managed/same-engine teacher path (a same-base --opd-teacher with no external
    # teacher URL, orbit.opd.opd_scoring.opd_score_sample via local_scoring_enabled)
    # scores through _score_top_k too, but only sets sample.opd_reverse_kl -- it never
    # calls opd_sglang._extract_teacher_topk, so teacher_topk_ids/teacher_topk_logprobs
    # would stay None. Only the external-URL path (opd_sglang.post_process's top-k
    # branch, Task 1) retains them.
    from orbit.opd.opd_scoring import local_scoring_enabled

    if local_scoring_enabled(args):
        raise ValueError(
            "--loss-type opd_topk_loss requires an external teacher (--opd-teacher-url, "
            "--opd-teacher-urls, or --opd-serve-teacher, which resolves to --opd-teacher-url once "
            "its engines are up): the managed/same-engine teacher path (a same-base --opd-teacher "
            "with no external teacher URL) scores through orbit.opd.opd_scoring.opd_score_sample, "
            "which does not retain teacher_topk_ids/teacher_topk_logprobs -- that transport lives "
            "only on the external-URL scoring path (orbit.opd.opd_sglang.post_process) in v1."
        )

    temperature = float(getattr(args, "rollout_temperature", 1.0))
    if temperature != 1.0:
        raise ValueError(
            "--loss-type opd_topk_loss requires --rollout-temperature == 1.0: top-k log-probs "
            f"cannot be re-tempered client-side, got {temperature}."
        )

    cp_size = getattr(args, "context_parallel_size", 1) or 1
    if cp_size != 1:
        raise ValueError(
            "--loss-type opd_topk_loss requires --context-parallel-size == 1: the retained top-k "
            f"transport is not CP-slice-aware in v1, got {cp_size}."
        )

    if getattr(args, "allgather_cp", False):
        raise ValueError(
            "--loss-type opd_topk_loss is incompatible with --allgather-cp: the CP redistribution "
            "helper only handles 1D per-token tensors, not the [R, K] student_topk_log_probs "
            "tensor (get_log_probs_and_entropy raises the same NotImplementedError at compute time)."
        )

    if getattr(args, "opd_topk_tail_bucket", False):
        raise ValueError(
            "--loss-type opd_topk_loss is incompatible with --opd-topk-tail-bucket: tail-bucket is "
            "a PG-arm reward feature whose own startup validation requires --opd-top-k-strategy "
            "only-student or intersection, structurally incompatible with opd_topk_loss's required "
            "only-teacher strategy."
        )

    # opd_topk_loss bypasses needs_opd_teacher() in the common case (default
    # advantage_estimator=grpo, use_opd=False), exactly like full_vocab above, so the
    # legacy hook check further down never runs for it either -- enforce the scoring
    # transport here (mirrors the full_vocab block's own enforcement immediately above).
    # Without this, a missing/wrong hook silently falls through to the default reward
    # path: no teacher_topk_ids/logprobs ever get populated, and the run only dies after
    # a full rollout on a bare KeyError once training reads the missing transport keys.
    expected_rm = "orbit.opd.opd_sglang.reward_func"
    expected_post = "orbit.opd.opd_sglang.post_process"
    if (
        getattr(args, "custom_rm_path", None) != expected_rm
        or getattr(args, "custom_reward_post_process_path", None) != expected_post
    ):
        raise ValueError(
            "--loss-type opd_topk_loss scores samples through the OPD custom-reward hooks; set "
            f"--custom-rm-path {expected_rm} and --custom-reward-post-process-path {expected_post}."
        )

    # Resolve --opd-topk-zero-outside's default here (not at parse time): on for
    # reverse/mixed, off (with a warning) for forward, where the top-k KL never
    # leaves the teacher's own reported support so the correction is a no-op.
    # opd_topk_loss_function's own getattr(..., None) fallback mirrors this
    # resolution as defense only -- this is the source of truth.
    kl_type = getattr(args, "opd_kl_type", "reverse") or "reverse"
    if getattr(args, "opd_topk_zero_outside", None) is None:
        if kl_type == "forward":
            args.opd_topk_zero_outside = False
            logger.warning(
                "--opd-topk-zero-outside defaults to False with --opd-kl-type forward: the forward "
                "top-k KL only ever sums over the teacher's own reported support, so the "
                "out-of-support correction would have no effect."
            )
        else:
            args.opd_topk_zero_outside = True

    # Pure distillation: no PPO advantage/returns pipeline, exactly like opd_jsd_loss's
    # --teacher-score-mode full_vocab block above.
    args.compute_advantages_and_returns = False


def _validate_opd_args(args) -> None:
    """Validate on-policy distillation args. Mirrors slime arguments.py:1761-1791."""
    from orbit.opd.opd_teacher_spec import is_same_base, is_self_teacher, parse_teacher_spec

    opd_top_k = getattr(args, "opd_log_prob_top_k", 0) or 0
    if opd_top_k < 0:
        raise ValueError("--opd-log-prob-top-k must be non-negative.")
    if opd_top_k > 0 and getattr(args, "opd_type", None) != "sglang":
        raise ValueError("--opd-log-prob-top-k is currently supported only with --opd-type=sglang.")
    opd_kl_type = getattr(args, "opd_kl_type", "reverse") or "reverse"
    if opd_kl_type != "reverse" and opd_top_k <= 0:
        raise ValueError(
            f"--opd-kl-type {opd_kl_type!r} requires --opd-log-prob-top-k > 0: the sampled-token "
            "path stores teacher_log_probs and computes reverse KL in the trainer; forward/mixed "
            "need per-position top-k distributions from rollout-side scoring."
        )
    opd_mixed_kl_weight = getattr(args, "opd_mixed_kl_weight", 0.5)
    if not (0.0 <= opd_mixed_kl_weight <= 1.0):
        raise ValueError(f"--opd-mixed-kl-weight must be in [0, 1], got {opd_mixed_kl_weight}.")
    if getattr(args, "opd_teacher_urls", None):
        if getattr(args, "opd_type", None) != "sglang":
            raise ValueError("--opd-teacher-urls is only supported with --opd-type=sglang.")
        # Local import to keep miles.utils free of rollout imports at module load.
        from orbit.opd.opd_sglang import parse_teacher_urls

        url_map = parse_teacher_urls(args.opd_teacher_urls)  # fail fast on malformed/duplicate entries
        has_ensemble_group = any(len(targets) > 1 for targets in url_map.values())
        if (
            has_ensemble_group
            and opd_top_k > 0
            and getattr(args, "opd_top_k_strategy", "only-student") != "only-student"
        ):
            raise ValueError(
                "Teacher ensembles (--opd-teacher-urls groups with multiple URLs) require "
                "--opd-top-k-strategy only-student: every group member must be scored at the "
                f"same student top-k token ids, got {args.opd_top_k_strategy!r}."
            )
    if getattr(args, "opd_topk_tail_bucket", False):
        if opd_top_k <= 0:
            raise ValueError("--opd-topk-tail-bucket requires --opd-log-prob-top-k > 0.")
        if getattr(args, "opd_reward_weight_mode", "student_p") != "student_p":
            raise ValueError(
                "--opd-topk-tail-bucket uses raw student probabilities as bucket weights and is "
                f"incompatible with --opd-reward-weight-mode {args.opd_reward_weight_mode!r}; use student_p."
            )
        if getattr(args, "opd_top_k_strategy", "only-student") not in ("only-student", "intersection"):
            raise ValueError(
                "--opd-topk-tail-bucket requires --opd-top-k-strategy only-student or intersection "
                f"(single-softmax student logprobs), got {args.opd_top_k_strategy!r}."
            )

    # Pure MOPD (advantage estimator) and blend (--use-opd) are mutually exclusive:
    # blend is meant to sit on top of a reward-based estimator, not on pure distillation.
    if args.advantage_estimator == "on_policy_distillation" and getattr(args, "use_opd", False):
        raise ValueError(
            "--advantage-estimator on_policy_distillation (pure MOPD) and --use-opd (blend) are "
            "mutually exclusive. Pure MOPD is reward-free distillation; --use-opd blends a distillation "
            "KL onto a reward-based estimator. Pick one."
        )

    # Teacher pools: several named frozen teachers resolved onto the existing
    # multi-teacher router; served members are launched like --opd-serve-teacher.
    if getattr(args, "opd_teacher_pool", None) is not None:
        if getattr(args, "opd_type", None) != "sglang":
            raise ValueError("--opd-teacher-pool requires --opd-type sglang.")
        if (
            getattr(args, "opd_serve_teacher", False)
            or getattr(args, "opd_teacher_url", None)
            or getattr(args, "opd_teacher_urls", None)
        ):
            raise ValueError(
                "--opd-teacher-pool subsumes --opd-serve-teacher/--opd-teacher-url(s); "
                "declare every teacher in the manifest instead."
            )
        if getattr(args, "teacher_score_mode", "sampled_token") == "full_vocab":
            raise ValueError(
                "--opd-teacher-pool is sampled-token only: full-vocab reconstruction needs one "
                "trainer-side LM head per member; use --opd-serve-teacher/--opd-teacher-url for "
                "a single full-vocab teacher."
            )
        from orbit.opd.opd_teacher_pool import parse_teacher_pool

        parse_teacher_pool(args.opd_teacher_pool)  # fail fast on a malformed manifest

    # Managed teacher serving: the job launches the frozen teacher itself and publishes
    # its endpoint as opd_teacher_url once the engines are up (start_rollout_servers).
    if getattr(args, "opd_serve_teacher", False):
        if getattr(args, "opd_type", None) != "sglang":
            raise ValueError("--opd-serve-teacher requires --opd-type sglang.")
        if getattr(args, "opd_teacher_url", None) or getattr(args, "opd_teacher_urls", None):
            raise ValueError(
                "--opd-serve-teacher and --opd-teacher-url(s) are mutually exclusive: the managed "
                "teacher publishes its own endpoint after its engines start."
            )
        if not getattr(args, "teacher_hf_checkpoint", None):
            raise ValueError(
                "--opd-serve-teacher serves --teacher-hf-checkpoint; set it to the frozen teacher's "
                "HF checkpoint directory."
            )
        if args.opd_teacher_num_gpus < 1:
            raise ValueError("--opd-teacher-num-gpus must be >= 1.")
        if (
            getattr(args, "opd_teacher_max_running_requests", None) is not None
            and args.opd_teacher_max_running_requests < 1
        ):
            raise ValueError("--opd-teacher-max-running-requests must be >= 1.")
        if (
            getattr(args, "opd_teacher_max_prefill_tokens", None) is not None
            and args.opd_teacher_max_prefill_tokens < 1
        ):
            raise ValueError("--opd-teacher-max-prefill-tokens must be >= 1.")

    # Full-vocab OPD: --loss-type opd_jsd_loss and --teacher-score-mode full_vocab come as a
    # pair, on the external single-URL sglang teacher transport.
    score_mode = getattr(args, "teacher_score_mode", "sampled_token") or "sampled_token"
    if getattr(args, "opd_defer_full_vocab_scoring", False) and score_mode != "full_vocab":
        raise ValueError("--opd-defer-full-vocab-scoring requires --teacher-score-mode full_vocab.")
    if (getattr(args, "loss_type", None) == "opd_jsd_loss") != (score_mode == "full_vocab"):
        raise ValueError(
            "--loss-type opd_jsd_loss and --teacher-score-mode full_vocab must be used together, "
            f"got loss_type={getattr(args, 'loss_type', None)!r} with teacher_score_mode={score_mode!r}."
        )
    if score_mode == "full_vocab":
        if getattr(args, "opd_type", None) != "sglang":
            raise ValueError("--teacher-score-mode full_vocab requires --opd-type sglang.")
        if not getattr(args, "opd_teacher_url", None) and not getattr(args, "opd_serve_teacher", False):
            raise ValueError(
                "--teacher-score-mode full_vocab requires --opd-teacher-url (a single external "
                "teacher) or --opd-serve-teacher (managed in-job serving)."
            )
        if getattr(args, "opd_teacher_urls", None):
            raise ValueError(
                "--teacher-score-mode full_vocab does not support --opd-teacher-urls routing/ensembles: "
                "mixing reconstructed distributions needs per-member LM heads trainer-side."
            )
        if opd_top_k > 0:
            raise ValueError("--teacher-score-mode full_vocab is incompatible with --opd-log-prob-top-k > 0.")
        if not getattr(args, "teacher_hf_checkpoint", None):
            raise ValueError(
                "--teacher-score-mode full_vocab requires --teacher-hf-checkpoint to reconstruct "
                "the teacher distribution trainer-side."
            )
        if getattr(args, "use_opd", False) or args.advantage_estimator == "on_policy_distillation":
            raise ValueError(
                "--loss-type opd_jsd_loss is a pure distillation loss: it replaces the OPD "
                "advantage machinery, so --use-opd / --advantage-estimator on_policy_distillation "
                "must be off."
            )
        # full_vocab bypasses needs_opd_teacher() (no OPD advantage), so the legacy hook
        # check below never runs for it -- enforce the scoring transport here.
        expected_rm = "orbit.opd.opd_sglang.reward_func"
        expected_post = "orbit.opd.opd_sglang.post_process"
        if (
            getattr(args, "custom_rm_path", None) != expected_rm
            or getattr(args, "custom_reward_post_process_path", None) != expected_post
        ):
            raise ValueError(
                "--teacher-score-mode full_vocab scores samples through the OPD custom-reward "
                f"hooks; set --custom-rm-path {expected_rm} and "
                f"--custom-reward-post-process-path {expected_post}."
            )
        # Pure distillation: no PPO advantage/returns pipeline.
        args.compute_advantages_and_returns = False

    # Direct top-k OPD loss: sibling of the full_vocab block above (own transport,
    # own coupling), on the existing top-k API rather than full-vocab reconstruction.
    validate_opd_topk_loss_args(args)

    # Forced on-policy ratio (Stage-3 MOPD kernel): the PPO ratio is pinned to exactly 1
    # (REINFORCE semantics), so every knob that would reintroduce a behaviour/actor
    # mismatch is checked with exact types -- silent coercion here changes the objective.
    force_on_policy_ratio = getattr(args, "force_on_policy_ratio", False)
    if type(force_on_policy_ratio) is not bool:
        raise ValueError("--force-on-policy-ratio must be an exact boolean.")
    use_tis = getattr(args, "use_tis", False)
    if type(use_tis) is not bool:
        raise ValueError("--use-tis must be an exact boolean.")
    if use_tis:
        tis_clip_low = getattr(args, "tis_clip_low", 0.0)
        tis_clip = getattr(args, "tis_clip", 2.0)
        if type(tis_clip_low) is not float or type(tis_clip) is not float:
            raise ValueError("--tis-clip-low and --tis-clip must be exact float values.")
        if not math.isfinite(tis_clip_low) or not math.isfinite(tis_clip):
            raise ValueError("--tis-clip-low and --tis-clip must be finite float values.")
        if not 0.0 <= tis_clip_low < tis_clip:
            raise ValueError("TIS clipping bounds must satisfy 0 <= --tis-clip-low < --tis-clip.")
    if force_on_policy_ratio:
        if getattr(args, "use_opd", False):
            raise ValueError("--force-on-policy-ratio forbids --use-opd blend mode.")
        if args.advantage_estimator != "on_policy_distillation":
            raise ValueError("--force-on-policy-ratio requires --advantage-estimator on_policy_distillation.")
        if getattr(args, "use_rollout_logprobs", False):
            raise ValueError("--force-on-policy-ratio forbids --use-rollout-logprobs.")
        steps_per_rollout = getattr(args, "num_steps_per_rollout", None)
        # Dev semantics: None means one optimizer pass over the rollout (the
        # ultra default was a literal 1); anything beyond one step reuses data
        # off-policy and contradicts the forced ratio.
        if steps_per_rollout is not None and (type(steps_per_rollout) is not int or steps_per_rollout != 1):
            raise ValueError(
                "--force-on-policy-ratio requires exactly one training step per "
                "rollout (--num-steps-per-rollout 1 or unset)."
            )

    # sglang-teacher OPD blend is only safe when the teacher scores through the
    # local rollout-engine adapter slot (same-base): the external-URL sampled-token
    # teacher's reward_func returns 0.0 and occupies the single --custom-rm-path
    # slot, so blending it with a reward-based estimator would degrade to a
    # KL-only signal with ~0 base advantage. (The full-vocab direct-loss path is
    # not eligible for --use-opd and may retain task rewards for metrics.)
    if getattr(args, "use_opd", False) and args.opd_type == "sglang":
        spec_for_blend = parse_teacher_spec(getattr(args, "opd_teacher", None), args.opd_teacher_load)
        external = (
            getattr(args, "opd_teacher_url", None)
            or getattr(args, "opd_teacher_urls", None)
            or getattr(args, "opd_serve_teacher", False)
            or getattr(args, "opd_teacher_pool", None)
        )
        if external or not is_same_base(spec_for_blend):
            raise ValueError(
                "--use-opd (blend) with --opd-type sglang requires a same-base teacher scored by "
                "the local engine (--opd-teacher base/adapter:<path>/self:*): the external-URL "
                "teacher's sampled-token reward_func occupies the single --custom-rm-path slot and "
                "returns 0.0, so blend would degrade to a KL-only signal. Use --opd-type megatron "
                "or a same-base local teacher for the blend."
            )

    # --opd-icepop gates the OPD advantage by the train/rollout importance ratio,
    # so it only applies when OPD is on and requires the trainer-recomputed student
    # log-probs (mirrors how the PG icepop/TIS path requires --use-rollout-logprobs off).
    if getattr(args, "opd_icepop", False):
        if not needs_opd_teacher(args):
            raise ValueError(
                "--opd-icepop only applies to on-policy distillation; enable it via "
                "--advantage-estimator on_policy_distillation (pure MOPD) or --use-opd (blend)."
            )
        if getattr(args, "use_rollout_logprobs", False):
            raise ValueError(
                "--opd-icepop is incompatible with --use-rollout-logprobs: the ICE-POP ratio needs "
                "the trainer-recomputed student log-probs vs the rollout log-probs, but "
                "--use-rollout-logprobs makes them identical (ratio == 1, no correction). "
                "Drop --use-rollout-logprobs."
            )

    if not needs_opd_teacher(args):
        return

    spec = parse_teacher_spec(getattr(args, "opd_teacher", None), args.opd_teacher_load)
    args.opd_teacher_spec = spec

    ema_decay = getattr(args, "opd_ema_decay", 0.999)
    if not (0.0 < ema_decay < 1.0):
        raise ValueError(f"--opd-ema-decay must be in (0, 1), got {ema_decay}.")
    if getattr(args, "opd_self_teacher_interval", 1) < 1:
        raise ValueError("--opd-self-teacher-interval must be >= 1.")
    promote = getattr(args, "opd_promote_interval", None)
    if promote is not None and promote < 1:
        raise ValueError("--opd-promote-interval must be >= 1.")
    if is_self_teacher(spec) and args.opd_type == "sglang" and promote is None:
        raise ValueError(
            "--opd-teacher self:* with --opd-type sglang requires --opd-promote-interval N: "
            "without promotion the engine's orbit_teacher slot would stay frozen at init."
        )

    if args.opd_type is None:
        raise ValueError(
            "On-policy distillation is enabled (advantage_estimator=on_policy_distillation or --use-opd), "
            "so --opd-type {megatron,sglang} is required to select the teacher producer."
        )

    if args.opd_type == "megatron":
        if spec is None:
            raise ValueError(
                "--opd-type megatron requires a teacher: --opd-teacher "
                "{base,adapter:<path>,self:ema,self:lag,load:<ckpt>} (or legacy --opd-teacher-load)."
            )
        if spec.source == "load":
            if _is_peft_enabled(args):
                raise ValueError(
                    "--opd-teacher load:<ckpt> loads a full in-process teacher model (like the ref "
                    "model), which is incompatible with PEFT (--peft-method != none). For PEFT runs "
                    "use a same-base teacher (--opd-teacher base/adapter:<path>/self:ema/self:lag) "
                    "or --opd-type sglang."
                )
            if not os.path.exists(spec.path):
                raise FileNotFoundError(f"--opd-teacher load: {spec.path} does not exist, please check the path.")
            if not os.path.exists(os.path.join(spec.path, "latest_checkpointed_iteration.txt")):
                logger.info(
                    f"--opd-teacher load: {spec.path} does not have latest_checkpointed_iteration.txt, "
                    "please make sure it is a valid megatron checkpoint directory."
                )
        else:
            if not _is_peft_enabled(args):
                raise ValueError(
                    f"--opd-teacher {args.opd_teacher!r} shares the student's base weights and needs "
                    "an adapter structure to swap the teacher onto (--peft-method != none); with full "
                    "fine-tuning use --opd-teacher load:<ckpt>."
                )
            if spec.source == "adapter":
                if not os.path.isdir(spec.path):
                    raise FileNotFoundError(f"--opd-teacher adapter: {spec.path} does not exist.")
                _validate_teacher_adapter_config(spec.path, args.peft_method)
    elif args.opd_type == "sglang":
        if spec is not None and spec.source == "load":
            raise ValueError(
                "--opd-type sglang scores via the rollout engine or an external SGLang server; "
                "--opd-teacher load:<ckpt> (in-process second model) requires --opd-type megatron."
            )
        external = (
            args.opd_teacher_url
            or getattr(args, "opd_teacher_urls", None)
            or getattr(args, "opd_serve_teacher", False)
            or getattr(args, "opd_teacher_pool", None)
        )
        if external:
            # Legacy external-teacher path: unchanged hook requirements.
            expected_rm = "orbit.opd.opd_sglang.reward_func"
            expected_post = "orbit.opd.opd_sglang.post_process"
            if (
                getattr(args, "custom_rm_path", None) != expected_rm
                or getattr(args, "custom_reward_post_process_path", None) != expected_post
            ):
                raise ValueError(
                    "--opd-type sglang with an external teacher URL scores samples through its "
                    f"custom-reward hooks; set --custom-rm-path {expected_rm} and "
                    f"--custom-reward-post-process-path {expected_post}."
                )
        else:
            # Local mode: the rollout engine scores its own teacher slot.
            if not is_same_base(spec):
                raise ValueError(
                    "--opd-type sglang without --opd-teacher-url needs a same-base local teacher: "
                    "--opd-teacher {base,adapter:<path>,self:ema,self:lag} (the rollout engine "
                    "scores it via the orbit_teacher adapter slot), or provide an external "
                    "--opd-teacher-url."
                )
            if not _is_peft_enabled(args):
                raise ValueError(
                    "--opd-type sglang local-teacher mode needs PEFT enabled (--peft-method != "
                    "none): the teacher is an adapter over the student's base."
                )
            if getattr(args, "peft_method", "none") == "lora":
                raise ValueError(
                    "--opd-type sglang local-teacher mode does not support --peft-method lora: "
                    "SGLang's unified PEFT LoRA path is single-active and applies one adapter "
                    "to the whole batch, so it cannot independently select the student and "
                    "--opd-teacher {base,adapter:<path>,self:ema,self:lag}. Use --peft-method "
                    "oft, --opd-type megatron, or an external SGLang teacher "
                    "(--opd-teacher-url)."
                )
            if (
                getattr(args, "peft_method", "none") == "oft"
                and is_self_teacher(spec)
                and getattr(args, "adapter_double_buffer", False)
            ):
                raise ValueError(
                    "--opd-teacher self:* with local --opd-type sglang is incompatible with "
                    "--peft-method oft and --adapter-double-buffer: NCCL double-buffering has "
                    "one fixed active OFT slot, so promoting orbit_teacher would overwrite the "
                    "student adapter instead of creating an independently routable teacher. "
                    "Use --opd-teacher base or adapter:<path>, a non-double-buffer Ray/IPC "
                    "transport, or an external teacher."
                )
            if spec.source == "adapter":
                if not os.path.isdir(spec.path):
                    raise FileNotFoundError(f"--opd-teacher adapter: {spec.path} does not exist.")
                _validate_teacher_adapter_config(spec.path, args.peft_method)
            if getattr(args, "custom_rm_path", None) == "orbit.opd.opd_sglang.reward_func":
                raise ValueError(
                    "Local-teacher mode scores through the built-in rollout stage; do not set "
                    "--custom-rm-path orbit.opd.opd_sglang.reward_func (it would double-score). "
                    "Leave --custom-rm-path free or point it at a real reward model."
                )


def _is_default_rollout_function_path(path: str | None) -> bool:
    return path is None or path in DEFAULT_ROLLOUT_FUNCTION_PATHS


def _apply_training_mode_args(args) -> None:
    training_mode = getattr(args, "training_mode", "rl")
    if training_mode not in {"rl", "sft"}:
        raise ValueError(f"--training-mode must be one of ['rl', 'sft'], got {training_mode!r}.")

    if training_mode == "rl":
        args.use_rollout_engines = getattr(args, "use_rollout_engines", True)
        return

    if getattr(args, "debug_rollout_only", False):
        raise ValueError("--training-mode sft is incompatible with --debug-rollout-only.")

    if getattr(args, "advantage_estimator", "grpo") == "ppo":
        raise ValueError("--training-mode sft is incompatible with --advantage-estimator ppo.")
    if getattr(args, "kl_coef", 0) != 0 or getattr(args, "use_kl_loss", False):
        raise ValueError("--training-mode sft is incompatible with KL reward/loss settings.")
    if getattr(args, "use_rollout_logprobs", False):
        raise ValueError("--training-mode sft is incompatible with --use-rollout-logprobs.")
    if getattr(args, "dynamic_sampling_filter_path", None) is not None:
        raise ValueError("--training-mode sft is incompatible with --dynamic-sampling-filter-path.")

    if _is_default_rollout_function_path(getattr(args, "rollout_function_path", None)):
        args.rollout_function_path = SFT_ROLLOUT_FUNCTION_PATH

    args.loss_type = "sft_loss"
    args.compute_advantages_and_returns = False
    args.n_samples_per_prompt = 1
    args.advantage_estimator = "grpo"

    eval_enabled = getattr(args, "eval_interval", None) is not None
    if eval_enabled and getattr(args, "eval_function_path", None) is None:
        raise ValueError(
            "--training-mode sft with --eval-interval requires an explicit --eval-function-path "
            "for generation-based evaluation."
        )

    args.use_rollout_engines = eval_enabled
    if not args.use_rollout_engines:
        args.rollout_num_gpus = 0
        args.offload_rollout = False
        if hasattr(args, "check_weight_update_equal"):
            args.check_weight_update_equal = False


def _is_peft_enabled(args) -> bool:
    """Local PEFT-enabled predicate.

    Avoids importing orbit.megatron.peft_utils here (that
    module pulls in megatron-core, which would force CPU CI to install
    GPU-only deps just to validate args).
    """
    return getattr(args, "peft_method", "none") != "none"


def _validate_teacher_adapter_config(adapter_dir: str, peft_method: str) -> None:
    """CPU-safe mirror of peft_utils.validate_peft_checkpoint_type (that module
    imports megatron.bridge at import time, unavailable at arg-parse time)."""
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path) as f:
        actual_type = json.load(f).get("peft_type")
    if actual_type is not None and actual_type.upper() != peft_method.upper():
        raise ValueError(
            f"--opd-teacher adapter: checkpoint at {adapter_dir} has peft_type={actual_type}, "
            f"expected {peft_method.upper()} (the active --peft-method)."
        )


_MEGATRON_FULL_MODEL_OFFLOAD_ERROR = (
    "Megatron --offload-train currently requires --peft-method lora or oft; "
    "full-model train offload needs a dedicated implementation."
)


def _normalize_peft_args(args):
    peft_method = getattr(args, "peft_method", "none")
    assert peft_method in _PEFT_METHODS, "--peft-method must be one of none, oft, lora."

    if getattr(args, "adapter_double_buffer", False) and peft_method == "none":
        raise AssertionError("--adapter-double-buffer requires --peft-method lora or oft.")
    peft_distributed_transport = getattr(args, "peft_distributed_transport", "nccl")
    if peft_distributed_transport not in {"nccl", "ray"}:
        raise AssertionError("--peft-distributed-transport must be one of nccl or ray.")
    if getattr(args, "adapter_double_buffer", False) and peft_distributed_transport != "nccl":
        raise AssertionError("--adapter-double-buffer requires --peft-distributed-transport nccl.")

    target_modules = getattr(args, "target_modules", None)
    exclude_modules = getattr(args, "exclude_modules", None)

    if peft_method != "none":
        assert target_modules is not None, "'--target-modules' is required when PEFT is enabled."

        peft_variant = getattr(args, "peft_variant", "standard")
        if target_modules in ("all-linear", "all"):
            if peft_variant == "dsv4":
                modules = ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"]
            elif peft_variant == "mla":
                modules = [
                    "q_a_proj",
                    "q_b_proj",
                    "kv_a_proj_with_mqa",
                    "kv_b_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            else:
                modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            if target_modules == "all":
                # "all" extends "all-linear" with the input embedding and the
                # language-modeling head. MLA / DSV4 are out of scope until those
                # families adopt --target-modules all.
                if peft_variant not in ("standard", "canonical"):
                    raise AssertionError(
                        "--target-modules all currently supports peft_variant in "
                        "{standard, canonical}; got "
                        f"{peft_variant!r}. Use 'all-linear' or an explicit list."
                    )
                modules = modules + ["embed_tokens", "lm_head"]
        elif isinstance(target_modules, str) and "," in target_modules:
            modules = [m.strip() for m in target_modules.split(",")]
        elif isinstance(target_modules, str):
            modules = [target_modules]
        else:
            modules = list(target_modules)

        if exclude_modules:
            exclude_set = (
                set(m.strip() for m in exclude_modules.split(",")) if "," in exclude_modules else {exclude_modules}
            )
            modules = [m for m in modules if m not in exclude_set]

        args.target_modules = modules

    peft_adapter_path = getattr(args, "peft_adapter_path", None)
    lora_adapter_path = getattr(args, "lora_adapter_path", None)
    oft_adapter_path = getattr(args, "oft_adapter_path", None)
    if peft_method == "lora":
        if peft_adapter_path is not None:
            if lora_adapter_path is not None and lora_adapter_path != peft_adapter_path:
                raise AssertionError("--peft-adapter-path and --lora-adapter-path must match when both are set.")
            args.lora_adapter_path = peft_adapter_path
            lora_adapter_path = peft_adapter_path
        if getattr(args, "lora_rank", 0) <= 0 and lora_adapter_path is None:
            raise AssertionError(
                "--peft-method lora requires --lora-rank > 0 or --lora-adapter-path/--peft-adapter-path."
            )
    elif peft_method == "oft":
        if peft_adapter_path is not None:
            if oft_adapter_path is not None and oft_adapter_path != peft_adapter_path:
                raise AssertionError("--peft-adapter-path and --oft-adapter-path must match when both are set.")
            args.oft_adapter_path = peft_adapter_path
            oft_adapter_path = peft_adapter_path
        if getattr(args, "oft_block_size", 0) <= 0 and oft_adapter_path is None:
            raise AssertionError(
                "--peft-method oft requires --oft-block-size > 0 or --oft-adapter-path/--peft-adapter-path."
            )
    elif peft_method != "oft" and peft_adapter_path is not None:
        raise AssertionError("--peft-adapter-path is only supported when --peft-method is lora or oft.")

    if peft_method != "lora":
        for name, default in _PEFT_LORA_DEFAULTS.items():
            assert getattr(args, name, default) == default, "LoRA flags require --peft-method lora."

    if peft_method != "oft":
        for name, default in _PEFT_OFT_DEFAULTS.items():
            assert getattr(args, name, default) == default, "OFT flags require --peft-method oft."

    return args


def _normalize_and_validate_peft_args(args):
    _normalize_peft_args(args)

    if args.peft_method != "none":
        assert args.megatron_to_hf_mode == "bridge", "PEFT requires --megatron-to-hf-mode bridge."

    return args


def _validate_dsv4_cp_args(args):
    if (getattr(args, "context_parallel_size", 1) or 1) <= 1:
        return args
    if getattr(args, "peft_variant", "standard") != "dsv4":
        return args

    requirement = (
        "DeepSeek V4 CP currently requires qkv_format='thd', allgather_cp=False, "
        "and Orbit dsv4_cu_seqlens metadata for Megatron packed THD zigzag CP"
    )
    if getattr(args, "qkv_format", None) != "thd":
        raise ValueError(f"{requirement}; got qkv_format={getattr(args, 'qkv_format', None)!r}.")
    if getattr(args, "allgather_cp", False):
        raise ValueError(f"{requirement}; got allgather_cp=True.")
    if int(getattr(args, "dsv4_cp_chunk_size_multiple", 128) or 0) <= 0:
        raise ValueError("--dsv4-cp-chunk-size-multiple must be positive.")

    return args


def _finalize_train_offload_args(args) -> None:
    if getattr(args, "offload_train_frozen_base_mode", None) is None:
        args.offload_train_frozen_base_mode = "auto"
    if args.offload_train_frozen_base_mode not in {"auto", "flat", "tms"}:
        raise ValueError("--offload-train-frozen-base-mode must be one of: auto, flat, tms")
    if args.offload_train is None:
        args.offload_train = False

    # Full fine-tuning frees train memory by a different route than PEFT, and
    # the difference is not a preference -- it is the only route that works.
    #
    # PEFT's saving comes from `offload_megatron_frozen_base_to_cpu`, whose
    # selector skips any parameter with `requires_grad`. Under full fine-tuning
    # that is every parameter, so the frozen-base path plans empty groups, logs
    # "after offload model", and frees zero bytes. Enabling --offload-train
    # without these two flags would therefore be a silent no-op that reads as a
    # success -- which is why this used to raise outright rather than allow it.
    #
    # Gradients and optimizer state are what full fine-tuning actually has to
    # give back. Measured on 8xH100 with Llama-3.1-8B: the FullFT arm sat at
    # 66.69 GB used / 12.48 GB free against 16.00 GB of paused SGLang K+V, and
    # died in `torch_memory_saver ... func=resume`; the LoRA arms sat at 43.88 /
    # 35.30 and resumed. The ~22.8 GB between them is exactly this state.
    #
    # Parameters stay resident on purpose. `update_weights` pushes Megatron
    # weights into the rollout engine every rollout and does not wake the train
    # state, so a zero-sized `param_data` would surface as corrupt rollouts
    # rather than an error. Megatron's own `offload_grad_buffers` passes
    # `move_params=False` for the same reason.
    #
    # Scoped to megatron because these two flags drive megatron-specific
    # primitives; other backends were never refused and are left alone.
    if args.train_backend == "megatron" and args.offload_train and not _is_peft_enabled(args):
        for _flag in ("offload_train_grad_buffers", "offload_train_optimizer"):
            if getattr(args, _flag, None) is False:
                raise ValueError(
                    f"--{_flag.replace('_', '-')} cannot be disabled for full fine-tuning "
                    "with --offload-train: the frozen-base path has nothing to offload when "
                    "every parameter is trainable, so the offload would free nothing."
                )
            setattr(args, _flag, True)

    if args.offload_train_grad_buffers is None:
        args.offload_train_grad_buffers = False
    if args.offload_train_optimizer is None:
        args.offload_train_optimizer = False
    if args.offload_train_grad_buffers and not args.offload_train:
        raise ValueError("--offload-train-grad-buffers requires --offload-train")
    if args.offload_train_optimizer and not args.offload_train:
        raise ValueError("--offload-train-optimizer requires --offload-train")
    if args.offload_train_async is None:
        args.offload_train_async = False
    if args.offload_train_async and not args.offload_train:
        raise ValueError("--offload-train-async requires --offload-train")
    if args.offload_rollout is None:
        args.offload_rollout = False

    adapter_unavailable_reason = None
    if not args.offload_train:
        adapter_unavailable_reason = "--offload-train is disabled"
    elif args.train_backend != "megatron":
        adapter_unavailable_reason = "--train-backend is not megatron"
    elif args.peft_method not in {"lora", "oft"}:
        adapter_unavailable_reason = "--peft-method is not lora or oft"
    elif args.megatron_to_hf_mode != "bridge":
        adapter_unavailable_reason = "--megatron-to-hf-mode is not bridge"

    adapter_offload_request = args.offload_train_adapter
    if adapter_offload_request is None:
        args.offload_train_adapter = False
    elif adapter_offload_request and adapter_unavailable_reason is not None:
        logger.warning(
            "Disabling --offload-train-adapter because %s. Adapter offload is only supported "
            "for Megatron bridge-mode LoRA/OFT train offload.",
            adapter_unavailable_reason,
        )
        args.offload_train_adapter = False

    # Full-model train offload used to raise here. It is now supported, by
    # offloading gradients and optimizer state while parameters stay resident --
    # see the block above `offload_train_grad_buffers` for why that split is the
    # only one that works. `_MEGATRON_FULL_MODEL_OFFLOAD_ERROR` is kept as the
    # record of what the old failure said, since operators will find it in logs.


def _apply_critic_args(args) -> None:
    args.use_critic = args.advantage_estimator == "ppo"
    if args.critic_mode in ("adapter", "head"):
        mode = args.critic_mode
        if args.advantage_estimator != "ppo":
            raise ValueError(f"--critic-mode {mode} requires --advantage-estimator ppo.")
        if mode == "adapter" and args.peft_method == "none":
            raise ValueError("--critic-mode adapter requires an enabled --peft-method: the critic is an adapter.")
        if args.train_backend != "megatron":
            raise ValueError(f"--critic-mode {mode} requires the megatron train backend.")
        if args.keep_old_actor:
            raise ValueError(
                f"--critic-mode {mode} is incompatible with --keep-old-actor: the value "
                "forward would run under the old-actor trunk (shared via aliasing) while "
                "the value phase trains under the current trunk."
            )
        if getattr(args, "use_rollout_routing_replay", False):
            raise ValueError(
                f"--critic-mode {mode} is incompatible with --use-rollout-routing-replay: "
                "critic forwards would hit the actor's routing-replay buffers."
            )
        for flag in ("critic_num_gpus_per_node", "critic_num_nodes"):
            if getattr(args, flag):
                raise ValueError(
                    f"--{flag.replace('_', '-')} is meaningless with --critic-mode {mode}: "
                    "the critic shares the actor workers' GPUs."
                )
        args.critic_num_gpus_per_node = 0
        args.critic_num_nodes = 0
        if args.critic_lr is None:
            args.critic_lr = args.lr
        return
    if getattr(args, "critic_num_gpus_per_node", None) is None:
        args.critic_num_gpus_per_node = args.actor_num_gpus_per_node
    if getattr(args, "critic_num_nodes", None) is None:
        args.critic_num_nodes = args.actor_num_nodes
    if getattr(args, "critic_load", None) is None:
        args.critic_load = args.load
    if getattr(args, "critic_lr", None) is None:
        args.critic_lr = args.lr


def _validate_ppo_args(args) -> None:
    if not getattr(args, "use_critic", False):
        return

    if getattr(args, "num_critic_only_steps", 0) < 0:
        raise ValueError("--num-critic-only-steps must be nonnegative.")

    if getattr(args, "num_critic_only_steps", 0) > 0 and getattr(args, "kl_coef", 0.0) != 0:
        raise ValueError(
            "--num-critic-only-steps is incompatible with nonzero --kl-coef: "
            "critic-only rollouts do not run the actor/reference forwards required "
            "for reward-level KL shaping. Set --kl-coef 0 or disable critic-only warmup."
        )

    if uses_separate_critic(args):
        actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
        critic_world_size = args.critic_num_nodes * args.critic_num_gpus_per_node
        if actor_world_size != critic_world_size:
            raise ValueError(
                "Separate-critic PPO requires equal actor and critic worker counts for "
                "one-to-one data synchronization; "
                f"got actor={actor_world_size} and critic={critic_world_size}."
            )

    if getattr(args, "offload_train", False):
        raise ValueError(
            "--advantage-estimator ppo is incompatible with --offload-train in Orbit's "
            "Megatron backend because the critic is a full-model trainer. Remove "
            "--offload/--offload-train and allocate separate actor, critic, and rollout GPUs."
        )


def _apply_custom_config_args(args) -> None:
    if not getattr(args, "custom_config_path", None):
        return

    with open(args.custom_config_path) as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    elif not isinstance(data, dict):
        raise ValueError(f"--custom-config-path must contain a mapping at the root; got {type(data).__name__}.")
    for k, v in data.items():
        if hasattr(args, k):
            logger.info(f"Warning: Argument {k} is already set to {getattr(args, k)}, will override with {v}.")
        setattr(args, k, v)


# ---------------------------------------------------------------------------
# Lifted out of the body of miles_validate_args() / hf_validate_args(). Each is
# called from a stamped ORBIT-SEAM at exactly the point it used to be inlined.
# ---------------------------------------------------------------------------


def _validate_peft_ref_args(args) -> None:
    if _is_peft_enabled(args):
        if args.ref_load is not None:
            raise ValueError(
                "args.ref_load is incompatible with peft_method != 'none'. Under PEFT, "
                "the reference policy is the base model and is computed by disabling the "
                "adapter at ref-forward time. Set --load to point at the base checkpoint "
                "instead (or set peft_method=none if you want the legacy ref-model path)."
            )
        if getattr(args, "ref_update_interval", None) is not None:
            raise ValueError(
                "args.ref_update_interval has no meaning under PEFT: the reference is the "
                "frozen base, which never updates. Remove --ref-update-interval (or set "
                "peft_method=none for sliding-window full-FT semantics)."
            )


def _apply_bridge_load_path(args) -> None:
    from orbit.megatron.low_precision_bootstrap import (
        load_hf_config,
        resolve_bridge_load_path,
        validate_low_precision_bootstrap_args,
    )

    hf_config = load_hf_config(args) if args.hf_checkpoint is not None else None
    if args.load is None:
        args.load = resolve_bridge_load_path(args, hf_config=hf_config)
    if hf_config is not None:
        validate_low_precision_bootstrap_args(args, hf_config=hf_config)


def _validate_eval_nll_args(args) -> None:
    # getattr, not attribute access: several unit tests call validate_args with a
    # hand-built Namespace that only carries the fields under test.
    eval_nll_data = getattr(args, "eval_nll_data", None)
    if eval_nll_data is not None:
        eval_nll_interval = getattr(args, "eval_nll_interval", 0)
        eval_nll_micro_batch_size = getattr(args, "eval_nll_micro_batch_size", None)
        assert os.path.exists(eval_nll_data), f"--eval-nll-data file does not exist: {eval_nll_data}"
        assert eval_nll_interval > 0, (
            "--eval-nll-data was given but --eval-nll-interval is "
            f"{eval_nll_interval}; set a positive interval or drop --eval-nll-data."
        )
        assert eval_nll_micro_batch_size is None or eval_nll_micro_batch_size > 0, (
            f"--eval-nll-micro-batch-size must be positive, got {eval_nll_micro_batch_size}"
        )


def orbit_normalize_peft_args(args) -> None:
    _normalize_and_validate_peft_args(args)
    _validate_dsv4_cp_args(args)

    # Expand --true-on-policy into its derived flags/env vars (no-op when off).
    # After PEFT normalization (the contract validates the adapter) and before
    # megatron/sglang validation (it mutates their dests).
    from orbit.true_on_policy import apply_true_on_policy_parse_defaults

    apply_true_on_policy_parse_defaults(args)


def orbit_validate_args(args) -> None:
    # Upstream code added after the flag rename (dashboard, examples) still reads
    # args.use_miles_router; keep it as a read-only alias of orbit's renamed flag.
    args.use_miles_router = args.use_orbit_router
    """Orbit's reward-side validator bundle, called from one seam in miles_validate_args."""
    _validate_opd_args(args)
    _validate_judge_args(args)
    _validate_genrm_args(args)
    _validate_reward_router_args(args)


def _apply_nested_rope_theta(hf_config) -> None:
    # Gemma-4 nests rope_theta per attention type; take the first.
    for _entry in hf_config.rope_parameters.values():
        if isinstance(_entry, dict) and "rope_theta" in _entry:
            hf_config.rope_theta = _entry["rope_theta"]
            break


# ---------------------------------------------------------------------------
# Registration.
# ---------------------------------------------------------------------------


def _override_arg(parser, name, **kwargs):
    """Rewrite an already-registered argument's ``default``/``choices``/``help``.

    miles' own ``reset_arg`` is the same idea but only ever rewrites ``default``
    (and falls back to ``add_argument`` when the option is absent). Orbit also
    narrows ``choices`` and rewrites ``help`` on miles-registered arguments, and
    it always wants a hard failure if the option it means to override is gone --
    so this is a deliberate superset rather than a reuse. Importing miles'
    helper here would also invert the seam's import direction (miles -> orbit).
    """
    for action in parser._actions:
        if name in action.option_strings:
            for key, value in kwargs.items():
                setattr(action, key, value)
            return parser
    raise ValueError(f"{name} is not registered on this parser; orbit cannot override it.")


def add_peft_arguments(parser):
    parser.add_argument(
        "--peft-method",
        type=str,
        choices=["none", "oft", "lora"],
        default="none",
        help=(
            "Parameter-efficient tuning method. OFT is the recommended PEFT path for colocated RL runs; "
            "use 'lora' for existing LoRA adapters or 'none' to disable PEFT."
        ),
    )
    parser.add_argument(
        "--peft-adapter-path",
        type=str,
        default=None,
        help="Path to a PEFT adapter checkpoint for resume.",
    )
    parser.add_argument(
        "--peft-variant",
        type=str,
        choices=["standard", "canonical", "mla", "dsv4"],
        default="standard",
        help=(
            "PEFT module-name variant. Use 'standard' for merged-QKV models, "
            "'mla' for DeepSeek V2/V3-style MLA, 'dsv4' for DeepSeek V4 native "
            "wq_a/wq_b/wkv/wo_a/wo_b targets, and 'canonical' for split-QKV LoRA."
        ),
    )
    parser.add_argument(
        "--adapter-double-buffer",
        action="store_true",
        default=False,
        help=(
            "Enable fixed two-slot adapter double buffering for distributed PEFT rollout engines. "
            "Only supported with --peft-method lora or oft on the NCCL PEFT transport."
        ),
    )
    parser.add_argument(
        "--peft-distributed-transport",
        type=str,
        choices=["nccl", "ray"],
        default=os.getenv("ORBIT_PEFT_DISTRIBUTED_TRANSPORT", "nccl"),
        help=(
            "Transport for distributed PEFT adapter updates. 'nccl' broadcasts adapter tensors "
            "through the SGLang update group; 'ray' serializes CPU adapter tensors through Ray "
            "and SGLang tensor-load endpoints."
        ),
    )
    parser.add_argument(
        "--oft-type",
        type=str,
        choices=["oft", "canonical_oft"],
        default="canonical_oft",
        help=(
            "OFT variant to use. 'canonical_oft' is the default and uses split rotations "
            "on fused QKV / gate-up layers with automatic fallback on unmerged layers. "
            "'oft' uses the legacy shared-R OFT wrapper."
        ),
    )
    parser.add_argument(
        "--oft-adapter-path",
        type=str,
        default=None,
        help="Path to load pre-trained OFT adapter weights (default: None)",
    )
    parser.add_argument(
        "--oft-block-size",
        type=int,
        default=0,
        help="OFT block size. Must be set with --peft-method oft.",
    )
    parser.add_argument("--oft-coft", action="store_true", default=False)
    parser.add_argument("--oft-eps", type=float, default=1e-5)
    parser.add_argument("--oft-block-share", action="store_true", default=False)
    return parser


def add_orbit_arguments(parser):
    """Every orbit-added argument, plus orbit's overrides of miles defaults."""
    parser.add_argument(
        "--offload-train-grad-buffers",
        action=argparse.BooleanOptionalAction,
        help=("Whether --offload-train moves Megatron DDP grad buffers to CPU. " "Defaults to false."),
    )
    parser.add_argument(
        "--offload-train-optimizer",
        action=argparse.BooleanOptionalAction,
        help=("Whether --offload-train moves Megatron optimizer params/state to CPU. " "Defaults to false."),
    )
    parser.add_argument(
        "--offload-train-adapter",
        action=argparse.BooleanOptionalAction,
        help=(
            "Whether bridge-mode Megatron PEFT adapter params/buffers are offloaded to CPU "
            "after rollout weight sync. Defaults to false."
        ),
    )
    parser.add_argument(
        "--offload-train-async",
        action=argparse.BooleanOptionalAction,
        help=(
            "When set with --offload-train, prefetch the train-state H2D wake-up on a "
            "dedicated CUDA stream and overlap with rollout cleanup. This includes the "
            "frozen base and, when enabled, the PEFT adapter. Diagnostic flag for the "
            "train-offload speed-up bake-off; default: off."
        ),
    )
    parser.add_argument(
        "--offload-train-frozen-base-mode",
        type=str,
        choices=["auto", "flat", "tms"],
        default="auto",
        help=(
            "Frozen base weight offload backend for Megatron PEFT train offload. "
            "'tms' uses torch_memory_saver pause/resume, 'flat' uses Orbit's pinned "
            "CPU flat-buffer mover, and 'auto' uses TMS when available for CUDA tensors."
        ),
    )
    parser.add_argument(
        "--offload-rollout-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether to offload rollout OFT adapter buffers through SGLang "
            "adapter CPU backup. Defaults to false; pass "
            "--offload-rollout-adapter to opt in."
        ),
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["rl", "sft"],
        default="rl",
        help="Training objective mode. RL is the default; SFT is an explicit opt-in mode.",
    )
    parser.add_argument(
        "--true-on-policy",
        action="store_true",
        default=False,
        help=(
            "Enable the deterministic true-on-policy ladder via a named contract "
            "(orbit/true_on_policy/). The current Phase 1-4 implementation aligns "
            "scoring and measures the remaining train/rollout kernel gap; it does not "
            "claim bit-exact parity until a contract enables the Phase-5 "
            "SGLang-in-Megatron backend. Expands at parse time into rollout and "
            "training determinism flags and validates model/topology/precision/adapter."
        ),
    )
    parser.add_argument(
        "--true-on-policy-contract",
        type=str,
        default=None,
        help="Override the contract selected by the model profile (e.g. qwen3_dense_true_on_policy_v1).",
    )
    # --recompute-logprobs-via-prefill was upstreamed (registered in miles/utils/arguments.py
    # with identical semantics); orbit's duplicate registration is retired.
    parser.add_argument(
        "--eval-pass-k-values",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit pass@k values to log for eval datasets. "
            "When unset, Orbit falls back to powers of two filtered by n_samples_per_eval_prompt."
        ),
    )
    parser.add_argument(
        "--eval-generate-max-concurrency",
        type=int,
        default=None,
        help=(
            "Maximum number of concurrent eval generation requests. "
            "Unset or non-positive values leave eval limited only by the rollout server concurrency."
        ),
    )

    parser.add_argument(
        "--eval-nll-data",
        type=str,
        default=None,
        help=(
            "JSONL of held-out examples for forward-only negative-log-likelihood eval. "
            "Uses the same chat schema and the same loss masking as --prompt-data under the "
            "SFT rollout function, so training and eval score identical tokens. Independent "
            "of the generation-based eval above: it needs no rollout engine and works with "
            "--eval-interval unset. Unset disables NLL eval (default: None)"
        ),
    )
    parser.add_argument(
        "--eval-nll-interval",
        type=int,
        default=0,
        help=(
            "Run held-out NLL eval every N rollout steps, plus once before training and "
            "always on the final rollout. 0 disables (default: 0)"
        ),
    )
    parser.add_argument(
        "--eval-nll-micro-batch-size",
        type=int,
        default=None,
        help=(
            "Rows per micro-batch during held-out NLL eval. Every row of --eval-nll-data is "
            "scored regardless of this value; it only controls peak eval memory. "
            "Defaults to --micro-batch-size."
        ),
    )
    parser.add_argument(
        "--critic-mode",
        type=str,
        choices=["full", "adapter", "head"],
        default="full",
        help=(
            "PPO critic topology: 'full' (default) runs the legacy separate full-model "
            "critic workers; 'head' runs a one-trunk value-head-only critic whose "
            "frozen critic-side trunk view aliases the actor's (works with a full-FT "
            "actor: the value backward produces no trunk gradients). "
            "'head' is EXPERIMENTAL: at 3B benchmark settings the linear head could "
            "not track the full-FT trunk (value loss ~20x the full critic's) and the "
            "policy collapsed by ~rollout 200 — see docs/reports/2026-08-10-ppo-"
            "critic-comparison; expect to need a deeper head or a KL anchor. "
            "'adapter' runs a one-trunk critic (PEFT adapter + value "
            "head aliasing the actor's frozen trunk) inside the actor workers. "
            "'adapter' requires --advantage-estimator ppo, an enabled --peft-method, "
            "and the megatron train backend."
        ),
    )
    parser.add_argument(
        "--lora-a-init-method",
        type=str,
        default="xavier",
        choices=["xavier", "normal", "kaiming", "zero"],
        help="Initialization for LoRA matrix A, forwarded to Megatron-Bridge's "
        "ParallelLinearAdapter (the path Orbit's Megatron linears actually take). "
        "'kaiming' is kaiming_uniform_(a=sqrt(5)), matching HF PEFT and the "
        "LoRA-without-regret paper; 'xavier' is xavier_normal_ and is Bridge's "
        "default. These differ by ~2.4x in std, which shifts the optimal learning "
        "rate (default: xavier)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help=(
            "Name for this run. Defaults to --wandb-group, which is the historical "
            "behaviour. Set it when several runs share a group -- a learning-rate "
            "sweep grouped by method otherwise puts every arm under one name."
        ),
    )
    if not any("--dsv4-moe-dispatcher" in action.option_strings for action in parser._actions):
        parser.add_argument(
            "--dsv4-moe-dispatcher",
            type=str,
            default="naive",
            choices=("naive", "deepep"),
            help="DeepSeek V4 MoE dispatcher override forwarded to the Megatron bridge provider.",
        )
    if not any("--dsv4-cp-chunk-size-multiple" in action.option_strings for action in parser._actions):
        parser.add_argument(
            "--dsv4-cp-chunk-size-multiple",
            type=int,
            default=128,
            help=(
                "For DeepSeek V4 packed THD CP, pad each sample so each zigzag CP "
                "chunk is a multiple of this value. Keep 128 for DSV4-Pro "
                "compressed KV; smaller debug models may override it."
            ),
        )

    parser = add_on_policy_distillation_arguments(parser)
    parser = add_peft_arguments(parser)

    # Orbit's overrides of miles-registered arguments. These run last, so every
    # option below is already on the parser.
    _override_arg(parser, "--train-backend", choices=["megatron"])
    _override_arg(
        parser,
        "--true-on-policy-mode",
        help=(
            "Internal true-on-policy scoring-mode flag. The exact per-token CI gate "
            "activates only for a contract with the Phase-5 SGLang-in-Megatron backend. "
            "Set automatically by --true-on-policy."
        ),
    )
    _override_arg(
        parser,
        "--loss-type",
        choices=["policy_loss", "sft_loss", "opd_jsd_loss", "opd_topk_loss", "custom_loss"],
        help=(
            "Choose loss type, currently support ppo policy_loss, sft_loss, "
            "opd_jsd_loss (full-vocab on-policy distillation, requires "
            "--teacher-score-mode full_vocab), or opd_topk_loss (direct top-k "
            "on-policy distillation on Orbit's own raw-mass semantics -- the "
            "teacher's top-k log-probs are used as-is, not renormalized within the "
            "subset; requires --opd-log-prob-top-k > 0 and --opd-top-k-strategy "
            "only-teacher); "
            "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
        ),
    )
    _override_arg(parser, "--tis-clip-low", default=0.0)
    _override_arg(
        parser,
        "--target-modules",
        help="Target modules for LoRA/OFT. Use 'all-linear' (attention + "
        "MLP/MoE linears only), 'all' (all-linear plus embed_tokens + lm_head; "
        "OFT only, peft_variant in {standard, canonical}), or comma-separated "
        "module names (e.g., 'q_proj,k_proj,v_proj,o_proj' for HF naming or "
        "'linear_qkv,linear_proj' for Megatron naming).",
    )
    _override_arg(
        parser,
        "--custom-rm-path",
        help=(
            "Path to the custom reward model function. "
            "If set, we will use this function to calculate the reward instead of the default one. "
            "The function should have the signature "
            "`async def custom_rm(args, sample, **kwargs) -> float`; kwargs carry "
            "`evaluation=True` for eval samples."
        ),
    )
    _override_arg(parser, "--loss-mask-type", choices=["qwen", "qwen3", "distill_qwen", "llama3"])

    return parser
