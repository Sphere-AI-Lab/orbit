"""Unit tests for the true-on-policy contract package (Phase 3).

Ported from miles ``tests/fast/true_on_policy/test_config.py`` (243 L) with the
orbit adaptations from the design doc (docs/plans/2026-07-06-true-on-policy-
design.md §4.4): Megatron-only simplification, schema extended with
``precision`` / ``supported_adapters`` / ``param_dtype_overrides``, contract
pins the triton attention backend (fa3 impossible on B200), train TP/CP
rejected until Phase 4 ports the TP-correct log-prob gather.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from orbit.true_on_policy import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    apply_true_on_policy_parse_defaults,
    build_true_on_policy_launch_plan,
    get_true_on_policy_contract,
    get_true_on_policy_model_profile,
    resolve_true_on_policy_model_name,
)


def _args(**overrides):
    values = {
        "true_on_policy": True,
        "true_on_policy_contract": None,
        "hf_checkpoint": "/models/Qwen3-4B",
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "rollout_num_gpus_per_engine": 1,
        "sequence_parallel": False,
        "peft_method": "none",
        "bf16": True,
        "fp16": False,
        "fp8": None,
        "true_on_policy_mode": False,
        "recompute_logprobs_via_prefill": False,
        "deterministic_mode": False,
        "sglang_enable_deterministic_inference": False,
        "sglang_attention_backend": None,
        "train_env_vars": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# ---------------------------------------------------------------------------
# Profiles and contracts
# ---------------------------------------------------------------------------


def test_qwen3_dense_profile_resolves_model_names():
    profile = get_true_on_policy_model_profile("Qwen3-4B")
    contract = get_true_on_policy_contract("qwen3_dense_true_on_policy_v1")

    assert profile.family == "qwen3_dense"
    assert profile.contract is QWEN3_DENSE_TRUE_ON_POLICY_V1
    assert profile.contract is contract
    assert contract.schema.name == "qwen3_dense_true_on_policy_v1"
    assert contract.schema.model_family == "qwen3_dense"
    # orbit deviation from miles: no "ulysses_cp" (orbit's CP loss-scaling
    # correction is unported); "tp" joined in Phase 4 with the TP-correct
    # full-vocab gather.
    assert profile.supported_train_layouts == ("dp", "tp", "pp")
    assert profile.supported_rollout_layouts == ("dp", "tp")
    assert profile.required_kernel_contracts == ("qwen3_dense_sglang_math",)
    assert profile.logprob_contract == "sglang_prefill"
    # orbit deviation from miles ("fa3"): fa3 requires SM 80-90, B200 is SM100.
    assert contract.sglang_attention_backend == "triton"


def test_contract_schema_carries_orbit_parity_matrix_fields():
    schema = QWEN3_DENSE_TRUE_ON_POLICY_V1.schema
    assert schema.precision == "bf16"
    assert schema.supported_adapters == ("full",)
    assert schema.param_dtype_overrides == ()


def test_unknown_true_on_policy_model_fails_early():
    with pytest.raises(ValueError, match="does not have a model profile"):
        get_true_on_policy_model_profile("unknown-model")


def test_model_name_resolved_from_hf_checkpoint_basename():
    assert resolve_true_on_policy_model_name("/models/Qwen3-4B") == "Qwen3-4B"
    assert resolve_true_on_policy_model_name("/models/Qwen3-4B/") == "Qwen3-4B"


def test_true_on_policy_contract_override_is_validated():
    args = _args(true_on_policy_contract="unknown_contract")
    with pytest.raises(ValueError, match="Unsupported true-on-policy contract"):
        build_true_on_policy_launch_plan(args)


# ---------------------------------------------------------------------------
# Off-mode: byte-for-byte no-op
# ---------------------------------------------------------------------------


def test_off_mode_builds_empty_plan_and_does_not_mutate_args():
    args = _args(true_on_policy=False, sequence_parallel=True, fp8="hybrid")
    before = dict(vars(args))

    apply_true_on_policy_parse_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert vars(args) == before
    assert not plan.enabled
    assert plan.train_args == ""
    assert plan.env_vars == {}


# ---------------------------------------------------------------------------
# On-mode: parse-time expansion
# ---------------------------------------------------------------------------


def test_switch_expands_rollout_and_mode_dests():
    args = _args()

    apply_true_on_policy_parse_defaults(args)

    assert args.true_on_policy_mode is True
    assert args.recompute_logprobs_via_prefill is True
    assert args.deterministic_mode is True
    assert args.sglang_enable_deterministic_inference is True
    assert args.sglang_attention_backend == "triton"
    assert args.train_env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    assert args.train_env_vars["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "NCCL_ALGO" in args.train_env_vars


def test_expansion_applies_training_side_determinism_flags():
    args = _args()

    apply_true_on_policy_parse_defaults(args)

    # Phase 4: batch-invariant kernels + fusion bans flow into TransformerConfig
    # via core_transformer_config_from_args (field-name matching).
    assert args.batch_invariant_mode is True
    assert args.apply_rope_fusion is False
    assert args.bias_swiglu_fusion is False


def test_conflicting_explicit_sglang_backend_is_rejected():
    args = _args(sglang_attention_backend="fa3")
    with pytest.raises(ValueError, match="attention backend"):
        apply_true_on_policy_parse_defaults(args)


def test_matching_explicit_sglang_backend_is_kept():
    args = _args(sglang_attention_backend="triton")
    apply_true_on_policy_parse_defaults(args)
    assert args.sglang_attention_backend == "triton"


def test_user_train_env_vars_win_over_contract_defaults():
    args = _args(train_env_vars={"CUBLAS_WORKSPACE_CONFIG": ":16:8"})
    apply_true_on_policy_parse_defaults(args)
    assert args.train_env_vars["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_expansion_exports_driver_process_env(monkeypatch):
    # Megatron's deterministic-mode validate_args asserts NCCL_ALGO in the
    # driver env, not just the actor env.
    import os

    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.delenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", raising=False)
    apply_true_on_policy_parse_defaults(_args())
    assert os.environ["NCCL_ALGO"] == "Ring"
    assert os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"


def test_nccl_algo_respects_ambient_environment(monkeypatch):
    monkeypatch.setenv("NCCL_ALGO", "Tree")
    args = _args()
    apply_true_on_policy_parse_defaults(args)
    assert args.train_env_vars["NCCL_ALGO"] == "Tree"

    monkeypatch.delenv("NCCL_ALGO")
    args = _args()
    apply_true_on_policy_parse_defaults(args)
    assert args.train_env_vars["NCCL_ALGO"] == "Ring"


# ---------------------------------------------------------------------------
# Topology validation
# ---------------------------------------------------------------------------


def test_sequence_parallel_is_rejected():
    args = _args(sequence_parallel=True)
    with pytest.raises(ValueError, match="sequence.parallel"):
        build_true_on_policy_launch_plan(args)


def test_train_tp_is_allowed_and_drives_tp_invariant_policy():
    args = _args(tensor_model_parallel_size=2)
    plan = build_true_on_policy_launch_plan(args)
    assert plan.parallel_layout.uses_train_tp
    assert plan.kernel_policy.tp_invariant_row_linear
    assert plan.kernel_policy.deterministic_tp_allreduce


def test_context_parallel_is_rejected():
    args = _args(context_parallel_size=2)
    with pytest.raises(ValueError, match="does not support 'cp'"):
        build_true_on_policy_launch_plan(args)


def test_pipeline_parallel_is_allowed():
    args = _args(pipeline_model_parallel_size=2)
    plan = build_true_on_policy_launch_plan(args)
    assert plan.parallel_layout.uses_train_pp


def test_rollout_tp_is_allowed_and_drives_tp_invariant_policy():
    args = _args(rollout_num_gpus_per_engine=2)
    plan = build_true_on_policy_launch_plan(args)
    assert plan.parallel_layout.uses_rollout_tp
    assert plan.kernel_policy.tp_invariant_row_linear
    assert plan.kernel_policy.deterministic_tp_allreduce


# ---------------------------------------------------------------------------
# Precision / adapter validation (§4.4 amendments)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"fp8": "hybrid"},
        {"bf16": False, "fp16": True},
        {"bf16": False},
    ],
)
def test_non_bf16_checkpoint_is_rejected(overrides):
    args = _args(**overrides)
    with pytest.raises(ValueError, match="precision"):
        build_true_on_policy_launch_plan(args)


@pytest.mark.parametrize("peft_method", ["oft", "lora"])
def test_peft_adapters_are_rejected_by_v1_contract(peft_method):
    args = _args(peft_method=peft_method)
    with pytest.raises(ValueError, match="adapter"):
        build_true_on_policy_launch_plan(args)


# ---------------------------------------------------------------------------
# Kernel policy and launch plan
# ---------------------------------------------------------------------------


def test_contract_object_owns_kernel_policy_values():
    plan = build_true_on_policy_launch_plan(_args())
    policy = plan.kernel_policy

    assert policy.contract is QWEN3_DENSE_TRUE_ON_POLICY_V1
    assert policy.deterministic_inference
    assert policy.deterministic_training
    assert policy.sglang_attention_backend == "triton"
    assert policy.batch_invariant_mode
    assert policy.disable_rope_fusion
    assert policy.disable_bias_swiglu_fusion
    # Phase-5 scope (SGLang-kernels-in-Megatron); never True in v1.
    assert policy.megatron_uses_sglang_backend is False
    # dp-only rollout: no TP-invariance machinery needed.
    assert policy.tp_invariant_row_linear is False
    assert policy.deterministic_tp_allreduce is False


def test_sglang_args_only_use_flags_our_fork_accepts():
    plan = build_true_on_policy_launch_plan(_args())

    assert plan.sglang_args.values == (
        "--sglang-enable-deterministic-inference",
        "--sglang-attention-backend",
        "triton",
    )
    # sglang-miles vocabulary; orbit's fork has no such server arg (§3.2).
    assert "--sglang-true-on-policy-contract" not in plan.train_args


def test_megatron_args_are_declared_for_phase4():
    plan = build_true_on_policy_launch_plan(_args())
    assert plan.megatron_args.values == (
        "--batch-invariant-mode",
        "--no-bias-swiglu-fusion",
        "--no-rope-fusion",
    )


def test_orbit_args_carry_the_mode_flags():
    plan = build_true_on_policy_launch_plan(_args())
    assert plan.orbit_args.values == (
        "--deterministic-mode",
        "--true-on-policy-mode",
        "--recompute-logprobs-via-prefill",
    )
    assert "--recompute-logprobs-via-prefill" in plan.train_args
