from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Any

from miles.orbit.true_on_policy.contracts import TrueOnPolicyContract, get_true_on_policy_contract
from miles.orbit.true_on_policy.model_profiles import (
    TrueOnPolicyModelProfile,
    get_true_on_policy_model_profile,
    resolve_true_on_policy_model_name,
)


@dataclass(frozen=True)
class TrueOnPolicyArgList:
    """Structured command-line args that stringify only at launch boundaries."""

    values: tuple[str, ...] = ()

    def as_cli_string(self) -> str:
        if not self.values:
            return ""
        return " ".join(shlex.quote(value) for value in self.values) + " "

    def contains(self, flag: str) -> bool:
        return flag in self.values


@dataclass(frozen=True)
class TrueOnPolicyParallelLayout:
    """Training and rollout topology relevant to true-on-policy parity."""

    train_tensor_parallel_size: int
    train_context_parallel_size: int
    train_pipeline_parallel_size: int
    rollout_num_gpus_per_engine: int

    @property
    def uses_train_tp(self) -> bool:
        return self.train_tensor_parallel_size > 1

    @property
    def uses_train_cp(self) -> bool:
        return self.train_context_parallel_size > 1

    @property
    def uses_train_pp(self) -> bool:
        return self.train_pipeline_parallel_size > 1

    @property
    def uses_rollout_tp(self) -> bool:
        return self.rollout_num_gpus_per_engine > 1


@dataclass(frozen=True)
class TrueOnPolicyKernelPolicy:
    """Kernel/runtime switches required to keep SGLang and Megatron aligned."""

    contract: TrueOnPolicyContract
    deterministic_inference: bool
    deterministic_training: bool
    sglang_attention_backend: str
    megatron_uses_sglang_backend: bool
    disable_rope_fusion: bool
    disable_bias_swiglu_fusion: bool
    batch_invariant_mode: bool
    tp_invariant_row_linear: bool
    deterministic_tp_allreduce: bool

    def build_sglang_args(self) -> TrueOnPolicyArgList:
        # No --sglang-true-on-policy-contract: that is sglang-miles vocabulary;
        # orbit's SGLang fork has no such server arg (design §3.2).
        values = [
            "--sglang-attention-backend",
            self.sglang_attention_backend,
        ]
        if self.deterministic_inference:
            values.insert(0, "--sglang-enable-deterministic-inference")
        return TrueOnPolicyArgList(tuple(values))

    def build_megatron_args(self) -> TrueOnPolicyArgList:
        values: list[str] = []
        if self.megatron_uses_sglang_backend:
            # Phase 5: run SGLang kernel wrappers inside Megatron (local spec).
            values.extend(["--transformer-impl", "local", "--use-cpu-initialization"])
        if self.batch_invariant_mode:
            values.append("--batch-invariant-mode")
        if self.disable_bias_swiglu_fusion:
            values.append("--no-bias-swiglu-fusion")
        if self.disable_rope_fusion:
            values.append("--no-rope-fusion")
        return TrueOnPolicyArgList(tuple(values))

    def build_env_vars(self) -> dict[str, str]:
        return {
            "NCCL_ALGO": os.environ.get("NCCL_ALGO", "Ring"),
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }


@dataclass(frozen=True)
class TrueOnPolicyLaunchPlan:
    """Derived cross-engine launch contract for one true-on-policy run."""

    enabled: bool
    model_profile: TrueOnPolicyModelProfile | None = None
    contract: TrueOnPolicyContract | None = None
    parallel_layout: TrueOnPolicyParallelLayout | None = None
    kernel_policy: TrueOnPolicyKernelPolicy | None = None
    sglang_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    megatron_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    orbit_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    env_vars: dict[str, str] = field(default_factory=dict)

    @property
    def train_args(self) -> str:
        return self.sglang_args.as_cli_string() + self.megatron_args.as_cli_string() + self.orbit_args.as_cli_string()


@dataclass(frozen=True)
class TrueOnPolicyConfig:
    """Typed contract derived from the single public true-on-policy switch."""

    enabled: bool
    model_profile: TrueOnPolicyModelProfile
    tensor_model_parallel_size: int
    context_parallel_size: int
    pipeline_model_parallel_size: int
    rollout_num_gpus_per_engine: int
    sequence_parallel: bool
    # §4.4 amendments: the detected run identity, validated against the contract.
    adapter: str
    precision: str
    contract_override: str | None = None

    @property
    def parallel_layout(self) -> TrueOnPolicyParallelLayout:
        return TrueOnPolicyParallelLayout(
            train_tensor_parallel_size=self.tensor_model_parallel_size,
            train_context_parallel_size=self.context_parallel_size,
            train_pipeline_parallel_size=self.pipeline_model_parallel_size,
            rollout_num_gpus_per_engine=self.rollout_num_gpus_per_engine,
        )

    @property
    def requires_tp_invariant_rollout(self) -> bool:
        layout = self.parallel_layout
        return layout.uses_train_tp or layout.uses_rollout_tp

    @property
    def contract(self) -> TrueOnPolicyContract:
        if self.contract_override is not None:
            return get_true_on_policy_contract(self.contract_override)
        return self.model_profile.contract

    def validate(self) -> None:
        if not self.enabled:
            return
        contract = self.contract
        profile = self.model_profile
        if contract.model_family != profile.family:
            raise ValueError(
                f"Contract {contract.name!r} is for {contract.model_family}, but model profile is {profile.family}"
            )
        if self.sequence_parallel and contract.disable_megatron_sequence_parallel:
            # miles silently disables SP at script level; orbit parses args
            # in-process, so a silent flip would contradict the launcher.
            raise ValueError(
                "--true-on-policy requires sequence parallelism off (SP changes reduction "
                "order and breaks train/rollout parity); remove --sequence-parallel."
            )
        layout = self.parallel_layout
        if layout.uses_train_tp and "tp" not in profile.supported_train_layouts:
            raise ValueError(
                f"true-on-policy profile {profile.family!r} does not support 'tp' training layouts yet: "
                "the current log-prob kernel breaks under TP>1; Phase 4 ports the TP-correct gather."
            )
        if layout.uses_train_cp and "cp" not in profile.supported_train_layouts:
            raise ValueError(f"true-on-policy profile {profile.family!r} does not support 'cp' training layouts.")
        if layout.uses_train_pp and "pp" not in profile.supported_train_layouts:
            raise ValueError(f"true-on-policy profile {profile.family!r} does not support 'pp' training layouts.")
        if layout.uses_rollout_tp and "tp" not in profile.supported_rollout_layouts:
            raise ValueError(f"true-on-policy profile {profile.family!r} does not support 'tp' rollout layouts.")
        if self.precision != contract.precision:
            raise ValueError(
                f"Contract {contract.name!r} certifies precision {contract.precision!r}, but this run "
                f"is {self.precision!r}. A different precision needs its own contract (e.g. exact_fp8)."
            )
        if self.adapter not in contract.supported_adapters:
            raise ValueError(
                f"Contract {contract.name!r} certifies adapters {contract.supported_adapters}, but this "
                f"run uses adapter {self.adapter!r}. Adapter parity is unproven; run without "
                "--true-on-policy to measure the mismatch via train_rollout_logprob_abs_diff instead."
            )

    def build_kernel_policy(self) -> TrueOnPolicyKernelPolicy:
        return TrueOnPolicyKernelPolicy(
            contract=self.contract,
            **self.contract.kernel_policy_kwargs_for(tp_invariant_rollout=self.requires_tp_invariant_rollout),
        )

    def build_launch_plan(self) -> TrueOnPolicyLaunchPlan:
        self.validate()
        kernel_policy = self.build_kernel_policy()
        orbit_args = TrueOnPolicyArgList(
            (
                "--deterministic-mode",
                "--true-on-policy-mode",
                "--recompute-logprobs-via-prefill",
            )
        )
        return TrueOnPolicyLaunchPlan(
            enabled=True,
            model_profile=self.model_profile,
            contract=self.contract,
            parallel_layout=self.parallel_layout,
            kernel_policy=kernel_policy,
            sglang_args=kernel_policy.build_sglang_args(),
            megatron_args=kernel_policy.build_megatron_args(),
            orbit_args=orbit_args,
            env_vars=kernel_policy.build_env_vars(),
        )


def _get_required_int(args: Any, name: str) -> int:
    value = getattr(args, name)
    if value is None:
        raise ValueError(f"{name} must be initialized before deriving true-on-policy config")
    return int(value)


def _detect_precision(args: Any) -> str:
    if getattr(args, "fp8", None):
        return "fp8"
    if getattr(args, "fp16", False):
        return "fp16"
    if getattr(args, "bf16", False):
        return "bf16"
    return "fp32"


def _detect_adapter(args: Any) -> str:
    peft_method = getattr(args, "peft_method", None)
    return "full" if peft_method in (None, "none") else str(peft_method)


def build_true_on_policy_config(args: Any) -> TrueOnPolicyConfig | None:
    if not getattr(args, "true_on_policy", False):
        return None

    model_name = resolve_true_on_policy_model_name(args.hf_checkpoint)
    profile = get_true_on_policy_model_profile(model_name)
    return TrueOnPolicyConfig(
        enabled=True,
        model_profile=profile,
        tensor_model_parallel_size=_get_required_int(args, "tensor_model_parallel_size"),
        context_parallel_size=_get_required_int(args, "context_parallel_size"),
        pipeline_model_parallel_size=_get_required_int(args, "pipeline_model_parallel_size"),
        rollout_num_gpus_per_engine=_get_required_int(args, "rollout_num_gpus_per_engine"),
        sequence_parallel=bool(getattr(args, "sequence_parallel", False)),
        adapter=_detect_adapter(args),
        precision=_detect_precision(args),
        contract_override=getattr(args, "true_on_policy_contract", None),
    )


def build_true_on_policy_launch_plan(args: Any) -> TrueOnPolicyLaunchPlan:
    config = build_true_on_policy_config(args)
    if config is None:
        return TrueOnPolicyLaunchPlan(enabled=False)
    return config.build_launch_plan()


def apply_true_on_policy_parse_defaults(args: Any) -> None:
    """Expand the single --true-on-policy switch onto the parsed args.

    Orbit analog of miles' ``apply_true_on_policy_script_defaults``: miles
    expands at launch-script level into CLI strings; orbit parses args
    in-process, so the expansion mutates parsed dests directly. Applies the
    rollout-side dests, the mode flags, the Megatron training-side determinism
    flags (batch-invariant kernels, fusion bans — they reach TransformerConfig
    via core_transformer_config_from_args field-name matching), and env vars.
    Off-mode is a byte-for-byte no-op.
    """
    plan = build_true_on_policy_launch_plan(args)
    if not plan.enabled:
        return

    contract_backend = plan.contract.sglang_attention_backend
    explicit_backend = getattr(args, "sglang_attention_backend", None)
    if explicit_backend is not None and explicit_backend != contract_backend:
        raise ValueError(
            f"Contract {plan.contract.name!r} pins the SGLang attention backend to "
            f"{contract_backend!r}, but --sglang-attention-backend {explicit_backend!r} was given."
        )

    args.sglang_attention_backend = contract_backend
    args.sglang_enable_deterministic_inference = True
    args.true_on_policy_mode = True
    args.recompute_logprobs_via_prefill = True
    args.deterministic_mode = True

    kernel_policy = plan.kernel_policy
    # Lets log_utils.py's exact train/rollout parity CI gate self-activate the
    # day a contract flips megatron_uses_sglang_backend to True (Phase 5:
    # SGLang kernels running inside Megatron via the fork rebase); until then
    # it stays False and the parity gap is measured, not asserted exact.
    args.true_on_policy_megatron_uses_sglang_backend = kernel_policy.megatron_uses_sglang_backend
    args.batch_invariant_mode = kernel_policy.batch_invariant_mode
    if kernel_policy.disable_rope_fusion:
        args.apply_rope_fusion = False
    if kernel_policy.disable_bias_swiglu_fusion:
        args.bias_swiglu_fusion = False

    env_vars = dict(getattr(args, "train_env_vars", None) or {})
    for key, value in plan.env_vars.items():
        env_vars.setdefault(key, value)
        # Megatron's deterministic-mode validation asserts NCCL_ALGO in the
        # *driver* process env (validate_args); actors get --train-env-vars.
        os.environ.setdefault(key, value)
    args.train_env_vars = env_vars
