from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrueOnPolicyContractName = Literal["qwen3_dense_true_on_policy_v1"]
ModelFamily = Literal["qwen3_dense", "qwen3_moe", "qwen3_next"]
KernelContract = Literal["qwen3_dense_sglang_math"]
LogprobContract = Literal["sglang_prefill"]


@dataclass(frozen=True)
class TrueOnPolicyContractSchema:
    """Declarative cross-repo identity for a true-on-policy parity contract.

    A contract is a point in the (model_family x precision x adapter) parity
    matrix: it names the exact numeric regime both engines must run for the
    bitwise train/rollout parity claim to hold.

    Ported from miles ``true_on_policy/schema.py`` with orbit amendments
    (design doc §4.4): ``precision`` / ``supported_adapters`` /
    ``param_dtype_overrides`` added; ``fsdp_attention_implementation``
    dropped (orbit is Megatron-only).
    """

    name: TrueOnPolicyContractName
    model_family: ModelFamily
    required_kernel_contracts: tuple[KernelContract, ...]
    logprob_contract: LogprobContract
    sglang_attention_backend: str
    disable_megatron_sequence_parallel: bool
    # orbit additions (design §4.4):
    precision: str = "bf16"
    supported_adapters: tuple[str, ...] = ("full",)
    # Params pinned to a non-ambient dtype in training must have a declared,
    # matching treatment on the SGLang side (the A_log lesson), e.g.
    # (("A_log", "fp32"),) once a Mamba-hybrid contract exists.
    param_dtype_overrides: tuple[tuple[str, str], ...] = ()


QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA = TrueOnPolicyContractSchema(
    name="qwen3_dense_true_on_policy_v1",
    model_family="qwen3_dense",
    required_kernel_contracts=("qwen3_dense_sglang_math",),
    logprob_contract="sglang_prefill",
    # miles pins "fa3"; fa3 refuses to boot on B200 ("requires SM>=80 and
    # SM<=90", Blackwell is SM100) while triton passed the batch-invariance
    # harness byte-exact (tools/rollout_determinism_harness.py, 2026-07-06).
    sglang_attention_backend="triton",
    disable_megatron_sequence_parallel=True,
    precision="bf16",
    supported_adapters=("full",),
    param_dtype_overrides=(),
)
