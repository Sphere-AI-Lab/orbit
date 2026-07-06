from __future__ import annotations

from dataclasses import dataclass

from .schema import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA,
    KernelContract,
    LogprobContract,
    ModelFamily,
    TrueOnPolicyContractName,
    TrueOnPolicyContractSchema,
)


@dataclass(frozen=True)
class TrueOnPolicyContract:
    """Internal parity contract selected by orbit and implemented by each engine."""

    schema: TrueOnPolicyContractSchema

    @property
    def name(self) -> TrueOnPolicyContractName:
        return self.schema.name

    @property
    def model_family(self) -> ModelFamily:
        return self.schema.model_family

    @property
    def required_kernel_contracts(self) -> tuple[KernelContract, ...]:
        return self.schema.required_kernel_contracts

    @property
    def logprob_contract(self) -> LogprobContract:
        return self.schema.logprob_contract

    @property
    def sglang_attention_backend(self) -> str:
        return self.schema.sglang_attention_backend

    @property
    def disable_megatron_sequence_parallel(self) -> bool:
        return self.schema.disable_megatron_sequence_parallel

    @property
    def precision(self) -> str:
        return self.schema.precision

    @property
    def supported_adapters(self) -> tuple[str, ...]:
        return self.schema.supported_adapters

    def kernel_policy_kwargs_for(self, *, tp_invariant_rollout: bool) -> dict[str, object]:
        # orbit is Megatron-only, so the fusion bans and batch-invariant mode
        # miles keyed on `train_backend == "megatron"` are unconditional here.
        return {
            "deterministic_inference": True,
            "deterministic_training": True,
            "sglang_attention_backend": self.sglang_attention_backend,
            # Phase 5 (SGLang-kernels-in-Megatron via the fork rebase) flips
            # this to True; until then Megatron runs its own kernels and the
            # parity gap is measured, not closed.
            "megatron_uses_sglang_backend": False,
            "disable_rope_fusion": True,
            "disable_bias_swiglu_fusion": True,
            "batch_invariant_mode": True,
            "tp_invariant_row_linear": tp_invariant_rollout,
            "deterministic_tp_allreduce": tp_invariant_rollout,
        }


QWEN3_DENSE_TRUE_ON_POLICY_V1 = TrueOnPolicyContract(
    schema=QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA,
)


_CONTRACT_BY_NAME = {
    QWEN3_DENSE_TRUE_ON_POLICY_V1.name: QWEN3_DENSE_TRUE_ON_POLICY_V1,
}


def get_true_on_policy_contract(name: str) -> TrueOnPolicyContract:
    try:
        return _CONTRACT_BY_NAME[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_CONTRACT_BY_NAME))
        raise ValueError(f"Unsupported true-on-policy contract {name!r}. Supported contracts: {supported}") from exc
