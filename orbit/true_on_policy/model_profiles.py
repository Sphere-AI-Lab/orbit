from __future__ import annotations

import os
from dataclasses import dataclass

from .contracts import QWEN3_DENSE_TRUE_ON_POLICY_V1, LogprobContract, ModelFamily, TrueOnPolicyContract

ParallelLayout = str


@dataclass(frozen=True)
class TrueOnPolicyModelProfile:
    """Model-specific true-on-policy capabilities and launch defaults.

    Ported from miles ``true_on_policy/model_profiles.py``; orbit drops the
    ``megatron_model_types`` mapping and the ``supports_megatron``/
    ``supports_fsdp`` flags (orbit resolves models via --hf-checkpoint + the
    bridge, and has no FSDP backend).
    """

    family: ModelFamily
    model_names: tuple[str, ...]
    supported_train_layouts: tuple[ParallelLayout, ...]
    supported_rollout_layouts: tuple[ParallelLayout, ...]
    contract: TrueOnPolicyContract

    @property
    def required_kernel_contracts(self):
        return self.contract.required_kernel_contracts

    @property
    def logprob_contract(self) -> LogprobContract:
        return self.contract.logprob_contract

    @property
    def sglang_attention_backend(self) -> str:
        return self.contract.sglang_attention_backend

    @property
    def disable_megatron_sequence_parallel(self) -> bool:
        return self.contract.disable_megatron_sequence_parallel


QWEN3_DENSE_PROFILE = TrueOnPolicyModelProfile(
    family="qwen3_dense",
    model_names=(
        "Qwen3-0.6B",
        "Qwen3-4B",
        "Qwen3-4B-Base",
        "Qwen3-4B-Instruct-2507",
    ),
    # miles certifies ("dp", "tp", "pp", "ulysses_cp") for training; "tp"
    # joined in Phase 4 with the TP-correct full-vocab gather. "cp" stays out
    # until the CP loss-scaling correction is ported.
    supported_train_layouts=("dp", "tp", "pp"),
    supported_rollout_layouts=("dp", "tp"),
    contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
)


_MODEL_PROFILES = (QWEN3_DENSE_PROFILE,)
_PROFILE_BY_MODEL_NAME = {model_name: profile for profile in _MODEL_PROFILES for model_name in profile.model_names}


def resolve_true_on_policy_model_name(hf_checkpoint: str) -> str:
    """Model identity = the HF checkpoint directory basename."""
    return os.path.basename(str(hf_checkpoint).rstrip("/"))


def get_true_on_policy_model_profile(model_name: str) -> TrueOnPolicyModelProfile:
    try:
        return _PROFILE_BY_MODEL_NAME[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILE_BY_MODEL_NAME))
        raise ValueError(
            f"true-on-policy does not have a model profile for {model_name!r}. Supported models: {supported}"
        ) from exc
