"""True-on-policy launch contract helpers.

Port of miles ``true_on_policy/`` (design doc
docs/plans/2026-07-06-true-on-policy-design.md, §4.4 "A-shaped package,
C-shaped semantics"): Megatron-only, schema extended with the
precision/adapter parity-matrix fields, and the qwen3-dense v1 contract pins
the triton attention backend (fa3 is impossible on B200).
"""

from orbit.peft.true_on_policy.config import (
    TrueOnPolicyArgList,
    TrueOnPolicyKernelPolicy,
    TrueOnPolicyLaunchPlan,
    TrueOnPolicyParallelLayout,
    apply_true_on_policy_parse_defaults,
    build_true_on_policy_config,
    build_true_on_policy_launch_plan,
)
from orbit.peft.true_on_policy.contracts import QWEN3_DENSE_TRUE_ON_POLICY_V1, TrueOnPolicyContract, get_true_on_policy_contract
from orbit.peft.true_on_policy.model_profiles import (
    TrueOnPolicyModelProfile,
    get_true_on_policy_model_profile,
    resolve_true_on_policy_model_name,
)

__all__ = [
    "TrueOnPolicyLaunchPlan",
    "TrueOnPolicyArgList",
    "TrueOnPolicyKernelPolicy",
    "TrueOnPolicyContract",
    "TrueOnPolicyModelProfile",
    "TrueOnPolicyParallelLayout",
    "QWEN3_DENSE_TRUE_ON_POLICY_V1",
    "apply_true_on_policy_parse_defaults",
    "build_true_on_policy_config",
    "build_true_on_policy_launch_plan",
    "get_true_on_policy_contract",
    "get_true_on_policy_model_profile",
    "resolve_true_on_policy_model_name",
]
