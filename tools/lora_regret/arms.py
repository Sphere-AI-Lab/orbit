"""The LoRA-without-regret SFT experiment matrix.

Two LR grids, because the LoRA and FullFT optima sit a decade apart and one
shared grid would spend most of its points where nothing happens.
"""

from __future__ import annotations

from dataclasses import dataclass

from orbit.utils.peft_param_match import matched_oft_block_size

ALL_MODULES = "linear_qkv,linear_proj,linear_fc1,linear_fc2"
ATTN_MODULES = "linear_qkv,linear_proj"
MLP_MODULES = "linear_fc1,linear_fc2"

# Brackets every published LoRA optimum (1.2e-4 .. 3.5e-4) with >=2 points a side.
LORA_LR_GRID = [5e-5, 8e-5, 1.2e-4, 2e-4, 3e-4, 5e-4, 8e-4]
# Same shape, one decade down; brackets the FullFT optimum 2.5e-5.
FULL_LR_GRID = [5e-6, 8e-6, 1.2e-5, 2e-5, 3e-5, 5e-5, 8e-5]
# OFT's natural LR scale is unknown a priori: it parameterizes a rotation, not
# an additive update. Scout wide, then refine around the argmin.
OFT_SCOUT_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

LORA_ALPHA = 32
# LORA_A_INIT_METHOD is fixed at "kaiming" for the whole sweep, never
# "uniform" -- orbit/utils/arguments.py registers
# choices=["xavier","normal","kaiming","zero"], so "uniform" is rejected by
# argparse outright. Orbit's own default is "xavier"; PEFT-compatible init is
# "kaiming", and the two differ by ~2.4x in std (see the launcher's comment),
# which shifts the measured optimal LR, so this is pinned rather than left to
# the launcher's own default.
LORA_A_INIT_METHOD = "kaiming"


@dataclass(frozen=True)
class Arm:
    name: str
    method: str  # "full" | "lora" | "oft"
    rank: int | None
    oft_block_size: int | None
    target_modules: str
    lr: float
    seed: int


def _name(method: str, tag: str, modules: str, lr: float, seed: int) -> str:
    short = {ALL_MODULES: "all", ATTN_MODULES: "attn", MLP_MODULES: "mlp"}.get(modules, "na")
    return f"{method}-{tag}-{short}-lr{lr:g}-s{seed}"


def sft_arms(hidden_size: int, ffn_size: int, seed: int = 0) -> list[Arm]:
    """The 82-arm SFT matrix: 42 LoRA/FullFT plus 40 OFT (5 scout + 5x7).

    `ffn_size` is accepted (not just `hidden_size`) to keep the signature
    stable for a future per-module OFT match -- MLP's `linear_fc2` has
    `d_in == ffn_size`, not `hidden_size` -- but today's matched block size is
    deliberately solved against the square attention shape only (one shared
    `OFT_BLOCK_SIZE` knob per arm; Megatron-Bridge's `OFTRotationModule`
    silently snaps it to a divisor of each layer's own `d_in`, so the MLP
    layers still end up with a valid, if not perfectly matched, block size).
    See `orbit.utils.peft_param_match`'s module docstring for the accounting.
    """
    if hidden_size <= 0 or ffn_size <= 0:
        raise ValueError(f"hidden_size and ffn_size must be positive, got {hidden_size}, {ffn_size}")

    arms: list[Arm] = []

    for lr in FULL_LR_GRID:
        arms.append(Arm(_name("full", "na", "", lr, seed), "full", None, None, "", lr, seed))

    lora_configs = [
        (256, ALL_MODULES),
        (256, ATTN_MODULES),
        (256, MLP_MODULES),
        (16, ALL_MODULES),
        (1, ALL_MODULES),
    ]
    for rank, modules in lora_configs:
        for lr in LORA_LR_GRID:
            arms.append(
                Arm(_name("lora", f"r{rank}", modules, lr, seed), "lora", rank, None, modules, lr, seed)
            )

    # Matched OFT. Block size is solved against the square (attention) shape so
    # all arms share one OFT_BLOCK_SIZE; per-layer snapping handles the rest.
    oft_configs = [
        (1, ALL_MODULES),
        (16, ALL_MODULES),
        (256, ALL_MODULES),
        (16, ATTN_MODULES),
        (16, MLP_MODULES),
    ]
    scout_block = matched_oft_block_size(16, hidden_size, hidden_size)
    for lr in OFT_SCOUT_GRID:
        arms.append(
            Arm(_name("oftscout", f"b{scout_block}", ALL_MODULES, lr, seed),
                "oft", None, scout_block, ALL_MODULES, lr, seed)
        )
    for rank, modules in oft_configs:
        block = matched_oft_block_size(rank, hidden_size, hidden_size)
        for lr in LORA_LR_GRID:
            arms.append(
                Arm(_name("oft", f"b{block}", modules, lr, seed), "oft", None, block, modules, lr, seed)
            )

    return arms


def arm_env(arm: Arm) -> dict[str, str]:
    """Environment overrides for one launcher invocation.

    Deliberately does not set ROLLOUT_SEED: the launcher ties it to SEED
    itself (scripts/lib/train.sh + scripts/lib/rollout.sh), which is exactly
    what makes a seed sweep vary training data order along with init -- an
    override here would silently defeat that.
    """
    env = {"LR": f"{arm.lr:g}", "SEED": str(arm.seed)}
    if arm.method == "full":
        env["PEFT_METHOD"] = "none"
        return env
    env["TARGET_MODULES"] = arm.target_modules
    if arm.method == "lora":
        env["PEFT_METHOD"] = "lora"
        env["LORA_RANK"] = str(arm.rank)
        env["LORA_ALPHA"] = str(LORA_ALPHA)
        env["LORA_A_INIT_METHOD"] = LORA_A_INIT_METHOD
    elif arm.method == "oft":
        env["PEFT_METHOD"] = "oft"
        env["OFT_BLOCK_SIZE"] = str(arm.oft_block_size)
    else:
        raise ValueError(f"unknown method {arm.method!r}")
    return env
