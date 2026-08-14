"""Behavioral contract for the dedicated Math OFT BS128 low-LR sweep."""

from tools.lora_regret.arms import ALL_MODULES, MATRICES, e4_arms

HIDDEN, FFN, QKV = 4096, 14336, 6144
EXPECTED_LRS = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5)
EXPECTED_NAMES = (
    "oftlow-b128-all-math-lr1e-07-s0",
    "oftlow-b128-all-math-lr3e-07-s0",
    "oftlow-b128-all-math-lr1e-06-s0",
    "oftlow-b128-all-math-lr3e-06-s0",
    "oftlow-b128-all-math-lr1e-05-s0",
)


def _arms():
    from tools.lora_regret.arms import e4_math_oft_b128_low_arms

    return e4_math_oft_b128_low_arms(HIDDEN, FFN, QKV)


def test_matrix_builds_exactly_the_requested_five_math_bs128_arms():
    arms = _arms()

    assert tuple(arm.lr for arm in arms) == EXPECTED_LRS
    assert tuple(arm.name for arm in arms) == EXPECTED_NAMES
    assert {arm.method for arm in arms} == {"oft"}
    assert {arm.oft_block_size for arm in arms} == {128}
    assert {arm.target_modules for arm in arms} == {ALL_MODULES}
    assert {arm.dataset for arm in arms} == {"math"}
    assert {arm.seed for arm in arms} == {0}
    assert all(arm.matched_ratio is not None for arm in arms)


def test_matrix_is_disjoint_from_the_existing_e4_campaign():
    assert not ({arm.name for arm in _arms()} & {arm.name for arm in e4_arms()})


def test_registry_builds_the_same_five_arms():
    registered = MATRICES["e4oftb128low"](HIDDEN, FFN, QKV, 0, None, None)

    assert registered == _arms()


def test_matrix_routes_through_the_rl_accuracy_stack_with_its_own_project():
    from tools.lora_regret.preflight import EXPECTED_ARMS, STAGE_GPU_REQUIREMENTS
    from tools.lora_regret.sweep import (
        MATRIX_LAUNCHERS,
        MATRIX_METRICS,
        MATRIX_PROJECTS,
        RL_LAUNCHER,
        wandb_project,
    )

    assert MATRIX_LAUNCHERS["e4oftb128low"] == RL_LAUNCHER
    assert MATRIX_METRICS["e4oftb128low"] == "accuracy"
    assert MATRIX_PROJECTS["e4oftb128low"] == "rl-b128-low-lr"
    assert wandb_project("e4oftb128low", None, "math", "oft") == (
        "math-rl-b128-low-lr-oft"
    )
    assert EXPECTED_ARMS["e4oftb128low"] == 5
    assert STAGE_GPU_REQUIREMENTS["e4oftb128low"] == 8
