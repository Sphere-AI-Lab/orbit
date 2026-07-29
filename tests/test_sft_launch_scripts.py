from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SFT_EXAMPLES = REPO_ROOT / "examples" / "sft"


def test_sft_launch_folder_contains_qwen_dataset_launchers():
    expected = {
        "run-qwen2_5-0_5b-bf16-sft-numinamath.sh",
        "run-qwen2_5-0_5b-bf16-sft-magicoder.sh",
        "run-qwen2_5-0_5b-bf16-sft-commonsenseqa.sh",
        "run-qwen2_5-0_5b-bf16-sft-socialiqa.sh",
        "run-qwen2_5-0_5b-bf16-sft-scienceqa-text.sh",
    }

    assert {path.name for path in SFT_EXAMPLES.glob("*.sh")} >= expected


def test_llama_oft_sft_launchers_exist():
    expected = {
        "run-llama3_1-8b-bf16-oft-sft-magicoder.sh",
        "run-llama3_1-8b-bf16-oft-sft-commonsenseqa.sh",
        "run-llama3_1-8b-bf16-oft-sft-scienceqa-text.sh",
        "run-llama3_1-8b-bf16-oft-sft-numinamath.sh",
    }

    assert {path.name for path in SFT_EXAMPLES.glob("*.sh")} >= expected


def test_sft_launchers_are_standalone():
    for path in SFT_EXAMPLES.glob("run-*.sh"):
        content = path.read_text(encoding="utf-8")
        assert "-sft-common.sh" not in content
        assert 'source "${SCRIPT_DIR}/' not in content
        assert 'source "${ORBIT_ROOT}/scripts/lib/launcher.sh"' in content


def test_qwen_sft_launchers_use_messages_sft_mode_and_no_sglang_args():
    for path in SFT_EXAMPLES.glob("run-qwen2_5-0_5b-bf16-sft-*.sh"):
        content = path.read_text(encoding="utf-8")
        assert "--training-mode sft" in content
        assert "--loss-type sft_loss" in content
        assert "--input-key messages" in content
        assert "SGLANG_ARGS=()" in content
        assert "--rollout-function-path orbit.rollout.sft_rollout.generate_rollout" in content
        assert "--loss-mask-type \"${LOSS_MASK_TYPE:-qwen}\"" in content


def test_llama_launchers_use_oft_and_response_only_mask():
    for path in SFT_EXAMPLES.glob("run-llama3_1-8b-bf16-oft-sft-*.sh"):
        content = path.read_text(encoding="utf-8")
        assert "llama3.1-8B-Instruct.sh" in content
        assert "--training-mode sft" in content
        assert "--loss-type sft_loss" in content
        assert "--input-key messages" in content
        assert "--loss-mask-type \"${LOSS_MASK_TYPE:-response_only}\"" in content
        assert "--peft-method oft" in content
        assert "--oft-type canonical_oft" in content
        assert "--oft-block-size \"${OFT_BLOCK_SIZE:-32}\"" in content
        assert "--oft-eps \"${OFT_EPS:-6e-5}\"" in content
        assert "--target-modules \"${TARGET_MODULES:-all-linear}\"" in content
        assert "SGLANG_ARGS=()" in content


def test_sft_dataset_wrappers_set_dataset_defaults():
    wrappers = {
        "run-qwen2_5-0_5b-bf16-sft-numinamath.sh": "numinamath",
        "run-qwen2_5-0_5b-bf16-sft-magicoder.sh": "magicoder",
        "run-qwen2_5-0_5b-bf16-sft-commonsenseqa.sh": "commonsenseqa",
        "run-qwen2_5-0_5b-bf16-sft-socialiqa.sh": "socialiqa",
        "run-qwen2_5-0_5b-bf16-sft-scienceqa-text.sh": "scienceqa-text",
    }

    for filename, dataset_name in wrappers.items():
        content = (SFT_EXAMPLES / filename).read_text(encoding="utf-8")
        assert f'SFT_DATASET_NAME="{dataset_name}"' in content
        assert f"/{dataset_name}/train.jsonl" in content
        assert "LAUNCHER_NAME=${LAUNCHER_NAME:-run_qwen25_05b_bf16_sft_${SFT_DATASET_SAFE}}" in content


def test_llama_wrappers_set_dataset_defaults():
    wrappers = {
        "run-llama3_1-8b-bf16-oft-sft-magicoder.sh": "magicoder",
        "run-llama3_1-8b-bf16-oft-sft-commonsenseqa.sh": "commonsenseqa",
        "run-llama3_1-8b-bf16-oft-sft-scienceqa-text.sh": "scienceqa-text",
        "run-llama3_1-8b-bf16-oft-sft-numinamath.sh": "numinamath",
    }

    for filename, dataset_name in wrappers.items():
        content = (SFT_EXAMPLES / filename).read_text(encoding="utf-8")
        assert f'SFT_DATASET_NAME="{dataset_name}"' in content
        assert f"/{dataset_name}/train.jsonl" in content
        assert "LAUNCHER_NAME=${LAUNCHER_NAME:-run_llama31_8b_bf16_oft_sft_${SFT_DATASET_SAFE}}" in content


# ---------------------------------------------------------------------------
# LoRA-without-regret repro launcher (gate G4).
#
# This repo has no scripts/lib/{peft,rollout,train}.sh, so there is no shared
# lib to hold the campaign's knobs and no shared default that could drift.
# These assertions therefore pin the knobs where they actually live -- in the
# launcher -- and pin the two that are load-bearing for the study's numbers:
# the no-colon LABEL_KEY form and the SEED/ROLLOUT_SEED tie.
# ---------------------------------------------------------------------------

LORA_REGRET_LAUNCHER = "run-llama3_1-8b-bf16-lora-sft-tulu3.sh"


def _lora_regret_launcher_text() -> str:
    return (SFT_EXAMPLES / LORA_REGRET_LAUNCHER).read_text(encoding="utf-8")


def test_lora_regret_launcher_exists_and_is_standalone():
    assert (SFT_EXAMPLES / LORA_REGRET_LAUNCHER).is_file()
    content = _lora_regret_launcher_text()
    assert 'source "${ORBIT_ROOT}/scripts/lib/launcher.sh"' in content
    assert 'source "${SCRIPT_DIR}/' not in content


def test_lora_regret_launcher_is_llama_tulu3_not_qwen_norobots():
    """The campaign re-anchored from Qwen3-4B/No-Robots to Llama-3.1-8B/Tulu3."""
    content = _lora_regret_launcher_text()
    assert "Llama-3.1-8B" in content
    assert "tulu3" in content
    assert "no_robots" not in content


def test_lora_regret_launcher_pins_the_llama31_chat_template():
    """Llama-3.1-8B *base* ships no chat_template, so apply_chat_template would
    raise and MultiTurnLossMaskGenerator could not even be constructed."""
    content = _lora_regret_launcher_text()
    assert "orbit/utils/chat_template_utils/templates/llama3.1_pinned.jinja" in content


def test_lora_regret_launcher_uses_the_llama3_loss_mask_and_raw_messages():
    content = _lora_regret_launcher_text()
    assert "LOSS_MASK_TYPE=${LOSS_MASK_TYPE:-llama3}" in content
    # sft_rollout hands sample.prompt straight to the mask generator, which
    # wants the raw messages list -- a rendered chat string would be
    # re-tokenized as text.
    assert "APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-0}" in content
    assert "--input-key prompt" in content


def test_lora_regret_launcher_label_key_uses_the_no_colon_form():
    """SFT rows are {"prompt": [...]} with no label field. ${LABEL_KEY:-...}
    would also fire on a set-but-empty value and re-default it, pointing the
    loader at a column that does not exist."""
    content = _lora_regret_launcher_text()
    assert "LABEL_KEY=${LABEL_KEY-}" in content
    assert "LABEL_KEY=${LABEL_KEY:-" not in content


def test_lora_regret_launcher_ties_rollout_seed_to_seed():
    """Only here, never in a shared default: --rollout-seed also seeds SGLang
    generation, so moving its 42 default would silently change other RL runs."""
    content = _lora_regret_launcher_text()
    assert "SEED=${SEED:-1234}" in content
    assert "ROLLOUT_SEED=${ROLLOUT_SEED:-${SEED}}" in content
    assert "--rollout-seed" in content


def test_lora_regret_launcher_uses_kaiming_lora_init():
    """xavier_normal_ and kaiming_uniform_(a=sqrt(5)) differ by ~2.4x in std,
    which shifts the measured optimal learning rate."""
    content = _lora_regret_launcher_text()
    assert '--lora-a-init-method "${LORA_A_INIT_METHOD:-kaiming}"' in content


def test_lora_regret_launcher_wires_the_held_out_nll_eval():
    content = _lora_regret_launcher_text()
    assert "--eval-nll-data" in content
    assert "--eval-nll-interval" in content
    assert "SGLANG_ARGS=()" in content


def test_lora_regret_launcher_still_passes_prompt_data():
    """train.py calls create_rollout_manager() unconditionally and
    RolloutManager.__init__ loads the dataset, so a pure-SFT run is not exempt
    from the loader's contract."""
    content = _lora_regret_launcher_text()
    assert "--prompt-data" in content
    assert "--training-mode sft" in content


def test_lora_regret_launcher_dispatches_lora_oft_and_full_finetune():
    """One launcher serves all three arms, because tools/lora_regret/arms.py
    drives a single script by environment override (PEFT_METHOD=lora|oft|none).

    A separate run-llama3_1-8b-bf16-oft-sft-tulu3.sh is NOT an option: it would
    be caught by test_llama_launchers_use_oft_and_response_only_mask's
    `run-llama3_1-8b-bf16-oft-sft-*.sh` glob, which requires
    `--input-key messages` and the response_only mask -- both wrong for this
    campaign, whose rows are {"prompt": [...]} scored by the llama3 mask.
    """
    content = _lora_regret_launcher_text()
    assert 'case "${PEFT_METHOD}" in' in content
    assert "--peft-method lora" in content
    assert "--peft-method oft" in content
    assert "--oft-type canonical_oft" in content


def test_lora_regret_launcher_oft_arm_passes_no_lora_flags():
    """orbit/utils/arguments.py cross-validates the two flag families: OFT flags
    must be at their defaults unless --peft-method is oft. Keep the converse
    true too, so an OFT arm's command line carries no LoRA rank/alpha that
    would read as if it had one."""
    content = _lora_regret_launcher_text()
    oft_branch = content.split("    oft)", 1)[1].split(";;", 1)[0]
    # Comments excluded deliberately: the branch explains this constraint in
    # prose that necessarily names the LoRA flags it is refusing to pass.
    code = [line for line in oft_branch.splitlines() if not line.lstrip().startswith("#")]
    assert any("--oft-block-size" in line for line in code)
    assert not any("--lora-rank" in line for line in code)
    assert not any("--lora-alpha" in line for line in code)


def test_lora_regret_launcher_requires_an_explicit_oft_block_size():
    """A silent OFT_BLOCK_SIZE default would defeat the matched-parameter
    comparison E5 exists to make: the block size IS the parameter budget, and
    it must come from peft_param_match.matched_oft_block_size, not from a
    number that happens to be in the script."""
    content = _lora_regret_launcher_text()
    assert "${OFT_BLOCK_SIZE:?" in content
    assert "${OFT_BLOCK_SIZE:-" not in content


def test_lora_regret_launcher_rejects_an_unknown_peft_method():
    content = _lora_regret_launcher_text()
    assert "Unsupported PEFT_METHOD" in content


def test_lora_regret_launcher_guards_full_finetune_against_too_few_gpus():
    """P0's arithmetic: 32 GB + 96 GB/N per GPU under the distributed optimizer
    (forced on in orbit/backends/megatron_utils/arguments.py), so N=1 is 128 GB
    and N=2 is 80 GB before activations. Failing at launch beats OOMing twenty
    minutes into a reserved node."""
    content = _lora_regret_launcher_text()
    assert "ALLOW_SMALL_FULLFT" in content
    assert "GPUS_PER_NODE < 4" in content


def test_lora_regret_launcher_keeps_nan_checks_on():
    """Megatron only offers the negative spelling, and asserts it when
    full_iteration CUDA graphs are on. A silently-NaN arm would read as a bad
    learning rate, so this launcher forgoes the CUDA graph instead."""
    # Comment lines are excluded deliberately: the launcher explains this
    # choice in prose that necessarily names both flags.
    code = [
        line for line in _lora_regret_launcher_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    assert not any("--no-check-for-nan-in-loss-and-grad" in line for line in code)
    assert not any("full_iteration" in line for line in code)
