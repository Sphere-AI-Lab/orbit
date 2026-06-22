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
