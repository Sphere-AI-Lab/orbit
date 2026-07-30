"""Contract for the LoRA-without-regret RL launcher (prerequisite P5, drives E4).

E4 decides claim C5 -- "LoRA matches FullFT under policy gradient even at rank
1, with a wider band of performant LRs". Every assertion here pins something
that would silently change what C5 measures, rather than the launcher's
cosmetics.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RL_LAUNCHER = REPO_ROOT / "examples" / "high_precision" / "run-llama3_1-8b-bf16-rl-math-gsm8k.sh"


def _text() -> str:
    return RL_LAUNCHER.read_text(encoding="utf-8")


def _code() -> list[str]:
    """Non-comment lines only. The launcher documents each choice in prose that
    necessarily names the alternative it rejected, so a bare substring check
    against the whole file would match the explanation and not the flag."""
    return [line for line in _text().splitlines() if not line.lstrip().startswith("#")]


def test_rl_launcher_exists_and_is_standalone():
    assert RL_LAUNCHER.is_file()
    content = _text()
    assert 'source "${ORBIT_ROOT}/scripts/lib/launcher.sh"' in content
    assert 'source "${SCRIPT_DIR}/' not in content


def test_rl_launcher_drives_train_py_not_train_async():
    """train_async.py refuses --eval-nll-data by design: its loop overlaps
    next-rollout generation with current-rollout training, so "the weights at
    the moment of measurement" is undefined. The same objection applies to any
    weights-referenced measurement, so this campaign uses the synchronous
    loop."""
    code = _code()
    assert any("${ORBIT_ROOT}/train.py" in line for line in code)
    assert not any("train_async.py" in line for line in code)


def test_rl_launcher_uses_thirty_two_samples_per_problem():
    """The post's setting. It is also what makes the GRPO-style baseline a
    per-problem mean rather than noise."""
    content = _text()
    assert "N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-32}" in content
    assert '--n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"' in content


def test_rl_launcher_uses_grpo_centering_with_no_kl_penalty_by_default():
    """The post runs plain policy gradient with importance sampling and
    GRPO-like centering. A KL penalty is an extra force on the update whose
    strength interacts with the learning rate -- exactly the axis E4 sweeps --
    so it defaults off and is opt-in."""
    content = _text()
    assert "--advantage-estimator grpo" in content
    assert '--kl-loss-coef "${KL_LOSS_COEF:-0.0}"' in content
    assert '--entropy-coef "${ENTROPY_COEF:-0.0}"' in content


def test_rl_launcher_grades_with_the_boxed_math_verifier():
    """rm_hub dispatches `boxed_math` to extract_boxed_answer + grade_answer_verl.
    `deepscaler` is the wrong choice here: it returns 0 unless the response
    contains `</think>` or `###Response`, neither of which a Llama-3.1 *base*
    policy emits."""
    assert '--rm-type "${RM_TYPE:-boxed_math}"' in _text()
    assert not any("deepscaler" in line for line in _code())


def test_rl_launcher_keeps_the_blogs_optimizer_protocol():
    content = _text()
    assert '--lr-decay-style "${LR_DECAY_STYLE:-constant}"' in content
    assert '--weight-decay "${WEIGHT_DECAY:-0.0}"' in content


def test_rl_launcher_dispatches_lora_oft_and_full_finetune():
    content = _text()
    assert 'case "${PEFT_METHOD}" in' in content
    assert "--peft-method lora" in content
    assert "--peft-method oft" in content
    assert "${OFT_BLOCK_SIZE:?" in content


def test_rl_launcher_reaches_rank_one():
    """E4-2's rank-1 arm is the claim's whole point, so nothing in the launcher
    may floor or round the rank."""
    content = _text()
    assert '--lora-rank "${LORA_RANK:-256}"' in content
    assert '--lora-a-init-method "${LORA_A_INIT_METHOD:-kaiming}"' in content
    assert '--lora-alpha "${LORA_ALPHA:-32}"' in content


def test_rl_launcher_pins_the_llama31_chat_template():
    """Llama-3.1-8B *base* ships no chat_template, so load_tokenizer raises
    before training starts (prerequisite P2)."""
    content = _text()
    assert "orbit/utils/chat_template_utils/templates/llama3.1_pinned.jinja" in content


def test_rl_launcher_ties_rollout_seed_to_seed():
    content = _text()
    assert "ROLLOUT_SEED=${ROLLOUT_SEED:-${SEED}}" in content


def test_rl_launcher_measures_accuracy_not_held_out_nll():
    """E4-3 reads validation-accuracy curves. Held-out NLL is not the metric
    for an RL arm: the policy's own distribution shifts, so NLL on a fixed
    reference set stops being comparable across arms."""
    code = _code()
    assert any("--eval-prompt-data" in line for line in code)
    assert not any("--eval-nll-data" in line for line in code)


SFT_LAUNCHER = REPO_ROOT / "examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh"


def _sft_text() -> str:
    return SFT_LAUNCHER.read_text(encoding="utf-8")


def test_both_launchers_read_the_gpu_floor_from_the_environment():
    """The hardcoded `< 4` is right for Llama-3.1-8B and wrong for every other
    model: Qwen3-0.6B FullFT is 9.6 GB and fits on one card. The registry
    computes the floor; the launcher must not second-guess it."""
    for text in (_text(), _sft_text()):
        assert "MIN_GPUS_FULLFT:-4" in text
        assert "GPUS_PER_NODE < 4" not in text
        assert 'GPUS_PER_NODE < MIN_GPUS_FULLFT' in text


def test_sft_launcher_uses_the_no_colon_form_for_the_chat_template():
    """Empty must mean "omit the flag" (Qwen3 ships its own template), while
    unset must mean "use the pinned Llama one". The colon form collapses those
    two into one, which is the LABEL_KEY bug, one flag over."""
    text = _sft_text()
    assert "${CHAT_TEMPLATE_PATH-" in text
    assert "${CHAT_TEMPLATE_PATH:-" not in text


def test_sft_launcher_still_defaults_to_the_pinned_llama_template():
    """Llama-3.1-8B base ships no chat template at all. A run with none applied
    would train on raw concatenated text."""
    assert "llama3.1_pinned.jinja" in _sft_text()
