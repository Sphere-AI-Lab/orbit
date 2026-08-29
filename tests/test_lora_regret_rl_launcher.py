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


def test_rl_launcher_grades_with_the_verl_math_verifier():
    """`math` dispatches to grade_answer_verl, which extracts the final
    \\boxed{...} from the response itself.

    NOT `boxed_math`, which double-extracts and can never return 1 -- see
    tests/test_lora_regret_reward_grading.py, which asserts the behaviour
    rather than the spelling. NOT `deepscaler`, which returns 0 unless the
    response contains `</think>` or `###Response`, neither of which a
    Llama-3.1 *base* policy emits."""
    assert '--rm-type "${RM_TYPE:-math}"' in _text()
    assert not any("deepscaler" in line for line in _code())
    assert not any("boxed_math" in line for line in _code())


def test_rl_launcher_does_not_wrap_a_base_policy_in_the_instruct_chat_template():
    """The policy is Llama-3.1-8B *base*. The pinned template is Instruct's, so
    applying it conditions the base model on turn-delimiter tokens it was never
    trained to read: the 2026-07-31 probe logged degenerate continuations and
    reward 0 on every rollout. `render_prompt` writes the frame into the jsonl
    instead, so the prompt string reaches the engine unmodified."""
    assert not any("--apply-chat-template" in line for line in _code())


def test_rl_launcher_stops_generation_at_the_frame_the_data_uses():
    """A base policy continues the pattern into a next problem and runs to the
    token cap; a truncated response has lost its \\boxed{...} and grades 0. The
    stop word therefore has to be exactly the frame prepare_data emits."""
    from tools.lora_regret.prepare_data import COMPLETION_STOP

    assert any("--rollout-stop" in line for line in _code())
    # The launcher spells the default with bash ANSI-C quoting; compare against
    # the escaped form so the two definitions cannot drift apart.
    escaped = COMPLETION_STOP.replace("\n", "\\n")
    assert any(f"$'{escaped}'" in line for line in _code())


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


def test_fullft_keeps_train_offload():
    """The contract inverted on 2026-07-31, and this test with it.

    It previously asserted the opposite -- that the `none)` branch passes
    `--no-offload-train` -- because orbit refused `--offload-train` for full
    fine-tuning outright, so every FullFT arm died in argument finalisation
    before a single rollout.

    That refusal is gone: full fine-tuning now offloads gradients and optimizer
    state (parameters stay resident, since `update_weights` pushes them to the
    rollout engine every rollout). With the refusal removed, disabling the
    offload is what breaks the arm rather than what saves it -- in colocate mode
    SGLang cannot resume its paused KV cache, measured at 12.48 GB free against
    16.00 GB of K+V:

        [torch_memory_saver.cpp] cudaError error: 2 (out of memory)
        file=csrc/core.cpp func=resume line=182

    So the flag must be ABSENT from the `none)` branch now.
    """
    none_block = _text().split("    none)", 1)[1].split(";;", 1)[0]
    assert "--no-offload-train" not in none_block, (
        "the none) branch must not disable train offload; full fine-tuning "
        "needs it to share the node with the rollout engine"
    )


def test_the_peft_arms_keep_train_offload():
    """PEFT_METHOD=lora/oft must not disable train offload either: colocate mode
    shares the GPUs with SGLang, and holding training weights resident is what
    the offload exists to avoid. Every arm now keeps it, by different means --
    PEFT offloads the frozen base, FullFT the gradients and optimizer state."""
    text = _text()
    assert "--no-offload-train" not in text, (
        "no branch of this launcher should disable train offload"
    )


def test_rl_launcher_names_each_wandb_run_after_its_arm():
    """The sweep sets WANDB_GROUP to the METHOD, so seven FullFT arms share one
    group. Without an explicit run name the name IS the group, and all seven
    appear as "full" with the learning rate visible only inside each config."""
    content = _text()
    assert '--wandb-run-name "${WANDB_RUN_NAME:-${LAUNCHER_NAME}}"' in content
    assert "--disable-wandb-random-suffix" in content


def test_rl_launcher_can_switch_checkpointing_off_entirely():
    """`SAVE_INTERVAL=` (empty) must drop --save-interval, not pass a large one.
    `should_run_periodic_action` short-circuits on `interval is None` and only
    then checks the final rollout, so any non-None interval still writes one
    checkpoint -- 616 s and 15 GB for a FullFT arm."""
    content = _text()
    assert "SAVE_INTERVAL=${SAVE_INTERVAL-50}" in content, "must use `-`, not `:-`"
    assert 'if [[ -n "${SAVE_INTERVAL}" ]]; then' in content
    assert '--save-interval "${SAVE_INTERVAL:-' not in content
