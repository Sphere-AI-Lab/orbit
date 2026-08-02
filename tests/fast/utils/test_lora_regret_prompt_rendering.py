"""How E4's prompts are rendered for a *base* policy.

The campaign fine-tunes `Llama-3.1-8B`, the base checkpoint, not Instruct. Until
2026-08-02 the launcher passed `--apply-chat-template` with the pinned Llama-3.1
Instruct template, so every prompt reached the policy as

    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Cutting Knowledge Date: December 2023
    Today Date: 26 Jul 2024

    <|eot_id|><|start_header_id|>user<|end_header_id|>

    {problem}

    Put your final answer in \\boxed{}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Those control tokens exist in the base vocabulary but the base model was never
trained to condition on them as turn delimiters, so the continuation after the
assistant header is off-distribution. The 2026-07-31 probe logged what that
produces: web-scrape noise ("Back to Index", runs of private-use codepoints),
and where the text was coherent it answered in prose without a \\boxed{}, which
grades 0 regardless.

The renderer here is a plain completion instead: a `Problem:` / `Solution:`
frame, which is ordinary pretraining text for a base model, with the boxed
instruction inside the problem block. It is frozen across FullFT and every LoRA
rank -- comparing arms rendered differently would confound the axis E4 sweeps.
"""

import pytest

from tools.lora_regret.prepare_data import (
    ANSWER_INSTRUCTION,
    COMPLETION_STOP,
    PROMPT_STYLES,
    render_prompt,
)

PROBLEM = "What is 2+2?"


def test_completion_style_frames_the_problem_without_chat_control_tokens():
    prompt = render_prompt(PROBLEM, answer_instruction=ANSWER_INSTRUCTION)
    assert "<|start_header_id|>" not in prompt
    assert "<|eot_id|>" not in prompt
    assert prompt.startswith("Problem:")
    assert PROBLEM in prompt
    # Ends on the cue the model completes, with no trailing space: a trailing
    # space is its own token and splits the first word of the continuation.
    assert prompt.endswith("Solution:")
    assert not prompt.endswith(" ")


def test_completion_style_keeps_the_boxed_instruction():
    """`--rm-type math` requires a \\boxed{...} in the response, so the prompt
    has to ask for one. Without it a correct answer in prose scores 0."""
    prompt = render_prompt(PROBLEM, answer_instruction=ANSWER_INSTRUCTION)
    assert "\\boxed{}" in prompt


def test_the_stop_word_matches_the_frame_the_renderer_emits():
    """A base model continues the pattern, so after its solution it writes the
    next `Problem:` itself. Without a stop word every rollout runs to the token
    cap -- 10.2% of them truncated at 2048 in the probe -- and a truncated
    response has lost its box, so it grades 0 whatever it argued.

    Pins stop word and frame together: renaming the block to `Question:`
    without moving the stop word would silently restore the runaway."""
    prompt = render_prompt(PROBLEM, answer_instruction=ANSWER_INSTRUCTION)
    assert prompt.startswith(COMPLETION_STOP.strip())


def test_raw_style_is_still_available_and_is_the_old_behaviour():
    """Kept so the chat-template path remains expressible: with
    `--apply-chat-template` the renderer must not also frame the text."""
    assert render_prompt(PROBLEM, answer_instruction=ANSWER_INSTRUCTION, style="raw") == (
        PROBLEM + ANSWER_INSTRUCTION
    )


def test_unknown_style_raises_rather_than_silently_falling_back():
    """A typo'd style that silently rendered `raw` would put the whole sweep
    back on the prompt that scores 0, and every arm would still run."""
    with pytest.raises(ValueError, match="prompt style"):
        render_prompt(PROBLEM, answer_instruction=ANSWER_INSTRUCTION, style="complettion")


def test_every_declared_style_renders():
    for style in PROMPT_STYLES:
        assert render_prompt(PROBLEM, answer_instruction=ANSWER_INSTRUCTION, style=style)
