from miles.utils.mask_utils import MultiTurnLossMaskGenerator


class FakeChatTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}

    def apply_chat_template(
        self,
        messages,
        add_special_tokens=False,
        tokenize=False,
        return_dict=False,
        add_generation_prompt=False,
        tools=None,
    ):
        text = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return [ord(ch) for ch in text]
        return text

    def get_added_vocab(self):
        return {}


class FakeNoChatTemplateTokenizer:
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}

    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("Cannot use chat template functions because tokenizer.chat_template is not set")

    def get_added_vocab(self):
        return {}


def test_response_only_loss_mask_trains_only_final_assistant_response():
    tokenizer = FakeChatTokenizer()
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="response_only")
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Where would a person store soup?"},
        {"role": "assistant", "content": "A. bowl"},
    ]

    token_ids, loss_mask = generator.get_loss_mask(messages)

    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    expected_prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    expected_response_ids = tokenizer("A. bowl", add_special_tokens=False)["input_ids"]
    assert token_ids == expected_prompt_ids + expected_response_ids
    assert loss_mask == [0] * len(expected_prompt_ids) + [1] * len(expected_response_ids)
    assert generator.get_response_lengths([loss_mask]) == [len(expected_response_ids)]


def test_response_only_loss_mask_honors_step_loss_mask_zero():
    tokenizer = FakeChatTokenizer()
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="response_only")
    messages = [
        {"role": "user", "content": "Write a function."},
        {"role": "assistant", "content": "def f(): pass", "step_loss_mask": 0},
    ]

    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert len(token_ids) == len(loss_mask)
    assert set(loss_mask) == {0}


def test_response_only_loss_mask_uses_llama_fallback_when_chat_template_missing():
    tokenizer = FakeNoChatTemplateTokenizer()
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="response_only")
    messages = [
        {"role": "user", "content": "Pick the answer."},
        {"role": "assistant", "content": "A. choice"},
    ]

    token_ids, loss_mask = generator.get_loss_mask(messages)

    expected_prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Pick the answer.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    expected_prompt_ids = tokenizer(expected_prompt, add_special_tokens=False)["input_ids"]
    expected_response_ids = tokenizer("A. choice", add_special_tokens=False)["input_ids"]
    assert token_ids == expected_prompt_ids + expected_response_ids
    assert loss_mask == [0] * len(expected_prompt_ids) + [1] * len(expected_response_ids)
