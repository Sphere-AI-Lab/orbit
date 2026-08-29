from transformers import AutoTokenizer


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    # return the lengths starting from the first occurrence of 1 to the end of each loss mask
    return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]


class MultiTurnLossMaskGenerator:
    def __init__(self, tokenizer: AutoTokenizer, tokenizer_type: str = "qwen"):
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        if tokenizer_type == "response_only":
            self.system_message_length = 0
            self.gen_token_length = 0
        else:
            self.system_message_length, self.gen_token_length = self.get_system_message_length()

    def get_response_lengths(self, loss_masks: list[list[int]]) -> list[int]:
        return get_response_lengths(loss_masks)

    def find_all_sublist_indices(self, main_list, sublist):
        sublist_len = len(sublist)
        indices = []
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i : i + sublist_len] == sublist:
                indices.append(i)
        return indices

    def get_system_message_length(self) -> tuple[int, int]:
        test_string = "FOR TESTING ONLY"
        test_messages = [
            {"role": "user", "content": test_string},
            {"role": "user", "content": test_string},
        ]
        raw_token_ids = self.tokenizer(test_string, add_special_tokens=False)["input_ids"]
        chat_template_token = self.tokenizer.apply_chat_template(
            test_messages, add_special_tokens=False, tokenize=False
        )
        chat_template_token_ids = self.tokenizer(chat_template_token, add_special_tokens=False)["input_ids"]
        idx_1, idx_2 = self.find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
        gen_token_length = len(
            self.tokenizer.apply_chat_template(
                test_messages, add_special_tokens=False, tokenize=True, return_dict=False, add_generation_prompt=True
            )
        ) - len(chat_template_token_ids)

        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))
        return system_message_length, gen_token_length

    def _format_llama_messages_without_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = False,
    ) -> str:
        parts = ["<|begin_of_text|>"]
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        if add_generation_prompt:
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)

    def _apply_chat_template_or_llama_fallback(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        add_generation_prompt: bool = False,
    ) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
            )
        except (AttributeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "chat_template" not in str(exc):
                raise
            if tools:
                raise ValueError("response_only Llama fallback does not support tools without a chat template") from exc
            return self._format_llama_messages_without_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
            )

    def gen_multi_turn_loss_mask_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        for i, message in enumerate(messages):
            if i == 0:
                message_ids = self.tokenizer.apply_chat_template(
                    [message], tokenize=True, return_dict=False, tools=tools
                )
            else:
                message_ids = self.tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen3(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        # Tokenize the WHOLE conversation exactly once. Qwen3's chat template decides
        # whether to wrap an assistant turn in an empty "<think>\n\n</think>\n\n" block
        # based on whether that turn is the LAST assistant response following the LAST
        # real user turn in the WHOLE conversation. Re-tokenizing a per-message or
        # per-prefix slice in isolation (as an earlier version of this method did, via a
        # synthetic single-user "prefix" message) silently flips that decision: every
        # isolated slice's own last assistant message trivially looks "final" to the
        # template, so it gets the think-wrapper whether or not it actually is the final
        # turn of the real conversation. Locating turn boundaries within a single,
        # complete tokenization avoids the bug entirely.
        all_token_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, tools=tools
        )
        all_loss_masks = [0] * len(all_token_ids)

        # "<|im_start|>assistant\n" rendered on its own: a content-independent marker for
        # where an assistant turn's header ends and its scorable content begins. Rendered
        # as the sole message in a length-1 list so no think-wrapper logic can apply to
        # it (a lone assistant message is never "after" a later user turn); truncated to
        # gen_token_length since an empty-content render also includes the message's own
        # closing "<|im_end|>\n", which is not part of the header.
        header_ids = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": ""}], tokenize=True, return_dict=False
        )[: self.gen_token_length]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        trailing_newline_id = header_ids[-1]

        header_positions = self.find_all_sublist_indices(all_token_ids, header_ids)
        assistant_messages = [message for message in messages if message["role"] == "assistant"]

        if len(header_positions) != len(assistant_messages):
            raise ValueError(
                f"Found {len(header_positions)} '<|im_start|>assistant' header(s) in the "
                f"tokenized conversation but {len(assistant_messages)} assistant message(s) "
                "in `messages`; cannot align loss-mask spans to messages."
            )

        for message, header_pos in zip(assistant_messages, header_positions, strict=True):
            start = header_pos + self.gen_token_length
            end = start
            while end < len(all_token_ids) and all_token_ids[end] != im_end_id:
                end += 1
            if end < len(all_token_ids):
                end += 1  # include <|im_end|>
            if end < len(all_token_ids) and all_token_ids[end] == trailing_newline_id:
                end += 1  # include the "\n" that follows <|im_end|>

            if message.get("step_loss_mask", 1) == 1:
                for k in range(start, min(end, len(all_token_ids))):
                    all_loss_masks[k] = 1

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_llama3(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        # Same single-tokenization strategy as the qwen3 method, and for the same
        # reason: rendering a message in isolation can change what the template
        # emits for it. Llama-3's template has no context-sensitive reasoning
        # wrapper (nothing analogous to Qwen3's <think> block), but it DOES inject
        # an unconditional system block, so per-message rendering would still
        # mis-locate every span after the first.
        all_token_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, tools=tools
        )
        all_loss_masks = [0] * len(all_token_ids)

        # Content-independent marker for where an assistant turn's header ends.
        # Tokenized from the literal string: <|start_header_id|> and
        # <|end_header_id|> are added special tokens, so they are matched during
        # pre-tokenization regardless of add_special_tokens (which governs only
        # bos/eos wrapping).
        header_ids = self.tokenizer(
            "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
        )["input_ids"]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        header_positions = self.find_all_sublist_indices(all_token_ids, header_ids)
        assistant_messages = [m for m in messages if m["role"] == "assistant"]

        if len(header_positions) != len(assistant_messages):
            raise ValueError(
                f"Found {len(header_positions)} assistant header(s) in the tokenized "
                f"conversation but {len(assistant_messages)} assistant message(s) in "
                "`messages`; cannot align loss-mask spans to messages."
            )

        for message, header_pos in zip(assistant_messages, header_positions, strict=True):
            start = header_pos + len(header_ids)
            end = start
            while end < len(all_token_ids) and all_token_ids[end] != eot_id:
                end += 1
            if end < len(all_token_ids):
                end += 1  # <|eot_id|> is a target: the model must learn to stop.
            # NOTE: unlike Qwen's "<|im_end|>\n", Llama-3 emits no newline after
            # <|eot_id|> -- the next "<|start_header_id|>" follows immediately -- so
            # there is deliberately no trailing-newline step here.

            if message.get("step_loss_mask", 1) == 1:
                for k in range(start, min(end, len(all_token_ids))):
                    all_loss_masks[k] = 1

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_distill_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        prompt = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True, tools=tools
        )
        response = messages[-1]["content"]
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_tokens = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        response_length = len(response_tokens)
        token_ids = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * response_length

        if messages[-1].get("step_loss_mask", 1) != 1:
            loss_mask = [0] * len(token_ids)
        return token_ids, loss_mask

    def gen_response_only_loss_mask(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("response_only loss mask requires the final message to be from assistant")

        prompt = self._apply_chat_template_or_llama_fallback(
            messages[:-1], tools=tools, add_generation_prompt=True
        )
        response = messages[-1]["content"]
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_tokens = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        token_ids = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)

        if messages[-1].get("step_loss_mask", 1) != 1:
            loss_mask = [0] * len(token_ids)
        return token_ids, loss_mask

    def get_loss_mask(self, messages: list[dict], tools: list[dict] = None) -> tuple[list[int], list[int]]:
        if self.tokenizer_type == "qwen":
            if "<｜Assistant｜>" in self.tokenizer.get_added_vocab():
                return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)

            return self.gen_multi_turn_loss_mask_qwen(messages, tools)
        elif self.tokenizer_type == "qwen3":
            return self.gen_multi_turn_loss_mask_qwen3(messages, tools)
        elif self.tokenizer_type == "distill_qwen":
            return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)
        elif self.tokenizer_type == "llama3":
            return self.gen_multi_turn_loss_mask_llama3(messages, tools)
        elif self.tokenizer_type == "response_only":
            return self.gen_response_only_loss_mask(messages, tools)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def get_loss_mask_with_multimodal_alignment(
        self, messages: list[dict], input_ids: list[int], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        text = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                text_parts = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                text.append({"role": msg["role"], "content": " ".join(text_parts)})
            else:
                text.append(msg)

        _, loss_mask_text = self.get_loss_mask(text, tools=tools)

        diff = len(input_ids) - len(loss_mask_text)
        assert diff >= 0, (
            f"input_ids (length={len(input_ids)}) is shorter than text loss_mask (length={len(loss_mask_text)}) "
            f"Please check if processor and tokenizer tokenization are consistent."
        )
        loss_mask = [0] * diff + loss_mask_text

        return input_ids, loss_mask

    def get_text_from_loss_mask(self, token_ids: list[int], loss_masks: list[int]) -> list[str]:
        selected_texts = []
        current_tokens = []

        for idx, mask in enumerate(loss_masks):
            if mask == 1:
                current_tokens.append(token_ids[idx])
            elif current_tokens:
                selected_texts.append(self.tokenizer.decode(current_tokens))
                current_tokens = []

        if current_tokens:
            selected_texts.append(self.tokenizer.decode(current_tokens))

        return selected_texts
