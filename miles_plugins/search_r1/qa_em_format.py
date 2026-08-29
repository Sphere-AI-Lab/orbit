# Adapted from Search-R1:
# https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/verl/utils/reward_score/qa_em_format.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0.

import re
import string


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def em_check(prediction: str, golden_answers: str | list[str]) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    return int(any(normalize_answer(answer) == normalized_prediction for answer in golden_answers))


def is_valid_sequence(text: str) -> tuple[bool, str]:
    assistant_match = re.search(r"<\|im_start\|>assistant\s*", text)
    if not assistant_match:
        return False, "Missing assistant marker"

    content = text[assistant_match.end() :]
    for tag in ["think", "search", "information", "answer"]:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    state = "start"

    for part in parts:
        if not part.strip():
            continue

        if re.match(r"</?(?:think|search|information|answer)>", part):
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        elif state not in ["in_think", "in_search", "in_information", "in_answer"]:
            if state in ["start", "after_think", "after_search", "information"] and part.strip():
                return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            return False, f"Unexpected content in state {state}"

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str: str) -> str | None:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if len(matches) <= 1:
        return None
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    return [match.strip() for match in re.findall(r"<information>(.*?)</information>", text, re.DOTALL)]


def is_retrieval_correct(text: str, golden_answers: str | list[str]) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    for block in extract_information_blocks(text):
        normalized_block = normalize_answer(block)
        if any(normalize_answer(answer) in normalized_block for answer in golden_answers):
            return True
    return False


def compute_score_em(
    solution_str: str,
    ground_truth: dict,
    *,
    structure_format_score: float = 0.0,
    final_format_score: float = 0.0,
    retrieval_score: float = 0.0,
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Search-R1 exact-match reward with optional format/retrieval credit."""
    targets = ground_truth["target"]
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = is_valid_format and is_retrieval_correct(solution_str, targets)
    answer = extract_solution(solution_str)

    if answer is None:
        if is_valid_format:
            return structure_format_score + (retrieval_score if retrieval_correct else 0.0)
        return 0.0

    if em_check(answer, targets):
        return score if is_valid_format else score - structure_format_score

    if is_valid_format:
        return structure_format_score + (retrieval_score if retrieval_correct else 0.0)

    return final_format_score or format_score

