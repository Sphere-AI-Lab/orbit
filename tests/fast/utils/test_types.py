"""Unit tests for Sample per-token state lifecycle."""

from unittest.mock import MagicMock

import numpy
import pytest
import torch
from PIL import Image

from orbit.utils.types import Sample


def _make_sample(
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    log_probs: bool = False,
    loss_mask: bool = False,
    routed_experts: bool = False,
    indexer_topk: bool = False,
) -> Sample:
    """Create a Sample with the given prompt + response token IDs."""
    tokens = prompt_ids + response_ids
    s = Sample(
        tokens=tokens,
        response_length=len(response_ids),
        response="dummy",
    )
    if log_probs:
        s.rollout_log_probs = [-0.1] * len(response_ids)
    if loss_mask:
        s.loss_mask = [1] * len(response_ids)
    if routed_experts:
        # shape: (num_tokens - 1, ...)
        s.rollout_routed_experts = numpy.zeros((len(tokens) - 1, 2, 2), dtype=numpy.int32)
    if indexer_topk:
        # shape: (num_tokens - 1, ...)
        s.rollout_indexer_topk = numpy.zeros((len(tokens) - 1, 2, 3), dtype=numpy.int32)
    return s


def test_reset_for_retry_restores_captured_multimodal_containers_repeatedly():
    prompt_image = Image.new("RGB", (1, 1), "red")
    stale_image = Image.new("RGB", (1, 1), "green")
    sample = Sample(multimodal_inputs={"images": [prompt_image], "tag": "prompt"})

    sample.capture_multimodal_inputs_for_retry()
    sample.multimodal_inputs["images"].append(stale_image)
    sample.reset_for_retry()
    first_restored_list = sample.multimodal_inputs["images"]
    assert first_restored_list == [prompt_image]
    assert first_restored_list[0] is prompt_image

    first_restored_list.append(stale_image)
    sample.reset_for_retry()
    assert sample.multimodal_inputs["images"] == [prompt_image]
    assert sample.multimodal_inputs["images"] is not first_restored_list
    assert sample.multimodal_inputs["tag"] == "prompt"


def test_retry_snapshot_is_not_serialized():
    sample = Sample(multimodal_inputs={"images": []})
    sample.capture_multimodal_inputs_for_retry()

    payload = sample.to_dict()
    restored = Sample.from_dict(payload)

    assert "retry_multimodal_inputs_snapshot" not in payload
    assert restored.retry_multimodal_inputs_snapshot is None


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.decode = lambda ids: "".join(chr(65 + i) for i in ids)
    return tok


class TestStripLastOutputTokens:
    def test_strip_zero_is_noop(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5])
        original_tokens = list(s.tokens)
        s.strip_last_output_tokens(0, tokenizer)
        assert s.tokens == original_tokens
        assert s.response_length == 3

    def test_strip_basic(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5])
        s.strip_last_output_tokens(2, tokenizer)
        assert s.tokens == [1, 2, 3]
        assert s.response_length == 1

    def test_strip_all_response(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5])
        s.strip_last_output_tokens(3, tokenizer)
        assert s.tokens == [1, 2]
        assert s.response_length == 0
        assert s.response == ""

    def test_strip_too_many_raises(self, tokenizer):
        s = _make_sample([1, 2], [3, 4])
        with pytest.raises(AssertionError, match="cannot strip 3 tokens"):
            s.strip_last_output_tokens(3, tokenizer)

    def test_strip_truncates_log_probs(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5], log_probs=True)
        assert len(s.rollout_log_probs) == 3
        s.strip_last_output_tokens(2, tokenizer)
        assert len(s.rollout_log_probs) == 1

    def test_strip_truncates_loss_mask(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5], loss_mask=True)
        assert len(s.loss_mask) == 3
        s.strip_last_output_tokens(1, tokenizer)
        assert len(s.loss_mask) == 2

    def test_strip_truncates_routed_experts(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5], routed_experts=True)
        original_len = len(s.rollout_routed_experts)
        s.strip_last_output_tokens(2, tokenizer)
        assert len(s.rollout_routed_experts) == original_len - 2

    def test_strip_truncates_indexer_topk(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5], indexer_topk=True)
        original_len = len(s.rollout_indexer_topk)
        s.strip_last_output_tokens(2, tokenizer)
        assert len(s.rollout_indexer_topk) == original_len - 2

    def test_strip_updates_response_text(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5])
        s.strip_last_output_tokens(1, tokenizer)
        # response should be re-decoded from the remaining response tokens
        assert s.response == tokenizer.decode(s.tokens[-s.response_length :])

    def test_strip_negative_is_noop(self, tokenizer):
        s = _make_sample([1, 2], [3, 4])
        original_tokens = list(s.tokens)
        s.strip_last_output_tokens(-1, tokenizer)
        assert s.tokens == original_tokens


class TestTeacherTopKTargets:
    def test_validate_accepts_matching_sparse_targets(self):
        sample = _make_sample([1], [2, 3])
        sample.teacher_topk_token_ids = torch.tensor([[10, 11], [12, 13]], dtype=torch.long)
        sample.teacher_topk_log_probs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]], dtype=torch.float32)
        sample.teacher_topk_valid_mask = torch.ones((2, 2), dtype=torch.bool)

        sample.validate()

    def test_validate_requires_ids_log_probs_and_mask_as_a_contract(self):
        sample = _make_sample([1], [2, 3])
        sample.teacher_topk_token_ids = torch.tensor([[10, 11], [12, 13]], dtype=torch.long)

        with pytest.raises(AssertionError, match="must either all be present"):
            sample.validate()

    @pytest.mark.parametrize(
        ("ids", "log_probs", "valid_mask", "error"),
        [
            (
                torch.tensor([10, 11]),
                torch.tensor([-0.1, -0.2]),
                torch.tensor([True, True]),
                r"shape \[T, K\]",
            ),
            (
                torch.tensor([[10, 11], [12, 13]]),
                torch.tensor([[-0.1], [-0.2]]),
                torch.ones((2, 2), dtype=torch.bool),
                "tensor shape mismatch",
            ),
            (
                torch.tensor([[10, 11]]),
                torch.tensor([[-0.1, -0.2]]),
                torch.ones((1, 2), dtype=torch.bool),
                "response dimension",
            ),
        ],
    )
    def test_validate_rejects_invalid_sparse_target_shapes(self, ids, log_probs, valid_mask, error):
        sample = _make_sample([1], [2, 3])
        sample.teacher_topk_token_ids = ids.long()
        sample.teacher_topk_log_probs = log_probs.float()
        sample.teacher_topk_valid_mask = valid_mask.bool()

        with pytest.raises(AssertionError, match=error):
            sample.validate()

    def test_validate_accepts_masked_padding_sentinels(self):
        sample = _make_sample([1], [2, 3])
        sample.teacher_topk_token_ids = torch.tensor([[10, 0], [12, 13]], dtype=torch.long)
        sample.teacher_topk_log_probs = torch.tensor([[-0.1, -torch.inf], [-0.3, -0.4]], dtype=torch.float32)
        sample.teacher_topk_valid_mask = torch.tensor([[True, False], [True, True]], dtype=torch.bool)

        sample.validate()

    def test_validate_rejects_nonfinite_valid_log_probs(self):
        sample = _make_sample([1], [2])
        sample.teacher_topk_token_ids = torch.tensor([[10, 11]], dtype=torch.long)
        sample.teacher_topk_log_probs = torch.tensor([[-0.1, -torch.inf]], dtype=torch.float32)
        sample.teacher_topk_valid_mask = torch.ones((1, 2), dtype=torch.bool)

        with pytest.raises(AssertionError, match="must be finite"):
            sample.validate()

    def test_strip_slices_sparse_targets_by_response_position(self, tokenizer):
        sample = _make_sample([1], [2, 3, 4])
        sample.teacher_topk_token_ids = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.long)
        sample.teacher_topk_log_probs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]], dtype=torch.float32)
        sample.teacher_topk_valid_mask = torch.ones((3, 2), dtype=torch.bool)

        sample.strip_last_output_tokens(1, tokenizer)

        assert sample.teacher_topk_token_ids.tolist() == [[10, 11], [12, 13]]
        torch.testing.assert_close(
            sample.teacher_topk_log_probs,
            torch.tensor([[-0.1, -0.2], [-0.3, -0.4]], dtype=torch.float32),
        )
        assert sample.teacher_topk_valid_mask.tolist() == [[True, True], [True, True]]
        sample.validate()

    def test_reset_for_retry_clears_sparse_targets(self):
        sample = _make_sample([1], [2])
        sample.teacher_log_probs = [-0.3]
        sample.opd_reverse_kl = [0.2]
        sample.teacher_topk_token_ids = torch.tensor([[10, 11]], dtype=torch.long)
        sample.teacher_topk_log_probs = torch.tensor([[-0.1, -0.2]], dtype=torch.float32)
        sample.teacher_topk_valid_mask = torch.ones((1, 2), dtype=torch.bool)

        sample.reset_for_retry()

        assert sample.teacher_log_probs is None
        assert sample.opd_reverse_kl is None
        assert sample.teacher_topk_token_ids is None
        assert sample.teacher_topk_log_probs is None
        assert sample.teacher_topk_valid_mask is None
