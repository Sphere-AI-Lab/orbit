from argparse import Namespace

import numpy as np
import pytest
import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils import data as data_utils
from miles.orbit.opd import teacher_lm_head as teacher_lm_head_module
from miles.backends.training_utils.data import DataIterator, get_batch
from miles.backends.training_utils.loss import opd_jsd_loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.backends.training_utils.loss_hub.math_utils import apply_opd_kl_to_advantages

_ROLLOUT_LOG_PROBS = torch.tensor([-0.2, -1.3, -0.7, -2.1, -0.4, -3.2, -1.8])
_OPD_VALUES = torch.tensor([0.1, 0.9, -0.3, 1.7, -1.1, 0.4, 2.3])
_JSD_CP_CHECKPOINT_KEY = "<test-dsv4-padded-cp-jsd>"


def _parallel_state(*, cp_size: int, cp_rank: int = 0) -> ParallelState:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    return ParallelState(
        intra_dp=trivial_group,
        intra_dp_cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        tp=trivial_group,
        # upstream's ParallelState gained required pp/ep/etp/indep_dp groups; trivial here.
        pp=trivial_group,
        ep=trivial_group,
        etp=trivial_group,
        indep_dp=trivial_group,
    )


def _args(qkv_format: str, *, dsv4: bool = False) -> Namespace:
    return Namespace(
        # upstream's get_rollout_data now consults args.enable_witness and,
        # on the bshd path, args.compress_ratios.
        enable_witness=False,
        compress_ratios=[],
        qkv_format=qkv_format,
        data_pad_size_multiplier=16,
        allgather_cp=False,
        peft_variant="dsv4" if dsv4 else "standard",
        dsv4_cp_chunk_size_multiple=4,
        true_on_policy_mode=False,
        bf16=False,
        fp16=False,
    )


def _load_rollout_data(
    monkeypatch: pytest.MonkeyPatch,
    *,
    qkv_format: str,
    cp_size: int,
    cp_rank: int,
    opd_key: str,
    dsv4: bool = False,
) -> dict:
    parallel_state = _parallel_state(cp_size=cp_size, cp_rank=cp_rank)
    rollout_data = {
        "tokens": [list(range(11))],
        "loss_masks": [[1] * 7],
        "total_lengths": [11],
        "response_lengths": [7],
        "rollout_log_probs": [_ROLLOUT_LOG_PROBS.tolist()],
        # Raw list[float], mirroring the wire format from the sglang OPD teacher --
        # dev's _tensorize_cp_sliced_log_probs no-ops on already-tensor values (the
        # megatron OPD teacher populates tensors *later*, past this point), so an
        # already-tensor seed here would skip CP slicing entirely.
        opd_key: [_OPD_VALUES.tolist()],
    }

    # upstream's process_rollout_data/get_rollout_data now return
    # (rollout_data, object_store_get_result); only the first element is under test.
    monkeypatch.setattr(data_utils, "process_rollout_data", lambda *args, **kwargs: (rollout_data, object()))
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    loaded_rollout_data, _store_get_result = data_utils.get_rollout_data(_args(qkv_format, dsv4=dsv4), object())
    return loaded_rollout_data


@pytest.mark.parametrize("opd_key", ["teacher_log_probs", "opd_reverse_kl"])
@pytest.mark.parametrize(
    ("qkv_format", "cp_size", "cp_rank", "expected_indices"),
    [
        ("thd", 1, 0, [0, 1, 2, 3, 4, 5, 6]),
        ("thd", 2, 0, [6]),
        ("thd", 2, 1, [0, 1, 2, 3, 4, 5]),
        ("bshd", 2, 0, [0]),
        ("bshd", 2, 1, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_sglang_opd_response_fields_follow_rollout_log_prob_cp_slice(
    monkeypatch: pytest.MonkeyPatch,
    opd_key: str,
    qkv_format: str,
    cp_size: int,
    cp_rank: int,
    expected_indices: list[int],
) -> None:
    rollout_data = _load_rollout_data(
        monkeypatch,
        qkv_format=qkv_format,
        cp_size=cp_size,
        cp_rank=cp_rank,
        opd_key=opd_key,
    )

    expected_indices_tensor = torch.tensor(expected_indices)
    torch.testing.assert_close(
        rollout_data["rollout_log_probs"][0],
        _ROLLOUT_LOG_PROBS[expected_indices_tensor],
    )
    torch.testing.assert_close(
        rollout_data[opd_key][0],
        _OPD_VALUES[expected_indices_tensor],
    )
    assert rollout_data[opd_key][0].dtype == torch.float32
    assert rollout_data[opd_key][0].device.type == "cpu"

    advantages = [torch.ones(len(expected_indices), dtype=torch.float32)]
    student_log_probs = [rollout_data[opd_key][0] + 0.25]
    apply_opd_kl_to_advantages(
        0.5,
        rollout_data,
        advantages,
        student_log_probs,
    )

    if opd_key == "teacher_log_probs":
        torch.testing.assert_close(advantages[0], torch.full_like(advantages[0], 0.875))
    else:
        torch.testing.assert_close(
            advantages[0],
            1.0 - 0.5 * rollout_data[opd_key][0],
        )


@pytest.mark.parametrize("opd_key", ["teacher_log_probs", "opd_reverse_kl"])
@pytest.mark.parametrize(
    ("cp_rank", "expected_indices"),
    [
        (0, [0]),
        (1, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_dsv4_padded_thd_opd_fields_follow_rollout_log_prob_cp_slice(
    monkeypatch: pytest.MonkeyPatch,
    opd_key: str,
    cp_rank: int,
    expected_indices: list[int],
) -> None:
    # DSV4 aligns total_length=11 to max_seq_len=16 for CP=2. The padding
    # changes the mirrored THD chunks and therefore the response-token owner:
    # rank 0 owns response index 0, rank 1 owns indices 1..6.
    rollout_data = _load_rollout_data(
        monkeypatch,
        qkv_format="thd",
        cp_size=2,
        cp_rank=cp_rank,
        opd_key=opd_key,
        dsv4=True,
    )

    assert rollout_data["max_seq_lens"] == [16]
    expected_indices_tensor = torch.tensor(expected_indices)
    torch.testing.assert_close(
        rollout_data["rollout_log_probs"][0],
        _ROLLOUT_LOG_PROBS[expected_indices_tensor],
    )
    torch.testing.assert_close(
        rollout_data[opd_key][0],
        _OPD_VALUES[expected_indices_tensor],
    )
    assert rollout_data[opd_key][0].dtype == torch.float32


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_dsv4_padded_thd_teacher_hidden_states_align_with_actual_jsd_logits(
    monkeypatch: pytest.MonkeyPatch,
    cp_rank: int,
) -> None:
    parallel_state = _parallel_state(cp_size=2, cp_rank=cp_rank)
    set_parallel_state(parallel_state)

    hidden = np.arange(7 * 3, dtype=np.float32).reshape(7, 3) / 10
    rollout_data = {
        "tokens": [list(range(11))],
        "loss_masks": [[1] * 7],
        "total_lengths": [11],
        "response_lengths": [7],
        "rollout_log_probs": [_ROLLOUT_LOG_PROBS.tolist()],
        "teacher_hidden_states": [hidden],
    }
    # upstream's process_rollout_data returns (rollout_data, object_store_get_result).
    monkeypatch.setattr(data_utils, "process_rollout_data", lambda *args, **kwargs: (rollout_data, object()))
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    args = _args("thd", dsv4=True)
    args.opd_jsd_beta = 0.0
    args.rollout_temperature = 1.0
    args.opd_log_prob_min_clamp = -1e30
    args.opd_loss_max_clamp = 1e30
    args.opd_jsd_pointwise_clip = None
    args.opd_log_topk_overlap = False
    args.opd_topk_overlap_ks = []
    args.use_kl_loss = False
    args.teacher_hf_checkpoint = _JSD_CP_CHECKPOINT_KEY
    args.log_probs_chunk_size = -1
    args.vocab_size = 3

    batch, _store_get_result = data_utils.get_rollout_data(args, object())
    batch["unconcat_tokens"] = [torch.as_tensor(batch["tokens"][0])]

    # Rows 3..9 predict the seven response tokens.  Their real-vocabulary
    # logits exactly match the teacher reconstruction; a large padded column
    # verifies that both CP ranks exclude it from the JSD normalizer.
    full_logits = torch.zeros(11, 4)
    full_logits[3:10, :3] = torch.from_numpy(hidden)
    full_logits[:, 3] = 25.0
    local_logits = cp_utils.slice_with_cp(full_logits, 0.0, "thd", max_seq_len=16)
    local_logits = local_logits.unsqueeze(0).requires_grad_(True)

    teacher_lm_head_module._TEACHER_LM_HEAD_CACHE[_JSD_CP_CHECKPOINT_KEY] = torch.eye(3)
    teacher_lm_head_module._SHARDED.add(_JSD_CP_CHECKPOINT_KEY)
    try:
        loss, _ = opd_jsd_loss_function(args, batch, local_logits, lambda value: value.sum())
        loss.backward()
    finally:
        teacher_lm_head_module._TEACHER_LM_HEAD_CACHE.pop(_JSD_CP_CHECKPOINT_KEY, None)
        teacher_lm_head_module._SHARDED.discard(_JSD_CP_CHECKPOINT_KEY)

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)
    torch.testing.assert_close(local_logits.grad[..., 3], torch.zeros_like(local_logits.grad[..., 3]), atol=0, rtol=0)


# --- get_batch threads teacher_topk_ids/teacher_topk_logprobs through -------
#
# Gate-discovered defect: the megatron forward_step's get_batch(...) key list
# carried "teacher_hidden_states" but not "teacher_topk_ids"/"teacher_topk_logprobs",
# so opd_topk_loss's KeyError: 'teacher_topk_ids' surfaced only once training
# actually reached loss_function. This exercises the real get_batch/DataIterator
# path (cp_size=1, qkv_format="thd") with a synthetic 4-sample rollout split into
# two micro-batches, asserting both keys survive and stay aligned to the right
# sample per micro-batch. The single hard `.cuda()` call inside get_batch's thd
# cu_seqlens path (independent of torch.cuda.current_device) is monkeypatched to
# stay on CPU, mirroring this file's existing torch.cuda.current_device patch.


def test_get_batch_threads_teacher_topk_keys_with_micro_batch_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    parallel_state = _parallel_state(cp_size=1)
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)

    # 4 samples, response_lengths 3/2/1/2, each already tensorized (mirroring what
    # get_rollout_data's _tensorize_cp_sliced_log_probs does before get_batch runs).
    teacher_topk_ids = [
        torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long),
        torch.tensor([[7, 8], [9, 10]], dtype=torch.long),
        torch.tensor([[11, 12]], dtype=torch.long),
        torch.tensor([[13, 14], [15, 16]], dtype=torch.long),
    ]
    teacher_topk_logprobs = [
        torch.tensor([[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]]),
        torch.tensor([[-0.7, -0.8], [-0.9, -1.0]]),
        torch.tensor([[-1.1, -1.2]]),
        torch.tensor([[-1.3, -1.4], [-1.5, -1.6]]),
    ]
    rollout_data = {
        "tokens": [torch.arange(7), torch.arange(5), torch.arange(4), torch.arange(6)],
        "loss_masks": [torch.ones(3), torch.ones(2), torch.ones(1), torch.ones(2)],
        "total_lengths": [7, 5, 4, 6],
        "response_lengths": [3, 2, 1, 2],
        "max_seq_lens": [7, 5, 4, 6],
        "teacher_topk_ids": teacher_topk_ids,
        "teacher_topk_logprobs": teacher_topk_logprobs,
    }

    keys = [
        "tokens",
        "total_lengths",
        "response_lengths",
        "loss_masks",
        "teacher_topk_ids",
        "teacher_topk_logprobs",
        "max_seq_lens",
    ]
    # Same as model.py's forward_step: micro_batch_size=2 -> batch 1 gets samples
    # [0, 1], batch 2 gets samples [2, 3].
    iterator = DataIterator(rollout_data, micro_batch_size=2)

    batch1 = get_batch(iterator, keys, pad_multiplier=16, qkv_format="thd", allgather_cp=False)
    assert "teacher_topk_ids" in batch1
    assert "teacher_topk_logprobs" in batch1
    torch.testing.assert_close(batch1["teacher_topk_ids"], teacher_topk_ids[0:2])
    torch.testing.assert_close(batch1["teacher_topk_logprobs"], teacher_topk_logprobs[0:2])

    batch2 = get_batch(iterator, keys, pad_multiplier=16, qkv_format="thd", allgather_cp=False)
    assert "teacher_topk_ids" in batch2
    assert "teacher_topk_logprobs" in batch2
    torch.testing.assert_close(batch2["teacher_topk_ids"], teacher_topk_ids[2:4])
    torch.testing.assert_close(batch2["teacher_topk_logprobs"], teacher_topk_logprobs[2:4])
