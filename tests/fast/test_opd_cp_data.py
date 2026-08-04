from argparse import Namespace

import pytest
import torch

from orbit.backends.training_utils import cp_utils
from orbit.backends.training_utils import data as data_utils
from orbit.backends.training_utils.parallel import GroupInfo, ParallelState
from orbit.utils.ppo_utils import apply_opd_kl_to_advantages

_ROLLOUT_LOG_PROBS = torch.tensor([-0.2, -1.3, -0.7, -2.1, -0.4, -3.2, -1.8])
_OPD_VALUES = torch.tensor([0.1, 0.9, -0.3, 1.7, -1.1, 0.4, 2.3])


def _parallel_state(*, cp_size: int, cp_rank: int = 0) -> ParallelState:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    return ParallelState(
        intra_dp=trivial_group,
        intra_dp_cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        tp=trivial_group,
    )


def _args(qkv_format: str) -> Namespace:
    return Namespace(
        qkv_format=qkv_format,
        data_pad_size_multiplier=16,
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

    monkeypatch.setattr(data_utils, "process_rollout_data", lambda *args, **kwargs: rollout_data)
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    return data_utils.get_rollout_data(_args(qkv_format), object())


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
