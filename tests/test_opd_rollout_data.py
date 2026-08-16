"""Regression test for C1: sglang OPD teacher_log_probs must be tensorized +
CP-sliced on the train side inside get_rollout_data.

For the sglang teacher, sample.teacher_log_probs is a list[float]; it is
transferred rollout->train and reaches get_rollout_data as a raw
list[list[float]]. Before the fix, get_rollout_data converted rollout_log_probs
to list[torch.Tensor] via slice_log_prob_with_cp but had no analogous block for
teacher_log_probs, so it stayed a raw list and later crashed with
"'list' object has no attribute 'to'" (ppo_utils.opd_mopd_advantages). This
reproduces at CP=1 (the list->tensor conversion never happens).
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from orbit.backends.training_utils import cp_utils, data


class _Axis:
    def __init__(self, size, rank=0, group=None):
        self.size = size
        self.rank = rank
        self.group = group


def _fake_parallel_state():
    # CP=1, TP=1, single DP rank -- the simplest layout, and the one that
    # currently crashes because the list->tensor conversion is skipped.
    return SimpleNamespace(cp=_Axis(1), tp=_Axis(1), intra_dp=_Axis(1))


@pytest.fixture
def patched(monkeypatch):
    # get_rollout_data hardcodes device=torch.cuda.current_device(). A working
    # CUDA *context* is required — torch.cuda.is_available() is not enough (on
    # login shells it reports True while context creation fails with "device
    # busy or unavailable"), so probe with a real allocation.
    try:
        torch.zeros(1, device="cuda")
    except Exception as e:
        pytest.skip(f"CUDA context unavailable: {e}")
    fake = _fake_parallel_state()
    # slice_log_prob_with_cp (cp_utils) and get_rollout_data (data) each import
    # get_parallel_state from .parallel, so both references must be patched.
    monkeypatch.setattr(data, "get_parallel_state", lambda: fake)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: fake)
    # process_rollout_data would DP-split via ray; return the prepared dict as-is.
    monkeypatch.setattr(data, "process_rollout_data", lambda args, ref, rank, size: ref)


def _base_rollout_data():
    # One sample: total_length=4, response_length=3.
    return {
        "tokens": [[10, 11, 12, 13]],
        "loss_masks": [[1, 1, 1]],
        "total_lengths": [4],
        "response_lengths": [3],
        "rollout_log_probs": [[-0.1, -0.2, -0.3]],
        "teacher_log_probs": [[-1.1, -1.2, -1.3]],
    }


def test_get_rollout_data_tensorizes_teacher_log_probs(patched):
    args = Namespace(qkv_format="thd")
    rollout_data = _base_rollout_data()

    out = data.get_rollout_data(args, rollout_data)

    teacher = out["teacher_log_probs"]
    assert isinstance(teacher, list) and len(teacher) == 1
    # The regression: this element was a raw list[float] before the fix.
    assert isinstance(teacher[0], torch.Tensor)
    assert teacher[0].dtype == torch.float32
    torch.testing.assert_close(
        teacher[0].cpu(), torch.tensor([-1.1, -1.2, -1.3], dtype=torch.float32)
    )


def test_get_rollout_data_no_teacher_key_is_noop(patched):
    # Non-OPD path: no teacher_log_probs present -> key stays absent.
    args = Namespace(qkv_format="thd")
    rollout_data = _base_rollout_data()
    del rollout_data["teacher_log_probs"]

    out = data.get_rollout_data(args, rollout_data)

    assert "teacher_log_probs" not in out


def test_get_rollout_data_leaves_already_tensor_teacher_untouched(patched):
    # Megatron path guard: if teacher_log_probs are already tensors (as produced
    # by compute_log_prob), they must not be re-processed.
    args = Namespace(qkv_format="thd")
    rollout_data = _base_rollout_data()
    existing = torch.tensor([-1.1, -1.2, -1.3], dtype=torch.float32)
    rollout_data["teacher_log_probs"] = [existing]

    out = data.get_rollout_data(args, rollout_data)

    assert out["teacher_log_probs"][0] is existing


# --- _tensorize_cp_sliced_log_probs: CPU-only paths (no CUDA context needed) ---


def test_tensorize_helper_noop_on_empty_list():
    # A DP rank can receive zero samples (rollout batch not divisible by the
    # training DP size); indexing [0] to sniff the element type IndexErrors.
    args = Namespace(qkv_format="thd")
    rollout_data = {"teacher_log_probs": [], "total_lengths": [], "response_lengths": []}

    data._tensorize_cp_sliced_log_probs(args, rollout_data, "teacher_log_probs")

    assert rollout_data["teacher_log_probs"] == []


def test_tensorize_helper_noop_on_absent_key():
    args = Namespace(qkv_format="thd")
    rollout_data = {"total_lengths": [3], "response_lengths": [3]}

    data._tensorize_cp_sliced_log_probs(args, rollout_data, "teacher_log_probs")

    assert "teacher_log_probs" not in rollout_data


def test_tensorize_helper_noop_on_already_tensor_entries():
    # Megatron OPD teacher populates tensors later via compute_log_prob; the
    # helper must leave already-tensorized entries untouched.
    args = Namespace(qkv_format="thd")
    t = torch.tensor([-1.0, -2.0])
    rollout_data = {"teacher_log_probs": [t], "total_lengths": [3], "response_lengths": [2]}

    data._tensorize_cp_sliced_log_probs(args, rollout_data, "teacher_log_probs")

    assert rollout_data["teacher_log_probs"][0] is t


def test_tensorize_helper_leaves_teacher_provenance_as_plain_dicts():
    args = Namespace(qkv_format="thd")
    provenance = {"request_id": "5:request-5"}
    rollout_data = {
        "teacher_log_probs": [],
        "teacher_scoring_provenance": [provenance],
        "total_lengths": [],
        "response_lengths": [],
    }

    data._tensorize_cp_sliced_log_probs(
        args,
        rollout_data,
        "teacher_log_probs",
    )

    assert rollout_data["teacher_scoring_provenance"] == [provenance]
    assert rollout_data["teacher_scoring_provenance"][0] is provenance
    assert type(rollout_data["teacher_scoring_provenance"][0]) is dict
