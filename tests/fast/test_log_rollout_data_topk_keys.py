"""Regression test for finding 1 (final-review fixes): `log_rollout_data`'s skip-list
must exclude `teacher_topk_ids`/`teacher_topk_logprobs` (opd_topk_loss's retained
teacher transport).

Before the fix, `teacher_topk_ids` (a list of `torch.long` `[R, K]` tensors) fell
through to the generic `val.mean() * cp_size` branch in `log_rollout_data`, and
`.mean()` on an integer tensor raises `RuntimeError: mean(): could not infer output
dtype ... Got: Long` -- crashing every opd_topk_loss run at the first rollout log.
"""

from argparse import Namespace

import torch

from orbit.backends.training_utils import log_utils
from orbit.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _single_process_state() -> None:
    # group=None short-circuits GroupInfo's post-init verification (no real process
    # group needed): gather_log_data is monkeypatched below, so no distributed calls
    # actually happen.
    single = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(intra_dp=single, intra_dp_cp=single, cp=single, tp=single, is_pp_last_stage=True)
    )


def test_log_rollout_data_skips_teacher_topk_keys(monkeypatch):
    _single_process_state()

    captured = {}

    def _fake_gather_log_data(metric_name, args, rollout_id, log_dict):
        captured["log_dict"] = log_dict
        return None

    monkeypatch.setattr(log_utils, "gather_log_data", _fake_gather_log_data)

    args = Namespace(
        ci_test=False,
        log_multi_turn=False,
        log_passrate=False,
        log_correct_samples=False,
        qkv_format="thd",
    )

    rollout_data = {
        "response_lengths": [2, 0],
        "total_lengths": [4, 2],
        "loss_masks": [torch.ones(2, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)],
        "teacher_topk_ids": [
            torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        ],
        "teacher_topk_logprobs": [
            torch.tensor([[-0.1, -0.2], [-0.3, -0.4]], dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        ],
    }

    # Before the fix this raised RuntimeError: mean(): could not infer output dtype
    # for Long input; use input.to(...) to cast to a floating point type.
    log_utils.log_rollout_data(rollout_id=0, args=args, rollout_data=rollout_data)

    assert "log_dict" in captured, "gather_log_data was never called"
    assert "teacher_topk_ids" not in captured["log_dict"]
    assert "teacher_topk_logprobs" not in captured["log_dict"]
