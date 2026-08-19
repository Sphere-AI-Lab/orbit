from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils import data as data_utils
from miles.backends.training_utils import log_utils


def test_true_on_policy_rollout_logprob_dtype_follows_training_precision():
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=True, bf16=True, fp16=False)) is torch.bfloat16
    )
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=True, bf16=False, fp16=True)) is torch.float16
    )
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=False, bf16=True, fp16=False)) is torch.float32
    )


def test_true_on_policy_log_checker_passes_when_values_and_dtype_match(monkeypatch):
    captured = {}
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        cp=SimpleNamespace(size=1),
        is_pp_last_stage=True,
    )

    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(
        log_utils,
        "gather_log_data",
        lambda metric_name, args, rollout_id, log_dict, **kwargs: captured.setdefault("log_dict", log_dict),
    )

    rollout_data = {
        "tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1], dtype=torch.int32)],
        "log_probs": [torch.tensor([-13.25, -13.5], dtype=torch.bfloat16)],
        "rollout_log_probs": [torch.tensor([-13.25, -13.5], dtype=torch.bfloat16)],
        "teacher_topk_token_ids": [torch.tensor([[10, 11], [12, 13]], dtype=torch.long)],
        "teacher_topk_log_probs": [torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])],
        "teacher_topk_valid_mask": [torch.ones((2, 2), dtype=torch.bool)],
    }

    log_utils.log_rollout_data(
        1,
        Namespace(
            ci_test=True,
            ci_disable_logprobs_checker=False,
            true_on_policy_mode=True,
            qkv_format="thd",
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        ),
        rollout_data,
    )

    assert captured["log_dict"]["log_probs"] == captured["log_dict"]["rollout_log_probs"]
    assert "teacher_topk_token_ids" not in captured["log_dict"]
    assert "teacher_topk_log_probs" not in captured["log_dict"]
    assert "teacher_topk_valid_mask" not in captured["log_dict"]


def test_opd_dagger_targets_move_to_training_device_with_stable_dtypes():
    rollout_data = {
        "teacher_topk_token_ids": [torch.tensor([[10, 0]], dtype=torch.int32)],
        "teacher_topk_log_probs": [torch.tensor([[-0.1, -torch.inf]], dtype=torch.float64)],
        "teacher_topk_valid_mask": [torch.tensor([[1, 0]], dtype=torch.int32)],
    }

    data_utils._move_opd_dagger_targets_to_device(rollout_data, torch.device("cpu"))

    assert rollout_data["teacher_topk_token_ids"][0].dtype is torch.long
    assert rollout_data["teacher_topk_log_probs"][0].dtype is torch.float32
    assert rollout_data["teacher_topk_valid_mask"][0].dtype is torch.bool
    assert rollout_data["teacher_topk_valid_mask"][0].tolist() == [[True, False]]


def test_rollout_opd_kl_statistics_report_k1_k2_k3_min_mean_max(monkeypatch):
    parallel_state = SimpleNamespace(cp=SimpleNamespace(size=1))
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)

    rollout_data = {
        "total_lengths": [4],
        "response_lengths": [3],
        "loss_masks": [torch.tensor([1, 1, 0], dtype=torch.int32)],
        # The masked value must not participate in exp() for k3.
        "opd_reverse_kl": [torch.tensor([0.2, -0.4, float("inf")], dtype=torch.float32)],
    }
    metrics, reductions = log_utils._compute_rollout_kl_statistics(
        Namespace(qkv_format="thd", opd_log_prob_top_k=0),
        rollout_data,
        cp_size=1,
    )

    k1 = torch.tensor([0.2, -0.4], dtype=torch.float64)
    k2 = 0.5 * k1.square()
    k3 = torch.expm1(-k1) + k1
    for name, values in (("k1", k1), ("k2", k2), ("k3", k3)):
        assert metrics[f"opd_kl/{name}/mean"] == pytest.approx(values.mean().item())
        assert metrics[f"opd_kl/{name}/min"] == pytest.approx(values.min().item())
        assert metrics[f"opd_kl/{name}/max"] == pytest.approx(values.max().item())
        assert reductions[f"opd_kl/{name}/min"] == "min"
        assert reductions[f"opd_kl/{name}/max"] == "max"
    assert not any(key.startswith("kl/") for key in metrics)


def test_rollout_train_kl_statistics_report_mismatch_without_a_reference(monkeypatch):
    """`rollout_train_kl/*` = KL(rollout engine || trainer recompute): the
    training/inference mismatch, emitted on ordinary RL runs (no ref model)."""
    parallel_state = SimpleNamespace(cp=SimpleNamespace(size=1))
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)

    rollout_data = {
        "total_lengths": [4],
        "response_lengths": [3],
        "loss_masks": [torch.tensor([1, 1, 0], dtype=torch.int32)],
        "rollout_log_probs": [torch.tensor([-1.0, -2.0, -9.0], dtype=torch.float32)],
        "log_probs": [torch.tensor([-1.2, -1.6, -3.0], dtype=torch.float32)],
    }
    metrics, reductions = log_utils._compute_rollout_kl_statistics(
        Namespace(qkv_format="thd", opd_log_prob_top_k=0),
        rollout_data,
        cp_size=1,
    )

    # d = sampled - recomputed over active tokens only
    k1 = torch.tensor([0.2, -0.4], dtype=torch.float64)
    k2 = 0.5 * k1.square()
    k3 = torch.expm1(-k1) + k1
    for name, values in (("k1", k1), ("k2", k2), ("k3", k3)):
        assert metrics[f"rollout_train_kl/{name}/mean"] == pytest.approx(values.mean().item(), rel=1e-6)
        assert metrics[f"rollout_train_kl/{name}/min"] == pytest.approx(values.min().item(), rel=1e-6)
        assert metrics[f"rollout_train_kl/{name}/max"] == pytest.approx(values.max().item(), rel=1e-6)
        assert reductions[f"rollout_train_kl/{name}/min"] == "min"
        assert reductions[f"rollout_train_kl/{name}/max"] == "max"
    # no reference model -> no policy/ref group; mismatch group must not squat on it
    assert not any(key.startswith("kl/") for key in metrics)


def test_rollout_kl_statistics_do_not_relabel_legacy_topk_scalar(monkeypatch):
    parallel_state = SimpleNamespace(cp=SimpleNamespace(size=1))
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    rollout_data = {
        "total_lengths": [2],
        "response_lengths": [1],
        "loss_masks": [torch.tensor([1], dtype=torch.int32)],
        "opd_reverse_kl": [torch.tensor([0.2])],
    }

    assert log_utils._compute_rollout_kl_statistics(
        Namespace(qkv_format="thd", opd_log_prob_top_k=2),
        rollout_data,
        cp_size=1,
    ) == ({}, {})


def test_rollout_kl_statistics_are_reusable_for_policy_reference_kl(monkeypatch):
    parallel_state = SimpleNamespace(cp=SimpleNamespace(size=1))
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    rollout_data = {
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1], dtype=torch.int32)],
        "log_probs": [torch.tensor([-0.2, -0.5])],
        "ref_log_probs": [torch.tensor([-0.4, -0.6])],
    }

    metrics, _ = log_utils._compute_rollout_kl_statistics(
        Namespace(qkv_format="thd", opd_log_prob_top_k=0),
        rollout_data,
        cp_size=1,
    )

    assert metrics["kl/k1/mean"] == pytest.approx(0.15)
    assert metrics["kl/k1/min"] == pytest.approx(0.1)
    assert metrics["kl/k1/max"] == pytest.approx(0.2)


def test_rollout_policy_ref_kl_uses_behavior_log_probs_when_configured(monkeypatch):
    parallel_state = SimpleNamespace(cp=SimpleNamespace(size=1))
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    rollout_data = {
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1], dtype=torch.int32)],
        "log_probs": [torch.tensor([-1.2, -1.5])],
        "rollout_log_probs": [torch.tensor([-0.3, -0.6])],
        "ref_log_probs": [torch.tensor([-0.4, -0.6])],
    }

    metrics, _ = log_utils._compute_rollout_kl_statistics(
        Namespace(qkv_format="thd", opd_log_prob_top_k=0, use_rollout_logprobs=True),
        rollout_data,
        cp_size=1,
    )

    assert metrics["kl/k1/mean"] == pytest.approx(0.05)
    assert metrics["kl/k1/min"] == pytest.approx(0.0)
    assert metrics["kl/k1/max"] == pytest.approx(0.1)


def test_rollout_kl_statistics_keep_policy_ref_and_opd_separate(monkeypatch):
    parallel_state = SimpleNamespace(cp=SimpleNamespace(size=1))
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    rollout_data = {
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1], dtype=torch.int32)],
        "log_probs": [torch.tensor([-0.2, -0.5])],
        "ref_log_probs": [torch.tensor([-0.4, -0.6])],
        "opd_reverse_kl": [torch.tensor([-0.7, 0.3])],
    }

    metrics, _ = log_utils._compute_rollout_kl_statistics(
        Namespace(qkv_format="thd", opd_log_prob_top_k=0),
        rollout_data,
        cp_size=1,
    )

    assert metrics["kl/k1/mean"] == pytest.approx(0.15)
    assert metrics["opd_kl/k1/mean"] == pytest.approx(-0.2)
    assert metrics["kl/k3/mean"] != pytest.approx(metrics["opd_kl/k3/mean"])


def test_gathered_metric_reduction_uses_global_min_and_max():
    gathered = [
        {
            "kl/k1/mean": 0.2,
            "kl/k1/min": -0.3,
            "kl/k1/max": 0.7,
            "opd_kl/k1/mean": 0.6,
            "opd_kl/k1/min": -0.9,
            "opd_kl/k1/max": 1.1,
        },
        {
            "kl/k1/mean": 0.4,
            "kl/k1/min": -0.8,
            "kl/k1/max": 0.5,
            "opd_kl/k1/mean": 0.2,
            "opd_kl/k1/min": -0.4,
            "opd_kl/k1/max": 0.8,
        },
    ]
    reduced = log_utils.reduce_gathered_log_dict(
        gathered,
        dp_size=2,
        reduction_by_key={
            "kl/k1/min": "min",
            "kl/k1/max": "max",
            "opd_kl/k1/min": "min",
            "opd_kl/k1/max": "max",
        },
    )

    assert reduced["kl/k1/mean"] == pytest.approx(0.3)
    assert reduced["kl/k1/min"] == -0.8
    assert reduced["kl/k1/max"] == 0.7
    assert reduced["opd_kl/k1/mean"] == pytest.approx(0.4)
    assert reduced["opd_kl/k1/min"] == -0.9
    assert reduced["opd_kl/k1/max"] == 1.1


def test_multi_turn_metrics_emit_compact_interaction_section(monkeypatch):
    captured = {}
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        is_pp_last_stage=True,
    )

    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)

    def fake_gather(metric_name, args, rollout_id, log_dict, reduction_by_key=None):
        captured["metric_name"] = metric_name
        captured["rollout_id"] = rollout_id
        captured["log_dict"] = log_dict
        captured["reduction_by_key"] = reduction_by_key

    monkeypatch.setattr(log_utils, "gather_log_data", fake_gather)

    log_utils.log_multi_turn_data(
        rollout_id=7,
        args=Namespace(rollout_max_response_len=8),
        rollout_data={
            "loss_masks": [
                torch.tensor([1, 0, 1], dtype=torch.int32),
                torch.tensor([1, 0], dtype=torch.int32),
            ],
            "round_number": [1, 3],
        },
    )

    assert captured["metric_name"] == "interaction"
    assert captured["rollout_id"] == 7
    assert captured["log_dict"] == pytest.approx(
        {
            "raw_tokens/max": 3.0,
            "length_cap_ratio": 0.0,
            "observation_tokens/mean": 1.0,
            "observation_token_ratio": 5 / 12,
            "rounds/mean": 2.0,
            "rounds/max": 3.0,
            "rounds/min": 1.0,
        }
    )
    assert captured["reduction_by_key"] == {
        "raw_tokens/max": "max",
        "rounds/max": "max",
        "rounds/min": "min",
    }


def test_opd_dagger_train_metrics_keep_an_independent_top_level_section():
    metrics = log_utils.log_train_step(
        args=Namespace(),
        loss_dict={
            "loss": 1.5,
            "opd_dagger/explicit_ce": 0.7,
            "opd_dagger/teacher_topk_mass": 0.8,
        },
        grad_norm=2.0,
        rollout_id=3,
        step_id=0,
        num_steps_per_rollout=1,
        should_log=False,
    )

    assert metrics["train/loss"] == 1.5
    assert metrics["train/grad_norm"] == 2.0
    assert metrics["opd_dagger/explicit_ce"] == 0.7
    assert metrics["opd_dagger/teacher_topk_mass"] == 0.8
    assert "train/opd_dagger/explicit_ce" not in metrics
    assert metrics["train/step"] == 3
