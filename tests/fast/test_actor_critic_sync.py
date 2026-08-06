from argparse import Namespace

import torch
import torch.distributed as dist

from orbit.backends.training_utils.data import sync_actor_critic_data
from tests.fast.dist_utils import init_gloo, run_multiprocess


def _clone_tensors(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.clone() for tensor in tensors]


def _assert_tensors_equal(actual: list[torch.Tensor], expected: list[torch.Tensor], dtype: torch.dtype) -> None:
    assert all(tensor.dtype == dtype for tensor in actual)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)


def _sync_worker(rank: int, world_size: int, port: int) -> None:
    assert world_size == 2
    init_gloo(rank, world_size, port=port)
    try:
        sync_cases = [
            (True, 0.0, False),
            (False, 0.1, False),
            (True, 0.0, True),
        ]
        for logprob_dtype in (torch.bfloat16, torch.float16):
            expected_values = [
                torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32),
                torch.tensor([-1.25, 0.75], dtype=torch.float32),
            ]
            expected_log_probs = [
                torch.tensor([-1.0, -2.0, -3.0], dtype=logprob_dtype),
                torch.tensor([-4.0, -5.0], dtype=logprob_dtype),
            ]
            expected_ref_log_probs = [
                torch.tensor([-1.5, -2.5, -3.5], dtype=logprob_dtype),
                torch.tensor([-4.5, -5.5], dtype=logprob_dtype),
            ]
            for use_rollout_logprobs, kl_coef, use_kl_loss in sync_cases:
                args = Namespace(
                    use_rollout_logprobs=use_rollout_logprobs,
                    kl_coef=kl_coef,
                    use_kl_loss=use_kl_loss,
                    true_on_policy_mode=True,
                    bf16=logprob_dtype == torch.bfloat16,
                    fp16=logprob_dtype == torch.float16,
                )
                log_probs_key = "rollout_log_probs" if use_rollout_logprobs else "log_probs"

                # Rollout log-probs are transported to both roles even when the
                # actor recomputes train-time `log_probs` for PPO.
                rollout_data = {"rollout_log_probs": _clone_tensors(expected_log_probs)}
                if rank == 0:
                    if not use_rollout_logprobs:
                        rollout_data[log_probs_key] = _clone_tensors(expected_log_probs)
                    if kl_coef != 0 or use_kl_loss:
                        rollout_data["ref_log_probs"] = _clone_tensors(expected_ref_log_probs)
                else:
                    rollout_data["values"] = _clone_tensors(expected_values)

                sync_actor_critic_data(args, rollout_data, dist.group.WORLD)

                _assert_tensors_equal(rollout_data[log_probs_key], expected_log_probs, logprob_dtype)
                _assert_tensors_equal(rollout_data["values"], expected_values, torch.float32)
                if kl_coef != 0 or use_kl_loss:
                    _assert_tensors_equal(rollout_data["ref_log_probs"], expected_ref_log_probs, logprob_dtype)
    finally:
        dist.destroy_process_group()


def test_actor_critic_sync_uses_matching_true_on_policy_wire_dtypes() -> None:
    run_multiprocess(_sync_worker)
