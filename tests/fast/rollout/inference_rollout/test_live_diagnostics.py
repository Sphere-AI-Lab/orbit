from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from miles.rollout.inference_rollout.live_diagnostics import (
    initial_live_log_at,
    maybe_log_all_samples_live_diagnostics,
)


def _live_hook(args, samples, data_source, *, live=False, rollout_id=None):
    args.seen["samples"] = samples
    args.seen["data_source"] = data_source
    args.seen["live"] = live
    args.seen["rollout_id"] = rollout_id
    return {"rollout/pre_filter_solve_rate": 0.25}


class LiveDiagnosticsTest(unittest.TestCase):
    def test_logs_pre_filter_metrics_at_interval(self) -> None:
        seen = {}
        args = SimpleNamespace(
            rollout_all_samples_process_path=f"{__name__}._live_hook",
            rollout_all_samples_live_log_interval=None,
            dynamic_sampling_filter_path="test:filter",
            n_samples_per_prompt=8,
            wandb_always_use_train_step=False,
            seen=seen,
        )
        all_samples = [["group-1"], ["group-2"]]

        with patch("miles.rollout.inference_rollout.live_diagnostics.tracking_utils.log") as log:
            next_log_at = initial_live_log_at(args, target_groups=2)
            next_log_at = maybe_log_all_samples_live_diagnostics(
                args,
                rollout_id=7,
                all_samples=all_samples,
                data_source="source",
                kept_groups=0,
                target_groups=2,
                pending_groups=1,
                next_log_at=next_log_at,
                extra_metrics={"rollout/dynamic_filter/drop_none_solved": 2},
            )

        self.assertEqual(seen["samples"], all_samples)
        self.assertEqual(seen["data_source"], "source")
        self.assertTrue(seen["live"])
        self.assertEqual(seen["rollout_id"], 7)
        self.assertEqual(next_log_at, 4)

        log.assert_called_once()
        logged_args, metrics = log.call_args[0][0], log.call_args[0][1]
        self.assertIs(logged_args, args)
        self.assertEqual(log.call_args.kwargs["step_key"], "rollout/step")
        self.assertEqual(metrics["rollout/step"], 7)
        self.assertEqual(metrics["rollout/pre_filter_solve_rate"], 0.25)
        self.assertEqual(metrics["rollout/refill/completed_prompt_groups"], 2)
        self.assertEqual(metrics["rollout/refill/kept_prompt_groups"], 0)
        self.assertEqual(metrics["rollout/refill/target_prompt_groups"], 2)
        self.assertEqual(metrics["rollout/refill/pending_prompt_groups"], 1)
        self.assertEqual(metrics["rollout/refill/keep_prompt_group_frac"], 0.0)
        self.assertEqual(metrics["rollout/dynamic_filter/drop_none_solved"], 2)


if __name__ == "__main__":
    unittest.main()
