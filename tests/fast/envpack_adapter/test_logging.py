from __future__ import annotations

import unittest
from types import SimpleNamespace

from miles_plugins.envpack_adapter.logging import (
    ALL_SAMPLES_PROCESS_PATH,
    add_all_sample_rollout_metrics,
    add_bucket_solve_rate_metrics,
    add_prompt_group_distribution_metrics,
    add_rollout_prompt_group_summary_metrics,
    build_all_sample_rollout_metrics,
    log_rollout_data,
    process_all_samples,
)


class _Sample:
    def __init__(self, metadata, group_index=None):
        self.metadata = metadata
        self.group_index = group_index


class EnvpackLoggingTest(unittest.TestCase):
    def test_bucket_solve_rate_uses_explicit_success_signal(self) -> None:
        samples = [
            _Sample({"envpack": {"env_name": "sokoban", "bucket_name": "b1_solve_5", "success": True}}),
            _Sample({"envpack": {"env_name": "sokoban", "bucket_name": "b1_solve_5", "success": False}}),
            _Sample({"envpack": {"env_name": "sokoban", "bucket_name": "b2_solve_5", "success": True}}),
            _Sample({"envpack": {"env_name": "sokoban", "bucket_name": "b2_solve_5"}}),
        ]
        log_dict = {}

        add_bucket_solve_rate_metrics(samples, log_dict, prefix="envpack_eval_bucket")

        self.assertEqual(log_dict["envpack_eval_bucket/sokoban/b1_solve_5/solve_rate"], 0.5)
        self.assertEqual(log_dict["envpack_eval_bucket/sokoban/b1_solve_5/count"], 2)
        self.assertEqual(log_dict["envpack_eval_bucket/sokoban/b2_solve_5/solve_rate"], 1.0)
        self.assertEqual(log_dict["envpack_eval_bucket/sokoban/b2_solve_5/count"], 1)

    def test_prompt_group_distribution_counts_zero_mixed_and_all_solved_groups(self) -> None:
        samples = []
        for group_index, solved_count in enumerate((0, 3, 8)):
            for rollout_index in range(8):
                samples.append(
                    _Sample(
                        {
                            "envpack": {
                                "env_name": "sokoban",
                                "bucket_name": "6x6_b1_solve_3",
                                "success": rollout_index < solved_count,
                            }
                        },
                        group_index=group_index,
                    )
                )
        log_dict = {}

        counts = add_prompt_group_distribution_metrics(samples, log_dict, prefix="envpack_prompt_groups")
        add_rollout_prompt_group_summary_metrics(log_dict, counts)

        base = "envpack_prompt_groups/sokoban/_overall"
        self.assertEqual(log_dict[f"{base}/groups"], 3)
        self.assertEqual(log_dict[f"{base}/none_solved_groups"], 1)
        self.assertEqual(log_dict[f"{base}/mixed_groups"], 1)
        self.assertEqual(log_dict[f"{base}/all_solved_groups"], 1)
        self.assertAlmostEqual(log_dict[f"{base}/none_solved_frac"], 1 / 3)
        self.assertAlmostEqual(log_dict[f"{base}/mixed_frac"], 1 / 3)
        self.assertAlmostEqual(log_dict[f"{base}/all_solved_frac"], 1 / 3)

        bucket_base = "envpack_prompt_groups/sokoban/6x6_b1_solve_3"
        self.assertEqual(log_dict[f"{bucket_base}/groups"], 3)

        self.assertAlmostEqual(log_dict["rollout/all_unsolved_prompt_frac"], 1 / 3)
        self.assertAlmostEqual(log_dict["rollout/all_solved_prompt_frac"], 1 / 3)
        self.assertEqual(log_dict["rollout/all_unsolved_prompts"], 1)
        self.assertEqual(log_dict["rollout/all_solved_prompts"], 1)
        self.assertNotIn("rollout/prompt_groups", log_dict)

    def test_all_sample_rollout_metrics_use_pre_filter_groups(self) -> None:
        kept_mixed_group = [
            _Sample(
                {"envpack": {"env_name": "sokoban", "bucket_name": "6x6_b1_solve_3", "success": rollout_index < 3}},
                group_index=1,
            )
            for rollout_index in range(8)
        ]
        dropped_none_group = [
            _Sample(
                {"envpack": {"env_name": "sokoban", "bucket_name": "6x6_b1_solve_3", "success": False}},
                group_index=2,
            )
            for _ in range(8)
        ]
        dropped_all_group = [
            _Sample(
                {"envpack": {"env_name": "sokoban", "bucket_name": "6x6_b1_solve_3", "success": True}},
                group_index=3,
            )
            for _ in range(8)
        ]
        log_dict = {}

        add_all_sample_rollout_metrics([kept_mixed_group, dropped_none_group, dropped_all_group], log_dict)

        self.assertAlmostEqual(log_dict["rollout/all_unsolved_prompt_frac"], 1 / 3)
        self.assertAlmostEqual(log_dict["rollout/all_solved_prompt_frac"], 1 / 3)
        self.assertEqual(log_dict["rollout/all_unsolved_prompts"], 1)
        self.assertEqual(log_dict["rollout/all_solved_prompts"], 1)
        self.assertAlmostEqual(log_dict["rollout/pre_filter_solve_rate"], 11 / 24)
        bucket_base = "envpack_prompt_groups/sokoban/6x6_b1_solve_3"
        self.assertEqual(log_dict[f"{bucket_base}/none_solved_groups"], 1)
        self.assertEqual(log_dict[f"{bucket_base}/mixed_groups"], 1)
        self.assertEqual(log_dict[f"{bucket_base}/all_solved_groups"], 1)
        pre_filter_bucket = "envpack_rollout_pre_filter_bucket/sokoban/6x6_b1_solve_3"
        self.assertAlmostEqual(log_dict[f"{pre_filter_bucket}/solve_rate"], 11 / 24)

    def test_build_all_sample_rollout_metrics_returns_pre_filter_metrics(self) -> None:
        none_group = [
            _Sample(
                {"envpack": {"env_name": "sokoban", "bucket_name": "6x6_b1_solve_3", "success": False}},
                group_index=1,
            )
            for _ in range(8)
        ]
        mixed_group = [
            _Sample(
                {"envpack": {"env_name": "sokoban", "bucket_name": "6x6_b1_solve_3", "success": rollout_index < 4}},
                group_index=2,
            )
            for rollout_index in range(8)
        ]

        metrics = build_all_sample_rollout_metrics([none_group, mixed_group])

        self.assertAlmostEqual(metrics["rollout/all_unsolved_prompt_frac"], 1 / 2)
        self.assertEqual(metrics["rollout/all_unsolved_prompts"], 1)
        self.assertEqual(metrics["rollout/all_solved_prompts"], 0)
        self.assertAlmostEqual(metrics["rollout/pre_filter_solve_rate"], 4 / 16)
        self.assertNotIn("rollout/step", metrics)

    def test_live_process_all_samples_returns_metrics_without_dumping(self) -> None:
        none_group = [
            _Sample(
                {"envpack": {"env_name": "sokoban", "bucket_name": "6x6_b1_solve_3", "success": False}},
                group_index=1,
            )
            for _ in range(8)
        ]
        mixed_group = [
            _Sample(
                {"envpack": {"env_name": "sokoban", "bucket_name": "6x6_b1_solve_3", "success": rollout_index < 2}},
                group_index=2,
            )
            for rollout_index in range(8)
        ]

        metrics = process_all_samples(
            SimpleNamespace(),
            [none_group, mixed_group],
            data_source=None,
            live=True,
            rollout_id=0,
            n_samples_per_group=8,
        )

        assert metrics is not None
        self.assertAlmostEqual(metrics["rollout/all_unsolved_prompt_frac"], 1 / 2)
        self.assertEqual(metrics["rollout/all_unsolved_prompts"], 1)
        self.assertEqual(metrics["rollout/all_solved_prompts"], 0)
        self.assertAlmostEqual(metrics["rollout/pre_filter_solve_rate"], 2 / 16)

    def test_rollout_log_hook_keeps_train_solve_rate_on_filtered_samples(self) -> None:
        kept_samples = [
            _Sample(
                {"envpack": {"env_name": "sokoban", "bucket_name": "6x6_b1_solve_3", "success": rollout_index < 3}},
                group_index=1,
            )
            for rollout_index in range(8)
        ]
        args = SimpleNamespace(rollout_all_samples_process_path=ALL_SAMPLES_PROCESS_PATH)
        log_dict = {}

        handled = log_rollout_data(
            rollout_id=0,
            args=args,
            samples=kept_samples,
            rollout_extra_metrics=log_dict,
            rollout_time=1.0,
        )

        self.assertFalse(handled)
        self.assertAlmostEqual(log_dict["rollout/solve_rate"], 3 / 8)
        self.assertAlmostEqual(log_dict["rollout/kept_solve_rate"], 3 / 8)
        self.assertAlmostEqual(log_dict["envpack_rollout_bucket/sokoban/6x6_b1_solve_3/solve_rate"], 3 / 8)
        self.assertNotIn("rollout/all_unsolved_prompt_frac", log_dict)
        self.assertNotIn("envpack_prompt_groups/sokoban/6x6_b1_solve_3/groups", log_dict)


if __name__ == "__main__":
    unittest.main()
