from __future__ import annotations

import unittest

from miles_plugins.envpack_adapter.logging import add_bucket_solve_rate_metrics


class _Sample:
    def __init__(self, metadata):
        self.metadata = metadata


class EnvpackLoggingTest(unittest.TestCase):
    def test_bucket_solve_rate_uses_explicit_success_signal(self) -> None:
        samples = [
            _Sample(
                {"envpack": {"env_name": "sokoban", "bucket_name": "b1_solve_5"}, "vagen": {"traj_success": True}}
            ),
            _Sample(
                {"envpack": {"env_name": "sokoban", "bucket_name": "b1_solve_5"}, "vagen": {"traj_success": False}}
            ),
            _Sample(
                {"envpack": {"env_name": "sokoban", "bucket_name": "b2_solve_5"}, "vagen": {"traj_success": True}}
            ),
            _Sample({"envpack": {"env_name": "sokoban", "bucket_name": "b2_solve_5"}}),
        ]
        log_dict = {}

        add_bucket_solve_rate_metrics(samples, log_dict, prefix="envpack_eval_bucket")

        self.assertEqual(log_dict["envpack_eval_bucket/sokoban/b1_solve_5/solve_rate"], 0.5)
        self.assertEqual(log_dict["envpack_eval_bucket/sokoban/b1_solve_5/count"], 2)
        self.assertEqual(log_dict["envpack_eval_bucket/sokoban/b2_solve_5/solve_rate"], 1.0)
        self.assertEqual(log_dict["envpack_eval_bucket/sokoban/b2_solve_5/count"], 1)


if __name__ == "__main__":
    unittest.main()
