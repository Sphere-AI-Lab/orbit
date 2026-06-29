from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")

import unittest

try:
    from envpack.envs.sokoban.dataset import (
        SokobanCandidate,
        bucket_name_for_metrics,
        parse_sampling_spec,
        select_candidates_by_bucket,
        split_train_eval_by_bucket,
    )

    _ENVPACK_AVAILABLE = True
except ModuleNotFoundError:
    _ENVPACK_AVAILABLE = False


@unittest.skipUnless(_ENVPACK_AVAILABLE, "envpack Sokoban dataset helpers are not installed")
class EnvpackBuildDatasetTest(unittest.TestCase):
    def test_min_solve_steps_maps_to_level_bucket(self) -> None:
        sampling = parse_sampling_spec(
            {
                "total_train": 10,
                "validation_ratio": 0.1,
                "min_solve_steps": [3, 10],
                "allocation": "capped",
            }
        )

        # Difficulty is binned on min_solve_steps alone; critical_steps is observed,
        # not a binning axis.
        self.assertEqual(
            bucket_name_for_metrics(
                {"solver_status": "solved_within_depth", "min_solve_steps": 5, "critical_steps": 2},
                sampling,
            ),
            "solve_5",
        )
        # Board geometry (dim_room, num_boxes) becomes a task-family bucket prefix.
        self.assertEqual(
            bucket_name_for_metrics(
                {
                    "solver_status": "solved_within_depth",
                    "min_solve_steps": 5,
                    "num_boxes": 1,
                    "dim_room": [6, 6],
                },
                sampling,
            ),
            "6x6_b1_solve_5",
        )
        # Out-of-range difficulty and unsolved boards are dropped.
        self.assertIsNone(
            bucket_name_for_metrics(
                {"solver_status": "solved_within_depth", "min_solve_steps": 2},
                sampling,
            )
        )
        self.assertIsNone(
            bucket_name_for_metrics(
                {"solver_status": "unsolved", "min_solve_steps": 5},
                sampling,
            )
        )

    def test_removed_sampling_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "critical_steps is no longer supported"):
            parse_sampling_spec({"total_train": 10, "min_solve_steps": [3, 10], "critical_steps": [1, 8]})
        with self.assertRaisesRegex(RuntimeError, "buckets is no longer supported"):
            parse_sampling_spec({"total_train": 10, "min_solve_steps": [3, 10], "buckets": [{"name": "a"}]})
        with self.assertRaisesRegex(RuntimeError, "requires min_solve_steps"):
            parse_sampling_spec({"total_train": 10})

    def test_capped_selection_and_validation_split_preserve_counts(self) -> None:
        sampling = parse_sampling_spec(
            {
                "total_train": 4,
                "validation_ratio": 0.5,
                "min_solve_steps": [3, 4],
                "allocation": "capped",
            }
        )
        candidates = [
            _candidate("a1", "solve_3"),
            _candidate("a2", "solve_3"),
            _candidate("a3", "solve_3"),
            _candidate("b1", "solve_4"),
            _candidate("b2", "solve_4"),
            _candidate("b3", "solve_4"),
        ]

        selected = select_candidates_by_bucket(candidates, sampling)
        train, eval_rows = split_train_eval_by_bucket(selected, train_total=4, eval_total=2)

        self.assertEqual(len(train), 4)
        self.assertEqual(len(eval_rows), 2)
        self.assertEqual({row.bucket_name for row in train + eval_rows}, set(selected))


def _candidate(env_uuid: str, bucket_name: str) -> SokobanCandidate:
    return SokobanCandidate(
        env_uuid=env_uuid,
        env_uuid_kind="sokoban_state",
        seed=1,
        env_name="sokoban",
        env_config={},
        profile="vision_free_think_local",
        pool_id="sokoban-vision",
        solver_metrics={
            "solver_status": "solved_within_depth",
            "min_solve_steps": 3,
            "critical_steps": 1,
            "bucket_name": bucket_name,
        },
        bucket_name=bucket_name,
    )


if __name__ == "__main__":
    unittest.main()
