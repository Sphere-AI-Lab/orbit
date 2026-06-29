from __future__ import annotations

import unittest

from miles_plugins.envpack_adapter.build_env_dataset import (
    _merge_sokoban_capacity_reports,
    _parse_spec,
    _validate_no_env_uuid_overlap,
)

try:
    from envpack.envs.sokoban.dataset import (
        SokobanCandidate,
        bucket_name_for_metrics,
        parse_sampling_spec,
        select_candidates_by_bucket,
        split_train_eval_by_bucket,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional thirdparty checkout
    _ENVPACK_IMPORT_ERROR = exc
else:
    _ENVPACK_IMPORT_ERROR = None


@unittest.skipIf(
    _ENVPACK_IMPORT_ERROR is not None, f"envpack Sokoban dataset helpers unavailable: {_ENVPACK_IMPORT_ERROR}"
)
class EnvpackBuildDatasetTest(unittest.TestCase):
    def test_range_sampling_maps_to_min_solve_level(self) -> None:
        sampling = parse_sampling_spec(
            {
                "total_train": 10,
                "validation_ratio": 0.1,
                "min_solve_steps": [4, 10],
                "allocation": "capped",
            }
        )

        self.assertEqual(
            bucket_name_for_metrics(
                {"solver_status": "solved_within_depth", "min_solve_steps": 5, "critical_steps": 2},
                sampling,
            ),
            "solve_5",
        )
        self.assertEqual(
            bucket_name_for_metrics(
                {
                    "solver_status": "solved_within_depth",
                    "min_solve_steps": 5,
                    "critical_steps": 2,
                    "num_boxes": 2,
                    "dim_room": [7, 7],
                },
                sampling,
            ),
            "7x7_b2_solve_5",
        )
        self.assertIsNone(
            bucket_name_for_metrics(
                {"solver_status": "solved_within_depth", "min_solve_steps": 2, "critical_steps": 1},
                sampling,
            )
        )

    def test_critical_steps_and_buckets_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "critical_steps"):
            parse_sampling_spec({"total_train": 10, "min_solve_steps": [4, 10], "critical_steps": [1, 8]})
        with self.assertRaisesRegex(RuntimeError, "buckets"):
            parse_sampling_spec({"total_train": 10, "min_solve_steps": [4, 10], "buckets": [{"name": "a"}]})

    def test_sampling_defaults_to_full_allocation(self) -> None:
        sampling = parse_sampling_spec({"total_train": 10, "min_solve_steps": [3, 10]})

        self.assertEqual(sampling.allocation, "full")

    def test_capped_selection_water_fills_and_split_preserves_counts(self) -> None:
        sampling = parse_sampling_spec(
            {
                "total_train": 4,
                "validation_ratio": 0.5,
                "min_solve_steps": [4, 6],
                "allocation": "capped",
            }
        )
        candidates = (
            [_candidate(f"a{i}", "solve_4") for i in range(5)]
            + [_candidate(f"b{i}", "solve_5") for i in range(4)]
            + [_candidate("c0", "solve_6")]
        )

        selected = select_candidates_by_bucket(candidates, sampling)

        self.assertEqual(len(selected["solve_6"]), 1)
        self.assertEqual(sum(len(values) for values in selected.values()), 6)

        train, eval_rows = split_train_eval_by_bucket(selected, train_total=4, eval_total=2)

        self.assertEqual(len(train), 4)
        self.assertEqual(len(eval_rows), 2)
        self.assertEqual({row.bucket_name for row in train + eval_rows}, set(selected))

    def test_capacity_reports_are_merged_across_sokoban_families(self) -> None:
        report = _merge_sokoban_capacity_reports(
            sampling_payload={"min_solve_steps": [4, 10]},
            family_reports=[
                {
                    "spec_idx": 0,
                    "pool_id": "sokoban-b1",
                    "capacity_report": {
                        "candidate_seeds": 10,
                        "probed_candidates": 10,
                        "unique_env_uuid": 8,
                        "duplicate_env_uuid_count": 2,
                        "accepted_candidates": 5,
                        "selected_train": 4,
                        "selected_eval": 1,
                        "bucket_available": {"6x6_b1_solve_5": 5},
                        "bucket_selected": {"6x6_b1_solve_5": 5},
                        "bucket_train_counts": {"6x6_b1_solve_5": 4},
                        "bucket_eval_counts": {"6x6_b1_solve_5": 1},
                        "solver_status_counts": {"solved_within_depth": 8},
                    },
                },
                {
                    "spec_idx": 1,
                    "pool_id": "sokoban-b2",
                    "capacity_report": {
                        "candidate_seeds": 20,
                        "probed_candidates": 20,
                        "unique_env_uuid": 18,
                        "duplicate_env_uuid_count": 2,
                        "accepted_candidates": 6,
                        "selected_train": 5,
                        "selected_eval": 1,
                        "bucket_available": {"7x7_b2_solve_5": 6},
                        "bucket_selected": {"7x7_b2_solve_5": 6},
                        "bucket_train_counts": {"7x7_b2_solve_5": 5},
                        "bucket_eval_counts": {"7x7_b2_solve_5": 1},
                        "solver_status_counts": {"solved_within_depth": 18},
                    },
                },
            ],
        )

        self.assertEqual(report["num_env_specs"], 2)
        self.assertEqual(report["candidate_seeds"], 30)
        self.assertEqual(report["selected_train"], 9)
        self.assertEqual(report["bucket_available"], {"6x6_b1_solve_5": 5, "7x7_b2_solve_5": 6})
        self.assertEqual(report["bucket_train_counts"], {"6x6_b1_solve_5": 4, "7x7_b2_solve_5": 5})
        self.assertEqual(report["bucket_eval_counts"], {"6x6_b1_solve_5": 1, "7x7_b2_solve_5": 1})
        self.assertEqual(len(report["families"]), 2)

    def test_concat_rejects_train_eval_uuid_overlap(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "train/eval env_uuid overlap"):
            _validate_no_env_uuid_overlap(
                [_row("same")],
                [_row("same")],
            )

    def test_env_spec_accepts_bucket_prefix_and_sampling_override(self) -> None:
        spec = _parse_spec(
            {
                "name": "Sokoban",
                "n_envs": 8,
                "bucket_prefix": "7x7_b1",
                "sampling": {"total_train": 4},
            },
            0,
        )

        self.assertEqual(spec.bucket_prefix, "7x7_b1")
        self.assertEqual(spec.sampling, {"total_train": 4})


def _candidate(env_uuid: str, bucket_name: str):
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


def _row(env_uuid: str):
    return {"metadata": {"envpack": {"env_uuid": env_uuid}}}


if __name__ == "__main__":
    unittest.main()
