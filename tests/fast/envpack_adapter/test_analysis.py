from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from miles_plugins.envpack_adapter.analysis import summarize_dapo_groups


class EnvpackAnalysisTest(unittest.TestCase):
    def test_summarize_dapo_groups_by_step_and_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for group_idx, solved_count in enumerate((0, 3, 8)):
                for rollout_idx in range(8):
                    _write_record(
                        root
                        / "train"
                        / "step0000"
                        / f"prompt{group_idx:05d}_rollout{rollout_idx:02d}"
                        / "record.json",
                        step=0,
                        group_index=group_idx,
                        bucket="6x6_b1_solve_3",
                        solved=rollout_idx < solved_count,
                    )

            rows = summarize_dapo_groups(root)

        overall = _row(rows, 0, "_overall")
        self.assertEqual(overall["groups"], 3)
        self.assertEqual(overall["none_solved"], 1)
        self.assertEqual(overall["mixed"], 1)
        self.assertEqual(overall["all_solved"], 1)
        self.assertAlmostEqual(overall["dapo_keep_rate"], 1 / 3)
        self.assertAlmostEqual(overall["solve_rate"], 11 / 24)

        bucket = _row(rows, 0, "6x6_b1_solve_3")
        self.assertEqual(bucket["mixed"], 1)


def _write_record(path: Path, *, step: int, group_index: int, bucket: str, solved: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ids": {"step": step, "group_index": group_index},
        "env": {"bucket_name": bucket},
        "outcome": {"traj_success": solved},
    }
    path.write_text(json.dumps(record), encoding="utf-8")


def _row(rows, step: int, bucket: str):
    for row in rows:
        if row["step"] == step and row["bucket"] == bucket:
            return row
    raise AssertionError(f"row not found: step={step} bucket={bucket}")


if __name__ == "__main__":
    unittest.main()
