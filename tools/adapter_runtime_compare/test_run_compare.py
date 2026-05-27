import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.adapter_runtime_compare import run_compare


class ParseOutputTest(unittest.TestCase):
    def test_json_output_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            log_path = run_dir / "run.log"
            log_path.write_text("perf 0: {'perf/rollout_time': 1.0}\n")
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "branch": "runtime",
                        "model": "qwen3_4b",
                        "precision": "bf16",
                        "peft": "oft",
                        "mode": "sync",
                        "repeat": 0,
                        "gpu_ids": [0],
                        "num_rollout": 1,
                        "eval_enabled": False,
                        "env": {"RUN_LOG": str(log_path)},
                    }
                )
            )
            (run_dir / "status.json").write_text(json.dumps({"status": "ok", "returncode": 0, "wall_s": 1.0}))
            output = root / "summary.json"

            with contextlib.redirect_stdout(io.StringIO()):
                rc = run_compare.parse_runs(argparse.Namespace(path=str(root), format="json", output=str(output)))

            self.assertEqual(rc, 0)
            rows = json.loads(output.read_text())
            self.assertEqual(rows[0]["run_id"], "run")


class CaseConfigTest(unittest.TestCase):
    def test_qwen25_dense_lora_uses_csgmv_backend(self) -> None:
        case = next(case for case in run_compare.CASES if case.key == "qwen25_05b_bf16_lora")

        self.assertEqual(case.lora_backend, "csgmv")


if __name__ == "__main__":
    unittest.main()
