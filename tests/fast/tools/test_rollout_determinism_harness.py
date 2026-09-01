import importlib.util
import sys
from pathlib import Path

import pytest


_HARNESS_PATH = Path(__file__).resolve().parents[3] / "tools" / "rollout_determinism_harness.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("rollout_determinism_harness", _HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_rejects_num_sequences_that_cannot_vary_batch_composition(monkeypatch, capsys):
    harness = _load_harness()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollout_determinism_harness.py",
            "--base-url",
            "http://127.0.0.1:30700",
            "--hf-checkpoint",
            "/model",
            "--prompts",
            "/prompts.jsonl",
            "--num-sequences",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        harness.main()

    assert exc_info.value.code == 2
    assert "num-sequences must be at least 2" in capsys.readouterr().err
