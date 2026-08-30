"""Tests for orbit.opd.opd_dump (M1 correctness leg, GPU side of I-5).

``tokens`` here matches the real ``miles.utils.types.Sample.tokens`` field
(full prompt+response token ids) -- the brief's ``response_token_ids`` name
does not exist on the real Sample dataclass, so the record uses the real
attribute name instead, consistently across the dump writer, the compare
CLI, and this test.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from orbit.opd.opd_dump import (
    ENV_LIMIT,
    ENV_PATH,
    dump_teacher_logprob_records,
    maybe_dump_teacher_logprobs,
)

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "tools" / "compare_opd_teacher_logprobs.py"


def _write(path, records):
    dump_teacher_logprob_records(str(path), records)


def _records(lp):
    return [{"rollout": 0, "sample_index": 0, "tokens": [1, 2, 3], "teacher_log_probs": lp}]


def test_dump_appends_jsonl(tmp_path):
    out = tmp_path / "d.jsonl"
    _write(out, _records([-0.1, -0.2, -0.3]))
    _write(out, _records([-0.1, -0.2, -0.3]))
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["teacher_log_probs"] == [-0.1, -0.2, -0.3]


def test_cli_pass_and_fail(tmp_path):
    ref, ok, bad = tmp_path / "r.jsonl", tmp_path / "ok.jsonl", tmp_path / "bad.jsonl"
    _write(ref, _records([-0.1, -0.2, -0.3]))
    _write(ok, _records([-0.1001, -0.2, -0.3]))
    _write(bad, _records([-0.5, -0.2, -0.3]))
    assert subprocess.run([sys.executable, str(CLI), str(ref), str(ok), "--atol", "5e-3"]).returncode == 0
    assert subprocess.run([sys.executable, str(CLI), str(ref), str(bad), "--atol", "5e-3"]).returncode == 1


def test_cli_reports_no_common_keys(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(a, [{"rollout": 0, "sample_index": 0, "tokens": [1, 2, 3], "teacher_log_probs": [-0.1]}])
    _write(b, [{"rollout": 1, "sample_index": 0, "tokens": [1, 2, 3], "teacher_log_probs": [-0.1]}])
    assert subprocess.run([sys.executable, str(CLI), str(a), str(b)]).returncode == 2


def test_cli_reports_token_mismatch_as_hard_error(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(a, [{"rollout": 0, "sample_index": 0, "tokens": [1, 2, 3], "teacher_log_probs": [-0.1]}])
    _write(b, [{"rollout": 0, "sample_index": 0, "tokens": [9, 9, 9], "teacher_log_probs": [-0.1]}])
    assert subprocess.run([sys.executable, str(CLI), str(a), str(b)]).returncode == 2


def test_maybe_dump_is_inert_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_PATH, raising=False)
    out = tmp_path / "should_not_exist.jsonl"
    samples = [SimpleNamespace(tokens=[1, 2, 3], teacher_log_probs=[-0.1, -0.2, -0.3])]
    maybe_dump_teacher_logprobs(0, samples)
    assert not out.exists()


def test_maybe_dump_writes_and_respects_limit(tmp_path, monkeypatch):
    out = tmp_path / "d.jsonl"
    monkeypatch.setenv(ENV_PATH, str(out))
    monkeypatch.setenv(ENV_LIMIT, "1")
    samples = [SimpleNamespace(tokens=[1, 2, 3], teacher_log_probs=[-0.1, -0.2, -0.3])]
    maybe_dump_teacher_logprobs(0, samples)  # rollout 0 < limit 1: dumped
    maybe_dump_teacher_logprobs(1, samples)  # rollout 1 >= limit 1: no-op
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "rollout": 0,
        "sample_index": 0,
        "tokens": [1, 2, 3],
        "teacher_log_probs": [-0.1, -0.2, -0.3],
    }


def test_maybe_dump_skips_samples_without_teacher_log_probs(tmp_path, monkeypatch):
    out = tmp_path / "d.jsonl"
    monkeypatch.setenv(ENV_PATH, str(out))
    samples = [
        SimpleNamespace(tokens=[1, 2, 3], teacher_log_probs=None),
        SimpleNamespace(tokens=[4, 5, 6], teacher_log_probs=[-0.4, -0.5]),
    ]
    maybe_dump_teacher_logprobs(0, samples)
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["sample_index"] == 1
    assert rec["tokens"] == [4, 5, 6]
