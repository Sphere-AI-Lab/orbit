# Adapter-First Experiments — Phase 0 + Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish Phase 0 (smoke-qualify every launcher the program's Phases 1–2 need) and produce Phase 1's three headline systems deliverables: the A1 sync-cost scaling curve, the A2 rollout-throughput timeline, and the M1 teacher-cost collapse table.

**Architecture:** CPU-side code work first (Tasks 1–7: harness repair, missing launchers, analyzers, dump/compare tooling, the R-2 recipe port) — all verifiable without GPUs via `pytest`, `bash -n`, and `ORBIT_DRY_RUN_ARGV=1` argv dry runs. GPU work (Tasks 8–11) is **USER-RUN**: each such task ends in exact command blocks for the user and a follow-up CPU analysis step for the executor. Nothing in this plan trains to convergence; every GPU run is a short qualification or timing run.

**Tech Stack:** bash launchers over `scripts/lib/launcher.sh`, Python 3 stdlib + matplotlib for analysis, pytest for tests. All Python runs under the workspace venv.

**Spec:** `docs/plans/2026-08-17-adapter-first-experiments-design.md` (this plan implements its Phase 0 remainder and Phase 1: experiments A1, A2, M1; recipe gap R-2; plus two small gaps found while planning — the harness's broken 30B case scripts and the missing adapter-swap OPD smoke).

## Global Constraints

- Repo root: `/lustre/fast/fast/zqiu/clthegoat-orbit/orbit`, branch `orbit-main`. All relative paths below are from this root.
- Environment (required before ANY python/launcher command, including dry runs): `source /lustre/fast/fast/zqiu/clthegoat-orbit/uv_env_build/activate.sh` (venv → CUDA_HOME → orbit loader → PYTHONPATH, in that order).
- CPU tests: `python -m pytest <file> -v` under that venv. Run them yourself; report actual output.
- **GPU commands are USER-RUN.** Never launch training, Ray, or anything touching a GPU. Prepare the exact command block (wrapped in `codexlog NAME ...` — it tees to `/lustre/home/zqiu/log/NAME.log`), present it, and stop until the user reports results.
- Commits: single short generic sentence, conventional prefix matching `git log` style (`feat(tools):`, `fix(tools):`, `docs(plans):` …). No AI attribution trailers of any kind. Commit via HEREDOC. Never push.
- Smoke assets (0.5B): HF `/lustre/fast/fast/zqiu/orbit_env_build/models/Qwen2.5-0.5B-Instruct`, Megatron `/lustre/fast/fast/zqiu/orbit_env_build/megatron_checkpoints/Qwen2.5-0.5B-Instruct-torchdist`, data `/lustre/fast/fast/zqiu/orbit_env_build/data/{gsm8k_agentic_train_64.jsonl,math_test_200.jsonl}`.
- 3B assets (validated): HF `/fast/groups/ei-slm/hf_models/Qwen2.5-3B-Instruct`, Megatron `/lustre/fast/fast/zqiu/clthegoat-orbit/ppo_critic_benchmark_models/Qwen2.5-3B-Instruct_torch_dist`, train data `/lustre/fast/fast/zqiu/clthegoat-orbit/ppo_critic_benchmark_data/openr1_49990/train.jsonl`.
- Metric key names (verified in `orbit/backends/megatron_utils/update_weight/sync_metrics.py`): `perf/update_weights_time`, `perf/update_weights_pause_time`, `perf/update_weights_payload_bytes`, `perf/update_weights_payload_num_tensors`, `perf/update_weights_num_chunks`. Log metric lines match `run_compare.METRIC_RE`: `perf <rollout>: {…}` with a Python-literal dict payload.
- Every launcher supports `ORBIT_DRY_RUN_ARGV=1` (prints the python argv and exits 0 before Ray) — this is the CPU-side qualification lever.
- This cluster's `env.sh` sets `ORBIT_PEFT_ADAPTER_TRANSPORT=cpu_gather`; async-topology adapter sync is NCCL regardless. A1 outputs must state transport per point (spec requirement).

---

### Task 1: Repair the comparison harness for this workspace (broken 30B cases + env shim)

The harness `tools/adapter_runtime_compare/run_compare.py` references `examples/high_precision/run-qwen3-30b-a3b-bf16-math-oft.sh` and `...-math-lora.sh`, which do not exist (the real 30B launchers are the `openr1` family). It also resolves its python from `<env_root>/.venv/bin`, but this workspace's venv lives at `uv_env_build/venv`.

**Files:**
- Modify: `tools/adapter_runtime_compare/run_compare.py` (CASES entries for `qwen3_30b`)
- Test: `tools/adapter_runtime_compare/test_run_compare.py`

**Interfaces:**
- Produces: every `Case.script` (and `Case.fullft_script` when set) is a path that exists under the repo root; the workspace env shim directory `/lustre/fast/fast/zqiu/clthegoat-orbit/harness-env/.venv` → symlink to `uv_env_build/venv`. Tasks 8–9 invoke the harness with `ORBIT_COMPARE_RUNTIME_ROOT=/lustre/fast/fast/zqiu/clthegoat-orbit/orbit` and `ORBIT_COMPARE_RUNTIME_ENV=/lustre/fast/fast/zqiu/clthegoat-orbit/harness-env`.

- [ ] **Step 1: Write the failing test** — append to `tools/adapter_runtime_compare/test_run_compare.py`:

```python
def test_case_scripts_exist():
    from tools.adapter_runtime_compare import run_compare

    missing = []
    for case in run_compare.CASES:
        for attr in ("script", "fullft_script"):
            rel = getattr(case, attr, None)
            if rel and not (run_compare.REPO_ROOT / rel).exists():
                missing.append(f"{case.model}/{case.precision}: {rel}")
    assert not missing, f"CASES reference missing launchers: {missing}"
```

- [ ] **Step 2: Run it, expect failure on the two 30B math scripts**

Run: `cd /lustre/fast/fast/zqiu/clthegoat-orbit/orbit && python -m pytest tools/adapter_runtime_compare/test_run_compare.py::test_case_scripts_exist -v`
Expected: FAIL listing `run-qwen3-30b-a3b-bf16-math-oft.sh` and `run-qwen3-30b-a3b-bf16-math-lora.sh`.

- [ ] **Step 3: Repoint the two `qwen3_30b` CASES entries** at the launchers that exist: `examples/high_precision/run-qwen3-30b-a3b-bf16-openr1-oft-b32.sh` (or the plain `-oft.sh` if present — pick whichever `ls examples/high_precision/ | grep 30b-a3b-bf16-openr1` shows as the OFT default) and `run-qwen3-30b-a3b-bf16-openr1-lora.sh`. Do not change gpu counts or extra_env. Before committing, open the chosen launchers and confirm they source `scripts/lib/launcher.sh` and require `HF_CKPT`/`MEGATRON_LOAD`/`TRAIN_JSONL` via env like the 4B family (the harness supplies env; hardcoded data paths would be a blocker — if hardcoded, override-ability via env is the fix, applied in the same commit).

- [ ] **Step 4: Full test file green**

Run: `python -m pytest tools/adapter_runtime_compare/test_run_compare.py -v`
Expected: PASS (all tests, including pre-existing registry/arm regressions).

- [ ] **Step 5: Create the env shim and validate plan + dry-run end-to-end (CPU)**

```bash
mkdir -p /lustre/fast/fast/zqiu/clthegoat-orbit/harness-env
ln -sfn /lustre/fast/fast/zqiu/clthegoat-orbit/uv_env_build/venv \
        /lustre/fast/fast/zqiu/clthegoat-orbit/harness-env/.venv
cd /lustre/fast/fast/zqiu/clthegoat-orbit/orbit
export ORBIT_COMPARE_RUNTIME_ROOT=$PWD
export ORBIT_COMPARE_RUNTIME_ENV=/lustre/fast/fast/zqiu/clthegoat-orbit/harness-env
python tools/adapter_runtime_compare/run_compare.py plan --branches runtime --profile main
python tools/adapter_runtime_compare/run_compare.py run  --branches runtime --profile pilot --dry-run
```

Expected: `plan` prints waves for every case with no missing-launcher error; `run --dry-run` exits 0. If dry-run demands env the harness does not set (e.g. a checkpoint path for a rung), record the exact missing variable — it becomes part of Task 8/9's command blocks, not a code change.

- [ ] **Step 6: Commit**

```bash
git add tools/adapter_runtime_compare/run_compare.py tools/adapter_runtime_compare/test_run_compare.py
git commit -m "fix(tools): point 30B compare cases at launchers that exist"
```

---

### Task 2: Complete the A1 rungs — 3B case and full-FT arms for 0.5B/3B/30B

A1's full-model-broadcast arm needs a full-FT async launcher per rung; only 4B has one (`run-qwen3-4b-instruct-2507-bf16-math-fullft-async.sh`, from I-3). The design doc also puts 3B on the A1 x-axis, but the harness has no 3B case.

**Files:**
- Create: `examples/high_precision/run-qwen2_5-0_5b-bf16-math-fullft-async.sh`
- Create: `examples/high_precision/run-qwen2_5-3b-bf16-math-fullft-async.sh`
- Create: `examples/high_precision/run-qwen3-30b-a3b-bf16-openr1-fullft-async.sh`
- Modify: `tools/adapter_runtime_compare/run_compare.py` (add `qwen25_3b` OFT case; set `fullft_script` on the 0.5B-OFT, 3B, and 30B-OFT cases)
- Test: `tools/adapter_runtime_compare/test_run_compare.py`

**Interfaces:**
- Produces: harness model keys `qwen25_05b`, `qwen25_3b`, `qwen3_4b`, `qwen3_30b`, each with modes `sync,async,async_db,async_fullft` runnable. Task 9 selects them via `--models`.

- [ ] **Step 1: Extract the I-3 transformation.** Run `diff examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft-async.sh examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-fullft-async.sh`. The diff is the exact oft→fullft delta (expected: `LAUNCHER_NAME`, `SAVE_DIR`, `PEFT_ARGS` emptied or `--peft-method none`, possibly optimizer/batch tweaks). Record it.

- [ ] **Step 2: Write the failing registry test** — append to `test_run_compare.py`:

```python
def test_a1_rungs_have_fullft_arms():
    from tools.adapter_runtime_compare import run_compare

    by_key = {(c.model, c.precision, c.peft): c for c in run_compare.CASES}
    for key in [("qwen25_05b", "bf16", "oft"), ("qwen25_3b", "bf16", "oft"),
                ("qwen3_4b", "bf16", "oft"), ("qwen3_30b", "bf16", "oft")]:
        case = by_key.get(key)
        assert case is not None, f"missing A1 case {key}"
        assert case.fullft_script, f"A1 case {key} has no fullft_script"
        assert (run_compare.REPO_ROOT / case.fullft_script).exists()
```

Run: `python -m pytest tools/adapter_runtime_compare/test_run_compare.py::test_a1_rungs_have_fullft_arms -v` — Expected: FAIL.

- [ ] **Step 3: Create the three fullft-async launchers** by applying the Step-1 delta to the corresponding OFT sync launchers of each model (`run-qwen2_5-0_5b-bf16-math-oft.sh`, `run-qwen2_5-3b-bf16-math-oft.sh`, and the 30B OFT launcher chosen in Task 1) — same model-args plugin, same data contract, `ORBIT_ENTRYPOINT=train_async.py`, disjoint actor/rollout GPUs copied from each model's async twin where one exists (0.5B/3B have no async twin: copy the 4B async launcher's resource block and scale `GPUS_PER_NODE` to 1 for 0.5B, 2 for 3B; 30B uses 4+4). Every new launcher must end with `source "${ORBIT_ROOT}/scripts/lib/launcher.sh"`.

- [ ] **Step 4: Add the `qwen25_3b` case + `fullft_script` fields** in CASES, copying the shape of the existing 0.5B OFT entry (script `examples/high_precision/run-qwen2_5-3b-bf16-math-oft.sh`, `extra_env={"REQUIRE_MEGATRON_LOAD": "1"}`, gpu counts by analogy: total 4, 2+2 async).

- [ ] **Step 5: CPU validation — argv dry run of each new launcher**

```bash
export HF_CKPT=/lustre/fast/fast/zqiu/orbit_env_build/models/Qwen2.5-0.5B-Instruct
export MEGATRON_LOAD=/lustre/fast/fast/zqiu/orbit_env_build/megatron_checkpoints/Qwen2.5-0.5B-Instruct-torchdist
export TRAIN_JSONL=/lustre/fast/fast/zqiu/orbit_env_build/data/gsm8k_agentic_train_64.jsonl
ORBIT_DRY_RUN_ARGV=1 bash examples/high_precision/run-qwen2_5-0_5b-bf16-math-fullft-async.sh
```

Expected: prints a `train_async.py` argv containing no `--peft-method oft`, exit 0. Repeat for 3B (3B asset paths) and 30B (any existing HF/Megatron paths from the 30B launcher's own header comments; dry run does not read them). Then `bash -n` all three.

- [ ] **Step 6: Tests green, commit**

Run: `python -m pytest tools/adapter_runtime_compare/test_run_compare.py -v` — Expected: PASS.

```bash
git add examples/high_precision/run-*fullft-async.sh tools/adapter_runtime_compare/
git commit -m "feat(examples): full-FT async arms for the A1 sync-cost rungs"
```

---

### Task 3: A1 summarizer — per-arm sync-cost table with bandwidth-fraction column

**Files:**
- Create: `tools/adapter_runtime_compare/analyze_a1.py`
- Test: `tools/adapter_runtime_compare/test_analyze_a1.py`

**Interfaces:**
- Consumes: harness run logs (`<output-dir>/<run_id>/*.log`) whose metric lines match `run_compare.METRIC_RE`; run_id format `r00_runtime_<model>_<precision>_<peft>_<mode>_g…` (from Task 1's harness).
- Produces: CLI `python tools/adapter_runtime_compare/analyze_a1.py <output-dir> --link-gbps 400 [--csv out.csv]` printing a markdown table with columns `model, mode, n_updates, update_s_mean, update_s_p50, payload_mb_mean, pause_s_mean, bw_frac`; `bw_frac = (payload_bytes/update_s) / (link_gbps/8 * 1e9)`.

- [ ] **Step 1: Write the failing test**

```python
import textwrap

from tools.adapter_runtime_compare import analyze_a1


def test_summarize_run_log(tmp_path):
    run_dir = tmp_path / "r00_runtime_qwen3_4b_bf16_oft_async_g0123"
    run_dir.mkdir()
    (run_dir / "run.log").write_text(textwrap.dedent("""\
        noise line
        perf 1: {'perf/update_weights_time': 0.2, 'perf/update_weights_payload_bytes': 100000000.0, 'perf/update_weights_pause_time': 0.05}
        perf 2: {'perf/update_weights_time': 0.4, 'perf/update_weights_payload_bytes': 100000000.0, 'perf/update_weights_pause_time': 0.15}
    """))
    rows = analyze_a1.summarize(tmp_path, link_gbps=400.0)
    assert len(rows) == 1
    row = rows[0]
    assert (row["model"], row["mode"]) == ("qwen3_4b", "async")
    assert row["n_updates"] == 2
    assert abs(row["update_s_mean"] - 0.3) < 1e-9
    assert abs(row["update_s_p50"] - 0.3) < 1e-9
    assert abs(row["pause_s_mean"] - 0.1) < 1e-9
    # 1e8 bytes / 0.3 s over a 400 Gb/s = 5e10 B/s link
    assert abs(row["bw_frac"] - (1e8 / 0.3) / 5e10) < 1e-9
```

Run: `python -m pytest tools/adapter_runtime_compare/test_analyze_a1.py -v` — Expected: FAIL (module missing).

- [ ] **Step 2: Implement `analyze_a1.py`**

```python
#!/usr/bin/env python3
"""Summarize adapter-runtime-compare logs into the A1 sync-cost table.

Reads every ``<output_dir>/<run_id>/*.log``, extracts ``perf N: {...}``
records carrying ``perf/update_weights_time``, and reports per (model, mode):
update wall time (mean/p50), payload MB, pause seconds, and the achieved
fraction of link bandwidth — the column that makes the full-model arm
strawman-proof (spec: A1). Transport is not in the logs; state it in the
figure caption (async arms: NCCL; colocated on this cluster: cpu_gather).
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

from tools.adapter_runtime_compare.run_compare import METRIC_RE, parse_payload

RUN_ID_RE = re.compile(
    r"r\d+_(?P<branch>[^_]+)_(?P<model>.+)_(?P<precision>bf16|fp8|int4)_"
    r"(?P<peft>[^_]+)_(?P<mode>sync|async|async_db|async_fullft)_g"
)

TIME_KEY = "perf/update_weights_time"
BYTES_KEY = "perf/update_weights_payload_bytes"
PAUSE_KEY = "perf/update_weights_pause_time"


def iter_update_records(log_path: Path):
    for line in log_path.read_text(errors="replace").splitlines():
        match = METRIC_RE.search(line)
        if not match or match.group("kind") != "perf":
            continue
        payload = parse_payload(match.group("payload"))
        if payload and TIME_KEY in payload:
            yield payload


def summarize(output_dir: Path, link_gbps: float) -> list[dict]:
    rows = []
    for run_dir in sorted(Path(output_dir).iterdir()):
        id_match = RUN_ID_RE.match(run_dir.name)
        if not id_match or not run_dir.is_dir():
            continue
        times, bytes_, pauses = [], [], []
        for log_path in sorted(run_dir.glob("*.log")):
            for rec in iter_update_records(log_path):
                times.append(float(rec[TIME_KEY]))
                bytes_.append(float(rec.get(BYTES_KEY, 0.0)))
                pauses.append(float(rec.get(PAUSE_KEY, 0.0)))
        if not times:
            print(f"warning: no {TIME_KEY} records in {run_dir.name}", file=sys.stderr)
            continue
        mean_t = statistics.mean(times)
        link_bytes_per_s = link_gbps / 8.0 * 1e9
        rows.append({
            "model": id_match.group("model"),
            "mode": id_match.group("mode"),
            "n_updates": len(times),
            "update_s_mean": mean_t,
            "update_s_p50": statistics.median(times),
            "payload_mb_mean": statistics.mean(bytes_) / 1e6,
            "pause_s_mean": statistics.mean(pauses),
            "bw_frac": (statistics.mean(bytes_) / mean_t) / link_bytes_per_s,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--link-gbps", type=float, required=True,
                        help="Nominal interconnect bandwidth for bw_frac (e.g. 400 for NDR IB)")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path")
    args = parser.parse_args(argv)

    rows = summarize(args.output_dir, args.link_gbps)
    if not rows:
        print("no runs with update_weights records found", file=sys.stderr)
        return 1
    cols = list(rows[0])
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(
            f"{row[c]:.4g}" if isinstance(row[c], float) else str(row[c]) for c in cols) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Test green**

Run: `python -m pytest tools/adapter_runtime_compare/test_analyze_a1.py -v` — Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tools/adapter_runtime_compare/analyze_a1.py tools/adapter_runtime_compare/test_analyze_a1.py
git commit -m "feat(tools): A1 sync-cost summarizer with bandwidth-fraction column"
```

---

### Task 4: A2 enablement — `EXTRA_TRAIN_ARGS` launcher hook + timeline figure script

A2 needs `--sglang-enable-metrics` on the engines (a valid orbit flag — sglang's `ServerArgs` is embedded under the `--sglang-` prefix in `orbit/backends/sglang_utils/arguments.py`), but launchers expose no way to append args. Add one generic hook; then the figure script the timeline tooling is missing.

**Files:**
- Modify: `scripts/lib/launcher.sh` (parse `EXTRA_TRAIN_ARGS` env string → array, after contract validation ~line 47)
- Modify: `scripts/lib/driver.sh` (append the array in BOTH argv sites — the echo block ~lines 55–74 and the `python3` invocation ~lines 83+; the file's own comment demands the two stay in sync)
- Create: `tools/rollout_timeline/figure.py`
- Test: `tests/fast/test_rollout_timeline_figure.py`, `tests/fast/test_launcher_extra_train_args.py`

**Interfaces:**
- Consumes: `binning.load_jsonl(path)` and `binning.build_timeline(probe_records, event_records, counter=..., bin_s=...)` → `{"bins": [...], "per_engine": {...}, "windows": [...]}`; each bin dict has `t_start`, `t_end`, `tokens_per_s`, `has_gap`, `in_update` (verify exact bin field names against `Bin.to_dict()` in `tools/rollout_timeline/binning.py` before writing the figure code; adjust names to match).
- Produces: `EXTRA_TRAIN_ARGS="--flag1 --flag2"` env honored by every launcher; CLI `python tools/rollout_timeline/figure.py --probe p.jsonl --events e.jsonl --out fig.png [--counter <spec>] [--bin-s 0.1] [--label NAME]` writing a PNG and printing summary stats.

- [ ] **Step 1: Failing hook test** — `tests/fast/test_launcher_extra_train_args.py`:

```python
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_extra_train_args_reach_argv(tmp_path):
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text('{"prompt": "x", "label": "1"}\n')
    hf = tmp_path / "hf"; hf.mkdir()
    meg = tmp_path / "meg"; meg.mkdir()
    env = dict(os.environ)
    env.update({
        "ORBIT_DRY_RUN_ARGV": "1",
        "EXTRA_TRAIN_ARGS": "--sglang-enable-metrics",
        "HF_CKPT": str(hf), "MEGATRON_LOAD": str(meg),
        "TRAIN_JSONL": str(jsonl), "SAVE_DIR": str(tmp_path / "save"),
    })
    proc = subprocess.run(
        ["bash", str(REPO / "examples/high_precision/run-qwen2_5-0_5b-bf16-math-oft.sh")],
        env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--sglang-enable-metrics" in proc.stdout
```

Run: `python -m pytest tests/fast/test_launcher_extra_train_args.py -v` — Expected: FAIL (flag absent from argv). If the dry run itself fails on some other required env, add that env in the test rather than weakening the launcher.

- [ ] **Step 2: Implement the hook.** In `scripts/lib/launcher.sh`, after the array-contract loop (below `unset _name`):

```bash
# Optional cross-cutting extra args (string env, deliberately word-split).
read -r -a ORBIT_EXTRA_TRAIN_ARGS <<< "${EXTRA_TRAIN_ARGS:-}"
```

In `scripts/lib/driver.sh`, append to BOTH argv lists, after `"${PEFT_ARGS[@]}"`:

```bash
            ${ORBIT_EXTRA_TRAIN_ARGS[@]+"${ORBIT_EXTRA_TRAIN_ARGS[@]}"}
```

(The `${arr[@]+...}` guard keeps `set -u` safe when the env var is unset.)

- [ ] **Step 3: Hook test green** — rerun Step 1's pytest. Expected: PASS.

- [ ] **Step 4: Failing figure test** — `tests/fast/test_rollout_timeline_figure.py`:

```python
import json

from tools.rollout_timeline import figure


def _probe_record(t, tokens):
    return {"t_wall": t, "engine_url": "http://e1", "ok": True,
            "counters": {"sglang:realtime_tokens_total{mode=decode}": tokens}}


def test_figure_writes_png(tmp_path):
    probe = tmp_path / "probe.jsonl"
    probe.write_text("\n".join(json.dumps(_probe_record(t / 10.0, 100.0 * t))
                               for t in range(50)) + "\n")
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps({"t_wall": 2.0, "event": "update_start", "weight_version": 1, "mode": "peft"}) + "\n"
        + json.dumps({"t_wall": 2.5, "event": "update_end", "weight_version": 1, "mode": "peft"}) + "\n")
    out = tmp_path / "fig.png"
    stats = figure.render(str(probe), str(events), str(out))
    assert out.exists() and out.stat().st_size > 0
    assert stats["n_bins"] > 0 and stats["n_windows"] == 1
```

Run: `python -m pytest tests/fast/test_rollout_timeline_figure.py -v` — Expected: FAIL (module missing).

- [ ] **Step 5: Implement `tools/rollout_timeline/figure.py`**

```python
#!/usr/bin/env python3
"""Render the A2 rollout-throughput timeline from probe + event JSONL.

One trace per invocation (one arm); overlaying arms is the caller's job
(run once per arm with --label, or import render() and compose). Update
windows are shaded; bins flagged has_gap are marked — a scrape gap IS
signal (engine unresponsive during an update).
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.rollout_timeline.binning import DEFAULT_BIN_S, build_timeline, load_jsonl  # noqa: E402

DEFAULT_COUNTER = "sglang:realtime_tokens_total{mode=decode}"


def render(probe_path: str, events_path: str, out_path: str, *,
           counter: str = DEFAULT_COUNTER, bin_s: float = DEFAULT_BIN_S,
           label: str = "") -> dict:
    timeline = build_timeline(load_jsonl(probe_path), load_jsonl(events_path),
                              counter=counter, bin_s=bin_s)
    bins = timeline["bins"]
    windows = timeline["windows"]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    t0 = bins[0]["t_start"] if bins else 0.0
    xs = [(b["t_start"] + b["t_end"]) / 2.0 - t0 for b in bins]
    ys = [b["tokens_per_s"] for b in bins]
    ax.plot(xs, ys, lw=1.0, label=label or None)
    for b in bins:
        if b.get("has_gap"):
            ax.axvspan(b["t_start"] - t0, b["t_end"] - t0, color="0.85", zorder=0)
    for w in windows:
        if w.get("t_start") is not None and w.get("t_end") is not None:
            ax.axvspan(w["t_start"] - t0, w["t_end"] - t0, alpha=0.25, color="tab:red",
                       zorder=1, label="_update")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("rollout tokens/s")
    if label:
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {"n_bins": len(bins), "n_windows": len(windows),
            "gap_bins": sum(1 for b in bins if b.get("has_gap"))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--counter", default=DEFAULT_COUNTER)
    parser.add_argument("--bin-s", type=float, default=DEFAULT_BIN_S)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    stats = render(args.probe, args.events, args.out,
                   counter=args.counter, bin_s=args.bin_s, label=args.label)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Before running the test, check `Bin.to_dict()` and `update_windows()` in `binning.py` for the real field names (`tokens_per_s`, `t_start`, `t_end`, window start/end keys) and align the code above to them exactly.

- [ ] **Step 6: Figure test green; run the two pre-existing timeline test files too**

Run: `python -m pytest tests/fast/test_rollout_timeline_figure.py tests/fast/test_rollout_timeline_binning.py tests/fast/test_rollout_timeline_probe.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/lib/launcher.sh scripts/lib/driver.sh tools/rollout_timeline/figure.py \
        tests/fast/test_rollout_timeline_figure.py tests/fast/test_launcher_extra_train_args.py
git commit -m "feat(tools): EXTRA_TRAIN_ARGS hook and rollout timeline figure"
```

---

### Task 5: Adapter-swap OPD smoke launcher (closes a silent M1 gap)

M1's `adapter:<path>` teacher row has no smoke launcher (the design doc's "all arms have existing smoke launchers" was wrong for this row; every other row checks out).

**Files:**
- Create: `examples/on_policy_distillation/run-qwen2_5-0_5b-opd-adapter-swap-smoke.sh`
- Modify: `docs/plans/2026-08-17-adapter-first-experiments-design.md` (M1 section, one clause)

**Interfaces:**
- Produces: a smoke launcher requiring `OPD_TEACHER_ADAPTER` (path to an OFT adapter checkpoint; Task 8's harness pilot produces one under `tmp_ckpts/adapter_runtime_compare/`) and passing `--opd-teacher "adapter:${OPD_TEACHER_ADAPTER}"`.

- [ ] **Step 1: Copy the closest working smoke.** `cp examples/on_policy_distillation/run-qwen2_5-0_5b-opd-free-teacher-smoke.sh examples/on_policy_distillation/run-qwen2_5-0_5b-opd-adapter-swap-smoke.sh`, then edit: `LAUNCHER_NAME=smoke_qwen25_05b_opd_adapter_swap`; add `: "${OPD_TEACHER_ADAPTER:?set OPD_TEACHER_ADAPTER to an OFT adapter checkpoint dir}"` next to the existing required-env lines; replace `--opd-teacher base` with `--opd-teacher "adapter:${OPD_TEACHER_ADAPTER}"`. Keep everything else (`--opd-type megatron`, PEFT student config) identical — same-trunk adapter teachers require the PEFT path, which the free-teacher smoke already satisfies.

- [ ] **Step 2: CPU validation**

```bash
bash -n examples/on_policy_distillation/run-qwen2_5-0_5b-opd-adapter-swap-smoke.sh
HF_CKPT=/lustre/fast/fast/zqiu/orbit_env_build/models/Qwen2.5-0.5B-Instruct \
MEGATRON_LOAD=/lustre/fast/fast/zqiu/orbit_env_build/megatron_checkpoints/Qwen2.5-0.5B-Instruct-torchdist \
TRAIN_JSONL=/lustre/fast/fast/zqiu/orbit_env_build/data/gsm8k_agentic_train_64.jsonl \
OPD_TEACHER_ADAPTER=/tmp/nonexistent-adapter \
ORBIT_DRY_RUN_ARGV=1 bash examples/on_policy_distillation/run-qwen2_5-0_5b-opd-adapter-swap-smoke.sh
```

Expected: argv printed containing `--opd-teacher adapter:/tmp/nonexistent-adapter`, exit 0 (existence of the adapter is checked at run time, not dry-run — if the launcher framework rejects the missing path at dry-run, point it at any existing dir instead).

- [ ] **Step 3: Correct the design doc.** In the M1 section, change "All arms have existing smoke launchers under `examples/on_policy_distillation/`." to "All arms have smoke launchers under `examples/on_policy_distillation/` (the `adapter:` row's was added 2026-08-19; the rest predate this program)."

- [ ] **Step 4: Commit**

```bash
git add examples/on_policy_distillation/run-qwen2_5-0_5b-opd-adapter-swap-smoke.sh \
        docs/plans/2026-08-17-adapter-first-experiments-design.md
git commit -m "feat(examples): adapter-swap OPD teacher smoke launcher"
```

---

### Task 6: Teacher-logprob dump + compare CLI (M1 correctness leg, GPU side of I-5)

The CPU tests (`tests/fast/test_opd_teacher_equivalence.py`) pin `alias_ref` / `adapter_off` / `adapter_swap` bitwise. The remaining leg — trainer-computed vs externally-served teacher logprobs on a real batch — needs a dump hook and a compare CLI over `orbit/utils/logprob_compare.py`.

**Files:**
- Create: `orbit/utils/opd_dump.py`
- Modify: the single site where OPD teacher log-probs are attached to samples (locate with `grep -rn "teacher_log_probs" orbit/backends/training_utils/data.py orbit/backends/megatron_utils/actor.py orbit/rollout/opd_sglang.py` — the attach/assignment point, not the loss-consumption point; there is one per opd-type path, megatron and sglang: instrument both)
- Create: `tools/compare_opd_teacher_logprobs.py`
- Test: `tests/fast/test_opd_dump.py`

**Interfaces:**
- Produces: env `ORBIT_OPD_TEACHER_LOGPROB_DUMP=<path.jsonl>` makes rank 0 append one record per sample for the first `ORBIT_OPD_TEACHER_LOGPROB_DUMP_LIMIT` (default 1) rollouts: `{"rollout": int, "sample_index": int, "response_token_ids": [int...], "teacher_log_probs": [float...]}`. CLI: `python tools/compare_opd_teacher_logprobs.py ref.jsonl cand.jsonl --atol 5e-3` exits 0 iff all matched samples (keyed by `(rollout, sample_index)`, token ids must be identical) are within tolerance, printing the `summarize_reports` summary.

- [ ] **Step 1: Failing test for the dump writer + CLI**

```python
import json
import subprocess
import sys
from pathlib import Path

from orbit.utils.opd_dump import dump_teacher_logprob_records

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "tools" / "compare_opd_teacher_logprobs.py"


def _write(path, records):
    dump_teacher_logprob_records(str(path), records)


def _records(lp):
    return [{"rollout": 0, "sample_index": 0,
             "response_token_ids": [1, 2, 3], "teacher_log_probs": lp}]


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
```

Run: `python -m pytest tests/fast/test_opd_dump.py -v` — Expected: FAIL (imports missing).

- [ ] **Step 2: Implement `orbit/utils/opd_dump.py`**

```python
"""Env-gated JSONL dump of OPD teacher log-probs (M1 correctness leg).

Enabled by ORBIT_OPD_TEACHER_LOGPROB_DUMP=<path>. Only the first
ORBIT_OPD_TEACHER_LOGPROB_DUMP_LIMIT rollouts (default 1) are dumped, on
rank 0 only — this is a fixed-batch equivalence probe, not telemetry.
"""

from __future__ import annotations

import json
import os

ENV_PATH = "ORBIT_OPD_TEACHER_LOGPROB_DUMP"
ENV_LIMIT = "ORBIT_OPD_TEACHER_LOGPROB_DUMP_LIMIT"


def dump_teacher_logprob_records(path: str, records: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def maybe_dump_teacher_logprobs(rollout_id: int, samples) -> None:
    """Call at the point where samples carry .teacher_log_probs; no-op unless enabled."""
    path = os.environ.get(ENV_PATH)
    if not path or rollout_id >= int(os.environ.get(ENV_LIMIT, "1")):
        return
    records = []
    for index, sample in enumerate(samples):
        teacher_lp = getattr(sample, "teacher_log_probs", None)
        if teacher_lp is None:
            continue
        records.append({
            "rollout": rollout_id,
            "sample_index": index,
            "response_token_ids": [int(t) for t in getattr(sample, "response_token_ids", [])],
            "teacher_log_probs": [float(x) for x in teacher_lp],
        })
    if records:
        dump_teacher_logprob_records(path, records)
```

Then instrument the attach sites found in Step 1's grep: immediately after teacher log-probs are assigned onto the batch's samples, insert `from orbit.utils.opd_dump import maybe_dump_teacher_logprobs` + `maybe_dump_teacher_logprobs(rollout_id, samples)` guarded so it only runs on rank 0 (`torch.distributed.get_rank() == 0` when initialized, matching how neighboring rank-0-only logging in the same file does it). Adapt the two field names (`response_token_ids`, sample container) to what the site actually holds — inspect the `Sample` type there; if the token-id field differs, use the real one in both `opd_dump.py` and the test.

- [ ] **Step 3: Implement `tools/compare_opd_teacher_logprobs.py`**

```python
#!/usr/bin/env python3
"""Compare two OPD teacher-logprob dumps (see orbit/utils/opd_dump.py)."""

from __future__ import annotations

import argparse
import json
import sys

from orbit.utils.logprob_compare import compare_logprobs, summarize_reports


def load(path: str) -> dict:
    records = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            records[(rec["rollout"], rec["sample_index"])] = rec
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=5e-3)
    args = parser.parse_args(argv)

    ref, cand = load(args.reference), load(args.candidate)
    common = sorted(set(ref) & set(cand))
    if not common:
        print("no common (rollout, sample_index) keys", file=sys.stderr)
        return 2
    reports = []
    for key in common:
        if ref[key]["response_token_ids"] != cand[key]["response_token_ids"]:
            print(f"token ids differ at {key}: not the same batch", file=sys.stderr)
            return 2
        reports.append(compare_logprobs(ref[key]["teacher_log_probs"],
                                        cand[key]["teacher_log_probs"]))
    summary = summarize_reports(reports)
    print(f"samples={len(common)} {summary}")
    ok = summary.within(args.atol)
    print("PASS" if ok else f"FAIL (atol={args.atol})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Check `LogprobCompareReport.within(atol)` and `summarize_reports` signatures in `orbit/utils/logprob_compare.py` first and adapt the two call sites to the real API.

- [ ] **Step 4: Tests green (new + existing equivalence suite)**

Run: `python -m pytest tests/fast/test_opd_dump.py tests/fast/test_opd_teacher_equivalence.py tests/fast/test_logprob_compare.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orbit/utils/opd_dump.py tools/compare_opd_teacher_logprobs.py tests/fast/test_opd_dump.py \
        $(git diff --name-only)
git commit -m "feat(opd): env-gated teacher logprob dump and compare CLI"
```

---

### Task 7: R-2 — the 3B OPD teacher-cost recipe suite

Port the teacher-variant flag blocks from the 0.5B smokes onto one shared 3B math recipe, following the `ppo_critic_compare_common.sh` wrapper pattern (one common recipe owns science; thin wrappers select only the variant).

**Files:**
- Create: `examples/on_policy_distillation/opd_teacher_cost_common.sh`
- Create: `examples/on_policy_distillation/run-qwen2_5-3b-opd-cost-{served,load,adapter,base,ema}.sh` (five wrappers)

**Interfaces:**
- Consumes: the 3B GRPO math recipe (`examples/high_precision/run-qwen2_5-3b-math-oft-grpo.sh`) as the base; variant flag blocks copied verbatim from the working smokes: `served` ← `run-qwen2_5-0_5b-opd-full-vocab-smoke.sh` (`--opd-serve-teacher --opd-teacher-num-gpus N`), `load` ← `run-qwen2_5-0_5b-opd-mopd-smoke.sh` (`--opd-teacher-load`), `adapter` ← Task 5's smoke, `base` ← `run-qwen2_5-0_5b-opd-free-teacher-smoke.sh`, `ema` ← `run-qwen2_5-0_5b-opd-ema-smoke.sh`.
- Produces: `OPD_COST_VARIANT` selected by wrapper; env contract identical across wrappers (`HF_CKPT`, `MEGATRON_LOAD`, `TRAIN_JSONL`, plus per-variant extras `OPD_TEACHER_LOAD` / `OPD_TEACHER_ADAPTER`). Task 11 runs these.

- [ ] **Step 1: Build the common recipe.** Start from a copy of `run-qwen2_5-3b-math-oft-grpo.sh`. Keep its model/PEFT/optimizer/rollout blocks unchanged. Add a variant dispatch that appends to `RL_ARGS` before `source .../launcher.sh`:

```bash
: "${OPD_COST_VARIANT:?wrapper must set OPD_COST_VARIANT}"
case "${OPD_COST_VARIANT}" in
  served)
    RL_ARGS+=( --use-opd --opd-kl-coef "${OPD_KL_COEF:-0.1}"
               --opd-serve-teacher --opd-teacher-num-gpus "${OPD_TEACHER_NUM_GPUS:-1}" ) ;;
  load)
    : "${OPD_TEACHER_LOAD:?set OPD_TEACHER_LOAD to a Megatron teacher ckpt}"
    RL_ARGS+=( --use-opd --opd-kl-coef "${OPD_KL_COEF:-0.1}"
               --opd-teacher-load "${OPD_TEACHER_LOAD}" ) ;;
  adapter)
    : "${OPD_TEACHER_ADAPTER:?set OPD_TEACHER_ADAPTER to an OFT adapter dir}"
    RL_ARGS+=( --use-opd --opd-kl-coef "${OPD_KL_COEF:-0.1}"
               --opd-type megatron --opd-teacher "adapter:${OPD_TEACHER_ADAPTER}" ) ;;
  base)
    RL_ARGS+=( --use-opd --opd-kl-coef "${OPD_KL_COEF:-0.1}"
               --opd-type megatron --opd-teacher base ) ;;
  ema)
    RL_ARGS+=( --use-opd --opd-kl-coef "${OPD_KL_COEF:-0.1}"
               --opd-type sglang --opd-teacher self:ema
               --opd-ema-decay "${OPD_EMA_DECAY:-0.99}" ) ;;
  *) echo "unknown OPD_COST_VARIANT=${OPD_COST_VARIANT}" >&2; exit 2 ;;
esac
LAUNCHER_NAME="qwen25_3b_opd_cost_${OPD_COST_VARIANT}"
```

**The flag lists above are the porting TARGET, not the source of truth** — before finalizing each case-arm, open the corresponding 0.5B smoke and copy its complete OPD flag block (including `--opd-type`, top-k/full-vocab switches, and any teacher-mem flags), preserving each smoke's exact working combination. In particular the free-teacher smoke's header documents that `--opd-type megatron` rejects PEFT students only for `--opd-teacher load:` — if the 3B `load` variant hits that rejection at dry-run, copy the mopd smoke's opd-type choice for that variant verbatim and note the deviation in the recipe header.

- [ ] **Step 2: Five thin wrappers**, each exactly:

```bash
#!/usr/bin/env bash
# M1 teacher-cost arm: <variant>. Selects only the variant; the common
# recipe owns every scientific hyperparameter.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPD_COST_VARIANT=<variant>
source "${SCRIPT_DIR}/opd_teacher_cost_common.sh"
```

- [ ] **Step 3: CPU validation — argv parity across wrappers**

```bash
for v in served load adapter base ema; do
  HF_CKPT=/fast/groups/ei-slm/hf_models/Qwen2.5-3B-Instruct \
  MEGATRON_LOAD=/lustre/fast/fast/zqiu/clthegoat-orbit/ppo_critic_benchmark_models/Qwen2.5-3B-Instruct_torch_dist \
  TRAIN_JSONL=/lustre/fast/fast/zqiu/clthegoat-orbit/ppo_critic_benchmark_data/openr1_49990/train.jsonl \
  OPD_TEACHER_LOAD=$PWD OPD_TEACHER_ADAPTER=$PWD \
  ORBIT_DRY_RUN_ARGV=1 bash examples/on_policy_distillation/run-qwen2_5-3b-opd-cost-${v}.sh \
    > /tmp/claude-argv-${v}.txt
done
for v in load adapter base ema; do diff /tmp/claude-argv-served.txt /tmp/claude-argv-${v}.txt; done
```

Expected: every wrapper exits 0; each pairwise diff shows ONLY the variant flag block and `LAUNCHER_NAME`-derived strings. Any other diff line is a recipe bug — fix before committing. Also `bash -n` all six files.

- [ ] **Step 4: Commit**

```bash
git add examples/on_policy_distillation/opd_teacher_cost_common.sh \
        examples/on_policy_distillation/run-qwen2_5-3b-opd-cost-*.sh
git commit -m "feat(examples): 3B OPD teacher-cost recipe suite (R-2)"
```

---

### Task 8: Phase-0 GPU smoke wave — **USER-RUN**, then executor triage

Everything below runs on the user's B200s. Present the block, stop, wait for logs. All commands assume `cd /lustre/fast/fast/zqiu/clthegoat-orbit/orbit && source ../uv_env_build/activate.sh` first, plus the harness env exports from Task 1 Step 5.

- [ ] **Step 1: Present the smoke commands to the user (do not run):**

```bash
# (a) Harness pilot: 0.5B, all four arms, ~15 min
codexlog phase0-pilot python tools/adapter_runtime_compare/run_compare.py run \
  --branches runtime --profile pilot --modes sync,async,async_db,async_fullft \
  --num-rollout 4 --no-eval

# (b) 4B family, all four arms, bench batch profile
codexlog phase0-q3-4b python tools/adapter_runtime_compare/run_compare.py run \
  --branches runtime --profile q3_4b --pefts oft --precisions bf16 \
  --modes sync,async,async_db,async_fullft --num-rollout 4 --no-eval

# (c) 4B fully-async launcher (no harness arm) — direct, 4+4 GPUs
export HF_CKPT=/fast/groups/ei-slm/hf_models/Qwen3-4B-Instruct-2507
export MEGATRON_LOAD=<the 4B torch_dist your published async runs used>   # see note below
export TRAIN_JSONL=/lustre/fast/fast/zqiu/clthegoat-orbit/ppo_critic_benchmark_data/openr1_49990/train.jsonl
codexlog phase0-4b-fully-async env NUM_ROLLOUT=4 DISABLE_EVAL=1 \
  bash examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft-fully-async.sh

# (d) OPD smokes at 0.5B (1–2 GPUs each; free-teacher, ema, mopd, full-vocab-served, adapter-swap)
export HF_CKPT=/lustre/fast/fast/zqiu/orbit_env_build/models/Qwen2.5-0.5B-Instruct \
       MEGATRON_LOAD=/lustre/fast/fast/zqiu/orbit_env_build/megatron_checkpoints/Qwen2.5-0.5B-Instruct-torchdist \
       TRAIN_JSONL=/lustre/fast/fast/zqiu/orbit_env_build/data/gsm8k_agentic_train_64.jsonl
codexlog phase0-opd-free    bash examples/on_policy_distillation/run-qwen2_5-0_5b-opd-free-teacher-smoke.sh
codexlog phase0-opd-ema     bash examples/on_policy_distillation/run-qwen2_5-0_5b-opd-ema-smoke.sh
codexlog phase0-opd-mopd    env OPD_TEACHER_LOAD=${MEGATRON_LOAD} bash examples/on_policy_distillation/run-qwen2_5-0_5b-opd-mopd-smoke.sh
codexlog phase0-opd-served  bash examples/on_policy_distillation/run-qwen2_5-0_5b-opd-full-vocab-smoke.sh
# adapter-swap needs an OFT adapter ckpt: use one saved by (a) under tmp_ckpts/adapter_runtime_compare/
codexlog phase0-opd-adapter env OPD_TEACHER_ADAPTER=<adapter dir from (a)> \
  bash examples/on_policy_distillation/run-qwen2_5-0_5b-opd-adapter-swap-smoke.sh

# (e) 30B qualification, 8 GPUs, one arm each (oft async_db + fullft async), ~30 min
codexlog phase0-q3-30b python tools/adapter_runtime_compare/run_compare.py run \
  --branches runtime --profile q3_30b --pefts oft --modes async_db,async_fullft \
  --num-rollout 3 --no-eval
```

The 4B/30B `MEGATRON_LOAD` paths: the HF checkpoints are at `/fast/groups/ei-slm/hf_models/{Qwen3-4B-Instruct-2507,Qwen3-30B-A3B-Instruct-2507}` (verified on disk), but the torch_dist conversions were not found by search — ask the user for the paths their published async runs used. Fallback if none exists: one conversion per rung, e.g. `codexlog convert-4b python tools/convert_hf_to_torch_dist.py --hf-checkpoint ${HF_CKPT} --save <dest>` (check that script's exact flag names with `--help` before presenting; it is the same tool that produced the validated 3B conversion).

Scoping note: this wave deliberately covers only the launchers Phase 1 consumes. The design doc's remaining Phase-0 smokes — search-r1 0.5B and tau-bench (P2), SFT launchers (M3), low-precision rungs (X1/X2) — are qualified at the start of their own phases; the critic-compare suite is already qualified by the completed 2026-08-06 benchmark.

- [ ] **Step 2 (executor, after user reports): Triage every log.** For each run: exit code 0; loss lines finite (`grep -E "loss.*(nan|inf)" <log>` empty); for the async arms, `perf/update_weights_time`, `perf/update_weights_payload_bytes`, `perf/update_weights_pause_time` present (`grep -o "perf/update_weights_[a-z_]*" <log> | sort -u`); for the OPD runs, the OPD path is active (grep the log for `opd`-tagged metric keys or the teacher-plan log line — record which line proves it). Record a qualification ledger table (launcher × result × log path) in `docs/plans/2026-08-17-adapter-first-experiments-design.md` is NOT the place — write it to `docs/reports/_src/2026-08-XX-phase0-qualification.md` (date of completion), and update the design doc's Phase-0 line to point at it.

- [ ] **Step 3: Commit the ledger + doc pointer**

```bash
git add docs/reports/_src/ docs/plans/2026-08-17-adapter-first-experiments-design.md
git commit -m "docs(reports): phase-0 launcher qualification ledger"
```

---

### Task 9: A1 measured runs — **USER-RUN**, then executor analysis

- [ ] **Step 1: Present the A1 campaign (after Task 8 passes; same env preamble):**

```bash
# One command per rung; 3 timing repeats, no eval, short runs.
for MODELS in qwen25_05b qwen25_3b qwen3_4b qwen3_30b; do
  codexlog a1-${MODELS} python tools/adapter_runtime_compare/run_compare.py run \
    --branches runtime --models ${MODELS} --pefts oft --precisions bf16 \
    --modes sync,async,async_db,async_fullft \
    --num-rollout 8 --repeats 3 --no-eval --campaign a1
done
```

(GPU counts come from the case registry: 0.5B = 2, 3B = 4 [Task 2], 4B = 4 [2+2 async], 30B = 8; total ≈ 40 GPU-h, matching the spec's ~10 GPU-h/point.)

- [ ] **Step 2 (executor): Summarize.** `python tools/adapter_runtime_compare/analyze_a1.py logs/adapter_runtime_compare --link-gbps <nominal, ask user: NVLink vs IB per layout> --csv docs/reports/_src/a1_sync_cost.csv`. Sanity-check the spec's expected shape: full-FT `update_s` grows with model size toward seconds; adapter arms flat ≈0.1 s; pause time nonzero in ALL arms (constraint 7). Verify the standing guard: each adapter-sync run's log-probability-mismatch metric stays at its baseline (grep the log for the rollout/train logprob-diff key and eyeball the series; a jump after an update event indicates a corrupted push — stop and report, do not average over it).

- [ ] **Step 3: Record.** Per-point table (model, mode, transport [async arms: NCCL; any colocated point: cpu_gather per `env.sh`], update_s, payload_mb, pause_s, bw_frac) into `docs/reports/_src/2026-08-XX-a1-sync-cost.md` with the figure (matplotlib from the CSV — model size x-axis, log-y update seconds, one line per mode). Commit as in Task 8 Step 3 (`docs(reports): A1 sync-cost scaling results`).

---

### Task 10: A2 measured timeline — **USER-RUN**, then executor figure

- [ ] **Step 1: Present the three-arm timeline runs (4B, 4+4 GPUs, ~10 min each):**

```bash
export HF_CKPT=/fast/groups/ei-slm/hf_models/Qwen3-4B-Instruct-2507
export MEGATRON_LOAD=<same 4B torch_dist as Task 8(c)>
export TRAIN_JSONL=/lustre/fast/fast/zqiu/clthegoat-orbit/ppo_critic_benchmark_data/openr1_49990/train.jsonl
export EXTRA_TRAIN_ARGS="--sglang-enable-metrics"
mkdir -p logs/a2

run_arm () {  # $1=arm-name $2=launcher $3..=extra env
  local name=$1 launcher=$2; shift 2
  ORBIT_TIMELINE_EVENTS_FILE=$PWD/logs/a2/${name}.events.jsonl \
  NUM_ROLLOUT=12 DISABLE_EVAL=1 "$@" codexlog a2-${name} bash ${launcher} &
  TRAIN_PID=$!
  sleep 180   # engines up; find their URLs in the newest run log
  URLS=$(grep -ohE "http://[0-9a-zA-Z.\-]+:[0-9]+" logs/run_qwen3_4b*_$(date +%Y%m%d)*.log | sort -u | tr '\n' ' ')
  python tools/rollout_timeline/probe.py --urls ${URLS} \
    --out logs/a2/${name}.probe.jsonl --interval 0.1 &
  PROBE_PID=$!
  wait ${TRAIN_PID}; kill ${PROBE_PID}
}

run_arm fullft examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-fullft-async.sh
run_arm single examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft-async.sh
run_arm db     examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft-async.sh env ADAPTER_DOUBLE_BUFFER=1
```

Caveats for the user in the handoff message: the URL grep pattern is a best guess — if the probe JSONL shows only failed scrapes, run `grep -iE "router|:3[0-9]{4}" <run log> | head` and re-point `--urls` (router URL alone is sufficient); if `/metrics` 404s despite the flag, add `--endpoint server_info` to the probe (coarser bins — figure still valid, note it in the caption).

- [ ] **Step 2 (executor): Render one figure per arm + the stats**

```bash
for name in fullft single db; do
  python tools/rollout_timeline/figure.py --probe logs/a2/${name}.probe.jsonl \
    --events logs/a2/${name}.events.jsonl --out logs/a2/${name}.png --label ${name}
done
```

Expected: ≥2 update windows per trace; the fullft trace shows the deepest/longest throughput trough at each window; db vs single quantifies the I-7 prize (both currently pause — constraint 7; say so in the caption). Iterate `--bin-s` (0.1 → 0.25) if traces are too noisy — the spec budgeted for iteration until the figure is clean. Record to `docs/reports/_src/2026-08-XX-a2-timeline.md`, commit (`docs(reports): A2 rollout-throughput timeline`).

---

### Task 11: M1 measured table + correctness leg — **USER-RUN**, then executor table

- [ ] **Step 1: Present the five cost-arm runs (3B, short; after Tasks 7–8):**

```bash
export HF_CKPT=/fast/groups/ei-slm/hf_models/Qwen2.5-3B-Instruct \
       MEGATRON_LOAD=/lustre/fast/fast/zqiu/clthegoat-orbit/ppo_critic_benchmark_models/Qwen2.5-3B-Instruct_torch_dist \
       TRAIN_JSONL=/lustre/fast/fast/zqiu/clthegoat-orbit/ppo_critic_benchmark_data/openr1_49990/train.jsonl
# an OFT adapter for the adapter arm: reuse one from the A1 3B run's tmp_ckpts
for v in base ema load adapter served; do
  codexlog m1-${v} env NUM_ROLLOUT=20 DISABLE_EVAL=1 \
    OPD_TEACHER_LOAD=${MEGATRON_LOAD} OPD_TEACHER_ADAPTER=<3B OFT adapter dir> \
    bash examples/on_policy_distillation/run-qwen2_5-3b-opd-cost-${v}.sh
done

# Correctness leg: same-teacher realizations must produce identical teacher logprobs
# on the FIRST rollout (fixed seed). Teacher == the frozen base in all three.
for v in base load served; do
  codexlog m1-eq-${v} env NUM_ROLLOUT=1 DISABLE_EVAL=1 SEED=1234 \
    OPD_TEACHER_LOAD=${MEGATRON_LOAD} \
    ORBIT_OPD_TEACHER_LOGPROB_DUMP=$PWD/logs/m1_eq_${v}.jsonl \
    bash examples/on_policy_distillation/run-qwen2_5-3b-opd-cost-${v}.sh
done
```

(If the recipe does not accept `SEED` env, take the seed flag name from `run-qwen2_5-3b-math-oft-grpo.sh` and thread it through the common recipe in Task 7 — matched seeds are what make the first batch comparable.)

- [ ] **Step 2 (executor): Correctness verdicts**

```bash
python tools/compare_opd_teacher_logprobs.py logs/m1_eq_base.jsonl logs/m1_eq_load.jsonl   --atol 5e-3
python tools/compare_opd_teacher_logprobs.py logs/m1_eq_base.jsonl logs/m1_eq_served.jsonl --atol 5e-3
```

Expected: PASS both (this is what licenses "free" in the M1 table). A token-id mismatch (exit 2) means the arms did not see the same batch — fix seeding before interpreting anything.

- [ ] **Step 3 (executor): Assemble the M1 table.** One row per variant; columns and their sources: extra GPUs (topology: `served` = `--opd-teacher-num-gpus`; `load` = 0 extra GPUs but a second trunk in memory; others 0), extra memory (allocator-counter peak from the log — `grep -o "memory/[a-z_]*" <log> | sort -u` to find the exact keys, then extract; state the key used), extra forwards/step (structural: `base`+KL = 0, `adapter`/`ema` = 1 adapter-swapped forward, `load` = 1 second-model forward, `served` = external), step time (`perf/actor_train_time` or the step-total perf key present in the logs — never compare `timing_s/actor_train` across modes that overlap differently, per the spec's constraint 2 analogue). Write `docs/reports/_src/2026-08-XX-m1-teacher-cost.md` scoped to same-trunk teachers (spec wording), commit (`docs(reports): M1 teacher-cost collapse table`).

---

## Execution notes

- Task order: 1 → 2 → {3, 4, 5, 6, 7 in any order} → 8 → {9, 10, 11 in any order}. Tasks 3–7 are independent of each other.
- Tasks 8–11 each contain a hard USER-RUN gate: prepare the command block, present it, stop. Do not poll GPUs, do not run `nvidia-smi` loops, do not launch "just a tiny" GPU check.
- If any dry-run or smoke exposes a wrong assumption in this plan (env var names, checkpoint paths, opd-type combinations), fix the code/launcher, re-run the CPU validation, and note the deviation in the commit — do not silently drift the recipe science (batch sizes, LRs, PEFT config stay untouched throughout).
- The final Phase-1 write-up (one HTML report over the three deliverables) is out of scope here; it follows via the `html-reports` skill once Tasks 9–11 have results.
