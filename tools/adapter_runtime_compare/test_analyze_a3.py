import json
import textwrap
from pathlib import Path

from tools.adapter_runtime_compare import analyze_a3


def _write_run(root: Path, run_id: str, seed: int, rewards: list[float], step_s: float, t0: int) -> Path:
    """Synthetic run: one rollout/perf/step record per rollout, timestamps ``step_s`` apart."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"seed": seed, "env": {"ROLLOUT_BATCH_SIZE": "16", "N_SAMPLES_PER_PROMPT": "2"}})
    )
    lines = ["noise line"]
    for i, reward in enumerate(rewards):
        secs = t0 + int(i * step_s)
        # Real prefix: ANSI colour, ray actor tag, "[date time.ms logger]" stamp.
        stamp = f"[2026-09-02 18:{secs // 60:02d}:{secs % 60:02d}.887 actor_cell0_rank0]"
        lines.append(
            f"\x1b[36m(MegatronTrainRayActor pid=1)\x1b[0m {stamp} log_utils.py:259 - rollout {i}: "
            f"{{'rollout/response_lengths': 98.0, 'rollout/raw_reward': {reward}, 'rollout/rewards': 0.0}}"
        )
        # One rollout's perf keys arrive in two records: the rollout manager's (which prints
        # numpy scalars as np.float64(...)) and the trainer's.
        lines.append(
            f"(RolloutManager pid=2) {stamp} metrics.py:120 - perf {i}: "
            f"{{'rollout/num_training_samples': 32, 'perf/rollout_time': np.float64({step_s * 0.6}), "
            f"'perf/tokens_per_gpu_per_sec': np.float64(1000.0)}}"
        )
        lines.append(
            f"(MegatronTrainRayActor pid=1) {stamp} train_metric_utils.py:50 - perf {i}: "
            f"{{'perf/update_weights_pause_time': 0.05, 'perf/update_weights_time': 0.1, "
            f"'perf/train_wait_time': {step_s * 0.3}, 'perf/step_time': {step_s}}}"
        )
        lines.append(f"(MegatronTrainRayActor pid=1) {stamp} model.py:847 - step {i}: {{'train/train_rollout_logprob_abs_diff': 0.01}}")
    (run_dir / "console.log").write_text("\n".join(lines) + "\n")
    return run_dir


def _campaign(tmp_path: Path) -> Path:
    root = tmp_path / "a3"
    # sync: 8 s/step; async_db: 3 s/step; async_fullft: 4 s/step. 4 rollouts each, 32 samples per rollout.
    _write_run(root, "r00_runtime_qwen3_4b_bf16_oft_sync_g0123", 1234, [0.1, 0.2, 0.3, 0.4], 8.0, 0)
    _write_run(root, "r01_runtime_qwen3_4b_bf16_oft_sync_g0123", 1235, [0.1, 0.2, 0.3, 0.6], 8.0, 0)
    _write_run(root, "r00_runtime_qwen3_4b_bf16_oft_async_db_g0123", 1234, [0.1, 0.2, 0.3, 0.5], 3.0, 0)
    _write_run(root, "r00_runtime_qwen3_4b_bf16_none_async_fullft_g0123", 1234, [0.1, 0.1, 0.2, 0.2], 4.0, 0)
    return root


def test_parse_run_extracts_reward_wall_and_perf(tmp_path):
    root = _campaign(tmp_path)
    runs = {r.run_id: r for r in analyze_a3.iter_runs(root)}
    sync = runs["r00_runtime_qwen3_4b_bf16_oft_sync_g0123"]

    assert sync.mode == "sync" and sync.seed == 1234
    assert sync.rollouts == [0, 1, 2, 3]
    assert sync.samples_per_rollout == 32 and sync.samples(3) == 128
    assert sync.wall_s(0) == 0.0 and sync.wall_s(3) == 24.0
    assert sync.perf[2]["perf/step_time"] == 8.0
    assert sync.logprob_gap[1] == 0.01


def test_run_summary_uses_warm_rollouts_and_last_k(tmp_path):
    root = _campaign(tmp_path)
    runs = {r.run_id: r for r in analyze_a3.iter_runs(root)}
    s = analyze_a3.run_summary(runs["r00_runtime_qwen3_4b_bf16_oft_async_db_g0123"], last_k=2, warm_from=1)

    assert s["n_rollouts"] == 4 and s["samples"] == 128 and s["wall_s"] == 9.0
    assert abs(s["reward_last_k"] - 0.4) < 1e-9
    assert s["step_s"] == 3.0 and abs(s["train_wait_s"] - 0.9) < 1e-9
    # merged across the two perf records of each rollout
    assert s["tok_per_gpu_s"] == 1000.0 and abs(s["rollout_s"] - 1.8) < 1e-9


def test_arm_summary_aggregates_seeds_and_checkpoints(tmp_path):
    root = _campaign(tmp_path)
    rows = {r["mode"]: r for r in analyze_a3.arm_summary(
        analyze_a3.iter_runs(root), last_k=1, warm_from=1, checkpoints=[64, 128], window=1
    )}

    assert [m for m in rows] == ["sync", "async_fullft", "async_db"]
    assert rows["sync"]["n_seeds"] == 2
    assert abs(rows["sync"]["reward_last_k"] - 0.5) < 1e-9
    assert rows["sync"]["reward_last_k_std"] is not None
    assert rows["async_db"]["reward_last_k_std"] is None  # one seed: no spread
    # reward @64 samples = rollout 1 (2 rollouts x 32); @128 = rollout 3
    assert abs(rows["sync"]["reward@64"] - 0.2) < 1e-9
    assert abs(rows["async_db"]["reward@128"] - 0.5) < 1e-9


def test_attribution_decomposes_wall_clock(tmp_path):
    root = _campaign(tmp_path)
    arm_rows = analyze_a3.arm_summary(analyze_a3.iter_runs(root), last_k=1, warm_from=1, checkpoints=[], window=1)
    attr = {(r["from"], r["to"]): r for r in analyze_a3.attribution(arm_rows)}

    # sync wall 24 s, async_fullft 12 s, async_db 9 s
    assert abs(attr[("sync", "async_fullft")]["speedup"] - 2.0) < 1e-9
    assert abs(attr[("async_fullft", "async_db")]["speedup"] - 12 / 9) < 1e-9
    assert abs(attr[("sync", "async_db")]["speedup"] - 24 / 9) < 1e-9


def test_main_renders_markdown_and_writes_json(tmp_path, capsys):
    root = _campaign(tmp_path)
    out_json = tmp_path / "series.json"
    rc = analyze_a3.main([str(root), "--last-k", "2", "--checkpoints", "64,128", "--window", "1", "--json", str(out_json)])

    assert rc == 0
    text = capsys.readouterr().out
    assert "## Per arm" in text and "| sync |" in text and "## Wall-clock attribution" in text
    series = json.loads(out_json.read_text())
    assert len(series) == 4
    first = next(s for s in series if s["mode"] == "sync")
    assert first["rollouts"][3]["samples"] == 128 and first["rollouts"][3]["step_time"] == 8.0


def test_runs_without_records_are_kept_only_when_nothing_else_parsed(tmp_path):
    root = tmp_path / "empty"
    run_dir = root / "r00_runtime_qwen3_4b_bf16_oft_sync_g0123"
    run_dir.mkdir(parents=True)
    (run_dir / "console.log").write_text(textwrap.dedent("""\
        Traceback (most recent call last):
        Exception: Server process terminated unexpectedly.
    """))
    runs = analyze_a3.iter_runs(root)
    assert len(runs) == 1 and runs[0].rollouts == []
