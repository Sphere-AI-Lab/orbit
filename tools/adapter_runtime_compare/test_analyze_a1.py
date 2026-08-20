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
