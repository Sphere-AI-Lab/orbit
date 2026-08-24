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
