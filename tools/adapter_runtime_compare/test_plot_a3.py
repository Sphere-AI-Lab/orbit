import json
from pathlib import Path

from tools.adapter_runtime_compare import plot_a3


def _series():
    def run(mode, seed, rewards, step_s, gap):
        return {
            "run_id": f"r{seed - 1234:02d}_runtime_qwen3_4b_bf16_x_{mode}_g0123",
            "mode": mode,
            "seed": seed,
            "samples_per_rollout": 32,
            "rollouts": [
                {"rollout": i, "samples": 32 * (i + 1), "wall_s": i * step_s, "reward": r, "logprob_gap": gap}
                for i, r in enumerate(rewards)
            ],
        }

    return [
        run("sync", 1234, [0.1, 0.2, 0.3, 0.4], 8.0, 0.008),
        run("sync", 1235, [0.3, 0.4, 0.5, 0.6], 8.0, 0.010),
        run("async_db", 1234, [0.2, 0.2, 0.2], 3.0, 0.009),  # shorter: arm curve truncates to 3
        run("async_fullft", 1234, [0.1, 0.1, 0.1, 0.1], 4.0, 0.012),
    ]


def test_moving_average_is_centred_and_edge_shrinking():
    assert plot_a3.moving_average([1, 2, 3, 4, 5], 3) == [1.5, 2.0, 3.0, 4.0, 4.5]
    assert plot_a3.moving_average([1, 2, 3], 1) == [1, 2, 3]


def test_arm_curves_average_seeds_and_truncate_to_common_length():
    curves = plot_a3.arm_curves(_series(), smooth=1)

    assert list(curves) == ["sync", "async_fullft", "async_db"]  # fixed order, by identity
    sync = curves["sync"]
    assert sync["samples"] == [32, 64, 96, 128] and sync["n_seeds"] == [2]
    assert [round(v, 6) for v in sync["reward"]] == [0.2, 0.3, 0.4, 0.5]
    assert all(abs(s - 0.141421) < 1e-5 for s in sync["reward_std"])  # stdev of two seeds 0.2 apart
    assert sync["wall_s"] == [0.0, 8.0, 16.0, 24.0]
    assert abs(sync["gap"][0] - 0.009) < 1e-12
    assert len(curves["async_db"]["samples"]) == 3 and curves["async_db"]["reward_std"] == [0.0, 0.0, 0.0]


def test_main_writes_svg_and_png(tmp_path):
    series_path = tmp_path / "series.json"
    series_path.write_text(json.dumps(_series()))
    out = tmp_path / "figs" / "a3"

    rc = plot_a3.main([str(series_path), "--out", str(out), "--smooth", "1"])

    assert rc == 0
    assert (tmp_path / "figs" / "a3.svg").stat().st_size > 1000
    assert (tmp_path / "figs" / "a3.png").stat().st_size > 1000
    svg = (tmp_path / "figs" / "a3.svg").read_text()
    # identity is carried by label text as well as color
    assert "async full-FT" in svg and "sync OFT" in svg and "double-buffer" in svg
    assert Path(out).with_suffix(".svg").exists()
