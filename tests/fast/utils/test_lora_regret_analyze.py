"""Reading the ledger into claims.

Every detector here has a case it must REJECT. A detector with only passing
cases is untested, and these decide whether ~800 GPU-hours produced a result or
an artifact.
"""

import json

import pytest

from tools.lora_regret.analyze import (
    argmins,
    edge_of_grid,
    load_records,
    lr_grids,
    sigma,
)


ALL = "linear_qkv,linear_proj,linear_fc1,linear_fc2"
ATTN = "linear_qkv,linear_proj"
FULL_KEY = ("full", None, "")


def _key(method, size, modules=ALL):
    """The 3-tuple ArmKey. target_modules is part of it because E3 runs
    `lora r256 attn` and `lora r256 all` in one matrix."""
    return (method, size, modules)


def _record(method, rank, lr, nll, seed=0, status="ok", modules=None, **extra):
    record = {
        "arm": f"{method}-r{rank}-all-lr{lr:g}-s{seed}",
        "method": method,
        "rank": rank,
        "oft_block_size": None,
        "target_modules": ("" if method == "full" else ALL) if modules is None else modules,
        "lr": lr,
        "seed": seed,
        "metric": "nll",
        "test_nll": nll,
        "status": status,
        "trace_consistent": True,
        "trace_warning": None,
        "nll_trace": None,
        "adapter_params": None,
        "global_batch_size": None,
        "dataset": None,
        "steps": 2000,
    }
    record.update(extra)
    return record


def _ledger(tmp_path, name, records):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


class TestLoadRecords:
    def test_failed_arms_are_dropped(self, tmp_path):
        path = _ledger(tmp_path, "a.jsonl", [
            _record("lora", 16, 2.5e-4, 1.5),
            _record("lora", 16, 5.0e-4, 1.4, status="failed"),
        ])
        assert len(load_records([path])) == 1

    def test_non_zero_seeds_are_dropped_by_default(self, tmp_path):
        """E1-0's replicates share a ledger directory and are not grid points.

        The runbook records the concrete failure: a seed-1 replicate at
        LR 9.95e-4 stealing r256's argmin from the real 2.5e-4.
        """
        path = _ledger(tmp_path, "a.jsonl", [
            _record("lora", 256, 2.5e-4, 1.50),
            _record("lora", 256, 2.5e-4, 1.49, seed=1),
        ])
        assert [r["seed"] for r in load_records([path])] == [0]
        assert len(load_records([path], seed=None)) == 2

    def test_an_inconsistent_trace_disqualifies_the_arm(self, tmp_path):
        path = _ledger(tmp_path, "a.jsonl", [
            _record("lora", 16, 2.5e-4, 1.5),
            _record("lora", 16, 5.0e-4, 1.4, trace_consistent=False,
                    trace_warning="samples=[992, 1000]"),
        ])
        kept = load_records([path])
        assert [r["lr"] for r in kept] == [2.5e-4]

    def test_a_glob_reads_every_shard(self, tmp_path):
        _ledger(tmp_path, "e1_lora_a.jsonl", [_record("lora", 1, 2.5e-4, 1.9)])
        _ledger(tmp_path, "e1_lora_b.jsonl", [_record("lora", 16, 2.5e-4, 1.6)])
        assert len(load_records([str(tmp_path / "e1_*.jsonl")])) == 2


class TestSigma:
    def test_is_the_standard_deviation_of_the_replicates(self, tmp_path):
        path = _ledger(tmp_path, "s.jsonl", [
            _record("lora", 256, 2.5e-4, 1.200000, seed=0),
            _record("lora", 256, 2.5e-4, 1.201000, seed=1),
            _record("lora", 256, 2.5e-4, 1.202000, seed=2),
        ])
        assert sigma(load_records([path], seed=None)) == pytest.approx(0.001, rel=1e-6)

    def test_refuses_fewer_than_three_replicates(self, tmp_path):
        path = _ledger(tmp_path, "s.jsonl", [
            _record("lora", 256, 2.5e-4, 1.20, seed=0),
            _record("lora", 256, 2.5e-4, 1.21, seed=1),
        ])
        with pytest.raises(ValueError, match="at least 3"):
            sigma(load_records([path], seed=None))


class TestArgmins:
    def test_picks_the_lowest_nll_per_arm(self, tmp_path):
        path = _ledger(tmp_path, "a.jsonl", [
            _record("lora", 16, 1.0e-4, 1.60),
            _record("lora", 16, 2.5e-4, 1.50),
            _record("lora", 16, 5.0e-4, 1.55),
            _record("full", None, 2.5e-5, 1.45),
        ])
        best = argmins(load_records([path]))
        assert best[_key("lora", 16)]["lr"] == 2.5e-4
        assert best[FULL_KEY]["lr"] == 2.5e-5

    def test_same_rank_different_placement_are_different_arms(self, tmp_path):
        """E3's collision case. A (method, rank) key would report one r256.

        `lora r256 attention-only` and `lora r256 all-modules` are both in the
        e3 matrix, and C4 is precisely the comparison between placements -- so
        collapsing them would delete the claim while appearing to answer it.
        """
        path = _ledger(tmp_path, "e3.jsonl", [
            _record("lora", 256, 2.5e-4, 1.50, modules=ALL),
            _record("lora", 256, 2.5e-4, 1.44, modules=ATTN),
        ])
        best = argmins(load_records([path]))
        assert len(best) == 2
        assert best[_key("lora", 256, ALL)]["test_nll"] == 1.50
        assert best[_key("lora", 256, ATTN)]["test_nll"] == 1.44


class TestEdgeOfGrid:
    def _grid(self, tmp_path, best_index):
        lrs = [1.0e-4, 1.5e-4, 2.5e-4, 4.0e-4, 6.3e-4]
        records = [
            _record("lora", 16, lr, 1.5 + (0.0 if i == best_index else 0.1))
            for i, lr in enumerate(lrs)
        ]
        return load_records([_ledger(tmp_path, "a.jsonl", records)])

    def test_fires_on_the_lowest_grid_point(self, tmp_path):
        flagged = edge_of_grid(self._grid(tmp_path, 0))
        assert _key("lora", 16) in flagged
        assert "re-centre" in flagged[_key("lora", 16)]

    def test_fires_on_the_highest_grid_point(self, tmp_path):
        assert _key("lora", 16) in edge_of_grid(self._grid(tmp_path, 4))

    def test_silent_one_grid_point_in(self, tmp_path):
        """The non-tautology case: an interior argmin must NOT be flagged."""
        assert edge_of_grid(self._grid(tmp_path, 1)) == {}
        assert edge_of_grid(self._grid(tmp_path, 3)) == {}

    def test_a_single_point_grid_is_flagged(self, tmp_path):
        """One LR is simultaneously the lowest and highest point tried."""
        records = load_records([_ledger(tmp_path, "a.jsonl", [_record("lora", 16, 2.5e-4, 1.5)])])
        assert _key("lora", 16) in edge_of_grid(records)


class TestLrGrids:
    def test_reports_the_sorted_grid_actually_run(self, tmp_path):
        path = _ledger(tmp_path, "a.jsonl", [
            _record("lora", 16, 5.0e-4, 1.55),
            _record("lora", 16, 1.0e-4, 1.60),
        ])
        assert lr_grids(load_records([path]))[_key("lora", 16)] == [1.0e-4, 5.0e-4]
