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


from tools.lora_regret.analyze import (
    batch_gaps,
    departure_steps,
    lr_band,
    placement_deltas,
)
from tools.lora_regret.trace import PHASE_AFTER_TRAIN, NllPoint


def _trace(nlls, start=1):
    return [
        NllPoint(i, i, PHASE_AFTER_TRAIN, nll, nll, 308760, 1000)
        for i, nll in enumerate(nlls, start=start)
    ]


class TestDepartureSteps:
    SIGMA = 0.001

    def test_an_arm_that_tracks_the_envelope_never_departs(self):
        traces = {"r512": _trace([1.5, 1.4, 1.3, 1.2]), "r256": _trace([1.5, 1.4, 1.3, 1.2])}
        assert departure_steps(traces, self.SIGMA) == {"r512": None, "r256": None}

    def test_reports_the_first_step_of_three_consecutive_excursions(self):
        # r1 exceeds the envelope by 10 sigma from step 2 onward.
        traces = {
            "r512": _trace([1.50, 1.40, 1.30, 1.20, 1.10]),
            "r1": _trace([1.50, 1.41, 1.31, 1.21, 1.11]),
        }
        assert departure_steps(traces, self.SIGMA)["r1"] == 2

    def test_does_not_fire_on_two_consecutive_excursions(self):
        """The non-tautology case: the rule says three, so two must not count."""
        traces = {
            "r512": _trace([1.50, 1.40, 1.30, 1.20, 1.10]),
            "r1": _trace([1.50, 1.41, 1.31, 1.20, 1.10]),
        }
        assert departure_steps(traces, self.SIGMA)["r1"] is None

    def test_an_excursion_under_two_sigma_does_not_count(self):
        traces = {
            "r512": _trace([1.5000, 1.4000, 1.3000, 1.2000]),
            "r16": _trace([1.5000, 1.4015, 1.3015, 1.2015]),  # 1.5 sigma
        }
        assert departure_steps(traces, self.SIGMA)["r16"] is None

    def test_an_empty_trace_is_none_not_a_crash(self):
        traces = {"r512": _trace([1.5, 1.4, 1.3]), "r1": []}
        assert departure_steps(traces, self.SIGMA)["r1"] is None


class TestLrBand:
    def test_the_band_spans_every_lr_within_two_sigma_of_the_best(self, tmp_path):
        path = _ledger(tmp_path, "e4.jsonl", [
            _record("lora", 1, 1e-6, 0.30),
            _record("lora", 1, 1e-5, 0.44),
            _record("lora", 1, 1e-4, 0.4395),
            _record("lora", 1, 1e-3, 0.10),
        ])
        records = load_records([path])
        for record in records:  # an accuracy ledger, scored the other direction
            record["metric"] = "accuracy"
            record["accuracy"] = record.pop("test_nll")
        band = lr_band(records, 0.001, metric="accuracy")
        assert band[_key("lora", 1)] == (1e-5, 1e-4)


class TestBatchGaps:
    """C3: the LoRA-minus-FullFT gap at each batch size, in sigma.

    The claim is a gap that GROWS with batch. A constant offset at all three
    batch sizes is not the signature and must be distinguishable from it, which
    means grouping by batch -- impossible until global_batch_size reached the
    ledger (Task 2).
    """

    def test_groups_by_batch_size(self, tmp_path):
        rows = []
        for batch, full_nll, lora_nll in [(32, 1.50, 1.502), (512, 1.40, 1.45)]:
            rows.append(_record("full", None, 2.5e-5, full_nll, global_batch_size=batch))
            rows.append(_record("lora", 256, 2.5e-4, lora_nll, global_batch_size=batch))
        gaps = batch_gaps(load_records([_ledger(tmp_path, "e2.jsonl", rows)]), 0.001)
        assert gaps[(32, _key("lora", 256))] == pytest.approx(2.0, abs=1e-6)
        assert gaps[(512, _key("lora", 256))] == pytest.approx(50.0, abs=1e-6)

    def test_a_batch_with_no_fullft_arm_is_skipped_not_guessed(self, tmp_path):
        rows = [_record("lora", 256, 2.5e-4, 1.45, global_batch_size=512)]
        gaps = batch_gaps(load_records([_ledger(tmp_path, "e2.jsonl", rows)]), 0.001)
        assert gaps == {}


class TestPlacementDeltas:
    """C4: NLL(attention) - NLL(MLP) at matched parameters, in sigma."""

    def test_pairs_attention_against_mlp(self, tmp_path):
        rows = [
            _record("lora", 256, 2.5e-4, 1.500, modules=ATTN),
            _record("lora", 92, 2.5e-4, 1.503, modules="linear_fc1,linear_fc2"),
        ]
        deltas = placement_deltas(load_records([_ledger(tmp_path, "e3.jsonl", rows)]), 0.001)
        assert deltas["attn(r256) - mlp(r92)"] == pytest.approx(-3.0, abs=1e-6)

    def test_no_mlp_arm_yields_no_comparison(self, tmp_path):
        rows = [_record("lora", 256, 2.5e-4, 1.500, modules=ATTN)]
        assert placement_deltas(load_records([_ledger(tmp_path, "e3.jsonl", rows)]), 0.001) == {}


import sys

from tools.lora_regret.analyze import all_modules_deltas, main


class TestAllModulesDeltas:
    """C4's second half: all-modules must not beat MLP-only by more than 2 sigma.

    The post claims attention-only underperforms MLP-only *and* that all-modules
    adds nothing on top of MLP-only. `placement_deltas` answers the first; without
    this the second half of E3-2 has arms in the matrix and no reader.
    """

    def test_pairs_all_modules_against_mlp(self, tmp_path):
        rows = [
            _record("lora", 256, 2.5e-4, 1.498, modules=ALL),
            _record("lora", 92, 2.5e-4, 1.500, modules="linear_fc1,linear_fc2"),
        ]
        deltas = all_modules_deltas(load_records([_ledger(tmp_path, "e3.jsonl", rows)]), 0.001)
        assert deltas["all(r256) - mlp(r92)"] == pytest.approx(-2.0, abs=1e-6)

    def test_an_attention_only_arm_is_not_mistaken_for_all_modules(self, tmp_path):
        """The non-tautology case: E3 runs attn r256 and all r256 in one matrix.

        Keying on rank alone, or on "targets linear_qkv", would pick the
        attention arm up here and report it as the all-modules comparison.
        """
        rows = [
            _record("lora", 256, 2.5e-4, 1.400, modules=ATTN),
            _record("lora", 92, 2.5e-4, 1.500, modules="linear_fc1,linear_fc2"),
        ]
        assert all_modules_deltas(load_records([_ledger(tmp_path, "e3.jsonl", rows)]), 0.001) == {}

    def test_no_mlp_arm_yields_no_comparison(self, tmp_path):
        rows = [_record("lora", 256, 2.5e-4, 1.500, modules=ALL)]
        assert all_modules_deltas(load_records([_ledger(tmp_path, "e3.jsonl", rows)]), 0.001) == {}


class TestJsonOutput:
    """--json is the campaign's handoff to figures, so it must be machine-read.

    One JSON document on stdout and nothing else: a single stray human-readable
    line makes the whole output unparseable, which is a failure that only shows
    up in the consumer.
    """

    def _interior(self, tmp_path):
        """A ledger whose every argmin is one grid point in, so nothing is flagged."""
        rows = [_record("lora", 256, lr, nll)
                for lr, nll in [(1e-4, 1.60), (2.5e-4, 1.50), (6.3e-4, 1.58)]]
        rows += [_record("full", None, lr, nll)
                 for lr, nll in [(1e-5, 1.52), (2.5e-5, 1.47), (6.3e-5, 1.51)]]
        return _ledger(tmp_path, "e1.jsonl", rows)

    def _run(self, monkeypatch, capsys, *argv):
        monkeypatch.setattr(sys, "argv", ["analyze.py", *argv])
        code = main()
        return code, capsys.readouterr()

    def test_stdout_is_one_parseable_json_document(self, tmp_path, monkeypatch, capsys):
        path = self._interior(tmp_path)
        code, out = self._run(
            monkeypatch, capsys, "argmins", "--ledgers", str(path), "--sigma", "0.001", "--json"
        )
        assert code == 0
        payload = json.loads(out.out)  # raises if a human-readable line leaked in
        assert payload["command"] == "argmins"

    def test_the_argmins_reach_the_json(self, tmp_path, monkeypatch, capsys):
        path = self._interior(tmp_path)
        _, out = self._run(
            monkeypatch, capsys, "argmins", "--ledgers", str(path), "--sigma", "0.001", "--json"
        )
        by_arm = {row["arm"]: row for row in json.loads(out.out)["argmins"]}
        assert by_arm["lora r256 all"]["lr"] == 2.5e-4
        assert by_arm["full"]["test_nll"] == 1.47
        assert by_arm["lora r256 all"]["lr_grid"] == [1e-4, 2.5e-4, 6.3e-4]

    def test_without_json_stdout_is_not_json(self, tmp_path, monkeypatch, capsys):
        """The non-tautology case: the human tables must survive unchanged."""
        path = self._interior(tmp_path)
        _, out = self._run(
            monkeypatch, capsys, "argmins", "--ledgers", str(path), "--sigma", "0.001"
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(out.out)
        assert "argmin_lr" in out.out

    def test_an_edge_of_grid_argmin_still_exits_three_and_names_the_arm(
        self, tmp_path, monkeypatch, capsys
    ):
        """Fail-closed does not depend on the output format.

        The payload is still emitted, so a consumer sees *why* it was refused
        rather than only a bare exit code.
        """
        rows = [_record("lora", 256, lr, nll)
                for lr, nll in [(1e-4, 1.40), (2.5e-4, 1.50), (6.3e-4, 1.58)]]
        rows += [_record("full", None, lr, nll)
                 for lr, nll in [(1e-5, 1.52), (2.5e-5, 1.47), (6.3e-5, 1.51)]]
        path = _ledger(tmp_path, "e1.jsonl", rows)
        code, out = self._run(
            monkeypatch, capsys, "c2", "--ledgers", str(path), "--sigma", "0.001", "--json"
        )
        assert code == 3
        payload = json.loads(out.out)
        assert "lora r256 all" in payload["edge_of_grid"]
        assert "c2" not in payload  # refused, not quoted

    def test_both_halves_of_c4_reach_the_json(self, tmp_path, monkeypatch, capsys):
        rows = [
            _record("lora", 256, 2.5e-4, 1.500, modules=ATTN),
            _record("lora", 256, 2.5e-4, 1.498, modules=ALL),
            _record("lora", 92, 2.5e-4, 1.503, modules="linear_fc1,linear_fc2"),
        ]
        path = _ledger(tmp_path, "e3.jsonl", rows)
        _, out = self._run(
            monkeypatch, capsys, "c4", "--ledgers", str(path), "--sigma", "0.001",
            "--json", "--allow-edge-argmin",
        )
        payload = json.loads(out.out)
        assert payload["c4"]["attn_minus_mlp"]["attn(r256) - mlp(r92)"] == pytest.approx(-3.0, abs=1e-6)
        assert payload["c4"]["all_minus_mlp"]["all(r256) - mlp(r92)"] == pytest.approx(-5.0, abs=1e-6)

    def test_the_sigma_subcommand_emits_json_too(self, tmp_path, monkeypatch, capsys):
        path = _ledger(tmp_path, "s.jsonl", [
            _record("lora", 256, 2.5e-4, 1.200, seed=0),
            _record("lora", 256, 2.5e-4, 1.201, seed=1),
            _record("lora", 256, 2.5e-4, 1.202, seed=2),
        ])
        code, out = self._run(monkeypatch, capsys, "sigma", "--ledgers", str(path), "--json")
        assert code == 0
        payload = json.loads(out.out)
        assert payload["sigma"] == pytest.approx(0.001, rel=1e-6)
        assert payload["n"] == 3


class TestAccuracyEdgeOfGrid:
    """The edge rule has to reach accuracy ledgers, or C5 is unguarded.

    An E4 ledger is entirely metric="accuracy" with test_nll=null, so the NLL
    view of it is empty and a guard computed only on that view has nothing to
    fire on. E4's grid is 4 points at half-decade spacing -- deliberately wide
    rather than resolved -- so a peak landing on an end is likely, and C5's
    claim is precisely about the WIDTH of the performant band, which a grid
    edge truncates.
    """

    def _acc_rows(self, peak_index, accuracies=None):
        lrs = [1e-6, 1e-5, 1e-4, 1e-3]
        scores = accuracies or [
            0.55 if i == peak_index else 0.30 for i in range(len(lrs))
        ]
        return [
            _record("lora", 1, lr, None, metric="accuracy", accuracy=acc)
            for lr, acc in zip(lrs, scores, strict=True)
        ]

    def _run(self, monkeypatch, capsys, *argv):
        monkeypatch.setattr(sys, "argv", ["analyze.py", *argv])
        code = main()
        return code, capsys.readouterr()

    def test_edge_of_grid_reads_accuracy_in_the_right_direction(self, tmp_path):
        """The peak is the MAXIMUM accuracy, not the minimum.

        Scores are arranged so the two answers disagree: the max sits on the
        grid edge and the min sits one point in. Reading the wrong direction
        therefore returns no flag rather than the same flag by luck.
        """
        rows = self._acc_rows(0, accuracies=[0.55, 0.44, 0.20, 0.30])
        records = load_records([_ledger(tmp_path, "e4.jsonl", rows)], metric="accuracy")
        assert _key("lora", 1) in edge_of_grid(records, metric="accuracy")

    def test_a_peak_on_the_lowest_lr_is_refused(self, tmp_path, monkeypatch, capsys):
        path = _ledger(tmp_path, "e4.jsonl", self._acc_rows(0))
        code, out = self._run(
            monkeypatch, capsys, "c5", "--ledgers", str(path), "--sigma", "0.001"
        )
        assert code == 3
        assert "lora r1 all" in out.err

    def test_a_peak_on_the_highest_lr_is_refused(self, tmp_path, monkeypatch, capsys):
        path = _ledger(tmp_path, "e4.jsonl", self._acc_rows(3))
        code, _ = self._run(
            monkeypatch, capsys, "c5", "--ledgers", str(path), "--sigma", "0.001"
        )
        assert code == 3

    def test_an_interior_peak_still_reads(self, tmp_path, monkeypatch, capsys):
        """The non-tautology case: c5 must still work when the peak is interior."""
        path = _ledger(tmp_path, "e4.jsonl", self._acc_rows(1))
        code, out = self._run(
            monkeypatch, capsys, "c5", "--ledgers", str(path), "--sigma", "0.001"
        )
        assert code == 0
        assert "peak=0.5500" in out.out

    def test_the_override_still_lets_it_through(self, tmp_path, monkeypatch, capsys):
        path = _ledger(tmp_path, "e4.jsonl", self._acc_rows(0))
        code, out = self._run(
            monkeypatch, capsys, "c5", "--ledgers", str(path), "--sigma", "0.001",
            "--allow-edge-argmin",
        )
        assert code == 0
        assert "peak=0.5500" in out.out

    def test_the_flagged_accuracy_arm_reaches_the_json(self, tmp_path, monkeypatch, capsys):
        path = _ledger(tmp_path, "e4.jsonl", self._acc_rows(0))
        code, out = self._run(
            monkeypatch, capsys, "c5", "--ledgers", str(path), "--sigma", "0.001", "--json"
        )
        assert code == 3
        payload = json.loads(out.out)
        assert "lora r1 all" in payload["edge_of_grid"]
        assert "c5" not in payload
