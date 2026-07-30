"""Read the sweep ledgers into the campaign's claims.

Every difference this module prints is in units of sigma, measured by E1-0 --
never off absolute loss values. The constant Orbit-vs-HF precision offset
(0.0032 nats) cancels in every ratio, ordering and curve-shape claim the
campaign makes, and cancels in nothing else.
"""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

# (method, size, target_modules). `size` is the rank for LoRA, the block size
# for OFT, and None for full fine-tuning.
#
# target_modules is part of the key and must NOT be dropped: E3 runs
# `lora r256 attention-only` and `lora r256 all-modules` in the same matrix, so
# a (method, rank) key would silently collapse two different arms into one and
# report whichever happened to score better as "the r256 argmin". That is the
# exact class of bug the seed-0 filter exists to prevent, one axis over.
ArmKey = tuple[str, int | None, str]


def load_records(
    paths,
    *,
    seed: int | None = 0,
    require_ok: bool = True,
    metric: str = "nll",
) -> list[dict]:
    """Ledger records worth analysing, from files or globs.

    `seed=0` is the default and is not cosmetic: E1-0's replicates live in the
    same ledger directory at seeds 1 and 2 and are *not* grid points. Measured
    on a synthetic ledger, dropping this filter let a replicate at LR 9.95e-4
    win r256's argmin away from the real 2.5e-4 purely because that one run
    happened to score better. Pass `seed=None` to read replicates, which is
    what `sigma` wants and nothing else does.

    Arms whose trace was inconsistent are dropped: a held-out set that changed
    size mid-run makes that arm's NLL incomparable to the others.
    """
    records: list[dict] = []
    for entry in paths:
        matches = sorted(glob.glob(str(entry))) or [str(entry)]
        for match in matches:
            path = Path(match)
            if not path.exists():
                raise FileNotFoundError(f"no ledger at {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated final line from an interrupted write
                if require_ok and record.get("status") != "ok":
                    continue
                if seed is not None and record.get("seed") != seed:
                    continue
                if record.get("metric", "nll") != metric:
                    continue
                if record.get("trace_consistent") is False:
                    continue
                records.append(record)
    return records


def arm_key(record: dict) -> ArmKey:
    size = record.get("oft_block_size") if record["method"] == "oft" else record.get("rank")
    return (record["method"], size, record.get("target_modules") or "")


def score(record: dict, metric: str = "nll") -> float:
    return record["accuracy"] if metric == "accuracy" else record["test_nll"]


def sigma(records: list[dict]) -> float:
    """Seed-to-seed standard deviation, from E1-0's replicates.

    Load with `seed=None`: the replicates are seeds 0, 1 and 2 of one
    configuration, and the default seed-0 filter would leave one point.
    """
    values = [score(r) for r in records]
    if len(values) < 3:
        raise ValueError(
            f"sigma needs at least 3 replicates, got {len(values)}. "
            "Run E1-0 (runbook section 7) and load with seed=None."
        )
    return statistics.stdev(values)


def lr_grids(records: list[dict]) -> dict[ArmKey, list[float]]:
    """The learning rates actually run, per arm, sorted."""
    grids: dict[ArmKey, set[float]] = {}
    for record in records:
        grids.setdefault(arm_key(record), set()).add(record["lr"])
    return {key: sorted(values) for key, values in grids.items()}


def argmins(records: list[dict], metric: str = "nll") -> dict[ArmKey, dict]:
    """The best-scoring record per arm.

    Lower is better for NLL, higher for accuracy -- the direction is chosen by
    `metric` rather than assumed, because E4's ledgers score by accuracy.
    """
    better = (lambda a, b: a > b) if metric == "accuracy" else (lambda a, b: a < b)
    best: dict[ArmKey, dict] = {}
    for record in records:
        key = arm_key(record)
        if key not in best or better(score(record, metric), score(best[key], metric)):
            best[key] = record
    return best


def edge_of_grid(records: list[dict], metric: str = "nll") -> dict[ArmKey, str]:
    """Arms whose argmin sits on a boundary of the grid that was run.

    An argmin at either end means the true optimum may lie outside the grid, so
    any ratio quoted from it is a lower bound on an unknown. The runbook's rule
    is to **re-centre, not extend**: extending keeps the old points at the wrong
    spacing and leaves the grid asymmetric about the new estimate.

    A one-point grid is flagged, because a single LR is simultaneously the
    lowest and the highest that was tried.
    """
    grids = lr_grids(records)
    flagged: dict[ArmKey, str] = {}
    for key, best in argmins(records, metric).items():
        grid = grids[key]
        if best["lr"] in (grid[0], grid[-1]):
            flagged[key] = (
                f"argmin LR {best['lr']:g} is on the edge of the grid "
                f"[{grid[0]:g} .. {grid[-1]:g}] ({len(grid)} points); "
                "re-centre the grid on it and re-run before quoting a ratio"
            )
    return flagged
