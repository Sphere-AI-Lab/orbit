"""Ten rollouts per method, then a verdict on every link between GPU and figure.

This exists because the coverage probe was not enough. It ran, it passed, and
the campaign it cleared then spent ~40 node-hours producing ledgers in which
every single row read `accuracy: null, status: "failed"`. Nothing crashed.
Training was fine. Three separate defects sat downstream of the part the probe
checks:

  1. train.py's generation-eval call omitted `num_rollout`, so
     `should_run_periodic_action`'s final-rollout branch was unreachable. At
     EVAL_INTERVAL=100000 -- chosen to mean "evaluate once, at the end" -- the
     modulo never matched either, and the arms produced ZERO post-training
     evals. The only eval in those logs is rollout 0's: the untrained policy.
  2. `parse_final_accuracy` demanded ("math_test", "gsm8k_test") while
     `arm_env` had told the launcher to score gsm8k alone. It fails closed on a
     missing dataset, so even that rollout-0 eval parsed to None.
  3. RUN_LOG is a fixed path per arm and the launcher opens it with `tee -a`,
     so a retried arm appends to its predecessor and every parser answered
     about a run that never happened -- 258 rollout timings on a 150-rollout
     row.

The probe could not have caught any of them, because it asks "does this method
run, and how fast" and all three sit AFTER that. So the question here is
different, and it is the only question that matters before a node is booked:

    does a number measured on the GPU reach the ledger, correctly labelled?

Every check below is a defect that has actually happened, and the run is
deliberately tiny -- 10 rollouts, three arms -- because catching them costs
minutes and missing them costs a reservation.

    bash scripts/lora_regret/smoke_e4_8gpu.sh

`plan` prints the three arms; `check` reads the ledger and the logs afterwards
and exits non-zero with the specific broken link named.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tools.lora_regret.arms import MATRICES
from tools.lora_regret.models import DEFAULT_MODEL
from tools.lora_regret.models import get as get_model
from tools.lora_regret.probe_log import last_run_segment, parse_reward_trace, parse_rollout_seconds
from tools.lora_regret.sweep import rl_eval_datasets

# The matrix the campaign actually runs. Not a stand-in: two of the three
# defects above were in code reached only via e4's per-dataset arms, and a smoke
# against a different matrix would have passed while they were live.
SMOKE_MATRIX = "e4"
SMOKE_DATASET = "gsm8k"

# Ten rollouts and an eval every five, so the run contains TWO post-training
# evals. Two rather than one on purpose: one eval could be produced by either
# the periodic branch or the final-rollout branch, so a single eval cannot tell
# a working final-rollout branch from a broken one. At interval 5 with 10
# rollouts the periodic branch fires at rollout 4 and the final-rollout branch
# at rollout 9 -- if the second is missing, defect (1) is back.
SMOKE_ROLLOUTS = 10
SMOKE_EVAL_INTERVAL = 5
EXPECTED_POST_TRAIN_EVALS = 2

# rollout.py logs the eval as a dict repr; sweep.py parses the same shape.
EVAL_LINE = re.compile(r"eval (?P<rollout_id>\d+): \{(?P<body>.*)\}")

# wandb bolds the path in its shutdown banner, so the run directory arrives as
# `...-b69fjjty\x1b[0m`. An escape is not whitespace, so a `\S+` capture takes
# it along and every path built from one misses by four characters -- which
# presents as an EMPTY wandb directory rather than as a parse error.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def smoke_arms(model_key: str = DEFAULT_MODEL) -> list:
    """One arm per method, from the real matrix.

    Selected from `MATRICES` rather than named in a list, so the smoke cannot
    drift out of the campaign it is clearing: a renamed arm or a method whose
    cell moved shows up here as a missing method, not as a passing run of
    something else. The middle learning rate of each method's grid is taken --
    the choice does not change which code executes, and the middle is the one an
    operator recognises from the runbook.
    """
    model = get_model(model_key)
    arms = MATRICES[SMOKE_MATRIX](
        model.hidden_size, model.ffn_size, model.qkv_output_size, 0, None, None
    )
    chosen = []
    for method in ("full", "lora", "oft"):
        cell = sorted(
            (a for a in arms if a.method == method and a.dataset == SMOKE_DATASET),
            key=lambda a: (a.lr, a.name),
        )
        if not cell:
            raise SystemExit(
                f"{SMOKE_MATRIX} has no {method} arm on {SMOKE_DATASET}; the matrix changed "
                "and this smoke would silently cover two methods instead of three."
            )
        chosen.append(cell[len(cell) // 2])
    return chosen


def post_train_eval_rollouts(log_text: str, datasets: tuple[str, ...]) -> list[int]:
    """Rollout ids of evals that measured a TRAINED policy.

    Rollout 0 is excluded by id, not by position: train.py's eval-before-train
    branch fires on rollout 0 regardless of interval, so a log with exactly one
    eval line looks complete and describes the base model. That is precisely
    what the seven gsm8k columns produced.

    An eval only counts if it reports every dataset the arm configured -- the
    same fail-closed rule `parse_final_accuracy` applies -- so this also catches
    defect (2) rather than counting a line the ledger will reject.
    """
    found = []
    for match in EVAL_LINE.finditer(log_text):
        rollout_id = int(match["rollout_id"])
        names = set(re.findall(r"'eval/([^'/]+)': [0-9.eE+-]+", match["body"]))
        if rollout_id > 0 and set(datasets) <= names:
            found.append(rollout_id)
    return sorted(set(found))


def offline_run_dir(log_text: str, repo_root: Path) -> Path | None:
    """The wandb directory this run wrote, read out of the run's own log.

    Taken from the `wandb sync <path>` line wandb prints at shutdown rather than
    by matching timestamps against `wandb/`, because the campaign runs arms
    back-to-back and a directory listing cannot say which arm owns which
    directory -- the failure that would make a green wandb check meaningless.
    """
    match = re.search(r"wandb sync (?P<path>\S*offline-run-\S+)", ANSI_ESCAPE.sub("", log_text))
    if not match:
        return None
    path = Path(match["path"])
    return path if path.is_absolute() else repo_root / path


def check_arm(arm, row: dict | None, repo_root: Path) -> list[tuple[bool, str]]:
    """Every link, in the order the data travels. Returns (ok, description)."""
    log_path = repo_root / "logs" / "lora_regret" / f"{arm.name}.log"
    datasets = rl_eval_datasets({"EVAL_DATASETS": arm.dataset} if arm.dataset else {})
    results: list[tuple[bool, str]] = []

    if not log_path.exists():
        return [(False, f"no log at {log_path}")]
    segment = last_run_segment(log_path.read_text(encoding="utf-8", errors="replace"))

    trace = parse_reward_trace(segment)
    results.append((
        len(trace) == SMOKE_ROLLOUTS,
        f"trained {len(trace)}/{SMOKE_ROLLOUTS} rollouts",
    ))

    evals = post_train_eval_rollouts(segment, datasets)
    results.append((
        len(evals) >= EXPECTED_POST_TRAIN_EVALS,
        f"post-training evals at {evals or 'NONE'} "
        f"(need {EXPECTED_POST_TRAIN_EVALS}, scoring {'+'.join(datasets)})",
    ))

    seconds = parse_rollout_seconds(segment)
    results.append((
        len(seconds) == SMOKE_ROLLOUTS,
        f"{len(seconds)} rollout timings for {SMOKE_ROLLOUTS} rollouts"
        + ("" if len(seconds) == SMOKE_ROLLOUTS else " -- retry contamination"),
    ))

    if row is None:
        results.append((False, "NO LEDGER ROW -- the arm ran and recorded nothing"))
        return results

    results.append((row.get("accuracy") is not None, f"ledger accuracy = {row.get('accuracy')}"))
    results.append((row.get("status") == "ok", f"ledger status = {row.get('status')!r}"))
    results.append((
        set(row.get("accuracy_per_dataset") or {}) == set(datasets),
        f"scored on {sorted(row.get('accuracy_per_dataset') or {})}, configured {list(datasets)}",
    ))

    run_dir = offline_run_dir(segment, repo_root)
    if run_dir is None:
        results.append((False, "no wandb offline directory named in the log"))
    else:
        wandb_files = list(run_dir.glob("run-*.wandb"))
        results.append((
            bool(wandb_files) and wandb_files[0].stat().st_size > 0,
            f"wandb offline dir {run_dir.name}"
            + (f" ({wandb_files[0].stat().st_size // 1024} KB)" if wandb_files else " EMPTY"),
        ))
        # `wandb sync` drops a `<run>.wandb.synced` marker beside the file it
        # uploaded. Checking the marker rather than asking the API keeps this
        # runnable on a compute node, which has no egress -- and an unsynced
        # marker is exactly the state the 2026-08-02 arms were left in.
        results.append((
            bool(list(run_dir.glob("run-*.wandb.synced"))),
            "wandb synced" if list(run_dir.glob("run-*.wandb.synced"))
            else "NOT synced -- run scripts/lora_regret/sync_wandb.sh from the login node",
        ))
    return results


def load_rows(ledger: Path) -> dict[str, dict]:
    if not ledger.exists():
        return {}
    rows = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            rows[record["arm"]] = record  # last write wins: a retried arm's newest row
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("plan", "check"))
    parser.add_argument("--ledger", type=Path, default=Path("results/smoke/e4_smoke.jsonl"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)

    arms = smoke_arms()

    if args.command == "plan":
        for arm in arms:
            # method, arm name, an anchored regex selecting exactly it
            print(f"{arm.method}\t{arm.name}\t^{re.escape(arm.name)}$")
        return 0

    rows = load_rows(args.ledger)
    failures = 0
    for arm in arms:
        print(f"\n=== {arm.method}: {arm.name}")
        for ok, description in check_arm(arm, rows.get(arm.name), args.repo_root):
            print(f"  {'PASS' if ok else 'FAIL'}  {description}")
            failures += not ok

    print()
    if failures:
        print(f"{failures} check(s) FAILED. Do NOT book the node -- every one of these was a "
              "real defect that produced a full, healthy-looking run and an empty ledger.")
        return 1
    print("All checks passed: a number measured on the GPU reaches the ledger, "
          "labelled with the dataset it was measured on, and the curve is in wandb.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
