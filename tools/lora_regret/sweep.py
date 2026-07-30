"""Drive the LoRA-without-regret sweep, one launcher invocation per arm.

Resumable: every completed arm appends a record to the results JSONL, and a
restart skips arms already recorded as "ok". A failed arm is retried on the
next run rather than silently skipped.

Progress and diagnostic output go to stderr; stdout is reserved for the
dry-run command lines, so `--dry-run | wc -l` and `--dry-run | head` give the
raw, pipeable arm matrix with nothing else mixed in.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from orbit.utils.peft_param_match import match_report
from tools.lora_regret.arms import (  # noqa: F401  (sft_arms re-exported)
    MATRICES,
    Arm,
    adapter_param_count,
    arm_env,
    sft_arms,
)

# The eval-line regex and phase labels live in trace.py -- one definition,
# built from EVAL_NLL_METRIC_KEY. Imported under the existing private names so
# every call site and the TestLogFormatPins pins keep working unchanged.
from tools.lora_regret.trace import (  # noqa: F401  (parse_trace re-exported)
    NLL_LINE as _NLL_LINE,
    PHASE_AFTER_TRAIN as _PHASE_AFTER_TRAIN,
    PHASE_BEFORE_TRAIN as _PHASE_BEFORE_TRAIN,
    parse_trace,
    trace_is_consistent,
)

# The campaign re-anchored from Qwen3-4B/No-Robots to Llama-3.1-8B/Tulu3, and
# this constant was left pointing at the old repo's launcher, which does not
# exist here. Pinned by test_the_launcher_the_sweep_shells_out_to_exists, because
# the failure mode is every arm failing identically for one silent reason.
LAUNCHER = "examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh"
RL_LAUNCHER = "examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh"

# Which script each matrix shells out to, and which metric its arms are scored
# by. E4 is RL: there is no held-out NLL to read, because an RL policy's own
# output distribution shifts as it trains, so NLL against a fixed reference set
# stops being comparable across arms. Accuracy is the metric, and it comes from a
# different log line produced by different code.
MATRIX_LAUNCHERS = {
    "sft82": LAUNCHER,
    "e1": LAUNCHER,
    "e1long": LAUNCHER,
    "e2": LAUNCHER,
    "e3": LAUNCHER,
    "e4": RL_LAUNCHER,
    "e5scout": LAUNCHER,
    "e5": LAUNCHER,
}
MATRIX_METRICS = {
    "sft82": "nll",
    "e1": "nll",
    "e1long": "nll",
    "e2": "nll",
    "e3": "nll",
    "e4": "accuracy",
    "e5scout": "nll",
    "e5": "nll",
}
# The eval dataset names the RL launcher passes to --eval-prompt-data. Given
# explicitly so parse_final_accuracy matches them exactly instead of guessing the
# key shape; pinned against the launcher's own text by
# test_the_rl_launcher_configures_exactly_the_datasets_the_parser_expects.
RL_EVAL_DATASETS = ("math_test", "gsm8k_test")


def load_ledger(path: Path) -> set[str]:
    """Arm names already completed successfully. Tolerates a truncated tail."""
    if not Path(path).exists():
        return set()
    done = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated final line from an interrupted write
        if record.get("status") == "ok":
            done.add(record["arm"])
    return done


def append_result(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def parse_final_nll(log_text: str) -> tuple[float | None, int | None]:
    """The arm's reported test NLL: the last post-training measurement.

    Only `phase=after_train` lines are candidates -- `phase=before_train`
    rows are excluded at the regex-match level, not filtered out after the
    fact by "take the last line in the file". That matters because at
    rollout/step==0 the NLL is logged twice: once by the before-train hook
    (the pristine base model) and, if `eval_nll_interval` fires on the very
    first rollout, once more by the periodic hook (already one optimizer
    step in). Excluding before-train rows outright means that row can never
    be picked as the arm's result, even under adversarial log ordering (for
    example interleaved multi-rank buffering placing it physically after a
    real after_train row).

    Among the (possibly several) after_train rows, the one with the highest
    `step` wins -- not simply the last regex match in the text -- so an
    out-of-order log still reports the true final measurement. A real
    completed run always has at least one after_train row: train.py forces
    the periodic hook to fire on the final rollout regardless of interval.
    """
    candidates = [
        (int(m["step"]), float(m["nll"]))
        for m in _NLL_LINE.finditer(log_text)
        if m["phase"] == _PHASE_AFTER_TRAIN
    ]
    if not candidates:
        return None, None
    step, nll = max(candidates, key=lambda pair: pair[0])
    return nll, step


# The RL eval logs a Python dict repr rather than a formatted metric line:
# orbit/ray/rollout.py's `logger.info(f"eval {rollout_id}: {log_dict}")`. So this
# matches the prefix for ordering, then picks dataset scores out of the repr.
#
# `eval/<name>` is a dataset score; `eval/<name>/<metric>` and
# `eval/<name>-truncated_ratio` are sub-metrics of the same dataset and must NOT
# be counted -- averaging a response length in with two accuracies would produce
# a number that looks like an accuracy and is not one. Hence the closing quote in
# the key pattern: it makes the key terminal.
_EVAL_LINE = re.compile(r"eval (?P<rollout_id>\d+): \{(?P<body>.*)\}")
# Permissive on the name (anything but a "/" or the closing quote) so an explicit
# dataset list can match a hyphenated name exactly. Sub-metric keys are then
# excluded by name membership, not by the regex.
_EVAL_SCORE = re.compile(r"'eval/(?P<name>[^'/]+)': (?P<score>[0-9.eE+-]+)")
_EVAL_AVG_NAME = "avg"


def parse_final_accuracy(
    log_text: str,
    datasets: tuple[str, ...] | None = None,
) -> tuple[float | None, int | None, dict[str, float]]:
    """The arm's accuracy: the mean over datasets at the highest rollout id.

    With `--rm-type boxed_math` the reward is exactly 1 or 0, so rollout.py's
    `sum(rewards) / len(rewards)` per-dataset score *is* accuracy on that split.

    Returns `(mean_score, rollout_id, per_dataset_scores)`. The mean is recomputed
    from the per-dataset scores rather than read off `eval/avg`, because
    rollout.py only emits `eval/avg` when more than one dataset is configured --
    a single-dataset run would otherwise parse as None and read as a failed arm.

    The highest `rollout_id` wins, not the last line in the file: multi-rank log
    interleaving can place an earlier eval physically last.

    `datasets` should name the eval datasets the launcher configured, and the
    caller should pass it whenever it knows them. Then only exact `eval/<name>`
    keys are read, and a line missing any of them is skipped rather than averaged
    over what did report -- fail closed, because a half-reported eval quoted as
    the arm's accuracy is worse than no number.

    Without `datasets` the shape of the key has to be guessed, and both possible
    guesses are lossy: rollout.py emits sub-metrics as `eval/<name>/<metric>`
    *and* `eval/<name>-<metric>` (pass@k when `--log-passrate` and
    `n_samples_per_eval_prompt > 1`, plus `-truncated_ratio`). Banning `-` in the
    name keeps sub-metrics out of the mean but silently drops any dataset whose
    own name contains a hyphen. That trade is why the explicit form exists.
    """
    best: tuple[int, dict[str, float]] | None = None
    for match in _EVAL_LINE.finditer(log_text):
        rollout_id = int(match["rollout_id"])
        found = {
            score_match["name"]: float(score_match["score"])
            for score_match in _EVAL_SCORE.finditer(match["body"])
            if score_match["name"] != _EVAL_AVG_NAME
        }
        if datasets is None:
            # Lossy guess -- see the docstring. "-" separates a sub-metric from
            # its dataset in rollout.py's flat key space, so a name containing
            # one cannot be told apart from `eval/<dataset>-<metric>`.
            found = {name: score for name, score in found.items() if "-" not in name}
        else:
            if not set(datasets) <= set(found):
                continue
            found = {name: found[name] for name in datasets}
        if not found:
            continue
        if best is None or rollout_id > best[0]:
            best = (rollout_id, found)
    if best is None:
        return None, None, {}
    rollout_id, scores = best
    return sum(scores.values()) / len(scores), rollout_id, scores


def _oft_match_summary(hidden_size: int) -> str:
    """One line per matched-parameter LoRA rank, for the dry-run diagnostic.

    Uses `match_report` (not `matched_oft_block_size` alone) specifically to
    surface the realized parameter ratio: the snap to a divisor of
    `hidden_size` can move it away from 1.0 (badly, at large rank -- see the
    module's docstring), and that should be visible before a sweep burns
    compute on it, not discovered afterwards.
    """
    lines = []
    for rank in (1, 16, 256):
        report = match_report(rank, hidden_size, hidden_size)
        lines.append(
            f"oft match rank={rank}: block_size={report['block_size']} "
            f"(ideal {report['ideal_block_size']}) ratio={report['ratio']:.3f}"
        )
    return "\n".join(lines)


def run_arm(
    arm: Arm,
    repo_root: Path,
    results_path: Path,
    dry_run: bool,
    launcher: str = LAUNCHER,
    metric: str = "nll",
    adapter_params: int | None = None,
) -> None:
    log_path = repo_root / "logs" / "lora_regret" / f"{arm.name}.log"
    # One dict, used for both the real environment and the dry-run preview --
    # so a previewed line cannot omit the per-arm SAVE_DIR that keeps
    # concurrent arms from overwriting each other.
    overrides = dict(arm_env(arm))
    overrides.update(
        {
            "LAUNCHER_NAME": arm.name,
            "RUN_LOG": str(log_path),
            "WANDB_GROUP": f"lora-regret-{'rl' if metric == 'accuracy' else 'sft'}",
            "SAVE_DIR": str(repo_root / "orbit_ckpts" / "lora_regret" / arm.name),
        }
    )
    env = dict(os.environ)
    env.update(overrides)
    cmd = ["bash", str(repo_root / launcher)]
    if dry_run:
        printed = " ".join(f"{k}={v}" for k, v in sorted(overrides.items()))
        print(f"{printed} bash {launcher}")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, env=env, cwd=repo_root)
    nll, accuracy, per_dataset, steps = (None, None, {}, None)
    trace_points: list = []
    trace_ok: bool | None = None
    trace_why: str | None = None
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if metric == "accuracy":
            accuracy, steps, per_dataset = parse_final_accuracy(log_text, RL_EVAL_DATASETS)
        else:
            nll, steps = parse_final_nll(log_text)
            trace_points = parse_trace(log_text)
            ok, why = trace_is_consistent(trace_points)
            trace_ok, trace_why = ok, (why or None)
    measured = accuracy if metric == "accuracy" else nll

    append_result(
        results_path,
        {
            "arm": arm.name,
            "method": arm.method,
            "rank": arm.rank,
            "oft_block_size": arm.oft_block_size,
            "target_modules": arm.target_modules,
            "lr": arm.lr,
            "seed": arm.seed,
            "matched_ratio": arm.matched_ratio,
            "metric": metric,
            "test_nll": nll,
            "accuracy": accuracy,
            "accuracy_per_dataset": per_dataset,
            "adapter_params": adapter_params,
            "wandb_run_id": None,
            "steps": steps,
            # The whole curve, not only its last point: C1's departure step is
            # unrecoverable from a scalar, and logs/ is gitignored.
            "nll_trace": [p._asdict() for p in trace_points] or None,
            "trace_consistent": trace_ok,
            "trace_warning": trace_why,
            # C3 groups by batch size; without this the batch an E2 arm ran at
            # survives only inside its name.
            "global_batch_size": arm.global_batch_size,
            "dataset": arm.dataset,
            "status": "ok" if (proc.returncode == 0 and measured is not None) else "failed",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--ffn-size", type=int, required=True)
    parser.add_argument(
        "--num-layers",
        type=int,
        required=True,
        help="Decoder layers in the model (32 for Llama-3.1-8B). Required rather "
        "than defaulted, like --hidden-size and --ffn-size: a wrong value makes "
        "every adapter_params in the ledger wrong by a constant factor.",
    )
    parser.add_argument("--results", type=Path, default=Path("results/lora_regret_sft.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--matrix",
        choices=sorted(MATRICES),
        default="sft82",
        help=(
            "Which arm matrix to run. 'e1'/'e2'/'e3' are the campaign plan's "
            "centred grids; 'sft82' is the original bracketing matrix and the "
            "only one with OFT arms."
        ),
    )
    parser.add_argument(
        "--oft-lr-centre",
        type=float,
        default=None,
        help=(
            "Learning rate the e5scout matrix found. Required by --matrix e5 and "
            "meaningless elsewhere: OFT parameterizes a rotation, so no LoRA "
            "learning rate transfers to it and the refinement grid has nothing to "
            "centre on until the scout has run."
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Regex; run only arms whose name matches (e.g. '^lora-r256' or '^oftscout').",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands, run nothing.")
    args = parser.parse_args()

    if args.matrix == "e5" and args.oft_lr_centre is None:
        # A clean exit rather than a traceback: this is the one argument an
        # operator following the runbook can only supply after another run.
        parser.error("--matrix e5 requires --oft-lr-centre; run --matrix e5scout first and pass its argmin")
    if args.matrix != "e5" and args.oft_lr_centre is not None:
        parser.error(f"--oft-lr-centre is only meaningful for --matrix e5, not {args.matrix}")

    repo_root = Path(__file__).resolve().parents[2]
    arms = MATRICES[args.matrix](args.hidden_size, args.ffn_size, args.seed, args.oft_lr_centre)
    if args.only:
        pattern = re.compile(args.only)
        arms = [a for a in arms if pattern.search(a.name)]

    done = load_ledger(args.results)
    todo = [a for a in arms if a.name not in done]
    print(f"{len(arms)} arms selected, {len(done)} already done, {len(todo)} to run", file=sys.stderr)
    # Only where it means something: the realized-ratio diagnostic is about OFT
    # block sizes, and printing it for a LoRA-only matrix invites reading it as a
    # property of arms that are about to run.
    if any(arm.method == "oft" for arm in arms):
        print(_oft_match_summary(args.hidden_size), file=sys.stderr)

    launcher = MATRIX_LAUNCHERS[args.matrix]
    metric = MATRIX_METRICS[args.matrix]
    print(f"launcher={launcher} metric={metric}", file=sys.stderr)
    for i, arm in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {arm.name}", file=sys.stderr)
        run_arm(
            arm, repo_root, args.results, args.dry_run,
            launcher=launcher, metric=metric,
            adapter_params=adapter_param_count(
                arm, args.hidden_size, args.ffn_size, args.num_layers
            ),
        )


if __name__ == "__main__":
    main()
