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
import time
from dataclasses import replace
from pathlib import Path

from orbit.utils.peft_param_match import match_report
from tools.lora_regret.arms import (  # noqa: F401  (sft_arms re-exported)
    MATRICES,
    MATRICES_REQUIRING_OFT_CENTRE,
    Arm,
    adapter_param_count,
    arm_env,
    sft_arms,
)
from tools.lora_regret.models import DEFAULT_MODEL, MODELS, model_env
from tools.lora_regret.models import get as get_model

# The eval-line regex and phase labels live in trace.py -- one definition,
# built from EVAL_NLL_METRIC_KEY. Imported under the existing private names so
# every call site and the TestLogFormatPins pins keep working unchanged.
from tools.lora_regret.probe_log import parse_rollout_seconds, parse_save_seconds
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
    "e1ot": LAUNCHER,
    "e1short": LAUNCHER,
    "e2": LAUNCHER,
    "e3": LAUNCHER,
    "e4": RL_LAUNCHER,
    "e4place": RL_LAUNCHER,
    "e5rl": RL_LAUNCHER,
    "e5scout": LAUNCHER,
    "e5": LAUNCHER,
}
MATRIX_METRICS = {
    "sft82": "nll",
    "e1": "nll",
    "e1long": "nll",
    "e1ot": "nll",
    "e1short": "nll",
    "e2": "nll",
    "e3": "nll",
    "e4": "accuracy",
    "e4place": "accuracy",
    "e5rl": "accuracy",
    "e5scout": "nll",
    "e5": "nll",
}
# The eval dataset names the RL launcher passes to --eval-prompt-data. Given
# explicitly so parse_final_accuracy matches them exactly instead of guessing the
# key shape; pinned against the launcher's own text by
# test_the_rl_launcher_configures_exactly_the_datasets_the_parser_expects.
RL_EVAL_DATASETS = ("math_test", "gsm8k_test")

# One wandb project per task, one group per method inside it.
#
# The launchers default to a single project for the whole campaign, which is
# right for a hand-run smoke and wrong for 242 swept arms: E1's rank ladder,
# E3's placement pair and E5's OFT arms would be one flat namespace, and the run
# that decides C2 would be indistinguishable in the sidebar from the one that
# decides C6. The matrix is the unit an operator schedules, reads and re-runs,
# so it is the unit the dashboard is split on.
#
# The name spells out `<dataset>-<sft|rl>-<what is tested>-<method>` rather than
# the matrix code, because "e4place" is only meaningful to someone holding the
# plan and "gsm8k-rl-placement-lora" is meaningful to anyone opening the sidebar.
# The first two components are not decoration: they are checked against each
# matrix's own arms by test_the_project_name_describes_the_arms_it_routes, so a
# project cannot end up claiming a dataset or a training mode it does not run.
#
# Only `<sft|rl>-<what is tested>` lives here. The DATASET comes from the
# arm and the METHOD is appended, so a project is `<dataset>-<mode>-<task>-<method>`
# -- `gsm8k-rl-rank-lora`, `math-rl-rank-ft`. Two reasons the dataset cannot stay
# in this table: E4 now trains a separate arm per dataset, so one matrix spans
# two of them; and a hardcoded "math-gsm8k" would have gone on claiming the mix
# after that stopped being true.
MATRIX_PROJECTS = {
    "e1": "sft-rank",
    "e1long": "sft-curves",
    "e1short": "sft-lr-horizon",
    "e1ot": "sft-rank",
    "e2": "sft-batch",
    "e3": "sft-placement",
    "e4": "rl-rank",
    "e4place": "rl-placement",
    "e5rl": "rl-oft-match",
    "e5scout": "sft-oft-scout",
    "e5": "sft-oft-match",
    "sft82": "sft-bracket",
}

# What the sidebar calls each method. `full` is spelled `ft` because that is what
# the post and every plan document call it.
METHOD_LABELS = {"full": "ft", "lora": "lora", "oft": "oft"}

# Where an arm goes when no matrix routed it. Deliberately the launchers' own
# default rather than any task's name: `run_arm` is callable directly (tests,
# one-off reruns), and inventing a plausible task would write those runs into a
# dashboard whose numbers are being quoted. This way an unrouted arm lands
# exactly where a hand-run one does.
UNROUTED_WANDB_PROJECT = "lora-without-regret"

# What an arm that names no dataset is training on. `sft82`, the legacy matrix,
# sets no `dataset` on any of its 82 arms and so takes the SFT launcher's own
# default. Spelled out here because the project name has to state a dataset, and
# omitting the component would produce `sft-bracket-lora` -- a name that reads
# like a project rather than like a missing field.
LAUNCHER_DEFAULT_DATASET = "tulu3"

# Every probe run, whatever task it names. Smoke runs are three rollouts with a
# real-looking loss curve; mixed into `tulu3-sft-rank` they would sit beside the
# arms that decide C2 and be indistinguishable from them in the sidebar. One bin
# for all of them, with the task and method in the group instead -- and it is
# keyed off `probe_rollouts` rather than a flag, so a probe cannot be pointed at
# a real project even deliberately.
SMOKE_WANDB_PROJECT = "lora-regret-smoke"


def arm_capacity(arm: Arm) -> str:
    """The arm's capacity, as the wandb group inside a method's project.

    `r1`/`r16`/`r256` for LoRA, `b32`.. for OFT, `full` for full fine-tuning --
    which has no capacity knob, and labelling it `na` would read as a missing
    value rather than as the point.
    """
    if arm.rank is not None:
        return f"r{arm.rank}"
    if arm.oft_block_size is not None:
        return f"b{arm.oft_block_size}"
    return "full"


def wandb_project(
    matrix: str | None,
    model: str | None = None,
    dataset: str | None = None,
    method: str | None = None,
) -> str:
    """The wandb project for one arm: `<dataset>-<mode>-<task>-<method>`.

    Four components, each of which would otherwise be invisible or ambiguous in
    the sidebar:

    **dataset**, because E4 trains one arm per dataset now -- `gsm8k-rl-rank-lora`
    and `math-rl-rank-lora` are two panels of Figure 6 and pooling them would put
    two different y-axes in one project.

    **method**, because it is the comparison C5 IS. Splitting FullFT and LoRA into
    their own projects is what lets each be read, and re-read, without the other's
    runs in the list.

    **model**, because an arm name carries method, capacity, placement, learning
    rate and seed but not the base model -- every matrix was single-model when the
    names were designed. `lora-r1-all-lr1e-05-s0` on Qwen3-1.7B and the same arm
    on Llama-3.1-8B are two experiments with one run name. Only non-default models
    are suffixed, so the campaign's own dashboards keep the bare name and the
    `<dataset>-<sft|rl>-` head that
    `test_the_project_name_describes_the_arms_it_routes` pins stays first.

    `dataset` and `method` are optional so a caller holding only a matrix -- the
    probe, which routes everything to one smoke project anyway -- still gets a
    usable name.
    """
    if matrix is None:
        return UNROUTED_WANDB_PROJECT
    try:
        task = MATRIX_PROJECTS[matrix]
    except KeyError:
        raise KeyError(
            f"no wandb project for matrix {matrix!r}; add one to MATRIX_PROJECTS "
            f"(known: {sorted(MATRIX_PROJECTS)})"
        ) from None
    parts = [p for p in (dataset, task, METHOD_LABELS.get(method, method)) if p]
    project = "-".join(parts)
    if model is not None and model != DEFAULT_MODEL:
        project = f"{project}-{model}"
    return project


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

    With `--rm-type math` the reward is exactly 1 or 0, so rollout.py's
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
    matrix: str | None = None,
    probe_rollouts: int | None = None,
) -> None:
    log_path = repo_root / "logs" / "lora_regret" / f"{arm.name}.log"
    # One dict, used for both the real environment and the dry-run preview --
    # so a previewed line cannot omit the per-arm SAVE_DIR that keeps
    # concurrent arms from overwriting each other.
    #
    # The model's environment goes down FIRST and the arm's own settings on top:
    # the model contributes checkpoint, plugin, mask type and GPU floor, while
    # the arm contributes LR, seed and PEFT knobs. The two sets are disjoint
    # today, and the ordering makes the arm win if they ever overlap.
    overrides = dict(model_env(get_model(arm.model), repo_root))
    overrides.update(arm_env(arm))
    if probe_rollouts is None:
        project = wandb_project(
            matrix, arm.model, arm.dataset or LAUNCHER_DEFAULT_DATASET, arm.method
        )
        # The method is in the project now, so the group carries CAPACITY --
        # the rank or the OFT block size -- which is what a reader compares
        # inside one method's dashboard. FullFT has none, and says so.
        group = arm_capacity(arm)
    else:
        # The task moves into the group so one smoke project still separates
        # e4place/oft from e1/lora, without either polluting a real dashboard.
        project = SMOKE_WANDB_PROJECT
        group = f"{matrix or 'unrouted'}-{arm.method}"
    overrides.update(
        {
            "LAUNCHER_NAME": arm.name,
            "RUN_LOG": str(log_path),
            # Project = the task, group = the method. The old group was
            # sft-vs-rl, which the launcher already implies and the project now
            # states outright; grouping by method is what makes a task's
            # FullFT, LoRA and OFT arms separable inside its own dashboard.
            "WANDB_PROJECT": project,
            "WANDB_GROUP": group,
            "SAVE_DIR": str(repo_root / "orbit_ckpts" / "lora_regret" / arm.name),
        }
    )
    if probe_rollouts is not None:
        # Applied AFTER arm_env, which is the whole point: an e1ot arm sets
        # NUM_ROLLOUT="" to request a full epoch and an e1short arm sets 100,
        # so a probe that merely exported the variable would be overridden by
        # exactly the two matrices whose length it most needs to cut.
        overrides["NUM_ROLLOUT"] = str(probe_rollouts)
        overrides["EVAL_NLL_INTERVAL"] = "1"
    env = dict(os.environ)
    env.update(overrides)
    cmd = ["bash", str(repo_root / launcher)]
    if dry_run:
        printed = " ".join(f"{k}={v}" for k, v in sorted(overrides.items()))
        print(f"{printed} bash {launcher}")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proc = subprocess.run(cmd, env=env, cwd=repo_root)
    elapsed = time.monotonic() - started
    nll, accuracy, per_dataset, steps = (None, None, {}, None)
    trace_points: list = []
    trace_ok: bool | None = None
    trace_why: str | None = None
    rollout_seconds: list[float] = []
    save_seconds: list[float] = []
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        # Measured per rollout by train.py's own ETA tracker, for SFT and RL
        # alike. Recorded on every row, not only probes: it is the only place a
        # completed arm's pace survives, and logs/ is gitignored.
        rollout_seconds = parse_rollout_seconds(log_text)
        # Priced separately from the rollout it lands inside: a FullFT
        # checkpoint is ~10 minutes, and averaging it into a per-rollout
        # figure doubled the campaign estimate.
        save_seconds = parse_save_seconds(log_text)
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
            # Which base model produced this number. Not derivable from `arm`:
            # the name carries method, capacity, placement, LR and seed, and was
            # designed when every matrix was single-model. Without it, globbing a
            # Qwen ledger and a Llama ledger into `analyze` merges two models'
            # arms into one argmin and nothing in the output looks wrong.
            "model": arm.model,
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
            # Where this row's curves live. Without it a ledger read months
            # later cannot be traced back to the dashboard it was read off,
            # which is the point of splitting the projects in the first place.
            "wandb_project": project,
            "wandb_group": group,
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
            "seconds": elapsed,
            "rollout_seconds": rollout_seconds,
            "save_seconds": save_seconds,
            "matrix": matrix,
            "gpus": int(os.environ.get("GPUS_PER_NODE", 0)) or None,
            # Present ONLY on probe rows, and `analyze` refuses any ledger that
            # has it. Three rollouts produce a real-looking test_nll; without
            # this a globbed ledger could decide an argmin from a learning rate
            # that trained for ninety seconds.
            "probe_rollouts": probe_rollouts,
            "status": "ok" if (proc.returncode == 0 and measured is not None) else "failed",
        },
    )


def argmins_from(patterns: list[str], allow_edge: bool) -> dict[tuple[str, int | None], float]:
    """Each E1 arm's argmin learning rate, read from the E1-1 ledgers.

    Fails closed twice, because E1-2 is ~70 GPU-hours per arm and both failures
    are silent otherwise:

    - Fewer than 8 arms recovered means a partial ledger. Running the 3 arms
      that happen to be there would produce a stage that *looks* complete.
    - An argmin on a grid edge means the LR is a boundary value rather than an
      optimum. Spending 70 hours at it is the single most expensive way to act
      on an unchecked number, so it is refused unless overridden.
    """
    from tools.lora_regret.analyze import argmins, edge_of_grid, load_records

    records = load_records(patterns)
    best = argmins(records)
    # analyze keys on (method, size, target_modules); e1long keys on
    # (method, rank), because every E1 arm is all-modules and the long curves
    # inherit that. Project, and refuse to guess if the ledger actually holds
    # two placements at one rank -- that is an E3 ledger, not an E1 one.
    found: dict[tuple[str, int | None], float] = {}
    for (method, size, modules), record in best.items():
        key = (method, size)
        if key in found:
            sys.exit(
                f"--argmins-from found more than one placement for {key} "
                f"(latest: {modules!r}). These ledgers mix placements, so there is no "
                "single argmin per rank; point it at the E1 ledgers only."
            )
        found[key] = record["lr"]
    if len(found) < 8:
        sys.exit(
            f"--argmins-from recovered only {len(found)} arms from {patterns}: "
            f"{sorted(found)}. E1-2 needs all 8 (FullFT plus ranks "
            "1, 4, 16, 64, 128, 256, 512); finish E1-1 first."
        )
    flagged = edge_of_grid(records)
    if flagged and not allow_edge:
        lines = "\n".join(f"  {key}: {why}" for key, why in flagged.items())
        sys.exit(
            "--argmins-from refuses an edge-of-grid argmin:\n"
            f"{lines}\n"
            "Re-centre the grid and re-run those arms, or pass --allow-edge-argmin "
            "to spend ~70 GPU-hours per arm on a boundary value anyway."
        )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default=DEFAULT_MODEL,
        help=(
            "Which base model every arm in the matrix runs on. Selects the "
            "checkpoint, the Megatron plugin, the loss-mask type and the GPU "
            "floor together, and -- because they decide every matched-parameter "
            "block size and rank -- the hidden, FFN and fused-QKV widths. "
            f"Default {DEFAULT_MODEL}, the campaign's original anchor."
        ),
    )
    parser.add_argument("--hidden-size", type=int, default=None,
                        help="Deprecated: derived from the arm's model. Kept so the "
                             "runbook's existing commands still work; a value that "
                             "contradicts the model is an error, not a preference.")
    parser.add_argument("--ffn-size", type=int, default=None, help="Deprecated; see --hidden-size.")
    parser.add_argument("--num-layers", type=int, default=None, help="Deprecated; see --hidden-size.")
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
            "Learning rate the e5scout matrix found. REQUIRED by --matrix e5, "
            "which has nothing else to centre on. Optional for every other "
            "matrix: each carries an OFT cell that runs a wide `oftscout` grid "
            "without this and a centred `oft` grid with it. OFT parameterizes a "
            "rotation, so no LoRA learning rate transfers to it -- there is no "
            "default that would not be an invented answer."
        ),
    )
    parser.add_argument(
        "--argmins-from",
        nargs="+",
        default=None,
        help="E1-1 ledger paths or globs. Required by --matrix e1long and "
        "meaningless elsewhere: the long curves only mean anything at each "
        "rank's own argmin learning rate, which E1-1 is what finds.",
    )
    parser.add_argument(
        "--allow-edge-argmin",
        action="store_true",
        help="Let --argmins-from accept an argmin sitting on a grid edge.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Regex; run only arms whose name matches (e.g. '^lora-r256' or '^oftscout').",
    )
    parser.add_argument(
        "--probe-rollouts", type=int, default=None,
        help="Cut every arm to this many rollouts and evaluate each one. For the "
             "coverage probe (scripts/lora_regret/coverage_probe.sh): proves a "
             "method runs and measures its per-rollout pace. Rows written under "
             "this flag carry `probe_rollouts` and `analyze` refuses any ledger "
             "containing one -- they are not measurements.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands, run nothing.")
    args = parser.parse_args()

    if args.matrix in MATRICES_REQUIRING_OFT_CENTRE and args.oft_lr_centre is None:
        # A clean exit rather than a traceback: this is the one argument an
        # operator following the runbook can only supply after another run.
        # Where it comes from differs per matrix, so name the source rather than
        # printing one matrix's instructions for another's failure.
        source = {
            "e5": "run --matrix e5scout first and pass its argmin",
            "e5rl": "take it from E4's oftscout argmin (--argmins-from results/e4*.jsonl)",
        }[args.matrix]
        parser.error(f"--matrix {args.matrix} requires --oft-lr-centre; {source}")
    # No inverse guard. Every matrix now carries an OFT cell, so a centre is
    # meaningful everywhere; e5 is only special in having nothing to fall back
    # on. `e1long` is the one matrix with no OFT arms -- its arms come from an
    # E1-1 ledger -- and passing a centre there is inert rather than wrong.
    if args.matrix == "e1long" and args.argmins_from is None:
        parser.error(
            "--matrix e1long requires --argmins-from; run --matrix e1 to completion "
            "first and point this at its ledgers"
        )
    if args.matrix != "e1long" and args.argmins_from is not None:
        parser.error(f"--argmins-from is only meaningful for --matrix e1long, not {args.matrix}")

    # The dimensions come from the registry now. A supplied flag is accepted
    # only when it agrees, because silently preferring one of two sources is how
    # a ledger ends up with every adapter_params wrong by a constant factor and
    # nothing in the output looking suspicious.
    selected_model = get_model(args.model)
    for flag, given, derived in (
        ("--hidden-size", args.hidden_size, selected_model.hidden_size),
        ("--ffn-size", args.ffn_size, selected_model.ffn_size),
        ("--num-layers", args.num_layers, selected_model.num_layers),
    ):
        if given is not None and given != derived:
            parser.error(
                f"{flag}={given} contradicts model {selected_model.key!r}, which has "
                f"{derived}. These are derived from the arm's model now; drop the flag."
            )

    repo_root = Path(__file__).resolve().parents[2]
    recovered = (
        argmins_from(args.argmins_from, args.allow_edge_argmin) if args.argmins_from else None
    )
    arms = MATRICES[args.matrix](
        selected_model.hidden_size, selected_model.ffn_size, selected_model.qkv_output_size,
        args.seed, args.oft_lr_centre, recovered,
    )
    # The builders stamp every arm with the registry's default model, since a
    # matrix is single-model by construction. Re-stamp rather than teach twelve
    # builders a new argument: `run_arm` reads `arm.model` to pick the checkpoint,
    # so an unstamped arm would be solved for Qwen3-1.7B's shapes and then *run*
    # on Llama-3.1-8B, with nothing in the arm name to show it.
    if selected_model.key != DEFAULT_MODEL:
        arms = [replace(arm, model=selected_model.key) for arm in arms]
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
        print(_oft_match_summary(selected_model.hidden_size), file=sys.stderr)

    launcher = MATRIX_LAUNCHERS[args.matrix]
    metric = MATRIX_METRICS[args.matrix]
    print(f"launcher={launcher} metric={metric}", file=sys.stderr)
    for i, arm in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {arm.name}", file=sys.stderr)
        model = get_model(arm.model)
        run_arm(
            arm, repo_root, args.results, args.dry_run,
            launcher=launcher, metric=metric, matrix=args.matrix,
            probe_rollouts=args.probe_rollouts,
            adapter_params=adapter_param_count(
                arm, model.hidden_size, model.ffn_size, model.num_layers,
                qkv_output_size=model.qkv_output_size,
            ),
        )


if __name__ == "__main__":
    main()
