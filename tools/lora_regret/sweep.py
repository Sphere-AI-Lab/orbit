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

from orbit.utils.eval_nll import EVAL_NLL_METRIC_KEY
from orbit.utils.peft_param_match import match_report
from tools.lora_regret.arms import MATRICES, Arm, arm_env, sft_arms  # noqa: F401  (sft_arms re-exported)

# The campaign re-anchored from Qwen3-4B/No-Robots to Llama-3.1-8B/Tulu3, and
# this constant was left pointing at the old repo's launcher, which does not
# exist here. Pinned by test_the_launcher_the_sweep_shells_out_to_exists, because
# the failure mode is every arm failing identically for one silent reason.
LAUNCHER = "examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh"

# train.py:_log_eval_nll emits one line per held-out NLL measurement, e.g.:
#
#   eval/test_nll rollout_id=12 step=12 phase=after_train nll=1.845700 \
#       sample_mean=1.801234 tokens=4096 samples=32
#
# `phase` is "before_train" (the untouched base model -- gate G4's number,
# logged once at rollout/step==0 before any optimizer step runs) or
# "after_train" (a post-optimizer-step measurement from the periodic hook,
# which train.py forces to fire on the final rollout regardless of
# eval_nll_interval). The regex is built from EVAL_NLL_METRIC_KEY rather than
# a re-spelled "eval/test_nll" literal so a rename of that constant cannot
# silently desync the parser from the metric it is meant to track.
_NLL_LINE = re.compile(
    re.escape(EVAL_NLL_METRIC_KEY)
    + r" rollout_id=(?P<rollout_id>\d+) step=(?P<step>\d+) phase=(?P<phase>\S+)"
    r" nll=(?P<nll>[0-9.]+) sample_mean=(?P<sample_mean>[0-9.]+)"
    r" tokens=(?P<tokens>\d+) samples=(?P<samples>\d+)"
)
_PHASE_BEFORE_TRAIN = "before_train"
_PHASE_AFTER_TRAIN = "after_train"


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


def run_arm(arm: Arm, repo_root: Path, results_path: Path, dry_run: bool) -> None:
    log_path = repo_root / "logs" / "lora_regret" / f"{arm.name}.log"
    env = dict(os.environ)
    env.update(arm_env(arm))
    env.update(
        {
            "LAUNCHER_NAME": arm.name,
            "RUN_LOG": str(log_path),
            "WANDB_GROUP": "lora-regret-sft",
            "SAVE_DIR": str(repo_root / "orbit_ckpts" / "lora_regret" / arm.name),
        }
    )
    cmd = ["bash", str(repo_root / LAUNCHER)]
    if dry_run:
        overrides = " ".join(f"{k}={v}" for k, v in sorted(arm_env(arm).items()))
        print(f"{overrides} bash {LAUNCHER}")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, env=env, cwd=repo_root)
    nll, steps = (None, None)
    if log_path.exists():
        nll, steps = parse_final_nll(log_path.read_text(encoding="utf-8", errors="replace"))

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
            "test_nll": nll,
            "adapter_params": None,
            "wandb_run_id": None,
            "steps": steps,
            "status": "ok" if (proc.returncode == 0 and nll is not None) else "failed",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--ffn-size", type=int, required=True)
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
        "--only",
        default=None,
        help="Regex; run only arms whose name matches (e.g. '^lora-r256' or '^oftscout').",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands, run nothing.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    arms = MATRICES[args.matrix](args.hidden_size, args.ffn_size, args.seed)
    if args.only:
        pattern = re.compile(args.only)
        arms = [a for a in arms if pattern.search(a.name)]

    done = load_ledger(args.results)
    todo = [a for a in arms if a.name not in done]
    print(f"{len(arms)} arms selected, {len(done)} already done, {len(todo)} to run", file=sys.stderr)
    print(_oft_match_summary(args.hidden_size), file=sys.stderr)

    for i, arm in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {arm.name}", file=sys.stderr)
        run_arm(arm, repo_root, args.results, args.dry_run)


if __name__ == "__main__":
    main()
