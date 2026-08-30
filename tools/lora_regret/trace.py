"""The held-out NLL curve for one arm, extracted from its launcher log.

`sweep.parse_final_nll` answers "what did this arm score". This module answers
"how did it get there", which is what C1's departure step is measured from and
what no ledger field previously carried.

The line regex lives here and `sweep.py` imports it. It is built from
`EVAL_NLL_METRIC_KEY` rather than a re-spelled "eval/test_nll" literal so a
rename of that constant cannot silently desync the parser from the metric it
tracks -- and a second copy of the regex would reintroduce precisely that risk,
which is why this is a move rather than an addition.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from miles.orbit.utils.eval_nll import EVAL_NLL_METRIC_KEY

# train.py:_log_eval_nll emits one line per held-out NLL measurement, e.g.:
#
#   eval/test_nll rollout_id=12 step=12 phase=after_train nll=1.845700 \
#       sample_mean=1.801234 tokens=4096 samples=32
NLL_LINE = re.compile(
    re.escape(EVAL_NLL_METRIC_KEY)
    + r" rollout_id=(?P<rollout_id>\d+) step=(?P<step>\d+) phase=(?P<phase>\S+)"
    r" nll=(?P<nll>[0-9.]+) sample_mean=(?P<sample_mean>[0-9.]+)"
    r" tokens=(?P<tokens>\d+) samples=(?P<samples>\d+)"
)
# "before_train" is the untouched base model, logged once at rollout/step 0
# before any optimizer step -- gate G4's number. "after_train" is a
# post-optimizer-step measurement from the periodic hook.
PHASE_BEFORE_TRAIN = "before_train"
PHASE_AFTER_TRAIN = "after_train"


class NllPoint(NamedTuple):
    rollout_id: int
    step: int
    phase: str
    nll: float
    sample_mean: float
    tokens: int
    samples: int


def parse_trace(log_text: str) -> list[NllPoint]:
    """Every held-out measurement in the log, in measurement order.

    Both phases are retained: `before_train` is a meaningful number (the
    pristine base model), it simply must never be picked as an arm's *result* --
    that exclusion belongs in `parse_final_nll`, not here.

    Sorted by `(step, phase != before_train)` rather than by file position.
    Multi-rank log buffering can place the two step-0 rows in either physical
    order, and at equal step the base-model measurement is by construction the
    earlier one.
    """
    points = [
        NllPoint(
            rollout_id=int(m["rollout_id"]),
            step=int(m["step"]),
            phase=m["phase"],
            nll=float(m["nll"]),
            sample_mean=float(m["sample_mean"]),
            tokens=int(m["tokens"]),
            samples=int(m["samples"]),
        )
        for m in NLL_LINE.finditer(log_text)
    ]
    return sorted(points, key=lambda p: (p.step, p.phase != PHASE_BEFORE_TRAIN))


def parse_trace_file(path: str | Path) -> list[NllPoint]:
    return parse_trace(Path(path).read_text(encoding="utf-8", errors="replace"))


def trace_is_consistent(points: list[NllPoint]) -> tuple[bool, str]:
    """Whether every measurement scored the same held-out set.

    `get_data_iterator` floor-divides, so 1,000 rows at global batch 32 would
    silently become 992 and the metric would start depending on batch size --
    which is the axis E2 varies, so the gap E2 measures would be partly an
    artifact of its own instrument. Returns the reason as text so the caller can
    put it in a ledger rather than only in a traceback.
    """
    if not points:
        return False, "empty trace: no eval/test_nll lines in the log"
    tokens = sorted({p.tokens for p in points})
    samples = sorted({p.samples for p in points})
    if len(tokens) > 1 or len(samples) > 1:
        return False, (
            f"held-out set changed size mid-run: tokens={tokens} samples={samples}; "
            "get_data_iterator floor-divides, so this metric depends on batch size"
        )
    return True, ""
