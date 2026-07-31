"""Per-rollout wall time, read out of a launcher log.

A separate module from `probe.py` on purpose: `sweep.py` records these seconds
on every ledger row, and `probe.py` imports `sweep.py` for the matrix tables --
so the parser living in `probe.py` would make that import a cycle.

`train.py:261` already logs one `progress` line per rollout, built by
`orbit/utils/training_eta.py`, for SFT and RL alike. Reading `last=` from it is
strictly better than subtracting wall clocks: it is the loop's own measurement,
it excludes startup by construction, and it needs no timestamp parsing.
"""

from __future__ import annotations

import re

# progress rollout=2/2 completed=3/3 remaining=0 elapsed=00:04:10 last=00:01:30
#          avg=00:01:40 eta_remaining=00:00:00 eta_at=2026-07-31 09:20:00
#
# The optional `<n>d ` group is not hypothetical: `format_duration` switches to
# `2d 03:04:05` past 24 hours, and a 29,323-rollout Tulu3 epoch crosses that in
# its own ETA field within the first few rollouts.
PROGRESS_LINE = re.compile(
    r"progress .*?\blast=(?:(?P<days>\d+)d )?(?P<h>\d+):(?P<m>\d\d):(?P<s>\d\d)"
)


def parse_rollout_seconds(log_text: str) -> list[float]:
    """Each completed rollout's own duration, in order."""
    out: list[float] = []
    for match in PROGRESS_LINE.finditer(log_text):
        days = int(match["days"] or 0)
        out.append(
            float(
                days * 86400
                + int(match["h"]) * 3600
                + int(match["m"]) * 60
                + int(match["s"])
            )
        )
    return out
