"""Env-gated JSONL dump of OPD teacher log-probs (M1 correctness leg).

Enabled by ORBIT_OPD_TEACHER_LOGPROB_DUMP=<path>. Only the first
ORBIT_OPD_TEACHER_LOGPROB_DUMP_LIMIT rollouts (default 1) are dumped, on
rank 0 only -- this is a fixed-batch equivalence probe, not telemetry.

Pure stdlib module (mirrors miles.orbit.utils.logprob_compare) so it is cheap to
import unconditionally at the top of the instrumented files.

Record shape: ``{"rollout": int, "sample_index": int, "tokens": [int...],
"teacher_log_probs": [float...]}``. ``tokens`` matches the real
``miles.utils.types.Sample.tokens`` field -- the full prompt+response token
ids (Sample has no separate response-only token field) -- so it also serves
as the compare CLI's identity/join key across two dumps of "the same batch".
"""

from __future__ import annotations

import json
import os

ENV_PATH = "ORBIT_OPD_TEACHER_LOGPROB_DUMP"
ENV_LIMIT = "ORBIT_OPD_TEACHER_LOGPROB_DUMP_LIMIT"


def dump_teacher_logprob_records(path: str, records: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def maybe_dump_teacher_logprobs(rollout_id: int, samples) -> None:
    """Call at the point where samples carry .teacher_log_probs; no-op unless enabled.

    ``samples`` is any iterable of sample-like objects exposing ``.tokens``
    (full prompt+response token ids) and ``.teacher_log_probs`` (per-response-
    token floats, or None when this sample was not OPD-scored). Real
    ``miles.utils.types.Sample`` objects satisfy this directly (sglang attach
    site); the megatron attach site has no Sample objects in scope at the
    point teacher log-probs are computed (they land on the batch-level
    ``rollout_data``/``teacher_data`` dicts instead), so it synthesizes
    lightweight ``types.SimpleNamespace(tokens=..., teacher_log_probs=...)``
    stand-ins with the same two attributes before calling this.
    """
    path = os.environ.get(ENV_PATH)
    if not path:
        return
    limit = int(os.environ.get(ENV_LIMIT, "1"))
    if rollout_id >= limit:
        return
    records = []
    for index, sample in enumerate(samples):
        teacher_lp = getattr(sample, "teacher_log_probs", None)
        if teacher_lp is None:
            continue
        records.append(
            {
                "rollout": rollout_id,
                "sample_index": index,
                "tokens": [int(t) for t in getattr(sample, "tokens", [])],
                "teacher_log_probs": [float(x) for x in teacher_lp],
            }
        )
    if records:
        dump_teacher_logprob_records(path, records)
