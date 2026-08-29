"""Per-rollout wall time, read out of a launcher log.

A separate module from `probe.py` on purpose: `sweep.py` records these seconds
on every ledger row, and `probe.py` imports `sweep.py` for the matrix tables --
so the parser living in `probe.py` would make that import a cycle.

`train.py:261` already logs one `progress` line per rollout, built by
`orbit/peft/utils/training_eta.py`, for SFT and RL alike. Reading `last=` from it is
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


# (MegatronTrainRayActor pid=...) [ts] timer.py:32 - Timer save_model end (elapsed: 616.5s)
#
# The actor's own timer, for the same reason the rollout durations come from
# train.py's progress line rather than from wall clocks: it measures the save
# and nothing around it.
SAVE_TIMER_LINE = re.compile(r"Timer save_model end \(elapsed: (?P<sec>[0-9.]+)s\)")


def parse_save_seconds(log_text: str) -> list[float]:
    """Each checkpoint write's duration, in order.

    Priced separately from the rollouts it happens to land inside. A FullFT arm
    writes ~15 GB of weights plus distributed-optimizer state and took 616.5s on
    Lustre; folded into a per-rollout average it doubled the campaign estimate.
    LoRA and OFT write adapters only, so theirs are small -- but they are the
    same measurement and are read the same way.
    """
    return [float(m["sec"]) for m in SAVE_TIMER_LINE.finditer(log_text)]


# scripts/lib/launcher.sh:68 echoes this as the first line it writes, right
# after `exec > >(... tee -a "${RUN_LOG}")`. The -a is the reason this function
# exists: sweep.py points RUN_LOG at a FIXED path per arm
# (logs/lora_regret/<arm>.log), and a `failed` arm is retried by the next
# campaign invocation, so attempt N+1 APPENDS to attempt N's file.
#
# Pinned against the launcher's own text by
# test_the_run_start_marker_is_the_line_the_launcher_actually_writes.
RUN_START_MARKER = "Logging to "


def last_run_segment(log_text: str) -> str:
    """Only the most recent launcher invocation's output.

    Every parser here and in sweep.py reads a whole file and answers a question
    about ONE run, which silently stops being the same thing the moment an arm
    is retried. Measured on `full-na-na-gsm8k-lr5e-07-s0`: three invocations in
    one file -- 108 rollouts, a startup failure, then a complete 150 -- and the
    ledger row recorded `rollout_seconds` of length 258, a pace summary for a
    run that never happened. `parse_final_nll` has the same exposure, taking a
    max over `step` across attempts that trained different amounts.

    The last segment rather than the largest: a retry exists because the
    previous attempt did not finish, so the newest is the one the ledger row is
    about. A log with no marker at all is returned unchanged -- an older log, or
    a caller's synthetic text, should read as one run rather than as nothing.
    """
    index = log_text.rfind(f"\n{RUN_START_MARKER}")
    if index == -1:
        return log_text
    return log_text[index + 1 :]


# (MegatronTrainRayActor pid=...) [ts] log_utils.py:54 - rollout 100: {'rollout/
# response_lengths': 152.3, 'rollout/rewards': 0.0, 'rollout/truncated': 0.0009,
# 'rollout/raw_reward': 0.723, ...}
#
# `raw_reward` NOT `rewards`: with GRPO centring the advantage is the reward
# minus its group mean, so `rollout/rewards` is ~0 on every healthy rollout and
# reads as a dead run. `raw_reward` is the uncentred mean, and with --rm-type
# math the reward is exactly 1 or 0, so it IS accuracy on the training batch.
ROLLOUT_METRICS_LINE = re.compile(r"rollout (?P<rollout_id>\d+): \{(?P<body>[^}]*)\}")


def _metric(body: str, key: str) -> float | None:
    match = re.search(rf"'rollout/{key}': (?P<value>[0-9.eE+-]+)", body)
    return float(match["value"]) if match else None


def parse_reward_trace(log_text: str) -> list[dict]:
    """The training-reward curve: one entry per rollout, in rollout order.

    Not the study's headline number -- that is held-out accuracy, and this is
    the mean reward on the training batch the policy just generated. It is what
    survives when the eval does not: the E4 gsm8k columns ran to completion with
    no post-training eval at all (train.py's generation-eval call omitted
    `num_rollout`, so its final-rollout branch was dead), and this curve is the
    only record of what those 40 node-hours learned.

    `truncated` and `response_len` ride along because they are what makes the
    curve legible. Every collapsed arm in that campaign has the same signature:
    response length climbs into the 2,048-token cap, a truncated answer has lost
    its \\boxed{...} so it grades 0 however well it argued, reward goes to zero,
    and with it the advantages -- after which there is no gradient signal and
    the arm cannot recover. Reward alone shows a run dying; with these two it is
    clear what killed it.

    Later entries win on a duplicated rollout id, so a caller that skips
    `last_run_segment` still sees the newest attempt rather than a mixture --
    but it sees the old attempt's TAIL beyond the new one's length, which is
    exactly the silent mixture this returns a list to make visible.
    """
    by_id: dict[int, dict] = {}
    for match in ROLLOUT_METRICS_LINE.finditer(log_text):
        body = match["body"]
        reward = _metric(body, "raw_reward")
        if reward is None:
            continue  # an eval dict or a metrics line from before raw_reward existed
        by_id[int(match["rollout_id"])] = {
            "rollout": int(match["rollout_id"]),
            "reward": reward,
            "truncated": _metric(body, "truncated"),
            "response_len": _metric(body, "response_lengths"),
        }
    return [by_id[key] for key in sorted(by_id)]
