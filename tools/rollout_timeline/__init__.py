"""Rollout throughput timeline probe + binning (standalone, stdlib-only).

- probe.py: polls SGLang engine HTTP endpoints (~100 ms) during a run and
  appends JSONL counter samples; scrape failures are recorded as data points.
- binning.py: pure functions turning probe JSONL + trainer-side event JSONL
  (ORBIT_TIMELINE_EVENTS_FILE markers emitted by
  orbit/peft/megatron/sync_metrics.py) into binned
  tokens/s series with weight-publication annotations for the figure script.
"""
