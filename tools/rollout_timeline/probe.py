"""Low-invasiveness rollout throughput probe.

Polls SGLang engine (or router) HTTP endpoints every ~100 ms during a training
run and appends one JSONL record per scrape:

    {"t_wall": ..., "engine_url": ..., "ok": true,
     "counters": {"sglang:realtime_tokens_total{mode=decode}": 12345.0, ...}}

Failed scrapes are recorded too (``"ok": false`` with an ``error`` field) — a
scrape gap IS signal: the engine was unresponsive, e.g. paused for a weight
update.

Endpoints:
- ``metrics`` (default): scrapes the Prometheus ``/metrics`` endpoint
  (requires the engine to run with ``--enable-metrics``) and sums the
  configured cumulative counters over their label sets. Defaults to
  ``sglang:realtime_tokens_total{mode=decode}`` (incremented per forward
  pass — fine-grained enough for 100 ms bins) plus
  ``sglang:generation_tokens_total`` (incremented per finished request) as a
  cross-check/fallback.
- ``server_info``: scrapes ``/server_info`` (always available) and records
  the ``last_gen_throughput`` gauge summed over ``internal_states``. Coarser
  (smoothed over SGLang's own log interval) — use when ``--enable-metrics``
  is off.

Run alongside training:

    python tools/rollout_timeline/probe.py \
        --urls http://host1:30000 http://host2:30000 \
        --out /path/to/probe.jsonl --interval 0.1

Stdlib only; safe to run from anywhere (no orbit imports).
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request

DEFAULT_COUNTER_SPECS = (
    "sglang:realtime_tokens_total{mode=decode}",
    "sglang:generation_tokens_total",
)

_PROM_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?"
    r"\s+(?P<value>[^\s]+)"
    r"(?:\s+\d+)?$"
)
_PROM_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus_text(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Parse Prometheus text exposition into (name, labels, value) samples."""
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE_RE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        labels_raw = match.group("labels") or ""
        labels = {key: val.encode().decode("unicode_escape") for key, val in _PROM_LABEL_RE.findall(labels_raw)}
        samples.append((match.group("name"), labels, value))
    return samples


def parse_counter_spec(spec: str) -> tuple[str, dict[str, str]]:
    """``name{label=value,...}`` -> (name, required-label dict)."""
    if "{" not in spec:
        return spec, {}
    name, _, rest = spec.partition("{")
    rest = rest.rstrip("}")
    label_filter = {}
    for part in rest.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition("=")
        label_filter[key.strip()] = val.strip().strip('"')
    return name, label_filter


def sum_counter(
    samples: list[tuple[str, dict[str, str], float]],
    spec: str,
) -> float | None:
    """Sum sample values matching a counter spec; None when absent entirely."""
    name, label_filter = parse_counter_spec(spec)
    total = 0.0
    found = False
    for sample_name, labels, value in samples:
        if sample_name != name:
            continue
        if any(labels.get(key) != val for key, val in label_filter.items()):
            continue
        total += value
        found = True
    return total if found else None


# Engines/routers are cluster-internal endpoints: never route scrapes through
# http_proxy/https_proxy env vars (login hosts often set them, which would
# 403/black-hole every poll).
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http_get(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"Accept": "*/*"})
    with _OPENER.open(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def scrape_engine(
    base_url: str,
    *,
    endpoint: str = "metrics",
    counter_specs: tuple[str, ...] = DEFAULT_COUNTER_SPECS,
    timeout: float = 0.5,
) -> dict:
    """One scrape of one engine. Never raises; failures become data points."""
    base = base_url.rstrip("/")
    try:
        if endpoint == "metrics":
            text = _http_get(f"{base}/metrics", timeout)
            samples = parse_prometheus_text(text)
            counters = {}
            for spec in counter_specs:
                value = sum_counter(samples, spec)
                if value is not None:
                    counters[spec] = value
            return {"ok": True, "counters": counters}
        if endpoint == "server_info":
            text = _http_get(f"{base}/server_info", timeout)
            info = json.loads(text)
            states = info.get("internal_states") or []
            throughputs = [
                float(state["last_gen_throughput"])
                for state in states
                if isinstance(state, dict) and state.get("last_gen_throughput") is not None
            ]
            gauges = {}
            if throughputs:
                gauges["last_gen_throughput"] = sum(throughputs)
            return {"ok": True, "gauges": gauges}
        return {"ok": False, "error": f"unknown endpoint {endpoint!r}"}
    except Exception as exc:  # noqa: BLE001 — every failure is a data point
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


class _JsonlWriter:
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, sort_keys=True)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def _poll_one_engine(
    url: str,
    writer: _JsonlWriter,
    *,
    interval: float,
    endpoint: str,
    counter_specs: tuple[str, ...],
    timeout: float,
    stop_event: threading.Event,
    duration: float | None,
) -> None:
    started = time.monotonic()
    while not stop_event.is_set():
        t_wall = time.time()
        record = {"t_wall": t_wall, "engine_url": url}
        record.update(
            scrape_engine(url, endpoint=endpoint, counter_specs=counter_specs, timeout=timeout)
        )
        writer.write(record)
        if duration is not None and time.monotonic() - started >= duration:
            break
        elapsed = time.time() - t_wall
        stop_event.wait(max(0.0, interval - elapsed))


def run_probe(
    urls: list[str],
    out_path: str,
    *,
    interval: float = 0.1,
    endpoint: str = "metrics",
    counter_specs: tuple[str, ...] = DEFAULT_COUNTER_SPECS,
    timeout: float | None = None,
    duration: float | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Poll every url on its own thread until duration elapses or stop_event
    is set (blocks the caller until all pollers finish)."""
    if timeout is None:
        timeout = max(interval, 0.25)
    stop_event = stop_event or threading.Event()
    writer = _JsonlWriter(out_path)
    threads = [
        threading.Thread(
            target=_poll_one_engine,
            args=(url, writer),
            kwargs=dict(
                interval=interval,
                endpoint=endpoint,
                counter_specs=counter_specs,
                timeout=timeout,
                stop_event=stop_event,
                duration=duration,
            ),
            daemon=True,
            name=f"rollout-timeline-probe-{url}",
        )
        for url in urls
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=2 * timeout)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--urls", nargs="+", required=True, help="Engine/router base URLs.")
    parser.add_argument("--out", required=True, help="Output JSONL path (appended).")
    parser.add_argument("--interval", type=float, default=0.1, help="Poll interval seconds (default 0.1).")
    parser.add_argument(
        "--endpoint",
        choices=["metrics", "server_info"],
        default="metrics",
        help="metrics: Prometheus counters (needs --enable-metrics on the engine); "
        "server_info: last_gen_throughput gauge fallback.",
    )
    parser.add_argument(
        "--counters",
        nargs="+",
        default=list(DEFAULT_COUNTER_SPECS),
        help="Counter specs for the metrics endpoint, e.g. "
        "'sglang:realtime_tokens_total{mode=decode}'.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-request timeout seconds (default: max(interval, 0.25)).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds (default: run until interrupted).",
    )
    args = parser.parse_args(argv)
    run_probe(
        args.urls,
        args.out,
        interval=args.interval,
        endpoint=args.endpoint,
        counter_specs=tuple(args.counters),
        timeout=args.timeout,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
