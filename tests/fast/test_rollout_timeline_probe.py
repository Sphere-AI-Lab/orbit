"""CPU tests for tools/rollout_timeline/probe.py.

Prometheus text parsing on synthetic payloads, and the probe loop against a
local stdlib HTTP server (including failing endpoints -> failure records).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools.rollout_timeline.probe import (
    DEFAULT_COUNTER_SPECS,
    parse_counter_spec,
    parse_prometheus_text,
    run_probe,
    scrape_engine,
    sum_counter,
)

PROM_TEXT = """\
# HELP sglang:generation_tokens_total Number of generation tokens processed.
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="qwen"} 120.0
sglang:realtime_tokens_total{model_name="qwen",mode="decode"} 40.0
sglang:realtime_tokens_total{model_name="qwen",mode="prefill_compute"} 300.0
sglang:realtime_tokens_total{other="x",mode="decode"} 2.0
sglang:gen_throughput 15.5
malformed line without value or
"""


# ---------------------------------------------------------------------------
# Prometheus text parsing
# ---------------------------------------------------------------------------


def test_parse_prometheus_text_names_labels_values():
    samples = parse_prometheus_text(PROM_TEXT)
    by_name = {}
    for name, labels, value in samples:
        by_name.setdefault(name, []).append((labels, value))
    assert by_name["sglang:generation_tokens_total"] == [({"model_name": "qwen"}, 120.0)]
    assert len(by_name["sglang:realtime_tokens_total"]) == 3
    assert by_name["sglang:gen_throughput"] == [({}, 15.5)]
    assert "malformed" not in by_name


def test_parse_counter_spec():
    assert parse_counter_spec("sglang:generation_tokens_total") == (
        "sglang:generation_tokens_total",
        {},
    )
    assert parse_counter_spec("sglang:realtime_tokens_total{mode=decode}") == (
        "sglang:realtime_tokens_total",
        {"mode": "decode"},
    )


def test_sum_counter_filters_by_label_and_sums_across_series():
    samples = parse_prometheus_text(PROM_TEXT)
    assert sum_counter(samples, "sglang:realtime_tokens_total{mode=decode}") == pytest.approx(42.0)
    assert sum_counter(samples, "sglang:realtime_tokens_total") == pytest.approx(342.0)
    assert sum_counter(samples, "sglang:generation_tokens_total") == pytest.approx(120.0)
    assert sum_counter(samples, "sglang:missing_total") is None


# ---------------------------------------------------------------------------
# Fake engine HTTP server
# ---------------------------------------------------------------------------


class _FakeEngineHandler(BaseHTTPRequestHandler):
    # Class-level mutable state shared with the test.
    state = {"tokens": 0.0, "step": 10.0, "fail_metrics": False}

    def log_message(self, *args):  # silence test output
        pass

    def do_GET(self):
        if self.path == "/metrics":
            if self.state["fail_metrics"]:
                self.send_response(503)
                self.end_headers()
                return
            self.state["tokens"] += self.state["step"]
            body = (
                f'sglang:realtime_tokens_total{{mode="decode"}} {self.state["tokens"]}\n'
                f'sglang:generation_tokens_total{{model_name="m"}} {self.state["tokens"] / 2}\n'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/server_info":
            body = json.dumps(
                {"internal_states": [{"last_gen_throughput": 123.5}, {"last_gen_throughput": 6.5}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def fake_engine():
    _FakeEngineHandler.state = {"tokens": 0.0, "step": 10.0, "fail_metrics": False}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeEngineHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield url
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


# ---------------------------------------------------------------------------
# scrape_engine
# ---------------------------------------------------------------------------


def test_scrape_engine_metrics(fake_engine):
    record = scrape_engine(fake_engine, endpoint="metrics", timeout=2.0)
    assert record["ok"]
    counters = record["counters"]
    assert counters["sglang:realtime_tokens_total{mode=decode}"] == pytest.approx(10.0)
    assert counters["sglang:generation_tokens_total"] == pytest.approx(5.0)


def test_scrape_engine_server_info(fake_engine):
    record = scrape_engine(fake_engine, endpoint="server_info", timeout=2.0)
    assert record["ok"]
    assert record["gauges"]["last_gen_throughput"] == pytest.approx(130.0)


def test_scrape_engine_http_error_is_a_data_point(fake_engine):
    _FakeEngineHandler.state["fail_metrics"] = True
    record = scrape_engine(fake_engine, endpoint="metrics", timeout=2.0)
    assert record["ok"] is False
    assert "error" in record


def test_scrape_engine_connection_refused_is_a_data_point():
    record = scrape_engine("http://127.0.0.1:1", endpoint="metrics", timeout=0.2)
    assert record["ok"] is False
    assert "error" in record


# ---------------------------------------------------------------------------
# Probe loop
# ---------------------------------------------------------------------------


def test_run_probe_appends_monotonic_counter_records(fake_engine, tmp_path):
    out = tmp_path / "probe.jsonl"
    run_probe(
        [fake_engine],
        str(out),
        interval=0.02,
        endpoint="metrics",
        counter_specs=DEFAULT_COUNTER_SPECS,
        timeout=2.0,
        duration=0.15,
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) >= 3
    decode_key = "sglang:realtime_tokens_total{mode=decode}"
    values = []
    for record in records:
        assert record["engine_url"] == fake_engine
        assert record["ok"]
        assert isinstance(record["t_wall"], float)
        values.append(record["counters"][decode_key])
    assert values == sorted(values)  # cumulative counter is non-decreasing
    assert values[-1] > values[0]


def test_run_probe_records_failures_and_recovers(fake_engine, tmp_path):
    out = tmp_path / "probe.jsonl"
    bad_url = "http://127.0.0.1:1"
    run_probe(
        [fake_engine, bad_url],
        str(out),
        interval=0.02,
        endpoint="metrics",
        counter_specs=DEFAULT_COUNTER_SPECS,
        timeout=0.2,
        duration=0.1,
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    by_url = {}
    for record in records:
        by_url.setdefault(record["engine_url"], []).append(record)
    assert all(r["ok"] for r in by_url[fake_engine])
    assert all(not r["ok"] and "error" in r for r in by_url[bad_url])
    assert len(by_url[bad_url]) >= 1
