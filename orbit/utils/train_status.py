"""Liveness/terminal-state sentinel shared by train.py and train_async.py.

The slurm launcher's watchdog (scripts/slurm/lib/ray_lifecycle.sh) decides a job
is dead from `ray job status`. When the Ray Jobs API / dashboard is transiently
unreachable (a control-plane hiccup, independent of whether the training actors
are alive), the watchdog falls back to reading this sentinel.

So the driver writes:
  - "running" at startup and again as a HEARTBEAT every training step, so a job
    that is actually making progress keeps `updated_at` fresh — the watchdog
    treats a fresh "running" heartbeat as "alive, keep waiting" instead of
    declaring CLUSTER_DEAD;
  - a terminal "completed"/"failed" at exit, so the watchdog can attribute the
    final state correctly even if the Ray status API is down at the end.

Path resolution (so the launcher can point this at node-local disk instead of a
possibly-slow shared FS): ORBIT_TRAIN_STATUS_FILE if set, else
$ORBIT_RUN_DIR/train_status.json, else no-op.
"""

import json
import os
import sys
import threading
from datetime import datetime, timezone


# Latest training progress, set by the loop and read by the heartbeat thread, so a single
# sentinel carries BOTH signals that cross-validate each other:
#   - `updated_at` freshness  => the process is ALIVE (kept fresh even during warmup by the
#     background heartbeat thread, which is why a Ray-status outage during warmup no longer
#     false-kills);
#   - `step`                  => training is PROGRESSING (bumped by the per-step write).
# "alive + step advancing" = healthy; "alive + step frozen" = wedged (visible in the watchdog log).
_progress = {"step": None}


def set_progress(step):
    """Record the latest training step for the heartbeat to report."""
    _progress["step"] = step


def train_status_path():
    """Resolve the sentinel path, or None if neither env var is set."""
    path = os.environ.get("ORBIT_TRAIN_STATUS_FILE")
    if path:
        return path
    run_dir = os.environ.get("ORBIT_RUN_DIR")
    if run_dir:
        return os.path.join(run_dir, "train_status.json")
    return None


def write_train_status(state, rc=None, error=None):
    """Atomically write the sentinel. Best-effort: never raises into the caller.

    `state` is one of "running" (startup + per-step heartbeat), "completed", or
    "failed". `updated_at` is what makes the heartbeat meaningful to the watchdog.
    """
    path = train_status_path()
    if not path:
        return

    payload = {
        "state": state,
        "rc": rc,
        "step": _progress.get("step"),
        "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)[:4000]

    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception as write_error:
        print(f"[train-status] WARN: failed to write {path}: {write_error}", file=sys.stderr)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def start_heartbeat(interval=60.0):
    """Background daemon that refreshes the sentinel every `interval` seconds.

    This is the PROCESS-LIVENESS signal: it keeps `updated_at` fresh no matter what the
    main thread is doing (long warmup / bridge model-load / save / eval), so the launcher
    watchdog won't mistake a slow-but-alive driver for a dead cluster during a transient
    Ray-status outage. The per-step write_train_status() calls remain the PROGRESS signal
    (they bump `step`); together the reader can tell "alive AND progressing" from "alive
    but wedged".

    Returns a stop() callable. Call it BEFORE writing the terminal state so the thread
    can't overwrite the terminal record with a stale "running".
    """
    stop_event = threading.Event()

    def _loop():
        while not stop_event.is_set():
            write_train_status("running")
            if stop_event.wait(interval):
                break

    thread = threading.Thread(target=_loop, name="train-status-heartbeat", daemon=True)
    thread.start()

    def stop():
        stop_event.set()
        thread.join(timeout=interval + 5)

    return stop
