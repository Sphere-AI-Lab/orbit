#!/bin/bash
#
# lib/ib_probe.sh — tier-ib healthcheck: exit 0 iff every InfiniBand-layer
# rail is LinkUp. Missing/empty ibstat is a non-blocking WARN (exit 0).
# Pure bash + system `ibstat` (no torch). Why a Polling rail breaks NCCL
# and how this feeds requeue: see docs/launcher.md "Healthcheck".

set -uo pipefail

if ! command -v ibstat >/dev/null 2>&1; then
    echo "WARN: ibstat not found; skipping IB rail check on $(hostname)"
    exit 0
fi

# Capture first (|| true) so ibstat's own nonzero rc (it errors, or a
# `timeout` truncates it) can't trip pipefail and wrongly mark the node BAD;
# the awk parse below treats empty input as a benign WARN-skip.
ib_out="$(ibstat 2>/dev/null)" || true
printf '%s\n' "$ib_out" | awk '
    # Reset per-rail state at each CA so values cannot leak across blocks
    # (e.g. a `timeout`-truncated record). ibstat emits Physical state before
    # Link layer, so we evaluate pstate on the Link layer line.
    /^CA / { ca = $2; gsub(/'\''/, "", ca); pstate = ""; rate = ""; next }
    /Physical state:/ { pstate = $3; next }
    /Rate:/ { rate = $2; next }
    /Link layer:/ {
        layer = $3
        if (layer == "InfiniBand") {
            ib_total++
            if (pstate != "LinkUp") {
                bad = bad sprintf("%s(state=%s,rate=%s) ", ca, pstate, rate)
                nbad++
            } else {
                ib_ok++
            }
        }
    }
    END {
        if (ib_total == 0) {
            print "WARN: ibstat returned no InfiniBand rails; skipping"
            exit 0
        }
        if (nbad > 0) {
            printf "FAIL: %d/%d IB rails not LinkUp: %s\n", nbad, ib_total, bad
            exit 1
        }
        printf "OK: %d/%d IB rails LinkUp\n", ib_ok, ib_total
        exit 0
    }
'
