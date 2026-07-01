#!/bin/bash
#
# gpu_deep_probe.sh — deep GPU health that tier1 (count) and tier2 (set_device) MISS.
# A GPU can pass both yet be silently degrading and then fail or CRAWL under real
# training load — the "looks fine, kills/slows the run" class. Zero-intrusion
# (nvidia-smi queries only). Flags:
#   - uncorrectable ECC errors (aggregate) > 0        -> FAIL: failing GPU memory
#   - HW/thermal/power-brake clock throttling active  -> RISK: slow under load
#   - PCIe link below max gen/width                   -> RISK: silently slow H2D/D2H
#   - NVLink link(s) inactive                         -> RISK: slow intra-node collectives
# Benign throttle bits (GpuIdle 0x1, AppClocks 0x2, SwPowerCap 0x4) are NOT flagged;
# only HwSlowdown 0x8 | SwThermal 0x20 | HwThermal 0x40 | HwPowerBrake 0x80 (mask 0xE8).
#
# Exit 1 on a hard fault (uncorrectable ECC); else 0 with RISK lines surfaced (throttle
# can be transient under load, so it warns rather than hard-fails — interpret per state).

set -uo pipefail
host=$(hostname)
rc=0
command -v nvidia-smi >/dev/null 2>&1 || { echo "WARN: $host nvidia-smi not found"; exit 0; }

# (1) uncorrectable ECC (aggregate) — a hard, unambiguous fault.
ecc=$(nvidia-smi --query-gpu=index,ecc.errors.uncorrected.aggregate.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' '$2 ~ /^[0-9]+$/ && $2+0 > 0 {printf "gpu%s=%s ", $1, $2}')
[[ -n "$ecc" ]] && { echo "FAIL: $host uncorrectable ECC errors: $ecc"; rc=1; }

# (2) real clock throttling — mask out benign bits (bash arithmetic handles 0x hex).
# clocks_throttle_reasons.active was renamed clocks_event_reasons.active on newer
# nvidia-smi (555+); try the legacy field first, fall back to the new one, and use
# whichever returns hex reasons. If neither does (very new/old driver), skip silently
# rather than false-fail.
throttle_csv=""
for field in clocks_throttle_reasons.active clocks_event_reasons.active; do
    out=$(nvidia-smi --query-gpu=index,"$field" --format=csv,noheader 2>/dev/null) || continue
    [[ "$out" == *0x* ]] && { throttle_csv=$out; break; }
done
thr=""
while read -r idx active; do
    [[ "$active" =~ ^0x[0-9A-Fa-f]+$ ]] || continue
    (( active & 0xE8 )) && thr="$thr gpu${idx}=${active}"
done < <(printf '%s\n' "$throttle_csv" | awk -F', *' '{print $1, $2}')
[[ -n "$thr" ]] && echo "RISK: $host GPU thermal/power/HW throttling active (slow under load):$thr"

# (3) PCIe link degraded (current < max gen or width).
pcie=$(nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' '($2 ~ /^[0-9]+$/ && $2+0 < $3+0) || ($4 ~ /^[0-9]+$/ && $4+0 < $5+0) {printf "gpu%s(gen%s/%s,x%s/%s) ", $1,$2,$3,$4,$5}')
[[ -n "$pcie" ]] && echo "RISK: $host PCIe link below max (silently slow H2D/D2H): $pcie"

# (4) NVLink inactive links.
nvl=""
if nvidia-smi nvlink -s >/dev/null 2>&1; then
    inact=$(nvidia-smi nvlink -s 2>/dev/null | grep -ciE 'inactive')
    [[ "${inact:-0}" -gt 0 ]] && { echo "RISK: $host $inact NVLink(s) inactive (slow intra-node all-reduce)"; nvl="nvlink"; }
fi

[[ $rc -eq 0 && -z "$thr$pcie$nvl" ]] && echo "OK: $host GPU deep health clean (ECC 0, no bad throttle, PCIe/NVLink full)"
exit $rc
