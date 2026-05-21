#!/bin/bash
#
# lib/manifest.sh — per-run MANIFEST.json read/write helpers.
#
# Sourced by launch_miles.sbatch (write side) and submit.sh (read side).
# All JSON I/O goes through python3 heredocs because bash JSON is fragile.
# See docs/launcher.md "MANIFEST.json schema" for the field list.

# write_manifest <state>
#
# Merge-writes $RUN_DIR/MANIFEST.json with the given state and a fresh
# updated_at. Preserves started_at on subsequent calls. Accepts optional
# extra fields via the MANIFEST_EXTRA_JSON env var (a JSON object string).
#
# Required env: RUN_DIR
# Optional env: SLURM_JOB_ID, HEAD_NODE, RECIPE, SLURM_RESTART_COUNT,
#               MANIFEST_EXTRA_JSON
write_manifest() {
    local state="$1"
    local now
    now=$(date -Is)
    HEAD_NODE_VAR="${HEAD_NODE:-}" STATE_ARG="$state" NOW_ARG="$now" \
    MANIFEST_PATH="$RUN_DIR/MANIFEST.json" python3 - <<'PY'
import json, os
path = os.environ['MANIFEST_PATH']
state, now = os.environ['STATE_ARG'], os.environ['NOW_ARG']
data = {}
if os.path.exists(path):
    try:
        with open(path) as f: data = json.load(f)
    except Exception: pass
data['state'] = state
data['updated_at'] = now
data.setdefault('started_at', now)
data['job_id'] = os.environ.get('SLURM_JOB_ID', '')
data['head_node'] = os.environ.get('HEAD_NODE_VAR', '')
data['run_dir'] = os.environ.get('RUN_DIR', '')
data['recipe'] = os.environ.get('RECIPE', '')
data['restarts'] = int(os.environ.get('SLURM_RESTART_COUNT', '0') or 0)
extra = os.environ.get('MANIFEST_EXTRA_JSON', '')
if extra:
    try: data.update(json.loads(extra))
    except Exception: pass
with open(path, 'w') as f:
    json.dump(data, f, indent=2, sort_keys=True); f.write("\n")
PY
}

# read_recent_manifests <runs_parent> [<n>] [<exclude_stamp>]
#
# Print one `[submit] WARN: prior run <stamp> ended <state> ...` line for
# each of the most recent n manifests under <runs_parent> whose state is
# not SUCCEEDED. Used by submit.sh to flag a still-broken job before re-
# submitting on top of a known-failing config.
#
# n defaults to 3. exclude_stamp (typically the new run's own stamp) is
# skipped so submit.sh doesn't warn about itself.
read_recent_manifests() {
    local parent="$1" n="${2:-3}" exclude="${3:-}"
    [[ -d "$parent" ]] || return 0
    PARENT="$parent" N="$n" EXCLUDE="$exclude" python3 - <<'PY' || true
import json, os
parent = os.environ['PARENT']
n = int(os.environ['N'])
exclude = os.environ['EXCLUDE']
stamps = sorted(d for d in os.listdir(parent)
                if d != exclude and os.path.isdir(os.path.join(parent, d)))
for stamp in stamps[-n:][::-1]:
    m = os.path.join(parent, stamp, "MANIFEST.json")
    if not os.path.exists(m): continue
    try:
        with open(m) as f: d = json.load(f)
    except Exception: continue
    state = d.get("state", "UNKNOWN")
    if state == "SUCCEEDED": continue
    print(f"[submit] WARN: prior run {stamp} ended {state} "
          f"(job_id={d.get('job_id','?')} node={d.get('head_node','?')})")
PY
}
