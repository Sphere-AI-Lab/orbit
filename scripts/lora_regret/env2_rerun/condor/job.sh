#!/usr/bin/env bash
# HTCondor executable for one env2 rerun launcher on a whole 8xH100 node.
#
#   job.sh <launcher basename> <condor cluster id>
#
# Not called by hand; the .sub files next to it pass both arguments. It records
# a status file for the job, refuses an allocation whose GPUs are not eight
# healthy H100s, and runs the launcher with stderr merged into stdout so the
# progress bars and log lines land in one file in order.

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 <launcher basename> <cluster id>" >&2
    exit 2
fi
launcher=$1
cluster=$2

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ENV2_DIR="$(cd -- "${HERE}/.." && pwd -P)"
ORBIT_ICLR_ROOT="$(cd -- "${ENV2_DIR}/../../.." && pwd -P)"

if [[ ! -f "${ENV2_DIR}/${launcher}" ]]; then
    echo "no such launcher: ${ENV2_DIR}/${launcher}" >&2
    exit 2
fi

# env.sh activates env2, exports every output directory and creates them.
# shellcheck disable=SC1091
source "${ENV2_DIR}/env.sh"

name="${launcher#run_}"
name="${name%_8gpu.sh}"
status="${E4_ENV2_SCHEDULER_DIR}/${name}.${cluster}.status"

exec 2>&1
cd "${ORBIT_ICLR_ROOT}"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname -s)"

write_status() {
    printf '%s\n' "state=$1" "commit=${commit}" "job_host=${host}" \
        "cluster=${cluster}" "launcher=${launcher}" "started_at=${started_at}" \
        "${@:2}" > "${status}"
}

started_at="$(now)"
write_status starting
printf 'launch_started_at=%s\nexecution_host=%s\ncommit=%s\nlauncher=%s\n' \
    "${started_at}" "${host}" "${commit}" "${launcher}"

# Eight healthy H100s, or do not start. A node with one dead GPU still matches
# the allocation (measured 2026-08-21 on i104: NVML handle failure on GPU 6),
# and preflight only counts devices, so the check has to be here.
gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 || true)"
gpu_count="$(printf '%s\n' "${gpu_names}" | grep -c . || true)"
bad_count="$(printf '%s\n' "${gpu_names}" | grep -vc 'NVIDIA H100 80GB HBM3' || true)"
if [[ "${gpu_count}" -ne 8 || "${bad_count}" -ne 0 ]]; then
    echo "allocation unhealthy: ${gpu_count} GPU(s) reported, ${bad_count} not H100 80GB HBM3:"
    printf '%s\n' "${gpu_names}"
    write_status allocation_unhealthy "finished_at=$(now)" "exit_code=3"
    exit 3
fi

rc=0
bash "${ENV2_DIR}/${launcher}" || rc=$?
if [[ "${rc}" -eq 0 ]]; then
    write_status finished "finished_at=$(now)" "exit_code=0"
else
    write_status failed "finished_at=$(now)" "exit_code=${rc}"
fi
printf 'launch_finished_at=%s\nexit_code=%s\n' "$(now)" "${rc}"
exit "${rc}"
