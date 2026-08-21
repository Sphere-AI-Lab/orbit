#!/usr/bin/env bash
# Submit one or more env2 rerun jobs.
#
#   bash scripts/lora_regret/env2_rerun/condor/submit.sh [--bid N] NAME...
#
# NAME is a .sub basename without the extension, e.g. e4_math_ft_lr1_lr7.
# The bid defaults to 35, the value the interactive H100 allocations used.

set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
bid=35
if [[ "${1:-}" == "--bid" ]]; then
    bid=$2
    shift 2
fi
if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 [--bid N] NAME..." >&2
    echo "names:" >&2
    ls "${HERE}"/*.sub | sed 's#.*/##; s#\.sub$##; s#^#  #' >&2
    exit 2
fi
for name in "$@"; do
    sub="${HERE}/${name%.sub}.sub"
    if [[ ! -f "${sub}" ]]; then
        echo "no such submit file: ${sub}" >&2
        exit 2
    fi
    condor_submit_bid "${bid}" "${sub}"
done
