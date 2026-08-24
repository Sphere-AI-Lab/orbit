#!/usr/bin/env bash
# M1 teacher-cost arm: base. Selects only the variant; the common
# recipe owns every scientific hyperparameter.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPD_COST_VARIANT=base
source "${SCRIPT_DIR}/opd_teacher_cost_common.sh"
