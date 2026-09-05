# scripts/models/qwen2.5-3B.sh — fork compatibility shim (2026-08-18 sync).
#
# Upstream converted every scripts/models/*.sh into python model_args()
# scripts (see the sibling qwen2.5-3B.py, consumed via
# orbit.utils.external_utils.model_args_utils.load_model_args). Existing bash
# recipes keep sourcing this shim, which delegates to the python source of
# truth — so model args cannot drift between the two forms.
#
# NEW recipes must NOT add more shims: consume load_model_args('<model>')
# directly (python launchers) or inline the one-liner below (bash launchers).
# Env-assignment prefix: recipes set knobs like MODEL_ARGS_ROTARY_BASE as plain
# shell vars (the old sourced .sh saw them); the python child only sees exports,
# so forward them explicitly.
MODEL_ARGS=($(MODEL_ARGS_ROTARY_BASE="${MODEL_ARGS_ROTARY_BASE:-}" python3 -c "from orbit.utils.external_utils.model_args_utils import load_model_args; print(load_model_args('qwen2.5-3B'))"))
