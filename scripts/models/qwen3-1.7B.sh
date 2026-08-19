# scripts/models/qwen3-1.7B.sh — fork compatibility shim (2026-08-18 sync).
#
# Upstream converted every scripts/models/*.sh into python model_args()
# scripts (see the sibling qwen3-1.7B.py, consumed via
# miles.utils.external_utils.model_args_utils.load_model_args). Existing bash
# recipes keep sourcing this shim, which delegates to the python source of
# truth — so model args cannot drift between the two forms.
#
# NEW recipes must NOT add more shims: consume load_model_args('<model>')
# directly (python launchers) or inline the one-liner below (bash launchers).
MODEL_ARGS=($(python3 -c "from miles.utils.external_utils.model_args_utils import load_model_args; print(load_model_args('qwen3-1.7B'))"))
