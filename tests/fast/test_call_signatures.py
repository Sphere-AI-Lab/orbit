"""Every call whose callee resolves uniquely in-repo must agree with its signature.

Guards the isolation refactor's failure mode: a callee gains a required parameter
(upstream's `forward_only(..., rollout_id)`) and one mechanically-ported caller is
left behind, so the TypeError is only reachable on GPU. See
tools/check_call_signatures.py, which under-reports by design.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "check_call_signatures.py"

SPEC = importlib.util.spec_from_file_location("orbit_check_call_signatures", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="repo sources use PEP 695 syntax")
def test_all_resolvable_calls_match_their_callee_signature():
    errors = CHECKER.collect_errors()
    assert not errors, "call-signature mismatches:\n  " + "\n  ".join(errors)
