"""Every static miles.*/miles_plugins.*/orbit.* import must resolve to a real
module (top-level orbit.* no longer exists, so any such import is stale and fails).

Guards the isolation refactor's failure mode: a file moves, the old package still
imports (so nothing crashes at import-scan time), but `from pkg import module`
now names a ghost. See tools/check_import_integrity.py.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "check_import_integrity.py"

SPEC = importlib.util.spec_from_file_location("orbit_check_import_integrity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="repo sources use PEP 695 syntax")
def test_all_orbit_imports_resolve():
    errors = CHECKER.collect_errors()
    assert not errors, "dangling orbit imports:\n  " + "\n  ".join(errors)
