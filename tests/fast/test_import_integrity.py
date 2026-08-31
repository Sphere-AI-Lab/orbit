"""Every static orbit.*/miles_plugins.* import must resolve to a real module.

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


@pytest.mark.skipif(sys.version_info < (3, 12), reason="repo sources use PEP 695 syntax")
def test_the_checker_sees_untracked_files():
    """Reading the git index alone makes this guard pass VACUOUSLY on a file that
    has not been committed yet -- exactly when a dangling import is most likely
    to be there, since the file is new. The file list is therefore read per call,
    tracked and untracked alike."""
    probe = REPO_ROOT / "orbit" / "_import_integrity_untracked_probe.py"
    probe.write_text("from orbit.definitely_not_a_real_module import nothing\n")
    try:
        errors = CHECKER.collect_errors()
    finally:
        probe.unlink()
    assert any("_import_integrity_untracked_probe" in e for e in errors), (
        "the checker did not flag a dangling import in an untracked file; it is "
        "reading the index only again"
    )
