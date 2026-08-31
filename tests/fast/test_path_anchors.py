"""Every `Path(__file__)`-derived upward walk must anchor inside the repo, and
any all-literal join off it must resolve to a path that exists.

Guards the classic move-proof-anchor failure mode: a file moves, a fixed
`parents[N]` index still returns SOME directory, but silently the wrong one
-- invisible until whatever lives under it fails to load. See
tools/check_path_anchors.py.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "check_path_anchors.py"

SPEC = importlib.util.spec_from_file_location("orbit_check_path_anchors", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="repo sources use PEP 695 syntax")
def test_all_path_anchors_are_valid():
    errors = CHECKER.collect_errors()
    assert not errors, "broken path anchors:\n  " + "\n  ".join(errors)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="repo sources use PEP 695 syntax")
def test_the_checker_sees_untracked_files():
    """A guard that reads only the git index passes VACUOUSLY on a file that has
    not been committed yet -- exactly when a new anchor is most likely to be
    wrong. tests/fast/test_arming.py landed carrying a finding this guard could
    not see until the commit that added it, so the blind spot is not theoretical.
    """
    probe = REPO_ROOT / "tools" / "_path_anchor_untracked_probe.py"
    probe.write_text(
        "from pathlib import Path\n"
        'BAD = Path(__file__).resolve().parents[1] / "no" / "such" / "file.py"\n'
    )
    try:
        errors = CHECKER.collect_errors()
    finally:
        probe.unlink()
    assert any("_path_anchor_untracked_probe" in e for e in errors), (
        "the checker did not flag a bad anchor in an untracked file; it is "
        "reading the index only again"
    )
