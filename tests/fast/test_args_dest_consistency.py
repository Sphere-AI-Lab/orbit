"""Every bare `args.<name>` read must name a dest some parser registers.

Guards the mechanical-rename failure mode: an option is renamed
(`--miles-root` -> `--orbit-root`), argparse's dest moves with it, but a reader
still spells the old dest. No source token spells the dest, so a token-level
rename scan misses it and the AttributeError only surfaces at run time. See
tools/check_args_dest_consistency.py.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "check_args_dest_consistency.py"

SPEC = importlib.util.spec_from_file_location("orbit_check_args_dest_consistency", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="repo sources use PEP 695 syntax")
def test_every_namespace_read_names_a_registered_dest():
    errors = CHECKER.collect_errors()
    assert not errors, "argparse dest inconsistencies:\n  " + "\n  ".join(errors)
