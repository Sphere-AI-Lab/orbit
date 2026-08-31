"""No process may reach an orbit patch target without arming orbit first.

Unarmed, a patched function is upstream's and nothing says so -- the run just
produces slightly different numbers forever. Both converter CLIs shipped that
way once (fixed in 288afc89).

This replaces a much weaker rule that used to live in test_hf_export_patches.py:
"a file with a __main__ block that imports a patched module BY NAME, or its
immediate parent package, must import orbit". Reaching a patched module is
transitive, and __main__ is not the only way a process starts, so that rule
considered 8 of 652 files and called the repo clean while eight tests asserted
against upstream's converters. See tools/check_arming.py.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "check_arming.py"

SPEC = importlib.util.spec_from_file_location("orbit_check_arming", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="repo sources use PEP 695 syntax")
def test_every_process_that_reaches_a_patch_target_arms_orbit():
    errors = CHECKER.collect_errors()
    assert not errors, "unarmed reach into orbit's patches:\n  " + "\n  ".join(errors)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="repo sources use PEP 695 syntax")
def test_the_checker_catches_the_bug_it_was_written_for():
    """Falsify the guard rather than trust it.

    The historical failure is a script whose import closure reaches a patched
    module through files that never import orbit. The probe cannot be hardcoded:
    on this base every converter sits under a package whose ``__init__`` reaches
    ``update_weight.common``, which DOES import orbit, so importing one arms the
    process for real and the guard is right to stay quiet. So derive the probe
    from the checker's own graph -- pick a patch target that is genuinely
    unreachable-while-armed, and require a complaint about it.
    """
    files = CHECKER.tracked_py()
    paths, edges = CHECKER.build_graph(files)
    targets = CHECKER.patch_targets()
    assert targets, "no patches registered; this guard has nothing to protect"

    def probe_closure(target: str) -> set[str]:
        """What a one-line `import <target>` script would actually pull in.

        Importing a submodule executes every parent package too, which is what
        makes most targets armed here.
        """
        reached: set[str] = set()
        for prefix in CHECKER._repo_prefixes(target, paths):
            reached |= CHECKER.closure(prefix, edges)
        return reached

    candidates = [
        t for t in sorted(targets, key=lambda t: (t[0], t[1] or ""))
        if not CHECKER.arms_orbit(probe_closure(t[0]))
    ]
    if not candidates:
        pytest.skip(
            "every patch target arms orbit through its own import closure, so no "
            "unarmed reach can be constructed to falsify the guard here"
        )
    module, attr = candidates[0]
    target = f"{module}.{attr}" if attr else module
    probe = REPO_ROOT / "tools" / "_arming_falsification_probe.py"
    # Must CALL the patched name, not merely import its module: the guard stopped
    # flagging bare module reachability once ~39 upstream launchers proved that
    # over-approximates (they touch miles.utils.misc and never call get_free_port).
    probe.write_text(
        f"import {module}\n"
        'if __name__ == "__main__":\n'
        f"    print({target})\n"
    )
    try:
        errors = CHECKER.collect_errors()
    finally:
        probe.unlink()
    assert any("_arming_falsification_probe" in e for e in errors), (
        f"the guard did not flag an unarmed script that reaches patched {target}"
    )


def test_the_ray_worker_hook_names_something_importable():
    """The ray half of the guard rests on this string resolving in a worker.

    Ray loads it by dotted path in a fresh process; a rename that left the
    string behind would fail every worker at startup.
    """
    from ray._common.utils import load_class

    from orbit.ray_setup import WORKER_SETUP_HOOK

    assert callable(load_class(WORKER_SETUP_HOOK))
