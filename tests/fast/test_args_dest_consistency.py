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


def test_pure_mopd_is_still_a_valid_advantage_estimator():
    """miles @ dbbab1566 dropped ``on_policy_distillation`` from the choices.

    Every orbit validator and the OPD loss path key on that exact string, and
    examples/on_policy_distillation/ spells pure MOPD with it, so losing the
    choice makes those recipes die in argparse before they start. orbit re-offers
    it (orbit/arguments.py::_extend_arg_choices) rather than editing miles' list.
    """
    import importlib.util
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "orbit_dump_args_surface", repo / "tools" / "dump_args_surface.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    parser = module.build_parser()
    action = next(a for a in parser._actions if "--advantage-estimator" in a.option_strings)
    assert "on_policy_distillation" in action.choices
    # ...and orbit only ADDED: upstream's own choices must all survive.
    for upstream_choice in ("grpo", "gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline", "ppo"):
        assert upstream_choice in action.choices


@pytest.mark.skipif(sys.version_info < (3, 12), reason="repo sources use PEP 695 syntax")
def test_the_checker_sees_untracked_files():
    """Reading the git index alone makes this guard pass VACUOUSLY on a file that
    has not been committed yet. That is how the verify_env.py `args.miles_root`
    defect reached a branch in the first place: new file, unseen guard."""
    probe = REPO_ROOT / "orbit" / "_args_dest_untracked_probe.py"
    probe.write_text(
        "def go(args):\n"
        "    return args.orbit_definitely_not_a_registered_dest\n"
    )
    try:
        errors = CHECKER.collect_errors()
    finally:
        probe.unlink()
    assert any("_args_dest_untracked_probe" in e for e in errors), (
        "the checker did not flag an unregistered dest in an untracked file; it "
        "is reading the index only again"
    )
