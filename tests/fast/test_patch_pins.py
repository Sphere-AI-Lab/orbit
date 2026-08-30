"""Orbit's patch pins still match the upstream functions they replace.

Orbit can override a vendored miles function without editing the miles file, by
registering a replacement in orbit/patch/. The hazard of any such patch is that
it rots silently: upstream renames or rewrites the target and the patch simply
stops describing reality, with nothing failing. Each patch therefore pins a hash
of the upstream body, and this test verifies every pin statically -- in the CPU
gate, seconds, no torch import -- so drift surfaces here rather than at the top
of a GPU run. See tools/check_patch_pins.py.
"""

import hashlib
import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "check_patch_pins.py"

SPEC = importlib.util.spec_from_file_location("orbit_check_patch_pins", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="repo sources use PEP 695 syntax")
def test_every_patch_pin_matches_upstream():
    errors = CHECKER.collect_errors()
    assert not errors, "stale patch pins:\n  " + "\n  ".join(errors)


def test_static_and_runtime_hashing_agree(tmp_path):
    """The gate is only meaningful if its hash equals the one the runtime checks.

    They are computed by different means -- ast.get_source_segment here,
    inspect.getsource at runtime -- and those differ over decorators, which
    inspect includes and ast does not. normalize() is what reconciles them; if
    that ever breaks, every pin silently becomes uncheckable rather than wrong,
    so assert it directly.
    """
    import ast
    import inspect

    src = textwrap.dedent(
        '''
        import functools


        @functools.cache
        def target(a, b=1):
            """doc"""

            return a + b
        '''
    ).lstrip()
    mod_path = tmp_path / "probe_mod.py"
    mod_path.write_text(src)

    spec = importlib.util.spec_from_file_location("orbit_pin_probe", mod_path)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    tree = ast.parse(src)
    fn_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    static = CHECKER.normalize(ast.get_source_segment(src, fn_node))
    runtime = CHECKER.normalize(inspect.getsource(probe.target))

    assert static == runtime, "static and runtime normalization diverged"
    assert (
        hashlib.sha256(static.encode()).hexdigest()
        == hashlib.sha256(runtime.encode()).hexdigest()
    )


def test_a_changed_upstream_body_is_actually_detected(tmp_path, monkeypatch):
    """A pin gate that cannot fail is worse than no gate: prove it fires."""
    pkg = tmp_path / "vendored"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text("def f(x):\n    return x + 1\n")

    monkeypatch.setattr(CHECKER, "REPO", tmp_path)
    correct = CHECKER.upstream_sha("vendored.mod", "f")
    assert correct is not None

    (pkg / "mod.py").write_text("def f(x):\n    return x + 2\n")
    after = CHECKER.upstream_sha("vendored.mod", "f")
    assert after != correct, "a changed body must change the hash"

    assert CHECKER.upstream_sha("vendored.mod", "gone") is None
