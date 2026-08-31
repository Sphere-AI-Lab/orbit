"""Orbit's two ``miles.utils`` seams still happen now that miles/ is pristine.

Both changes used to live inside vendored files:

* ``reloadable_process_group.py`` defined and called ``_forward_remaining_collectives``;
* ``misc.py``'s ``get_free_port`` had an extra ``and _try_lock_port_range(...)``.

They now live in orbit/utils/miles_utils_patches.py. A seam that silently stops
firing looks exactly like a seam that fires, so each property below is asserted
against observable behaviour rather than against the presence of the code.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import orbit  # noqa: F401  -- importing orbit arms the seams

import torch.distributed as dist  # noqa: E402

from miles.utils import misc  # noqa: E402
from miles.utils.reloadable_process_group import ReloadableProcessGroup  # noqa: E402
from orbit.patch.on_import import check_seam_targets, fired, registry  # noqa: E402
from orbit.utils.miles_utils_patches import (  # noqa: E402
    _NOT_A_COLLECTIVE,
    forward_remaining_collectives,
)


REPO = Path(__file__).resolve().parents[2]
ORBIT_SEAM_MODULE = "orbit.utils.miles_utils_patches"


@pytest.fixture(autouse=True)
def _isolated_port_locks(tmp_path, monkeypatch):
    """Keep the flock files out of the shared /tmp directory real runs use."""
    monkeypatch.setenv("ORBIT_PORT_LOCK_DIR", str(tmp_path / "port-locks"))


# --------------------------------------------------------------------------
# reloadable_process_group: the lifted function and its import-time seam
# --------------------------------------------------------------------------


def test_the_vendored_module_no_longer_carries_the_forwarding_code():
    """The point of the move. If this fails the file went dirty again."""
    src = (REPO / "miles" / "utils" / "reloadable_process_group.py").read_text()
    assert "_forward_remaining_collectives" not in src
    assert "ORBIT" not in src


def test_the_seam_fires_when_the_vendored_module_is_imported():
    names = {module for module, _ in fired()}
    assert "miles.utils.reloadable_process_group" in names, (
        "orbit registered a callback for this module but it never ran; the "
        "collectives below are forwarded only by accident if so"
    )


def test_every_torch_collective_is_forwarded():
    """The bug the lifted function exists for: an un-forwarded collective reaches
    the C++ base, whose backend map is empty, and dies with "No backend type
    associated with device type cuda"."""
    unforwarded = [
        name
        for name in dir(dist.ProcessGroup)
        if not name.startswith("__")
        and name not in _NOT_A_COLLECTIVE
        and callable(getattr(dist.ProcessGroup, name, None))
        and name not in vars(ReloadableProcessGroup)
    ]
    assert not unforwarded, f"these fall through to the C++ base: {unforwarded}"


def test_orbit_is_what_forwarded_them_not_the_vendored_class():
    """Distinguishes "the seam worked" from "upstream happened to define it".

    ``vars(ReloadableProcessGroup)`` cannot tell the two apart once the seam has
    run, but the closure orbit installs carries orbit's module name.
    """
    from_orbit = [
        name
        for name, attr in vars(ReloadableProcessGroup).items()
        if getattr(attr, "__module__", None) == ORBIT_SEAM_MODULE
    ]
    assert from_orbit, "the seam installed nothing at all"
    # Sanity: upstream's own hand-written forwards are still upstream's.
    assert vars(ReloadableProcessGroup)["allreduce"].__module__ == (
        "miles.utils.reloadable_process_group"
    )


def test_a_forwarded_collective_delegates_to_the_inner_group():
    forwarded = next(
        name
        for name, attr in vars(ReloadableProcessGroup).items()
        if getattr(attr, "__module__", None) == ORBIT_SEAM_MODULE
    )

    calls = []

    class Stub:
        _fwd = lambda self, method, *a, **kw: calls.append((method, a, kw)) or "ok"  # noqa: E731

    bound = vars(ReloadableProcessGroup)[forwarded].__get__(Stub(), Stub)
    assert bound(1, x=2) == "ok"
    assert calls == [(forwarded, (1,), {"x": 2})]


def test_forwarding_is_idempotent():
    """It runs once per process today, but a second run must not double-wrap."""
    assert forward_remaining_collectives() == []


def test_every_import_seam_target_still_exists():
    """A callback hung off a renamed module never runs and never complains."""
    assert check_seam_targets() == []
    assert {module for module, _ in registry()} >= {
        "miles.utils.reloadable_process_group",
        "miles.backends.megatron_utils",
    }


def test_every_entrypoint_that_reaches_an_import_seam_arms_it():
    """Import seams fail the same silent way the function patches do.

    ``import orbit`` is what arms them, so a standalone script that imports a
    seam's module without importing orbit just does not get the seam -- no
    error, only the missing behaviour much later.
    tests/fast/test_hf_export_patches.py enforces this for ``patch_function``
    targets; the on_import registry is a second, separate registry and needs the
    same check or the rule is enforced for half the seams.
    """
    import ast

    reaching = set()
    for module, _ in registry():
        reaching.add(module)
        reaching.add(module.rsplit(".", 1)[0])

    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py"], capture_output=True, text=True
    ).stdout.split()

    offenders = []
    for rel in tracked:
        if rel.startswith(("miles/", "miles_plugins/", "orbit/", "tests/")):
            continue  # library code and tests run inside processes that import orbit
        src = (REPO / rel).read_text(errors="surrogateescape")
        if '__name__ == "__main__"' not in src and "__name__ == '__main__'" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        if not (imported & reaching):
            continue
        if not any(m == "orbit" or m.startswith("orbit.") for m in imported):
            offenders.append(rel)

    assert not offenders, (
        "entrypoint reaches an orbit import seam without arming it "
        "(add `import orbit`):\n  " + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------
# misc.get_free_port: the delegating patch
# --------------------------------------------------------------------------


def test_the_get_free_port_patch_is_installed():
    assert misc.get_free_port.__module__ == ORBIT_SEAM_MODULE
    assert hasattr(misc, "_orbit_unpatched_get_free_port"), (
        "the pristine upstream function must be kept so the patch can delegate"
    )


def test_orbit_never_hands_the_same_port_out_twice():
    """The race the seam exists for: ``is_port_available`` observes a port, it
    does not claim one, so two callers both saw the same range free."""
    first = misc.get_free_port(31000)
    second = misc.get_free_port(31000)
    assert first != second

    # ...and prove the patch is what did it: upstream alone repeats itself.
    assert misc._orbit_unpatched_get_free_port(32000) == (
        misc._orbit_unpatched_get_free_port(32000)
    )


def test_upstream_still_owns_what_available_means(monkeypatch):
    """The delegation property. Orbit's replacement never calls
    ``is_port_available`` itself, so if this skip happens, upstream's body ran.
    """
    blocked = set(range(33000, 33005))
    monkeypatch.setattr(misc, "is_port_available", lambda port: port not in blocked)
    assert misc.get_free_port(33000) == 33005


def test_a_consecutive_range_is_claimed_as_a_whole():
    base = misc.get_free_port(34000, consecutive=3)
    # Every port in the returned range is now locked, so the next single-port
    # request cannot land inside it.
    assert misc.get_free_port(base) >= base + 3


def test_the_patch_survives_miles_being_imported_before_orbit():
    """Import ORDER must not decide whether the seams take effect: a Ray actor
    imports miles while unpickling its worker class, before its body runs
    ``import orbit``. Run it for real in a subprocess rather than simulating it.
    """
    probe = (
        "import miles.utils.misc as misc\n"  # BEFORE orbit
        "import miles.utils.reloadable_process_group as rpg\n"
        "import orbit\n"
        "print(misc.get_free_port.__module__)\n"
        "print(any(getattr(a, '__module__', None) == "
        f"{ORBIT_SEAM_MODULE!r} for a in vars(rpg.ReloadableProcessGroup).values()))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    lines = out.stdout.split()
    assert lines[-2:] == [ORBIT_SEAM_MODULE, "True"], out.stdout
