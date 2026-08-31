"""Orbit's ``miles.utils`` seam still happens now that the vendored file is pristine.

``misc.py``'s ``get_free_port`` had an extra ``and _try_lock_port_range(...)``
inside upstream's scan loop; it is a delegating patch in
orbit/utils/miles_utils_patches.py now. A seam that silently stops firing looks
exactly like a seam that fires, so the property below is asserted against
observable behaviour rather than against the presence of the code.

Orbit's other change to this package -- ``reloadable_process_group.py``'s
collective forwarding -- has no tests here because it has no orbit code: miles
@ dbbab1566 defines and calls ``_forward_remaining_collectives`` itself.
(orbit-main-isolated, on the older base, still lifts it and still tests it.)

The rule that no unarmed process may reach these seams is enforced in
tests/fast/test_arming.py, over the real transitive import graph.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import orbit  # noqa: F401  -- importing orbit arms the seams


from miles.utils import misc  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
ORBIT_SEAM_MODULE = "orbit.utils.miles_utils_patches"


@pytest.fixture(autouse=True)
def _isolated_port_locks(tmp_path, monkeypatch):
    """Keep the flock files out of the shared /tmp directory real runs use."""
    monkeypatch.setenv("ORBIT_PORT_LOCK_DIR", str(tmp_path / "port-locks"))


# --------------------------------------------------------------------------
# reloadable_process_group: the lifted function and its import-time seam
# --------------------------------------------------------------------------
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
        "import orbit\n"
        "print(misc.get_free_port.__module__)\n"
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
    assert lines[-1:] == [ORBIT_SEAM_MODULE], out.stdout
