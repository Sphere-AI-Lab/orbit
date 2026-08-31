"""Orbit's megatron.bridge plugins still load when the Megatron backend loads.

``miles/backends/megatron_utils/__init__.py`` used to carry a best-effort
``import miles_plugins.megatron_bridge.patches.bridges`` -- an import kept purely
for its side effects (the ``broadcast_obj_from_pp_rank`` unwrap that lets
megatron.bridge see through orbit's ReloadableProcessGroup, plus orbit's bridge
subclass registrations). The vendored file is pristine again and orbit runs that
import from its own side, at the same moment.

A side-effect import that stops happening produces no error at all -- just a
``"Group ... is not registered"`` failure much later, on multi-PP runs only --
so these assert the side effect, not the code.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import orbit  # noqa: F401  -- importing orbit arms the seam

from orbit.megatron.bridge_plugins import (  # noqa: E402
    MEGATRON_BACKEND,
    load_bridge_plugins,
)
from orbit.patch.on_import import fired, registry  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
PLUGIN_PACKAGE = "miles_plugins.megatron_bridge.patches.bridges"


def test_the_vendored_backend_init_no_longer_carries_the_import():
    src = (REPO / "miles" / "backends" / "megatron_utils" / "__init__.py").read_text()
    assert "miles_plugins" not in src
    assert "ORBIT" not in src


def test_orbit_registers_the_seam_against_the_backend_package():
    assert MEGATRON_BACKEND == "miles.backends.megatron_utils"
    assert (MEGATRON_BACKEND, "load_bridge_plugins") in [
        (module, callback.__name__) for module, callback in registry()
    ]


def test_importing_the_backend_runs_the_seam_and_installs_the_shim():
    import miles.backends.megatron_utils  # noqa: F401

    assert MEGATRON_BACKEND in {module for module, _ in fired()}
    assert PLUGIN_PACKAGE in sys.modules, (
        "the backend imported without orbit's bridge plugins coming with it"
    )

    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    assert getattr(MegatronParamMapping, "_orbit_pp_group_unwrap_installed", False), (
        "megatron.bridge's pp broadcast cannot see orbit's ReloadableProcessGroup "
        "without this shim; pp_size > 1 fails with 'Group ... is not registered'"
    )


def test_the_loader_is_best_effort_and_never_raises(monkeypatch, caplog):
    """Not every environment has megatron.bridge, and the backend must still
    import where it does not. A None entry in sys.modules makes the import fail
    the same way a missing dependency would."""
    monkeypatch.setitem(sys.modules, PLUGIN_PACKAGE, None)
    with caplog.at_level(logging.WARNING, logger="orbit.megatron.bridge_plugins"):
        assert load_bridge_plugins() is False
    assert "megatron.bridge plugins failed to load" in caplog.text


def test_the_seam_fires_even_when_the_backend_is_imported_before_orbit():
    """Import ORDER must not decide whether the plugins load: a Ray actor
    imports the backend while unpickling its worker class, before its body runs
    ``import orbit``. Run it for real rather than simulating it.
    """
    probe = (
        "import sys\n"
        "import miles.backends.megatron_utils  # BEFORE orbit\n"
        f"print('before', {PLUGIN_PACKAGE!r} in sys.modules)\n"
        "import orbit\n"
        f"print('after', {PLUGIN_PACKAGE!r} in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    printed = [line for line in out.stdout.splitlines() if line.startswith(("before", "after"))]
    assert printed == ["before False", "after True"], out.stdout
