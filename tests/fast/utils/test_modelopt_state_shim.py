"""Megatron's `get_model()` imports a package this env does not ship.

    megatron/training/training.py:1324
        if has_nvidia_modelopt:
            from megatron.post_training.checkpointing import has_modelopt_state

`has_nvidia_modelopt` is True here (nvidia-modelopt 0.44.0 is installed for the
NVFP4/INT4 work) but the installed `megatron` namespace package carries only
`bridge`, `core` and `training`. So every full fine-tuning run dies with
`ModuleNotFoundError: No module named 'megatron.post_training'` while every
PEFT run is fine -- orbit routes PEFT through `_setup_peft_model_via_bridge`
and never reaches Megatron's `get_model()`.

Found by the coverage probe on 2026-07-31, on the first FullFT arm ever run.
"""

import sys

import pytest

from orbit.backends.megatron_utils.modelopt_state_shim import (
    MODULE_NAME,
    has_modelopt_state,
    install_if_missing,
)


def _sharded_checkpoint(root, iteration=0, with_modelopt=False):
    """A Megatron torch_dist checkpoint, shaped like the campaign's real one."""
    (root / "latest_checkpointed_iteration.txt").write_text(str(iteration))
    iter_dir = root / f"iter_{iteration:07d}"
    iter_dir.mkdir()
    (iter_dir / "__0_0.distcp").write_bytes(b"")
    if with_modelopt:
        (iter_dir / "modelopt_state").mkdir()
    return root


class TestHasModeloptState:
    def test_a_plain_checkpoint_has_none(self, tmp_path):
        """The campaign's Llama-3.1-8B_torch_dist: verified 0 modelopt_state
        directories. `False` is the correct answer, not a fallback."""
        assert has_modelopt_state(_sharded_checkpoint(tmp_path)) is False

    def test_a_missing_path_has_none(self, tmp_path):
        assert has_modelopt_state(tmp_path / "nope") is False

    def test_none_is_not_a_checkpoint(self):
        """Megatron calls this only when args.load is set, but the guard is
        cheap and a None here would otherwise raise inside a Ray actor."""
        assert has_modelopt_state(None) is False

    def test_modelopt_state_at_the_checkpoint_root_is_found(self, tmp_path):
        (tmp_path / "modelopt_state").mkdir()
        with pytest.raises(RuntimeError, match="megatron.post_training"):
            has_modelopt_state(tmp_path)

    def test_a_real_modelopt_checkpoint_refuses_rather_than_lying(self, tmp_path):
        """The shim answers only the question it can answer correctly.

        Returning False here would silently skip ModelOpt setup and train a
        model that is not the one on disk. Returning True is no better: the
        code that then runs needs more of `megatron.post_training` than this
        file provides, so it would fail later and less clearly. Refusing names
        the missing package and stops.
        """
        root = _sharded_checkpoint(tmp_path, with_modelopt=True)
        with pytest.raises(RuntimeError) as excinfo:
            has_modelopt_state(root)
        assert "megatron.post_training" in str(excinfo.value)
        assert str(root) in str(excinfo.value)

    def test_the_release_checkpoint_layout_is_understood(self, tmp_path):
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
        (tmp_path / "release").mkdir()
        assert has_modelopt_state(tmp_path) is False


class TestInstall:
    @pytest.fixture(autouse=True)
    def _clean_modules(self):
        """The shim mutates sys.modules; leaving it there would let a later
        test import a package this env does not have."""
        before = {k: v for k, v in sys.modules.items() if k.startswith("megatron.post_training")}
        yield
        for key in [k for k in sys.modules if k.startswith("megatron.post_training")]:
            del sys.modules[key]
        sys.modules.update(before)

    def test_it_makes_megatrons_own_import_line_work(self):
        """Exactly the statement at training.py:1325."""
        install_if_missing()
        from megatron.post_training.checkpointing import (  # noqa: F401
            has_modelopt_state as imported,
        )

        assert imported is has_modelopt_state

    def test_it_reports_whether_it_installed_anything(self, monkeypatch):
        """The install-then-no-op transition, from a known-clean start.

        `install_if_missing` is a process-wide one-shot, so the first call only
        returns True if nothing has already triggered it. Importing anything
        under `orbit.backends.megatron_utils` does trigger it -- `model.py`
        installs the shim at import -- so whether this test saw a clean slate
        used to depend on which other tests pytest happened to run first. It
        passed for as long as no earlier test in collection order imported that
        package, and broke the moment one did.

        Clearing the registration first makes the transition the test's own
        precondition rather than a property of the session."""
        for name in ("megatron.post_training.checkpointing", "megatron.post_training"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        assert install_if_missing() is True
        # Second call: already present, nothing to do.
        assert install_if_missing() is False

    def test_it_never_shadows_a_real_installation(self, monkeypatch):
        """If Megatron-LM's post_training is ever installed, the real one wins.
        A shim that overwrote it would silently downgrade a working install."""
        real = type(sys)("megatron.post_training.checkpointing")
        real.has_modelopt_state = lambda path: "REAL"
        monkeypatch.setitem(sys.modules, "megatron.post_training.checkpointing", real)
        assert install_if_missing() is False
        assert sys.modules["megatron.post_training.checkpointing"] is real


def test_the_module_name_is_the_one_megatron_actually_imports():
    """Pinned against the upstream source rather than retyped, so a rename in
    Megatron cannot leave this shim registering a name nobody imports."""
    from pathlib import Path

    import megatron.training.training as upstream

    text = Path(upstream.__file__).read_text(encoding="utf-8")
    assert f"from {MODULE_NAME} import has_modelopt_state" in text


def test_orbits_model_module_installs_the_shim_before_get_model_can_run():
    """The import in Megatron is inside `get_model()`, so the shim only has to
    be in sys.modules before that call -- but it must be, on every path that
    reaches it, including inside a Ray actor that imported orbit fresh."""
    import orbit.backends.megatron_utils.model as model_module

    assert hasattr(model_module, "_MODELOPT_SHIM_INSTALLED")
