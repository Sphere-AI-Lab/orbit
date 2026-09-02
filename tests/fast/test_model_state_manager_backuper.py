"""FullModelStateManager must construct against upstream's TensorBackuper API.

miles dbbab1566 retired the ``--enable-weights-backuper`` flag and with it the
``single_tag`` argument of ``TensorBackuper.create`` (now
``create(source_getter, main_cast_ctx=None)``). The full-FT actor init on this
branch died with ``TypeError: create() got an unexpected keyword argument
'single_tag'``; the PEFT arms never hit it because they use AdapterStateManager.
"""

import pytest

from orbit.megatron import model_state_manager


class _UpstreamBackuper:
    """Stub with the dbbab1566 signature: no single_tag parameter."""

    created = []

    @staticmethod
    def create(source_getter, main_cast_ctx=None):
        _UpstreamBackuper.created.append((source_getter, main_cast_ctx))
        return _UpstreamBackuper()

    backup_tags = ("actor",)

    def get(self, tag):
        return {}

    def backup(self, tag):
        pass

    def copy(self, *, src_tag, dst_tag):
        pass

    def restore(self, tag):
        pass


@pytest.fixture(autouse=True)
def _upstream_backuper(monkeypatch):
    _UpstreamBackuper.created.clear()
    monkeypatch.setattr(model_state_manager, "TensorBackuper", _UpstreamBackuper)


def test_full_model_manager_constructs_with_upstream_signature():
    source = lambda: []  # noqa: E731

    manager = model_state_manager.FullModelStateManager(source_getter=source, single_tag=None)

    assert manager.backup_tags == ("actor",)
    assert _UpstreamBackuper.created == [(source, None)]


def test_single_tag_is_rejected_loudly_on_this_base():
    with pytest.raises(ValueError, match="single_tag"):
        model_state_manager.FullModelStateManager(source_getter=lambda: [], single_tag="actor")
