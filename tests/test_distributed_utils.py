from orbit.utils import distributed_utils


def test_new_process_group_options_kwargs_prefers_backend_options(monkeypatch):
    def helper(*args, backend_options=None, timeout=None):
        return None

    monkeypatch.setattr(distributed_utils, "_new_process_group_helper", helper)

    assert distributed_utils._new_process_group_options_kwargs("options") == {
        "backend_options": "options",
    }


def test_new_process_group_options_kwargs_supports_legacy_pg_options(monkeypatch):
    def helper(*args, pg_options=None, timeout=None):
        return None

    monkeypatch.setattr(distributed_utils, "_new_process_group_helper", helper)

    assert distributed_utils._new_process_group_options_kwargs("options") == {
        "pg_options": "options",
    }


def test_new_process_group_options_kwargs_handles_helpers_without_options(monkeypatch):
    def helper(*args, timeout=None):
        return None

    monkeypatch.setattr(distributed_utils, "_new_process_group_helper", helper)

    assert distributed_utils._new_process_group_options_kwargs("options") == {}
