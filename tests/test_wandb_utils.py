from types import SimpleNamespace

from orbit.utils import wandb_utils


def _args(*, wandb_run_id=None):
    return SimpleNamespace(
        use_wandb=True,
        wandb_run_id=wandb_run_id,
        wandb_mode=None,
        wandb_key=None,
        wandb_host=None,
        wandb_random_suffix=False,
        wandb_group="test-group",
        rank=0,
        wandb_team="test-team",
        wandb_project="test-project",
        wandb_dir=None,
        env_report=None,
    )


def _stub_wandb(monkeypatch, *, initialized_run_id):
    init_calls = []
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: init_calls.append(kwargs),
        run=SimpleNamespace(id=initialized_run_id),
        Settings=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(wandb_utils, "wandb", fake_wandb)
    monkeypatch.setattr(wandb_utils, "_init_wandb_common", lambda: None)
    return init_calls


def test_init_wandb_primary_starts_fresh_run_without_explicit_id(monkeypatch):
    args = _args()
    init_calls = _stub_wandb(monkeypatch, initialized_run_id="generated-run-id")

    wandb_utils.init_wandb_primary(args)

    assert len(init_calls) == 1
    assert "id" not in init_calls[0]
    assert "resume" not in init_calls[0]
    assert args.wandb_run_id == "generated-run-id"


def test_init_wandb_primary_resumes_explicit_run_id(monkeypatch):
    args = _args(wandb_run_id="stable-run-id")
    init_calls = _stub_wandb(monkeypatch, initialized_run_id="stable-run-id")

    wandb_utils.init_wandb_primary(args)

    assert len(init_calls) == 1
    assert init_calls[0]["id"] == "stable-run-id"
    assert init_calls[0]["resume"] == "allow"
    assert args.wandb_run_id == "stable-run-id"
