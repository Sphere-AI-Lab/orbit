"""The seven per-learning-rate launch scripts must partition E4 exactly.

`run_e4_lr{1..7}_8gpu.sh` each select one column of the sweep: FullFT at the
i-th point of its grid, plus LoRA r1/r16/r256 at the i-th point of theirs. Two
things can go wrong silently, and both cost a node before anyone notices.

A **gap** -- a learning rate no script selects -- leaves a hole in one of C5's
four curves. Every script that does run still succeeds, `analyze` still reports
an argmin, and the missing point is invisible unless someone counts.

An **overlap** -- an arm two scripts both select -- is worse than wasted compute
if the two land in different ledgers: `analyze` globs them together and the arm
appears twice, so a duplicated point silently gets double weight in whatever the
claim is read off.

`campaign.sh` already refuses a selection that is not `EXPECT_ARMS` long, which
catches a script that selects 0 or 8. It cannot catch two scripts selecting the
same 4, or seven scripts of 4 that miss an arm between them. That is what this
file is for.
"""

import re
from pathlib import Path

from tools.lora_regret.arms import e4_arms

SCRIPTS = sorted(
    (Path(__file__).resolve().parents[3] / "scripts" / "lora_regret").glob("run_e4_*_lr*_8gpu.sh")
)


def _method_re(path: Path) -> str:
    match = re.search(r"METHOD_RE='([^']+)'", path.read_text(encoding="utf-8"))
    assert match, f"{path.name} has no single-quoted METHOD_RE"
    return match.group(1)


def _arms():
    return e4_arms()


def _pattern(path: Path):
    return re.compile(_method_re(path))


def _selected(path: Path) -> list[str]:
    """What `sweep.py --only` would select: re.search against the arm name."""
    return [a.name for a in _arms() if _pattern(path).search(a.name)]


def test_there_is_one_script_per_grid_point_per_panel():
    """Figure 6 is two panels, and each is schedulable on its own."""
    from tools.lora_regret.arms import RL_DATASETS

    assert len(SCRIPTS) == 7 * len(RL_DATASETS)
    assert {p.name for p in SCRIPTS} == {
        f"run_e4_{ds}_lr{i}_8gpu.sh" for ds in RL_DATASETS for i in range(1, 8)
    }


def test_each_script_selects_one_dataset_only():
    """A column that mixed panels would put two y-axes in one ledger, and
    `analyze` globs a panel's ledgers together."""
    for path in SCRIPTS:
        datasets = {a.dataset for a in _arms() if _pattern(path).search(a.name)}
        assert len(datasets) == 1, (path.name, datasets)
        assert path.name.startswith(f"run_e4_{datasets.pop()}_lr")


def test_each_script_selects_one_fullft_arm_and_three_lora_ranks():
    """One point on each of C5's four curves, at a single learning rate."""
    for path in SCRIPTS:
        names = _selected(path)
        assert len(names) == 4, (path.name, names)
        assert sum(n.startswith("full-") for n in names) == 1, (path.name, names)
        assert {n.split("-")[1] for n in names if n.startswith("lora-")} == {"r1", "r16", "r256"}, (
            path.name,
            names,
        )


def test_the_lora_arms_in_a_column_share_one_learning_rate():
    """A column is a vertical slice of the figure. Three ranks at three
    different LRs would not be one."""
    for path in SCRIPTS:
        # `.+`, not `[^-]+`: the learning rate itself carries a hyphen in
        # exponent form, so `lr5e-06-s0` splits wrong on a negated class.
        lora_lrs = {
            re.search(r"-lr(.+)-s\d+$", n).group(1) for n in _selected(path) if n.startswith("lora-")
        }
        assert len(lora_lrs) == 1, (path.name, lora_lrs)


def test_no_script_selects_an_oft_arm():
    """`analyze` reads a ledger as one comparable set, and an `oftscout` row
    carries a learning rate from a different search entirely."""
    for path in SCRIPTS:
        assert not any(n.startswith("oftscout") for n in _selected(path)), path.name


def test_the_seven_scripts_partition_e4_exactly():  # noqa: D401
    """The property neither `EXPECT_ARMS` nor a per-script check can see: no
    arm selected twice, and no non-OFT arm left unselected."""
    selected = [name for path in SCRIPTS for name in _selected(path)]
    expected = {a.name for a in e4_arms() if a.method != "oft"}
    assert len(selected) == len(set(selected)), "an arm is selected by two scripts"
    assert set(selected) == expected
    assert len(selected) == 56


def test_each_script_writes_its_own_ledger():
    """Seven nodes appending to one file would interleave partial rows. The
    ledgers are globbed back together at analysis time instead."""
    ledgers = {
        re.search(r"RESULTS=(\S+)", path.read_text(encoding="utf-8")).group(1) for path in SCRIPTS
    }
    assert len(ledgers) == 14
    assert all(led.startswith("results/e4_") for led in ledgers)


def test_each_script_asserts_its_own_arm_count():
    for path in SCRIPTS:
        assert "EXPECT_ARMS=4" in path.read_text(encoding="utf-8"), path.name


PROTOCOL = SCRIPTS[0].parent / "e4_protocol.sh" if SCRIPTS else None


def test_every_column_sources_the_shared_protocol():
    """Fourteen node bookings, one protocol. A sweep is only a sweep if every
    arm differs in the learning rate and nothing else, so the knobs that shape
    the update live in one file rather than in fourteen copies where a drift
    between two of them would be indistinguishable from a real effect."""
    for path in SCRIPTS:
        assert 'source "${HERE}/e4_protocol.sh"' in path.read_text(encoding="utf-8"), path.name


def test_the_protocol_sets_the_knobs_that_change_the_update():
    """Each of these alters the mathematics of the step, not just its cost, and
    each defaults the wrong way for this experiment in orbit or the launcher."""
    text = PROTOCOL.read_text(encoding="utf-8")
    assert ': "${RL_EXTRA_ARGS=--disable-grpo-std-normalization}"' in text
    assert ': "${EPS_CLIP=1e9}"' in text
    assert ': "${EPS_CLIP_HIGH=1e9}"' in text


def test_the_protocol_disables_checkpointing_with_an_empty_value():
    """`SAVE_INTERVAL=999999` would still write one checkpoint: orbit's
    `should_run_periodic_action` short-circuits on `interval is None` and only
    then checks the final rollout. Only the empty value drops the flag."""
    text = PROTOCOL.read_text(encoding="utf-8")
    assert ': "${SAVE_INTERVAL=}"' in text
    assert ': "${SAVE_INTERVAL=0' not in text


def test_every_protocol_value_is_a_default_not_a_lock():
    """`: "${VAR=x}"` assigns only when unset, so an operator can re-run one
    column at a different rollout count without editing the file. A bare
    `export VAR=x` would silently ignore the environment."""
    text = PROTOCOL.read_text(encoding="utf-8")
    for var in ("RL_EXTRA_ARGS", "EPS_CLIP", "NUM_ROLLOUT", "SAVE_INTERVAL", "EVAL_INTERVAL"):
        assert f': "${{{var}=' in text, var
        assert f"\nexport {var}=" not in text, var


def test_the_protocol_logs_wandb_offline():
    """The compute nodes have no egress. On 2026-08-02 seven arms ran 90
    minutes in the launcher's online path and nothing reached the server -- no
    project, no retry, no warning. Offline is also the only local format
    `wandb sync` can replay: a shared-mode directory comes back with config and
    summary and zero history rows, which is an empty dashboard."""
    assert ': "${WANDB_MODE=offline}"' in PROTOCOL.read_text(encoding="utf-8")


def test_the_sync_script_exists_and_refuses_to_run_offline():
    """`wandb sync` inheriting WANDB_MODE=offline from the shell would write
    the uploads straight back to disk."""
    script = PROTOCOL.parent / "sync_wandb.sh"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "unset WANDB_MODE" in text
    assert "--sync-all" in text


def test_the_arm_count_guard_survives_a_partial_ledger():
    """Resume was advertised and did not work. `campaign.sh` compared
    EXPECT_ARMS against the sweep's STDOUT, which lists only the arms still to
    run -- so a column that finished 1 of its 4 arms and was re-run saw 3,
    refused to start, and blamed a renamed arm. EXPECT_ARMS is a claim about
    which arms the script COVERS, and that does not shrink as they complete, so
    the count now comes from the sweep's "N arms selected" line on stderr."""
    campaign = PROTOCOL.parent / "campaign.sh"
    text = campaign.read_text(encoding="utf-8")
    assert "arms selected" in text and "SWEEP_ERR" in text
    assert 'SELECTED=$(printf' not in text, "the guard must not count the to-run list"
    assert '"${TODO}" -eq 0' in text, "a fully-done selection should exit cleanly, not run nothing"


def test_campaign_sources_the_protocol_itself():
    """So a one-off single-arm invocation cannot lose it.

    On 2026-08-03 a LoRA arm was launched as `MATRIX=e4 METHOD_RE=... bash
    campaign.sh` with the protocol left unsourced. It ran in wandb's online
    mode from a compute node with no egress and logged nothing, silently:
    correct project, correct run name, `wandb_mode = None`, and a `run-*`
    directory where an `offline-run-*` should have been."""
    text = (PROTOCOL.parent / "campaign.sh").read_text(encoding="utf-8")
    assert 'source "${ORBIT_ROOT}/scripts/lora_regret/e4_protocol.sh"' in text
    assert 'if [[ "${WANDB_MODE:-}" != "offline" ]]; then' in text


def test_the_sweep_syncs_wandb_after_every_arm():
    """"Offline" must not mean "manual". A directory nobody syncs is a
    dashboard nobody sees, so the upload runs inside the sweep after each arm
    rather than being left to the operator -- and after EACH arm, not at the
    end of a twelve-hour column."""
    text = (PROTOCOL.parent.parent.parent / "tools" / "lora_regret" / "sweep.py").read_text(
        encoding="utf-8"
    )
    assert "def sync_wandb_offline_runs" in text
    assert "sync_wandb_offline_runs(repo_root)" in text
    assert 'env.pop("WANDB_MODE", None)' in text, "the upload must not itself run offline"
    assert 'os.environ.get("WANDB_AUTOSYNC", "1")' in text, "must be defeatable"


def test_the_sweep_can_sync_wandb_during_arms_but_does_not_by_default():
    """An after-arm-only sync replays nothing but quiescent, complete
    directories -- the most stable sync there is, and what an unattended
    overnight column should run. But it also means a dashboard a full
    ~90-minute arm behind, so a watcher thread exists for the nights someone
    is actually watching: `wandb sync` replays a live offline directory up to
    its current tail and the next pass refreshes it (sync_wandb.sh documents
    the same property for the manual path). Opt-in via WANDB_SYNC_INTERVAL
    because the live replay's warts are cosmetic but real: the run shows as
    "finished" between passes."""
    text = (PROTOCOL.parent.parent.parent / "tools" / "lora_regret" / "sweep.py").read_text(
        encoding="utf-8"
    )
    assert "def start_wandb_sync_watcher" in text
    assert "start_wandb_sync_watcher(repo_root)" in text
    assert 'os.environ.get("WANDB_SYNC_INTERVAL", "0")' in text, "off unless asked for"
    assert "_WANDB_SYNC_LOCK" in text, "watcher and after-arm sync must not overlap"
    assert "daemon=True" in text, "the watcher must never keep a finished sweep alive"


def test_every_periodic_action_in_train_py_is_told_the_rollout_count():
    """The protocol's "one eval, at the end" depends entirely on this argument.

    `should_run_periodic_action(rollout_id, interval, per_epoch, num_rollout)`
    fires on the last rollout via `rollout_id == num_rollout - 1`, and that
    branch is unreachable when the fourth argument is omitted. EVAL_INTERVAL is
    100000 precisely so the modulo never matches and only the final-rollout
    branch fires -- so an omitted `num_rollout` does not degrade the eval
    cadence, it removes post-training eval entirely.

    That is not hypothetical. train.py's generation-eval call omitted it while
    the held-out-NLL call twenty lines above passed it, so E4's gsm8k columns
    ran 150 rollouts apiece and evaluated only the UNTRAINED policy, from the
    separate eval-before-train branch. Every ledger row read `accuracy: null,
    status: failed` beside a complete, healthy log.

    Checked over the AST rather than the text because the call spans lines and
    a grep for the argument name would pass on a comment mentioning it.
    """
    import ast

    train_py = PROTOCOL.parents[2] / "train.py"
    tree = ast.parse(train_py.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "should_run_periodic_action"
    ]
    assert len(calls) >= 3, f"expected the eval, eval-nll and save call sites, found {len(calls)}"
    for call in calls:
        passed = len(call.args) + len(call.keywords)
        assert passed == 4, (
            f"{train_py.name}:{call.lineno} passes {passed} arguments to "
            "should_run_periodic_action; without num_rollout its final-rollout "
            "branch is dead and the last rollout produces no measurement"
        )


def test_a_periodic_action_without_the_rollout_count_never_fires_on_the_last_rollout():
    """The behaviour the pin above protects, stated directly: at the protocol's
    own settings -- 150 rollouts, interval 100000 -- the fourth argument is the
    only thing standing between one eval and none."""
    from orbit.utils.misc import should_run_periodic_action

    fires = [
        rollout_id
        for rollout_id in range(150)
        if should_run_periodic_action(rollout_id, 100000, None, 150)
    ]
    assert fires == [149]
    assert not any(
        should_run_periodic_action(rollout_id, 100000, None) for rollout_id in range(150)
    )
