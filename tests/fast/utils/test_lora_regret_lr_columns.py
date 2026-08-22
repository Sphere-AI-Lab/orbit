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

import os
import re
import subprocess
from pathlib import Path

from tools.lora_regret.arms import ALL_MODULES, e4_arms, e4lr0_arms

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "lora_regret"
SCRIPTS = [
    SCRIPT_DIR / f"run_e4_{dataset}_lr{column}_8gpu.sh"
    for dataset in ("gsm8k", "math")
    for column in range(1, 8)
]
LR0_SCRIPTS = [
    SCRIPT_DIR / "run_e4_gsm8k_lr0_8gpu.sh",
    SCRIPT_DIR / "run_e4_math_lr0_8gpu.sh",
]
OFT_LRS = (2e-6, 5e-6, 1e-5, 3e-5, 7e-5, 2e-4, 4e-4)
OFT_SCRIPTS = [
    SCRIPT_DIR / f"run_e4_{dataset}_oft_lr{column}_8gpu.sh"
    for dataset in ("gsm8k", "math")
    for column in range(7)
]


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


def _lr0_selected(path: Path) -> list[str]:
    return [a.name for a in e4lr0_arms() if _pattern(path).search(a.name)]


def _oft_selected(path: Path):
    return [arm for arm in _arms() if arm.method == "oft" and _pattern(path).search(arm.name)]


def _fake_python(tmp_path: Path) -> Path:
    """Stand in only for the unavailable GPU Python stack at campaign's edge."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-c" ]]; then
    exit 0
fi
if [[ -n "${CAPTURE_FILE:-}" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${MATRIX:-}" "${METHOD_RE:-}" "${RESULTS:-}" \
        "${EXPECT_ARMS:-}" "${ALLOW_OFT:-}" >> "${CAPTURE_FILE}"
fi
if [[ -n "${CACHE_CAPTURE_FILE:-}" ]]; then
    printf '%s\n' "${TRITON_CACHE_DIR:-}" > "${CACHE_CAPTURE_FILE}"
fi
printf '%s\n' \
    'ARM=one PEFT_METHOD=oft' \
    'ARM=two PEFT_METHOD=oft' \
    'ARM=three PEFT_METHOD=oft'
printf '3 arms selected, 0 already done, 3 to run\n' >&2
""",
        encoding="utf-8",
    )
    python.chmod(0o755)
    return fake_bin


def test_the_fourteen_oft_scripts_exist():
    assert all(path.is_file() for path in OFT_SCRIPTS)


def test_each_oft_script_selects_one_dataset_lr_and_three_blocks():
    """A regex typo would silently run the wrong column on a booked node."""
    for path in OFT_SCRIPTS:
        selected = _oft_selected(path)
        dataset = path.name.split("_")[2]
        column = int(re.search(r"_oft_lr(\d)_", path.name).group(1))
        assert len(selected) == 3, (path.name, [arm.name for arm in selected])
        assert {arm.dataset for arm in selected} == {dataset}
        assert {arm.lr for arm in selected} == {OFT_LRS[column]}
        assert {arm.oft_block_size for arm in selected} == {8, 128, 1024}
        assert {arm.target_modules for arm in selected} == {ALL_MODULES}
        assert all(arm.name.startswith("oftscout-") for arm in selected)


def test_the_oft_scripts_partition_all_forty_two_arms_once():
    """No OFT arm may be skipped or run twice across the fourteen ledgers."""
    selected = [arm.name for path in OFT_SCRIPTS for arm in _oft_selected(path)]
    expected = {arm.name for arm in _arms() if arm.method == "oft"}
    assert len(selected) == len(set(selected)) == 42
    assert set(selected) == expected


def test_every_oft_wrapper_dry_runs_through_the_real_campaign(tmp_path):
    """Dedicated OFT ledgers opt in, while the campaign remains training-free."""
    fake_bin = _fake_python(tmp_path)
    capture = tmp_path / "wrapper-env.txt"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "VIRTUAL_ENV": str(tmp_path / "venv"),
            "CUDA_HOME": str(tmp_path),
            "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
            "SKIP_PREFLIGHT": "1",
            "DRY_RUN": "1",
            "CAPTURE_FILE": str(capture),
        }
    )

    for wrapper in OFT_SCRIPTS:
        result = subprocess.run(
            ["bash", str(wrapper)],
            cwd=SCRIPT_DIR.parents[1],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (wrapper.name, result.stdout, result.stderr)
        assert "dry run -- launcher commands only" in result.stdout

    rows = [line.split("\t") for line in capture.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 14
    assert {row[0] for row in rows} == {"e4"}
    assert {row[3] for row in rows} == {"3"}
    assert {row[4] for row in rows} == {"1"}
    assert {row[2] for row in rows} == {
        f"results/e4_{dataset}_oft_lr{column}.jsonl"
        for dataset in ("gsm8k", "math")
        for column in range(7)
    }


def test_campaign_still_refuses_oft_without_a_dedicated_ledger_opt_in(tmp_path):
    fake_bin = _fake_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "VIRTUAL_ENV": str(tmp_path / "venv"),
            "CUDA_HOME": str(tmp_path),
            "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
            "SKIP_PREFLIGHT": "1",
            "DRY_RUN": "1",
            "MATRIX": "e4",
            "METHOD_RE": "^oftscout-",
            "RESULTS": str(tmp_path / "not-dedicated.jsonl"),
            "EXPECT_ARMS": "3",
        }
    )
    env.pop("ALLOW_OFT", None)

    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "campaign.sh")],
        cwd=SCRIPT_DIR.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "REFUSING: the selection contains OFT arms" in result.stderr


def _run_dry_campaign(tmp_path: Path, extra_env: dict[str, str]) -> subprocess.CompletedProcess:
    fake_bin = _fake_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "VIRTUAL_ENV": str(tmp_path / "venv"),
            "CUDA_HOME": str(tmp_path),
            "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
            "SKIP_PREFLIGHT": "1",
            "DRY_RUN": "1",
            "MATRIX": "e4",
            "METHOD_RE": "^lora-",
            "RESULTS": str(tmp_path / "results.jsonl"),
            "EXPECT_ARMS": "3",
            "ALLOW_OFT": "1",
            "CACHE_CAPTURE_FILE": str(tmp_path / "cache-dir.txt"),
            **extra_env,
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT_DIR / "campaign.sh")],
        cwd=SCRIPT_DIR.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_campaign_defaults_triton_cache_to_node_local_tmp(tmp_path):
    """A missing override must not leave Triton's concurrent JIT cache on NFS."""
    test_user = f"orbit-campaign-test-{os.getpid()}"
    cache_dir = Path(f"/tmp/triton_cache_{test_user}")
    if cache_dir.is_dir():
        cache_dir.rmdir()
    try:
        result = _run_dry_campaign(tmp_path, {"USER": test_user})
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert (tmp_path / "cache-dir.txt").read_text(encoding="utf-8").strip() == str(cache_dir)
        assert cache_dir.is_dir()
    finally:
        if cache_dir.is_dir():
            cache_dir.rmdir()


def test_campaign_preserves_explicit_triton_cache_dir(tmp_path):
    """A caller-selected local cache remains authoritative."""
    cache_dir = tmp_path / "custom-triton-cache"
    result = _run_dry_campaign(tmp_path, {"TRITON_CACHE_DIR": str(cache_dir)})
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert (tmp_path / "cache-dir.txt").read_text(encoding="utf-8").strip() == str(cache_dir)
    assert cache_dir.is_dir()


def test_lr0_scripts_exist():
    assert all(path.is_file() for path in LR0_SCRIPTS)


def test_lr0_scripts_partition_the_lr0_matrix():
    selected = [name for path in LR0_SCRIPTS for name in _lr0_selected(path)]

    assert len(selected) == len(set(selected)) == 6
    assert set(selected) == {arm.name for arm in e4lr0_arms()}


def test_each_lr0_script_selects_one_dataset_and_all_three_ranks():
    for path in LR0_SCRIPTS:
        selected = [arm for arm in e4lr0_arms() if _pattern(path).search(arm.name)]
        assert len(selected) == 3, (path.name, [arm.name for arm in selected])
        assert {arm.dataset for arm in selected} == {path.name.split("_")[2]}
        assert {arm.rank for arm in selected} == {1, 16, 256}
        assert {arm.method for arm in selected} == {"lora"}


def test_lr0_scripts_use_separate_ledgers_and_the_shared_protocol():
    texts = [path.read_text(encoding="utf-8") for path in LR0_SCRIPTS]
    ledgers = {re.search(r"RESULTS=(\S+)", text).group(1) for text in texts}

    assert ledgers == {"results/e4_gsm8k_lr0.jsonl", "results/e4_math_lr0.jsonl"}
    assert all("EXPECT_ARMS=3" in text for text in texts)
    assert all('source "${HERE}/e4_protocol.sh"' in text for text in texts)


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
