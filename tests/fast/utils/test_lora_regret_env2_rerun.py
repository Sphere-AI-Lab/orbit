import os
import re
import shutil
import subprocess
from pathlib import Path

from tools.lora_regret.arms import ALL_MODULES, MATRICES, e4_arms, e4lr0_arms
from tools.lora_regret.models import DEFAULT_MODEL, get as get_model
from tools.lora_regret.run_paths import resolve_arm_paths


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts" / "lora_regret" / "env2_rerun"
DATASETS = ("math", "gsm8k")
FULLFT_LRS = (5e-8, 1e-7, 3e-7, 7e-7, 2e-6, 4e-6, 1e-5)
LORA_LRS = (2e-6, 5e-6, 1e-5, 3e-5, 7e-5, 2e-4, 4e-4)
OFT_LRS = (5e-7, 1e-6, 3e-6, 7e-6, 2e-5, 4e-5, 1e-4)


def _wrappers() -> list[Path]:
    return [
        SCRIPT_DIR / f"run_e4_{dataset}_lr{column}_8gpu.sh"
        for dataset in DATASETS
        for column in range(1, 8)
    ]


def _oft_wrappers() -> list[Path]:
    return [
        SCRIPT_DIR / f"run_e4_{dataset}_oft_lr{column}_8gpu.sh"
        for dataset in DATASETS
        for column in range(1, 8)
    ]


def _fake_python(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-c" ]]; then
    exit 0
fi
case "${METHOD_RE:-}" in
    ^full-*) count=1; peft=none ;;
    ^lora-*) count=3; peft=lora ;;
    ^oftenv2-*) count=1; peft=oft ;;
    *) exit 91 ;;
esac
todo=${count}
if [[ "${peft}" == "none" && "${FULLFT_ALREADY_DONE:-0}" == "1" ]]; then
    todo=0
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${MATRIX:-}" "${METHOD_RE:-}" "${RESULTS:-}" "${EXPECT_ARMS:-}" \
    "${LORA_REGRET_LOG_DIR:-}" "${WANDB_DIR:-}" \
    "${LORA_REGRET_CKPT_DIR:-}" "${VIRTUAL_ENV:-}" "${ALLOW_OFT:-}" \
    "${PREFLIGHT_STAGE:-}" >> "${CAPTURE_FILE}"
for ((i = 0; i < todo; i++)); do
    printf 'ARM=arm%s PEFT_METHOD=%s\n' "${i}" "${peft}"
done
printf '%s arms selected, %s already done, %s to run\n' \
    "${count}" "$((count - todo))" "${todo}" >&2
""",
        encoding="utf-8",
    )
    python.chmod(0o755)
    return fake_bin


def _selected(matrix: str, method_re: str):
    if matrix == "e4lr0":
        arms = e4lr0_arms()
    elif matrix == "e4":
        arms = e4_arms()
    else:
        model = get_model(DEFAULT_MODEL)
        arms = MATRICES[matrix](
            model.hidden_size,
            model.ffn_size,
            model.qkv_output_size,
            0,
            None,
            None,
        )
    pattern = re.compile(method_re)
    return [arm for arm in arms if pattern.search(arm.name)]


def test_env2_wrappers_run_the_shifted_lora_grid_in_clean_output_roots(tmp_path):
    """Catches a missing column, an old lr7 LoRA selection, or output reuse."""
    wrappers = _wrappers()
    assert all(path.is_file() for path in wrappers)

    run_root = tmp_path / "env2-rerun"
    env_root = tmp_path / "orbit_env_v2"
    activate = env_root / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(f'export VIRTUAL_ENV="{env_root}"\n', encoding="utf-8")
    fake_bin = _fake_python(tmp_path)
    capture = tmp_path / "campaign-boundary.tsv"

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "ORBIT_ENV2_ROOT": str(env_root),
            "ORBIT_ENV2_ACTIVATE": str(activate),
            "E4_ENV2_RUN_ROOT": str(run_root),
            "SKIP_PREFLIGHT": "1",
            "DRY_RUN": "1",
            "CAPTURE_FILE": str(capture),
        }
    )

    for wrapper in wrappers:
        result = subprocess.run(
            ["bash", str(wrapper)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (wrapper.name, result.stdout, result.stderr)

    rows = [line.split("\t") for line in capture.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 28
    assert {row[2] for row in rows} == {
        str(run_root / "results" / f"e4_{dataset}_lr{column}.jsonl")
        for dataset in DATASETS
        for column in range(1, 8)
    }
    assert {row[4] for row in rows} == {str(run_root / "logs" / "lora_regret")}
    assert {row[5] for row in rows} == {str(run_root / "wandb")}
    assert {row[6] for row in rows} == {str(run_root / "orbit_ckpts" / "lora_regret")}
    assert {row[7] for row in rows} == {str(env_root)}

    for dataset in DATASETS:
        for column, (fullft_lr, lora_lr) in enumerate(zip(FULLFT_LRS, LORA_LRS), start=1):
            ledger = str(run_root / "results" / f"e4_{dataset}_lr{column}.jsonl")
            column_rows = [row for row in rows if row[2] == ledger]
            assert len(column_rows) == 2

            fullft_row = next(row for row in column_rows if row[3] == "1")
            lora_row = next(row for row in column_rows if row[3] == "3")
            fullft = _selected(fullft_row[0], fullft_row[1])
            lora = _selected(lora_row[0], lora_row[1])

            assert len(fullft) == 1
            assert fullft[0].method == "full"
            assert fullft[0].dataset == dataset
            assert fullft[0].lr == fullft_lr
            assert len(lora) == 3
            assert {arm.method for arm in lora} == {"lora"}
            assert {arm.dataset for arm in lora} == {dataset}
            assert {arm.rank for arm in lora} == {1, 16, 256}
            assert {arm.lr for arm in lora} == {lora_lr}

    assert not any(arm.lr == 1e-3 for row in rows for arm in _selected(row[0], row[1]))
    assert (run_root / "results").is_dir()
    assert (run_root / "logs" / "lora_regret").is_dir()
    assert (run_root / "wandb").is_dir()
    assert (run_root / "orbit_ckpts" / "lora_regret").is_dir()
    assert (run_root / "scheduler").is_dir()

    capture.unlink()
    env["FULLFT_ALREADY_DONE"] = "1"
    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "run_e4_math_lr1_8gpu.sh")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    resumed_rows = capture.read_text(encoding="utf-8").splitlines()
    assert len(resumed_rows) == 2, "a completed FullFT phase must not prevent the LoRA phase"
    assert "every arm in this selection is already recorded ok" in result.stdout


def test_env2_oft_wrappers_center_lr4_on_the_historical_math_optimum(tmp_path):
    from tools.lora_regret.preflight import EXPECTED_ARMS, STAGE_GPU_REQUIREMENTS

    assert EXPECTED_ARMS["e4oftenv2"] == 14
    assert STAGE_GPU_REQUIREMENTS["e4oftenv2"] == 8

    wrappers = _oft_wrappers()
    assert all(path.is_file() for path in wrappers)

    run_root = tmp_path / "env2-rerun"
    env_root = tmp_path / "orbit_env_v2"
    activate = env_root / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(f'export VIRTUAL_ENV="{env_root}"\n', encoding="utf-8")
    fake_bin = _fake_python(tmp_path)
    capture = tmp_path / "oft-campaign-boundary.tsv"

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "ORBIT_ENV2_ROOT": str(env_root),
            "ORBIT_ENV2_ACTIVATE": str(activate),
            "E4_ENV2_RUN_ROOT": str(run_root),
            "SKIP_PREFLIGHT": "1",
            "DRY_RUN": "1",
            "CAPTURE_FILE": str(capture),
        }
    )

    for wrapper in wrappers:
        result = subprocess.run(
            ["bash", str(wrapper)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (wrapper.name, result.stdout, result.stderr)

    rows = [line.split("\t") for line in capture.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 14
    assert {row[2] for row in rows} == {
        str(run_root / "results" / f"e4_{dataset}_oft_lr{column}.jsonl")
        for dataset in DATASETS
        for column in range(1, 8)
    }
    assert {row[4] for row in rows} == {str(run_root / "logs" / "lora_regret")}
    assert {row[5] for row in rows} == {str(run_root / "wandb")}
    assert {row[6] for row in rows} == {str(run_root / "orbit_ckpts" / "lora_regret")}
    assert {row[7] for row in rows} == {str(env_root)}
    assert {row[8] for row in rows} == {"1"}
    assert {row[9] for row in rows} == {"e4oftenv2"}

    for dataset in DATASETS:
        for column, expected_lr in enumerate(OFT_LRS, start=1):
            ledger = str(run_root / "results" / f"e4_{dataset}_oft_lr{column}.jsonl")
            row = next(row for row in rows if row[2] == ledger)
            selected = _selected(row[0], row[1])

            assert row[3] == "1"
            assert len(selected) == 1
            arm = selected[0]
            assert arm.method == "oft"
            assert arm.dataset == dataset
            assert arm.oft_block_size == 128
            assert arm.target_modules == ALL_MODULES
            assert arm.lr == expected_lr

    assert OFT_LRS[3] == 7e-6


def test_env2_oft_probe_cost_matches_the_shell_protocol():
    import tools.lora_regret.arms as arms

    expected_rollouts = getattr(arms, "E4_ENV2_OFT_ROLLOUTS", None)
    assert expected_rollouts == 150

    protocol = (REPO_ROOT / "scripts/lora_regret/e4_protocol.sh").read_text(
        encoding="utf-8"
    )
    match = re.search(r'^: "\$\{NUM_ROLLOUT=([0-9]+)\}"$', protocol, re.MULTILINE)
    assert match
    assert int(match.group(1)) == expected_rollouts


def test_env2_oft_aggregate_wrappers_visit_each_column_once(tmp_path):
    aggregate_wrappers = [
        SCRIPT_DIR / f"run_e4_{dataset}_oft_lr1_lr7_8gpu.sh"
        for dataset in DATASETS
    ]
    assert all(path.is_file() for path in aggregate_wrappers)

    run_root = tmp_path / "env2-rerun"
    env_root = tmp_path / "orbit_env_v2"
    activate = env_root / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(f'export VIRTUAL_ENV="{env_root}"\n', encoding="utf-8")
    fake_bin = _fake_python(tmp_path)
    capture = tmp_path / "oft-aggregate-boundary.tsv"

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "ORBIT_ENV2_ROOT": str(env_root),
            "ORBIT_ENV2_ACTIVATE": str(activate),
            "E4_ENV2_RUN_ROOT": str(run_root),
            "SKIP_PREFLIGHT": "1",
            "DRY_RUN": "1",
            "CAPTURE_FILE": str(capture),
        }
    )

    for wrapper in aggregate_wrappers:
        result = subprocess.run(
            ["bash", str(wrapper)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (wrapper.name, result.stdout, result.stderr)

    rows = [line.split("\t") for line in capture.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 14
    assert [row[2] for row in rows] == [
        str(run_root / "results" / f"e4_{dataset}_oft_lr{column}.jsonl")
        for dataset in DATASETS
        for column in range(1, 8)
    ]


def test_sweep_honors_explicit_log_and_checkpoint_roots(tmp_path):
    """Catches silently writing new runs back into the checkout's old paths."""
    log_dir = tmp_path / "new-logs"
    ckpt_dir = tmp_path / "new-checkpoints"
    arm = next(arm for arm in e4_arms() if arm.method == "full")

    log_path, save_dir = resolve_arm_paths(
        REPO_ROOT,
        arm.name,
        {
            "LORA_REGRET_LOG_DIR": str(log_dir),
            "LORA_REGRET_CKPT_DIR": str(ckpt_dir),
        },
    )

    assert log_path == log_dir / f"{arm.name}.log"
    assert save_dir == ckpt_dir / arm.name


def test_dedicated_sync_scans_only_the_env2_wandb_root(tmp_path):
    """Catches the shared sync script changing back to the old checkout root."""
    copied_repo = tmp_path / "repo"
    copied_scripts = copied_repo / "scripts" / "lora_regret"
    copied_env2 = copied_scripts / "env2_rerun"
    copied_env2.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "scripts/lora_regret/sync_wandb.sh", copied_scripts)
    shutil.copy2(SCRIPT_DIR / "env.sh", copied_env2)
    shutil.copy2(SCRIPT_DIR / "sync_wandb.sh", copied_env2)

    run_root = tmp_path / "env2-rerun"
    offline_dir = run_root / "wandb" / "wandb" / "offline-run-env2"
    offline_dir.mkdir(parents=True)
    (offline_dir / "run-env2.wandb").write_text("offline", encoding="utf-8")

    env_root = tmp_path / "orbit_env_v2"
    activate = env_root / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(f'export VIRTUAL_ENV="{env_root}"\n', encoding="utf-8")

    fake_bin = tmp_path / "sync-bin"
    fake_bin.mkdir()
    wandb = fake_bin / "wandb"
    wandb.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$PWD" > "${SYNC_CAPTURE_DIR}/cwd"
printf '%s\n' "$@" > "${SYNC_CAPTURE_DIR}/args"
""",
        encoding="utf-8",
    )
    wandb.chmod(0o755)
    capture_dir = tmp_path / "sync-capture"
    capture_dir.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "ORBIT_ENV2_ROOT": str(env_root),
            "ORBIT_ENV2_ACTIVATE": str(activate),
            "E4_ENV2_RUN_ROOT": str(run_root),
            "SYNC_CAPTURE_DIR": str(capture_dir),
            "QUIESCE_MIN": "999999",
        }
    )
    result = subprocess.run(
        ["bash", str(copied_env2 / "sync_wandb.sh")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert (capture_dir / "cwd").read_text(encoding="utf-8").strip() == str(
        run_root / "wandb"
    )
    args = (capture_dir / "args").read_text(encoding="utf-8").splitlines()
    assert "wandb/offline-run-env2" in args

    legacy_offline = copied_repo / "wandb" / "offline-run-legacy"
    legacy_offline.mkdir(parents=True)
    (legacy_offline / "run-legacy.wandb").write_text("offline", encoding="utf-8")
    env["VIRTUAL_ENV"] = str(env_root)
    env.pop("WANDB_SYNC_ROOT", None)
    result = subprocess.run(
        ["bash", str(copied_scripts / "sync_wandb.sh")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert (capture_dir / "cwd").read_text(encoding="utf-8").strip() == str(copied_repo)
    args = (capture_dir / "args").read_text(encoding="utf-8").splitlines()
    assert "wandb/offline-run-legacy" in args
