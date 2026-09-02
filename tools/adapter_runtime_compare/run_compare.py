#!/usr/bin/env python3
"""Run and parse paired adapter-runtime comparison experiments.

This harness intentionally lives outside the two candidate worktrees. It keeps
branch root, uv environment, GPU slice, and launcher overrides explicit so the
comparison can be reproduced without relying on shell state.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import dataclasses
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "adapter_runtime_compare"
DEFAULT_CKPT_ROOT = REPO_ROOT / "tmp_ckpts" / "adapter_runtime_compare"
DEFAULT_WORKSPACE_ROOT = Path(os.environ.get("ORBIT_COMPARE_WORKSPACE", REPO_ROOT.parent))


@dataclasses.dataclass(frozen=True)
class Branch:
    name: str
    root: Path
    env_root: Path

    @property
    def venv_bin(self) -> Path:
        return self.env_root / ".venv" / "bin"


BRANCHES = {
    "runtime": Branch(
        name="runtime",
        root=Path(
            os.environ.get(
                "ORBIT_COMPARE_RUNTIME_ROOT",
                DEFAULT_WORKSPACE_ROOT / "orbit_versioned_adapter_runtime",
            )
        ),
        env_root=Path(os.environ.get("ORBIT_COMPARE_RUNTIME_ENV", DEFAULT_WORKSPACE_ROOT / "runtime-env")),
    ),
    "cc": Branch(
        name="cc",
        root=Path(
            os.environ.get(
                "ORBIT_COMPARE_CC_ROOT",
                DEFAULT_WORKSPACE_ROOT / "orbit_versioned_adapter_runtime_cc",
            )
        ),
        env_root=Path(os.environ.get("ORBIT_COMPARE_CC_ENV", DEFAULT_WORKSPACE_ROOT / "cc-env")),
    ),
}


@dataclasses.dataclass(frozen=True)
class Case:
    model: str
    precision: str
    peft: str
    script: str
    gpu_total: int
    train_gpus_async: int
    rollout_gpus_async: int
    # Per-engine TP for the async arms. None -> rollout_gpus_async (one engine
    # spanning all rollout GPUs). Small dense models must pin 1: OFT requires
    # every TP-sharded input dim to stay divisible by the block size
    # (Qwen2.5-0.5B hidden 896 / TP2 = 448, not a multiple of 128).
    rollout_gpus_per_engine: int | None = None
    target_modules: str | None = None
    lora_backend: str | None = None
    fullft_script: str | None = None
    extra_env: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.model}_{self.precision}_{self.peft}"


CASES = [
    Case(
        model="qwen25_05b",
        precision="bf16",
        peft="lora",
        script="examples/high_precision/run-qwen2_5-0_5b-bf16-math-lora.sh",
        gpu_total=4,
        train_gpus_async=2,
        rollout_gpus_async=2,
        lora_backend="csgmv",
    ),
    Case(
        model="qwen25_05b",
        precision="bf16",
        peft="oft",
        script="examples/high_precision/run-qwen2_5-0_5b-bf16-math-oft.sh",
        gpu_total=4,
        train_gpus_async=2,
        rollout_gpus_async=2,
        fullft_script="examples/high_precision/run-qwen2_5-0_5b-bf16-math-fullft-async.sh",
        extra_env={"OFT_BLOCK_SIZE": "64"},
    ),
    Case(
        model="qwen25_3b",
        precision="bf16",
        peft="oft",
        script="examples/high_precision/run-qwen2_5-3b-bf16-math-oft.sh",
        gpu_total=4,
        train_gpus_async=2,
        rollout_gpus_async=2,
        fullft_script="examples/high_precision/run-qwen2_5-3b-bf16-math-fullft-async.sh",
        extra_env={"REQUIRE_MEGATRON_LOAD": "1"},
    ),
    Case(
        model="qwen3_4b",
        precision="bf16",
        peft="oft",
        script="examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft.sh",
        gpu_total=4,
        train_gpus_async=2,
        rollout_gpus_async=2,
        target_modules="linear_qkv,linear_proj,linear_fc1,linear_fc2",
        fullft_script="examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-fullft-async.sh",
        extra_env={"REQUIRE_MEGATRON_LOAD": "1"},
    ),
    Case(
        model="qwen3_4b",
        precision="bf16",
        peft="lora",
        script="examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-lora.sh",
        gpu_total=4,
        train_gpus_async=2,
        rollout_gpus_async=2,
        target_modules="linear_qkv,linear_proj,linear_fc1,linear_fc2",
        lora_backend="triton",
        extra_env={"REQUIRE_MEGATRON_LOAD": "1"},
    ),
    Case(
        model="qwen3_4b",
        precision="int4",
        peft="oft",
        script="examples/infra_features/low_precision/run-qwen3-4b-int4-math-oft.sh",
        gpu_total=4,
        train_gpus_async=2,
        rollout_gpus_async=2,
        target_modules="linear_qkv,linear_proj,linear_fc1,linear_fc2",
        extra_env={"REQUIRE_MEGATRON_LOAD": "1"},
    ),
    Case(
        model="qwen3_4b",
        precision="int4",
        peft="lora",
        script="examples/infra_features/low_precision/run-qwen3-4b-int4-math-oft.sh",
        gpu_total=4,
        train_gpus_async=2,
        rollout_gpus_async=2,
        target_modules="linear_qkv,linear_proj,linear_fc1,linear_fc2",
        lora_backend="triton",
        extra_env={"REQUIRE_MEGATRON_LOAD": "1"},
    ),
    Case(
        model="qwen3_30b",
        precision="bf16",
        peft="oft",
        script="examples/high_precision/run-qwen3-30b-a3b-bf16-openr1-oft-b32.sh",
        gpu_total=8,
        train_gpus_async=4,
        rollout_gpus_async=4,
        target_modules="linear_qkv,linear_proj,linear_fc1,linear_fc2",
        fullft_script="examples/high_precision/run-qwen3-30b-a3b-bf16-openr1-fullft-async.sh",
    ),
    Case(
        model="qwen3_30b",
        precision="bf16",
        peft="lora",
        script="examples/high_precision/run-qwen3-30b-a3b-bf16-openr1-lora.sh",
        gpu_total=8,
        train_gpus_async=4,
        rollout_gpus_async=4,
        target_modules="linear_qkv,linear_proj,linear_fc1,linear_fc2",
        lora_backend="triton",
    ),
]

@dataclasses.dataclass(frozen=True)
class Arm:
    """A named comparison arm: which launcher a case runs (``script_field``
    selects the Case attribute holding it) plus the env overrides that pick
    the entrypoint/topology and the weight-sync path."""

    name: str
    entrypoint: str  # relative to the branch root
    colocate: bool
    script_field: str = "script"
    env: dict[str, str] = dataclasses.field(default_factory=dict)


ARMS = {
    "sync": Arm(name="sync", entrypoint="train.py", colocate=True, env={"ADAPTER_DOUBLE_BUFFER": "0"}),
    "async": Arm(name="async", entrypoint="train_async.py", colocate=False, env={"ADAPTER_DOUBLE_BUFFER": "0"}),
    "async_db": Arm(name="async_db", entrypoint="train_async.py", colocate=False, env={"ADAPTER_DOUBLE_BUFFER": "1"}),
    # Full-model sync arm: full fine-tuning (--peft-method none) exercising the
    # legacy full-parameter broadcast in update_weights instead of the PEFT
    # adapter push. Needs a dedicated launcher (Case.fullft_script) because the
    # adapter launchers hardcode their PEFT flags.
    "async_fullft": Arm(
        name="async_fullft",
        entrypoint="train_async.py",
        colocate=False,
        script_field="fullft_script",
        env={"ADAPTER_DOUBLE_BUFFER": "0", "PEFT_METHOD": "none"},
    ),
}

MODES = tuple(ARMS)
# Backward-compatible default: async_fullft runs only when selected via --modes.
DEFAULT_MODES = ("sync", "async", "async_db")

# Case env that only makes sense with a PEFT adapter; scrubbed when an arm
# forces PEFT_METHOD=none so manifests do not record misleading knobs.
PEFT_ONLY_ENV_KEYS = (
    "TARGET_MODULES",
    "SGLANG_LORA_BACKEND",
    "OFT_BLOCK_SIZE",
    "LORA_RANK",
    "LORA_ALPHA",
    "LORA_DROPOUT",
)


def arm_script(case: Case, mode: str) -> str | None:
    return getattr(case, ARMS[mode].script_field)


def arm_peft(case: Case, mode: str) -> str:
    return ARMS[mode].env.get("PEFT_METHOD", case.peft)


@dataclasses.dataclass(frozen=True)
class Job:
    run_id: str
    repeat: int
    branch: Branch
    case: Case
    mode: str
    script: str
    peft: str
    gpu_ids: tuple[int, ...]
    run_dir: Path
    num_rollout: int
    eval_enabled: bool
    eval_interval: int
    skip_eval_before_train: bool
    batch_profile: str
    # A3 axes: per-repeat seed (--seeds) and a per-arm learning rate
    # (--fullft-lr, applied to the async_fullft arm only). None -> launcher default.
    seed: int | None = None
    lr: str | None = None

    @property
    def run_log(self) -> Path:
        return self.run_dir / "run.log"

    @property
    def console_log(self) -> Path:
        return self.run_dir / "console.log"

    @property
    def status_path(self) -> Path:
        return self.run_dir / "status.json"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run.json"


def parse_csv_filter(value: str | None, allowed: set[str], label: str) -> set[str]:
    if not value:
        return set(allowed)
    selected = {item.strip() for item in value.split(",") if item.strip()}
    unknown = selected - allowed
    if unknown:
        raise SystemExit(f"Unknown {label}: {', '.join(sorted(unknown))}. Allowed: {', '.join(sorted(allowed))}")
    return selected


def selected_cases(args: argparse.Namespace) -> list[Case]:
    models = parse_csv_filter(args.models, {case.model for case in CASES}, "model")
    precisions = parse_csv_filter(args.precisions, {case.precision for case in CASES}, "precision")
    pefts = parse_csv_filter(args.pefts, {case.peft for case in CASES}, "peft")
    cases = [
        case
        for case in CASES
        if case.model in models and case.precision in precisions and case.peft in pefts
    ]
    if args.profile == "pilot":
        cases = [case for case in cases if case.key == "qwen25_05b_bf16_lora"]
    elif args.profile == "q25":
        cases = [case for case in cases if case.model == "qwen25_05b"]
    elif args.profile == "q3_4b":
        cases = [case for case in cases if case.model == "qwen3_4b"]
    elif args.profile == "q3_30b":
        cases = [case for case in cases if case.model == "qwen3_30b"]
    if not cases:
        raise SystemExit("No cases selected")
    return cases


def selected_branches(args: argparse.Namespace) -> list[Branch]:
    names = parse_csv_filter(args.branches, set(BRANCHES), "branch")
    return [BRANCHES[name] for name in ("runtime", "cc") if name in names]


def selected_modes(args: argparse.Namespace) -> list[str]:
    requested = args.modes or ",".join(DEFAULT_MODES)
    modes = parse_csv_filter(requested, set(MODES), "mode")
    ordered = [mode for mode in MODES if mode in modes]
    if args.profile == "pilot":
        ordered = [mode for mode in ordered if mode == "async"]
    if not ordered:
        raise SystemExit("No modes selected")
    return ordered


def default_rollouts(profile: str) -> int:
    if profile == "pilot":
        return 3
    return 30


def parse_seeds(value: str | None) -> tuple[int, ...]:
    """``--seeds 1234,1235`` -> (1234, 1235); empty/None -> () (no seed axis)."""
    if not value:
        return ()
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise SystemExit(f"--seeds must be comma-separated integers, got {value!r}") from exc
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"--seeds contains duplicates: {value!r}")
    return seeds


def parse_gpu_slice(value: str | None) -> tuple[int, ...]:
    """``--gpu-slice 0,1,2,3`` -> (0, 1, 2, 3); empty/None -> () (default alternation).

    The default wave layout alternates odd repeats onto GPUs 4-7 (two branches on an
    8-GPU node); a pinned slice runs every wave on the same GPUs, which is what a
    single-branch campaign on a 4-GPU node needs.
    """
    if not value:
        return ()
    try:
        gpus = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise SystemExit(f"--gpu-slice must be comma-separated integers, got {value!r}") from exc
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise SystemExit(f"--gpu-slice must be distinct non-negative GPU ids, got {value!r}")
    return gpus


def build_waves(args: argparse.Namespace) -> list[list[Job]]:
    branches = selected_branches(args)
    cases = selected_cases(args)
    modes = selected_modes(args)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    campaign = args.campaign or f"{args.profile}_{stamp}"
    output_root = Path(args.output_dir).resolve() / campaign
    rollouts = args.num_rollout if args.num_rollout is not None else default_rollouts(args.profile)
    eval_enabled = args.eval if args.eval is not None else args.profile != "pilot"
    seeds = parse_seeds(getattr(args, "seeds", None))
    fullft_lr = getattr(args, "fullft_lr", None)
    gpu_slice = parse_gpu_slice(getattr(args, "gpu_slice", None))
    # --seeds is the repeat axis when given: one repeat per seed, in order.
    repeats = len(seeds) if seeds else args.repeats
    waves: list[list[Job]] = []

    for repeat in range(repeats):
        seed = seeds[repeat] if seeds else None
        for case in cases:
            for mode in modes:
                if arm_script(case, mode) is None:
                    if repeat == 0:
                        print(
                            f"skipping {case.key} mode={mode}: case defines no {ARMS[mode].script_field}",
                            file=sys.stderr,
                        )
                    continue
                branch_order = branches
                if repeat % 2 == 1:
                    branch_order = list(reversed(branches))
                if case.gpu_total == 8:
                    for branch in branch_order:
                        waves.append(
                            [
                                make_job(
                                    output_root,
                                    branch,
                                    case,
                                    mode,
                                    repeat,
                                    gpu_slice or tuple(range(8)),
                                    rollouts,
                                    eval_enabled,
                                    args.eval_interval,
                                    args.skip_eval_before_train,
                                    args.batch_profile,
                                    seed=seed,
                                    fullft_lr=fullft_lr,
                                )
                            ]
                        )
                    continue

                low = (0, 1, 2, 3)
                high = (4, 5, 6, 7)
                if repeat % 2 == 1:
                    low, high = high, low
                gpu_slices = [gpu_slice] if gpu_slice else [low, high]
                wave = []
                for index, branch in enumerate(branch_order):
                    wave.append(
                        make_job(
                            output_root,
                            branch,
                            case,
                            mode,
                            repeat,
                            gpu_slices[index % len(gpu_slices)],
                            rollouts,
                            eval_enabled,
                            args.eval_interval,
                            args.skip_eval_before_train,
                            args.batch_profile,
                            seed=seed,
                            fullft_lr=fullft_lr,
                        )
                    )
                waves.append(wave)
    if not waves:
        raise SystemExit("No runnable jobs selected (selected modes need launchers on the selected cases)")
    return waves


def make_job(
    output_root: Path,
    branch: Branch,
    case: Case,
    mode: str,
    repeat: int,
    gpu_ids: tuple[int, ...],
    num_rollout: int,
    eval_enabled: bool,
    eval_interval: int,
    skip_eval_before_train: bool,
    batch_profile: str,
    *,
    seed: int | None = None,
    fullft_lr: str | None = None,
) -> Job:
    script = arm_script(case, mode)
    if script is None:
        raise SystemExit(f"Case {case.key} defines no {ARMS[mode].script_field} for mode {mode}")
    peft = arm_peft(case, mode)
    run_id = (
        f"r{repeat:02d}_{branch.name}_{case.model}_{case.precision}_{peft}_{mode}_"
        f"g{''.join(str(g) for g in gpu_ids)}"
    )
    return Job(
        run_id=run_id,
        repeat=repeat,
        branch=branch,
        case=case,
        mode=mode,
        script=script,
        peft=peft,
        gpu_ids=gpu_ids,
        run_dir=output_root / run_id,
        num_rollout=num_rollout,
        eval_enabled=eval_enabled,
        eval_interval=eval_interval,
        skip_eval_before_train=skip_eval_before_train,
        batch_profile=batch_profile,
        seed=seed,
        lr=fullft_lr if mode == "async_fullft" else None,
    )


def validate_job(job: Job) -> None:
    script = job.branch.root / job.script
    if not script.exists():
        raise SystemExit(f"Missing launcher for {job.run_id}: {script}")
    python = job.branch.venv_bin / "python3"
    if not python.exists():
        raise SystemExit(f"Missing uv venv python for {job.branch.name}: {python}")


def job_env(job: Job) -> dict[str, str]:
    env = os.environ.copy()
    venv_bin = str(job.branch.venv_bin)
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = str(job.branch.env_root / ".venv")
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in job.gpu_ids)
    env["ORBIT_RAY_LIFECYCLE"] = "private"
    env["ORBIT_LOG_FILTER"] = env.get("ORBIT_LOG_FILTER", "1")
    env["RAY_NUM_CPUS"] = env.get("RAY_NUM_CPUS", "64")

    env["RUN_TAG"] = job.run_id
    env["RUN_LOG"] = str(job.run_log)
    env["SAVE_DIR"] = str(DEFAULT_CKPT_ROOT / job.run_id)
    env["NUM_ROLLOUT"] = str(job.num_rollout)
    # Launchers read SEED / LR with their recipe literals as defaults, so an
    # unset axis leaves the recipe untouched (and the manifest omits the key).
    if job.seed is not None:
        env["SEED"] = str(job.seed)
    if job.lr is not None:
        env["LR"] = str(job.lr)
    env["DISABLE_EVAL"] = "0" if job.eval_enabled else "1"
    env["SKIP_EVAL_BEFORE_TRAIN"] = "1" if job.skip_eval_before_train else "0"
    env["EVAL_INTERVAL"] = str(job.eval_interval)
    env["ENABLE_WANDB"] = "0"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env["DATASET"] = "math"

    apply_batch_profile(env, job.batch_profile, job.case)
    apply_case_env(env, job.case)
    # Mode/arm env is applied last so arm overrides (e.g. PEFT_METHOD=none for
    # the full-FT arm) win over the case defaults.
    apply_mode_env(env, job)
    return env


def apply_batch_profile(env: dict[str, str], profile: str, case: Case) -> None:
    if profile == "recipe":
        return
    if profile not in {"bench", "tiny"}:
        raise SystemExit(f"Unknown batch profile {profile!r}")
    # Bench profile keeps wall-clock manageable across many branch/mode pairs.
    env["ROLLOUT_BATCH_SIZE"] = "16"
    env["N_SAMPLES_PER_PROMPT"] = "2"
    env["GLOBAL_BATCH_SIZE"] = "32"
    env["ROLLOUT_MAX_RESPONSE_LEN"] = "1024"
    env["EVAL_MAX_RESPONSE_LEN"] = "1024"
    env["N_SAMPLES_PER_EVAL_PROMPT"] = "1"
    if case.model == "qwen3_30b":
        env["SGLANG_MEM_FRACTION_STATIC"] = "0.50"
    if profile == "tiny":
        env["ROLLOUT_BATCH_SIZE"] = "4"
        env["N_SAMPLES_PER_PROMPT"] = "2"
        env["GLOBAL_BATCH_SIZE"] = "8"
        env["ROLLOUT_MAX_RESPONSE_LEN"] = "256"
        env["EVAL_MAX_RESPONSE_LEN"] = "256"


def apply_mode_env(env: dict[str, str], job: Job) -> None:
    arm = ARMS.get(job.mode)
    if arm is None:
        raise SystemExit(f"Unknown mode: {job.mode}")
    root = job.branch.root
    env["ORBIT_ENTRYPOINT"] = str(root / arm.entrypoint)
    if arm.colocate:
        env["ORBIT_COLOCATE"] = "1"
        env["GPUS_PER_NODE"] = str(job.case.gpu_total)
        env.pop("ROLLOUT_NUM_GPUS", None)
        env.pop("ROLLOUT_NUM_GPUS_PER_ENGINE", None)
    else:
        env["ORBIT_COLOCATE"] = "0"
        env["GPUS_PER_NODE"] = str(job.case.train_gpus_async)
        env["ROLLOUT_NUM_GPUS"] = str(job.case.rollout_gpus_async)
        env["ROLLOUT_NUM_GPUS_PER_ENGINE"] = str(
            job.case.rollout_gpus_per_engine
            if job.case.rollout_gpus_per_engine is not None
            else job.case.rollout_gpus_async
        )
    env.update(arm.env)
    if arm.env.get("PEFT_METHOD") == "none":
        for key in PEFT_ONLY_ENV_KEYS:
            env.pop(key, None)


def apply_case_env(env: dict[str, str], case: Case) -> None:
    env["PEFT_METHOD"] = case.peft
    if case.target_modules:
        env["TARGET_MODULES"] = case.target_modules
    if case.peft == "lora":
        env["LORA_RANK"] = env.get("LORA_RANK", "32")
        env["LORA_ALPHA"] = env.get("LORA_ALPHA", "64")
        env["LORA_DROPOUT"] = env.get("LORA_DROPOUT", "0.0")
    if case.lora_backend:
        env["SGLANG_LORA_BACKEND"] = case.lora_backend
    env.update(case.extra_env)


def write_manifest(job: Job, env: dict[str, str]) -> None:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    public_env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "VIRTUAL_ENV",
        "ORBIT_ENTRYPOINT",
        "ORBIT_COLOCATE",
        "GPUS_PER_NODE",
        "ROLLOUT_NUM_GPUS",
        "ROLLOUT_NUM_GPUS_PER_ENGINE",
        "ADAPTER_DOUBLE_BUFFER",
        "NUM_ROLLOUT",
        "DISABLE_EVAL",
        "SKIP_EVAL_BEFORE_TRAIN",
        "EVAL_INTERVAL",
        "ROLLOUT_BATCH_SIZE",
        "N_SAMPLES_PER_PROMPT",
        "GLOBAL_BATCH_SIZE",
        "ROLLOUT_MAX_RESPONSE_LEN",
        "EVAL_MAX_RESPONSE_LEN",
        "PEFT_METHOD",
        "TARGET_MODULES",
        "OFT_BLOCK_SIZE",
        "SGLANG_LORA_BACKEND",
        "SAVE_DIR",
        "RUN_LOG",
        "SEED",
        "LR",
        "SGLANG_ATTENTION_BACKEND",
    ]
    manifest = {
        "run_id": job.run_id,
        "repeat": job.repeat,
        "branch": job.branch.name,
        "branch_root": str(job.branch.root),
        "env_root": str(job.branch.env_root),
        "model": job.case.model,
        "precision": job.case.precision,
        "peft": job.peft,
        "mode": job.mode,
        "script": job.script,
        "gpu_ids": list(job.gpu_ids),
        "num_rollout": job.num_rollout,
        "eval_enabled": job.eval_enabled,
        "eval_interval": job.eval_interval,
        "batch_profile": job.batch_profile,
        "seed": job.seed,
        "lr": job.lr,
        "env": {key: env[key] for key in public_env_keys if key in env},
    }
    job.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run_job(job: Job) -> subprocess.Popen[bytes]:
    validate_job(job)
    env = job_env(job)
    write_manifest(job, env)
    status = {
        "run_id": job.run_id,
        "status": "running",
        "start_time": time.time(),
        "start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    job.status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    console = job.console_log.open("ab")
    command = ["bash", str(job.branch.root / job.script)]
    return subprocess.Popen(command, env=env, stdout=console, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def update_status(job: Job, process: subprocess.Popen[bytes], started: float) -> None:
    end = time.time()
    status = {
        "run_id": job.run_id,
        "status": "ok" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "start_time": started,
        "end_time": end,
        "wall_s": end - started,
    }
    job.status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def run_waves(args: argparse.Namespace) -> int:
    waves = build_waves(args)
    if args.dry_run:
        print_plan(waves)
        return 0

    failures = 0
    active: list[tuple[Job, subprocess.Popen[bytes], float]] = []

    def terminate_active(signum: int, _frame: Any) -> None:
        print(f"Received signal {signum}; terminating active jobs", file=sys.stderr)
        for active_job, proc, _started in active:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                active_job.status_path.write_text(
                    json.dumps({"run_id": active_job.run_id, "status": "terminated"}, indent=2) + "\n"
                )
        raise SystemExit(130)

    old_int = signal.signal(signal.SIGINT, terminate_active)
    old_term = signal.signal(signal.SIGTERM, terminate_active)
    try:
        for wave_index, wave in enumerate(waves, start=1):
            print(f"wave {wave_index}/{len(waves)}: {', '.join(job.run_id for job in wave)}", flush=True)
            active.clear()
            for job in wave:
                started = time.time()
                process = run_job(job)
                active.append((job, process, started))
                print(
                    f"  started {job.run_id} pid={process.pid} gpus={','.join(map(str, job.gpu_ids))}",
                    flush=True,
                )
            for job, process, started in active:
                process.wait()
                update_status(job, process, started)
                if process.returncode != 0:
                    failures += 1
                    print(f"  failed {job.run_id} rc={process.returncode} log={job.run_log}", flush=True)
                else:
                    print(f"  finished {job.run_id} log={job.run_log}", flush=True)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
    return 1 if failures else 0


def print_plan(waves: list[list[Job]]) -> None:
    for wave_index, wave in enumerate(waves, start=1):
        print(f"wave {wave_index}:")
        for job in wave:
            print(
                "  "
                f"{job.run_id} branch={job.branch.name} model={job.case.model} "
                f"precision={job.case.precision} peft={job.peft} mode={job.mode} "
                f"gpus={','.join(map(str, job.gpu_ids))} rollouts={job.num_rollout} "
                f"eval={int(job.eval_enabled)} script={job.script}"
            )


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
METRIC_RE = re.compile(r"\b(?P<kind>perf|eval|step) (?P<rollout>\d+): (?P<payload>\{.*\})")
TIMING_RE = re.compile(r"\b(?P<prefix>startup|rollout \d+|shutdown): (?P<phase>.+?) done elapsed=(?P<seconds>[0-9.]+)s")
ROLLOUT_PROF_RE = re.compile(
    r"\[rollout-prof\].*?rollout_id=(?P<rollout>\d+).*?SUMMARY:.*?wall=(?P<wall>[0-9.]+)s.*?throughput=(?P<throughput>[0-9.]+)"
)
EVAL_PROF_RE = re.compile(r"\[eval-prof\] dataset_total name=(?P<name>\S+) wall=(?P<wall>[0-9.]+)s")


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_payload(text: str) -> dict[str, Any] | None:
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def metric_bucket(kind: str, payload: dict[str, Any]) -> str:
    if kind == "eval":
        return "eval"
    if any(key.startswith("train/") or key.startswith("loss/") for key in payload):
        return "train_step"
    if any(key in payload for key in ("perf/train_time", "perf/actor_train_time", "perf/log_probs_time")):
        return "train_perf"
    if any(key.startswith("rollout/") for key in payload) or "perf/rollout_time" in payload:
        return "rollout_perf"
    return kind


def parse_log(path: Path) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    timings: dict[str, list[float]] = defaultdict(list)
    rollout_prof: list[dict[str, float]] = []
    eval_prof: list[dict[str, float | str]] = []

    if not path.exists():
        return {"missing_log": True}

    for raw_line in path.read_text(errors="ignore").splitlines():
        line = strip_ansi(raw_line)
        metric_match = METRIC_RE.search(line)
        if metric_match:
            payload = parse_payload(metric_match.group("payload"))
            if payload is not None:
                bucket = metric_bucket(metric_match.group("kind"), payload)
                buckets[bucket].append({"rollout": int(metric_match.group("rollout")), **payload})
            continue
        timing_match = TIMING_RE.search(line)
        if timing_match:
            phase = timing_match.group("phase").replace(" ", "_")
            timings[phase].append(float(timing_match.group("seconds")))
            continue
        rollout_match = ROLLOUT_PROF_RE.search(line)
        if rollout_match:
            rollout_prof.append(
                {
                    "rollout": float(rollout_match.group("rollout")),
                    "wall": float(rollout_match.group("wall")),
                    "throughput": float(rollout_match.group("throughput")),
                }
            )
            continue
        eval_match = EVAL_PROF_RE.search(line)
        if eval_match:
            eval_prof.append({"name": eval_match.group("name"), "wall": float(eval_match.group("wall"))})

    summary: dict[str, Any] = {
        "missing_log": False,
        "rollout_perf_count": len(buckets["rollout_perf"]),
        "train_perf_count": len(buckets["train_perf"]),
        "train_step_count": len(buckets["train_step"]),
        "eval_count": len(buckets["eval"]),
        "rollout_prof_count": len(rollout_prof),
        "eval_prof_count": len(eval_prof),
    }
    for bucket_name, rows in buckets.items():
        add_metric_summaries(summary, bucket_name, rows)
    for phase, values in timings.items():
        add_values(summary, f"timing/{phase}", values)
    if rollout_prof:
        add_values(summary, "rollout_prof/wall", [row["wall"] for row in rollout_prof])
        add_values(summary, "rollout_prof/throughput", [row["throughput"] for row in rollout_prof])
    if eval_prof:
        add_values(summary, "eval_prof/wall", [float(row["wall"]) for row in eval_prof])
    return summary


def add_metric_summaries(summary: dict[str, Any], bucket_name: str, rows: list[dict[str, Any]]) -> None:
    values_by_key: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if key == "rollout":
                continue
            float_value = as_float(value)
            if float_value is not None:
                values_by_key[key].append(float_value)
    for key, values in values_by_key.items():
        safe_key = key.replace("/", ".")
        add_values(summary, f"{bucket_name}/{safe_key}", values)


def add_values(summary: dict[str, Any], prefix: str, values: list[float]) -> None:
    if not values:
        return
    summary[f"{prefix}.last"] = values[-1]
    summary[f"{prefix}.mean"] = sum(values) / len(values)
    summary[f"{prefix}.min"] = min(values)
    summary[f"{prefix}.max"] = max(values)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def summarize_run_dir(run_dir: Path) -> dict[str, Any] | None:
    manifest = load_json(run_dir / "run.json")
    if not manifest:
        return None
    status = load_json(run_dir / "status.json")
    log_path = Path(manifest.get("env", {}).get("RUN_LOG", run_dir / "run.log"))
    parsed = parse_log(log_path)
    row = {
        "run_id": manifest.get("run_id", run_dir.name),
        "branch": manifest.get("branch"),
        "model": manifest.get("model"),
        "precision": manifest.get("precision"),
        "peft": manifest.get("peft"),
        "mode": manifest.get("mode"),
        "repeat": manifest.get("repeat"),
        "gpu_ids": ",".join(map(str, manifest.get("gpu_ids", []))),
        "num_rollout": manifest.get("num_rollout"),
        "eval_enabled": manifest.get("eval_enabled"),
        "status": status.get("status", "unknown"),
        "returncode": status.get("returncode"),
        "wall_s": status.get("wall_s"),
        "run_dir": str(run_dir),
        "run_log": str(log_path),
    }
    row.update(parsed)
    return row


def parse_runs(args: argparse.Namespace) -> int:
    root = Path(args.path).resolve()
    run_dirs = [path for path in root.rglob("*") if path.is_dir() and (path / "run.json").exists()]
    rows = [row for row in (summarize_run_dir(path) for path in sorted(run_dirs)) if row is not None]
    if not rows:
        print(f"No run.json files found under {root}", file=sys.stderr)
        return 1
    columns = ordered_columns(rows)
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    else:
        print_markdown(rows, columns)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "json":
            output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        elif args.format == "csv":
            with output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        else:
            with output.open("w") as handle:
                with contextlib.redirect_stdout(handle):
                    print_markdown(rows, columns)
    return 0


def ordered_columns(rows: list[dict[str, Any]]) -> list[str]:
    first = [
        "run_id",
        "branch",
        "model",
        "precision",
        "peft",
        "mode",
        "repeat",
        "gpu_ids",
        "status",
        "returncode",
        "wall_s",
        "rollout_perf_count",
        "train_perf_count",
        "train_step_count",
        "eval_count",
        "rollout_perf/perf.rollout_time.mean",
        "rollout_perf/perf.tokens_per_gpu_per_sec.mean",
        "rollout_perf/rollout.raw_reward.mean",
        "train_perf/perf.update_weights_time.mean",
        "train_perf/perf.actor_train_time.mean",
        "train_perf/perf.log_probs_time.mean",
        "train_perf/perf.step_time.mean",
        "train_step/train.train_rollout_logprob_abs_diff.mean",
        "eval/eval.math.last",
        "eval/eval.math.mean",
        "timing/generate.mean",
        "timing/actor_train.mean",
        "timing/actor_update_weights.mean",
        "timing/eval.mean",
        "run_log",
    ]
    all_keys = set().union(*(row.keys() for row in rows))
    return [key for key in first if key in all_keys] + sorted(all_keys - set(first))


def print_markdown(rows: list[dict[str, Any]], columns: list[str]) -> None:
    preferred = [column for column in columns if column not in {"run_dir"}]
    preferred = preferred[:28]
    print("| " + " | ".join(preferred) + " |")
    print("| " + " | ".join("---" for _ in preferred) + " |")
    for row in rows:
        values = [format_cell(row.get(column)) for column in preferred]
        print("| " + " | ".join(values) + " |")


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=("pilot", "q25", "q3_4b", "q3_30b", "main"), default="main")
    parser.add_argument("--branches", help="Comma-separated branch names: runtime,cc")
    parser.add_argument("--models", help="Comma-separated model keys")
    parser.add_argument("--precisions", help="Comma-separated precision keys")
    parser.add_argument("--pefts", help="Comma-separated PEFT methods")
    parser.add_argument(
        "--modes",
        help=(
            "Comma-separated modes: sync,async,async_db,async_fullft "
            "(async_fullft runs only when explicitly selected)"
        ),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--seeds",
        help=(
            "Comma-separated --seed values, one repeat per seed (overrides --repeats); "
            "stamped into the launcher env as SEED and recorded in run.json"
        ),
    )
    parser.add_argument(
        "--fullft-lr",
        help=(
            "Learning rate for the async_fullft arm only (launcher env LR); the adapter arms keep "
            "their recipe LR. A3 seeds it from the e4 full-FT window rather than reusing the OFT LR"
        ),
    )
    parser.add_argument(
        "--gpu-slice",
        help=(
            "Comma-separated GPU ids to run EVERY wave on (e.g. 0,1,2,3 on a 4-GPU node); "
            "default alternates odd repeats onto GPUs 4-7 for two-branch 8-GPU campaigns"
        ),
    )
    parser.add_argument("--num-rollout", type=int)
    parser.add_argument("--eval", dest="eval", action="store_true", default=None)
    parser.add_argument("--no-eval", dest="eval", action="store_false")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--skip-eval-before-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-profile", choices=("bench", "tiny", "recipe"), default="bench")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--campaign")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Print selected experiment waves")
    add_common_args(plan_parser)

    run_parser = subparsers.add_parser("run", help="Run selected experiment waves")
    add_common_args(run_parser)
    run_parser.add_argument("--dry-run", action="store_true")

    parse_parser = subparsers.add_parser("parse", help="Parse completed run logs")
    parse_parser.add_argument("path")
    parse_parser.add_argument("--format", choices=("markdown", "csv", "json"), default="markdown")
    parse_parser.add_argument("--output", help="Optional CSV output path")

    args = parser.parse_args(argv)
    if args.command == "plan":
        print_plan(build_waves(args))
        return 0
    if args.command == "run":
        return run_waves(args)
    if args.command == "parse":
        return parse_runs(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
