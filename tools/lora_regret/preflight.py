"""Audit everything a reservation can discover expensively -- before it starts.

    python -m tools.lora_regret.preflight --stage e1-lora

Each check prints what it found. Exits non-zero if any required check failed,
so this can gate a job script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

from tools.lora_regret.arms import DATA_DIR, MATRICES, MATRICES_REQUIRING_OFT_CENTRE

HF_CKPT = "/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B"
# Note: still under the *old* repo's path. Verified present (15 GB) on
# 2026-07-30; it is a cross-repo dependency rather than a break, which is
# exactly why it is checked here rather than assumed.
MEGATRON_LOAD = "/lustre/fast/fast/zqiu/orbit-infra/orbit/checkpoints/Llama-3.1-8B_torch_dist"

# Measured counts from the 2026-07-30 materialization, not expectations. MATH is
# 7,498 rather than 7,500 because two number_theory rows carry an empty \boxed{}
# and an empty label can never be earned honestly.
EXPECTED_ROWS = {
    "tulu3_train.jsonl": 938_343,
    "tulu3_test.jsonl": 1_000,
    "openthoughts3_train.jsonl": 10_000,
    "openthoughts3_test.jsonl": 100,
    "math_train.jsonl": 7_498,
    "math_test.jsonl": 5_000,
    "gsm8k_train.jsonl": 7_473,
    "gsm8k_test.jsonl": 1_319,
    "math_gsm8k_train.jsonl": 14_971,
}

# Counts with no `--oft-lr-centre`, which is what `check_matrices` builds and
# what an operator sees before the scout has run. Supplying a centre does not
# change any count -- every OFT cell keeps the width of the LoRA cell it mirrors,
# only its learning rates move from the scout span onto a centred grid.
EXPECTED_ARMS = {
    "e1": 45, "e2": 48, "e3": 35, "e4": 98, "e5scout": 5, "e5": 50, "sft82": 82,
    "e1ot": 45, "e1short": 21, "e4lr0": 6, "e4place": 35, "e5rl": 42,
    "e4oftb128low": 5,
    "e4oftb128refine": 6,
    "e4oftverify": 3,
    "e4oftenv2": 14,
}

# What each stage needs before it is worth starting. P3 is 2 rather than 1
# because DP=1 makes the reduction it tests a no-op; FullFT is 4 for the
# 32 GB + 96 GB/N optimizer-state arithmetic the launcher enforces.
STAGE_GPU_REQUIREMENTS = {
    "smoke": 1,
    "e1-lora": 1,
    "e3": 1,
    "e5": 1,
    "p3": 2,
    "e1-full": 4,
    "e2-full": 4,
    "e4": 8,
    "e4oftb128low": 8,
    "e4oftb128refine": 8,
    "e4oftverify": 8,
    "e4oftenv2": 8,
    # e1ot and e1short are LoRA-and-FullFT matrices, but their FullFT arms are
    # selected with --only and run on the e1-full allocation; the stage floor
    # here is the LoRA one, which is what an operator checks before the bulk of
    # the arms.
    "e1ot": 1,
    "e1short": 1,
    "e4place": 8,
}


class Check(NamedTuple):
    name: str
    ok: bool
    detail: str


def check_env() -> list[Check]:
    """Imports, and that each module has a real file behind it.

    `__file__ is not None` is the load-bearing half. The failure this guards
    against -- a venv of symlinks into a cleared uv cache -- *imports
    successfully*: Python treats a directory with no loadable __init__.py as a
    namespace package, so it presents as a missing attribute, not an ImportError.
    """
    checks = []
    for name in ("torch", "transformers", "megatron.core", "miles.orbit"):
        try:
            module = __import__(name, fromlist=["__file__"])
            path = getattr(module, "__file__", None)
            version = getattr(module, "__version__", "?")
            if path is None:
                checks.append(Check(f"import:{name}", False,
                                    "imported as a namespace package with no __file__ -- "
                                    "the venv's symlinks are dangling; rebuild per INSTALL.md"))
            else:
                checks.append(Check(f"import:{name}", True, f"{version} at {path}"))
        except Exception as exc:  # noqa: BLE001 -- report any import failure verbatim
            checks.append(Check(f"import:{name}", False, f"{type(exc).__name__}: {exc}"))
    return checks


def check_gpus(stage: str | None) -> list[Check]:
    try:
        import torch

        count = torch.cuda.device_count()
        names = {torch.cuda.get_device_name(i) for i in range(count)}
    except Exception as exc:  # noqa: BLE001
        return [Check("gpus", False, f"could not query CUDA: {exc}")]
    detail = f"{count} device(s): {', '.join(sorted(names)) or 'none'}"
    if stage is None:
        return [Check("gpus", True, detail)]
    needed = STAGE_GPU_REQUIREMENTS[stage]
    return [Check("gpus", count >= needed, f"{detail}; stage {stage!r} needs >= {needed}")]


def check_checkpoints(hf_ckpt: str | Path, megatron_load: str | Path) -> list[Check]:
    hf_path, mg_path = Path(hf_ckpt), Path(megatron_load)
    checks = [
        Check("hf_checkpoint", hf_path.is_dir(),
              f"{hf_path}" if hf_path.is_dir() else f"missing: {hf_path}")
    ]
    marker = mg_path / "latest_checkpointed_iteration.txt"
    if not mg_path.is_dir():
        checks.append(Check("megatron_load", False, f"missing: {mg_path}"))
    elif not marker.exists():
        checks.append(Check("megatron_load", False,
                            f"{mg_path} exists but has no latest_checkpointed_iteration.txt"))
    else:
        checks.append(Check("megatron_load", True,
                            f"{mg_path} at iteration {marker.read_text().strip()}"))
    return checks


def check_data(data_dir: str | Path) -> list[Check]:
    """Row counts, not just existence.

    A truncated split silently changes the denominator of every E1 number, and
    it is indistinguishable from a good one by `ls`.
    """
    root = Path(data_dir)
    checks = []
    for name, expected in EXPECTED_ROWS.items():
        path = root / name
        if not path.exists():
            checks.append(Check(name, False, f"missing: {path}"))
            continue
        with path.open("r", encoding="utf-8") as handle:
            rows = sum(1 for _ in handle)
        checks.append(
            Check(name, rows == expected,
                  f"{rows} rows" if rows == expected else f"{rows} rows, expected {expected}")
        )
    return checks


def check_matrices(hidden_size: int, ffn_size: int, qkv_output_size: int) -> list[Check]:
    """Every matrix builds, at the count the runbook documents.

    A matrix that raises does so here, in a second, rather than after Ray has
    started on a reserved node. `e1long` is excluded: it needs a real E1-1
    ledger, so its guard is tested by the sweep's own CLI instead.
    """
    checks = []
    for name, expected in EXPECTED_ARMS.items():
        try:
            centre = 1e-4 if name in MATRICES_REQUIRING_OFT_CENTRE else None
            built = MATRICES[name](hidden_size, ffn_size, qkv_output_size, 0, centre, None)
            checks.append(
                Check(f"matrix:{name}", len(built) == expected,
                      f"{len(built)} arms" if len(built) == expected
                      else f"{len(built)} arms, expected {expected}")
            )
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(f"matrix:{name}", False, f"{type(exc).__name__}: {exc}"))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGE_GPU_REQUIREMENTS), default=None)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--hf-checkpoint", default=HF_CKPT)
    parser.add_argument("--megatron-load", default=MEGATRON_LOAD)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-size", type=int, default=14336)
    # Fused q+k+v width. Not derivable from hidden_size under GQA, and it decides
    # every matched-parameter block size, so a matrix audited at the wrong value
    # passes while building arms for a different model.
    parser.add_argument("--qkv-output-size", type=int, default=6144)
    parser.add_argument("--skip-gpu", action="store_true", help="for CPU-only preflight")
    args = parser.parse_args()

    checks: list[Check] = []
    checks += check_env()
    if not args.skip_gpu:
        checks += check_gpus(args.stage)
    checks += check_checkpoints(args.hf_checkpoint, args.megatron_load)
    checks += check_data(args.data_dir)
    checks += check_matrices(args.hidden_size, args.ffn_size, args.qkv_output_size)

    width = max(len(c.name) for c in checks)
    for check in checks:
        print(f"[{'ok' if check.ok else 'FAIL':>4}] {check.name:{width}}  {check.detail}")

    failed = [c for c in checks if not c.ok]
    if failed:
        print(f"\n{len(failed)} check(s) failed -- do not start the reservation:", file=sys.stderr)
        for check in failed:
            print(f"  {check.name}: {check.detail}", file=sys.stderr)
        return 1
    print(f"\nall {len(checks)} checks passed"
          + (f" for stage {args.stage!r}" if args.stage else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
