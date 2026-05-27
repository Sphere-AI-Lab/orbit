from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path


PRECISION_PROFILES = {"fp8", "nvfp4", "int4"}


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from checkpoint_parity_core import (  # noqa: E402
    compare_expected_fp8_metadata,
    compare_expected_int4_metadata,
    compare_expected_nvfp4_metadata,
    write_report_json,
)


def _parse_precision_profile(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in PRECISION_PROFILES:
        supported = ", ".join(sorted(PRECISION_PROFILES))
        raise argparse.ArgumentTypeError(
            f"Unsupported checkpoint precision profile {value!r}; expected one of: {supported}."
        )
    return normalized


def parse_args(default_precision_profile: str = "fp8") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check checkpoint parity between a Hugging Face model and Megatron dist checkpoint")
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--megatron-dist", required=True)
    parser.add_argument(
        "--precision-profile",
        type=_parse_precision_profile,
        default=default_precision_profile,
        help="Checkpoint precision profile. Supported values: fp8, nvfp4, int4.",
    )
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()
    args.precision_profile = _parse_precision_profile(args.precision_profile)
    return args


def compare_checkpoint_metadata(*, hf_model_path: str, megatron_dist_path: str, precision_profile: str):
    if precision_profile == "fp8":
        return compare_expected_fp8_metadata(
            hf_model_path=hf_model_path,
            megatron_dist_path=megatron_dist_path,
        )
    if precision_profile == "nvfp4":
        return compare_expected_nvfp4_metadata(
            hf_model_path=hf_model_path,
            megatron_dist_path=megatron_dist_path,
        )
    if precision_profile == "int4":
        return compare_expected_int4_metadata(
            hf_model_path=hf_model_path,
            megatron_dist_path=megatron_dist_path,
        )
    raise AssertionError(f"Unhandled checkpoint precision profile: {precision_profile}")


def main(default_precision_profile: str = "fp8") -> None:
    args = parse_args(default_precision_profile=default_precision_profile)
    report = compare_checkpoint_metadata(
        hf_model_path=args.hf_model,
        megatron_dist_path=args.megatron_dist,
        precision_profile=args.precision_profile,
    )
    if args.report_json:
        write_report_json(args.report_json, asdict(report))
    print(f"checkpoint parity: {'ok' if report.ok else 'not ok'}")
    print(f"precision profile: {args.precision_profile}")
    print(f"missing checkpoint keys: {len(report.missing_checkpoint_keys)}")
    print(f"missing scale keys: {len(report.missing_scale_keys)}")
    print(f"missing extra_state keys: {len(report.missing_extra_state_keys)}")
    print(f"mismatched shapes: {len(report.mismatched_shapes)}")
    if not report.ok:
        raise SystemExit(1)
