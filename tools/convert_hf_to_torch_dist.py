"""Legacy CLI wrapper for HF -> Megatron checkpoint conversion.

This keeps orbit's historical `--hf-checkpoint` / `--save` interface while
delegating the actual conversion flow to the orbit-owned patch import
path.
"""

from __future__ import annotations

import argparse


def parse_legacy_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace checkpoint to Megatron torch_dist format.",
    )
    parser.add_argument("--hf-checkpoint", "--hf-model", dest="hf_model", required=True)
    parser.add_argument("--save", "--megatron-path", dest="megatron_path", required=True)
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None, import_fn=None) -> int:
    args, ignored = parse_legacy_args(argv)

    if import_fn is None:
        from orbit_plugins.megatron_bridge.patches.conversion.convert_checkpoints import (
            import_hf_to_megatron as import_fn,
        )

    if ignored:
        print(
            "Ignoring legacy Megatron CLI flags during HF->Megatron conversion: "
            + " ".join(ignored)
        )

    import_fn(
        hf_model=args.hf_model,
        megatron_path=args.megatron_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
