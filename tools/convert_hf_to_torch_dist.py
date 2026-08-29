# ORBIT-SEAM: whole-file rewrite - base's do-everything conversion script (Megatron arg parsing,
# distributed process-group init, model build, HF weight load via bridge, torch_dist checkpoint
# save) moved to miles_plugins.megatron_bridge.patches.conversion.convert_checkpoints.
# import_hf_to_megatron; this file keeps only a legacy-flag-compatible CLI wrapper around it.
"""Legacy CLI wrapper for HF -> Megatron checkpoint conversion.

This keeps orbit's historical `--hf-checkpoint` / `--save` interface while
delegating the actual conversion flow to the orbit-owned patch import
path.
"""

from __future__ import annotations

# ORBIT-SEAM: stdlib-only imports replace base's heavy megatron/mbridge/miles imports, which now
# live in the extracted import_hf_to_megatron implementation and are imported lazily inside main()
import argparse
import sys
from pathlib import Path


# ORBIT-SEAM: ensures the repo root precedes any other sys.path entry, so `import miles_plugins...`
# below resolves to this checkout rather than an installed/site-packages copy
_repo_root = str(Path(__file__).resolve().parents[1])
sys.path = [p for p in sys.path if p != _repo_root]
sys.path.insert(0, _repo_root)


# ORBIT-SEAM: translates orbit's historical --hf-checkpoint/--save CLI flags to
# import_hf_to_megatron()'s kwarg names, keeping backward CLI compatibility
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
        from miles_plugins.megatron_bridge.patches.conversion.convert_checkpoints import (
            import_hf_to_megatron as import_fn,
        )

    if ignored:
        print(
            "Ignoring legacy Megatron CLI flags during HF->Megatron conversion: "
            + " ".join(ignored)
        )

    # ORBIT-SEAM: delegates the actual conversion to the home implementation; legacy Megatron CLI
    # flags this parser doesn't recognize are only warned about above, not rejected
    import_fn(
        hf_model=args.hf_model,
        megatron_path=args.megatron_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    return 0


# ORBIT-SEAM: raises SystemExit(main()) for a real process exit code, instead of base's bare
# main() call which never propagated a non-zero status
if __name__ == "__main__":
    raise SystemExit(main())
