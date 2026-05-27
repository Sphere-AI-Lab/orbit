"""Orbit wrapper around the patch-owned Megatron-Bridge INT4 direct conversion."""

from __future__ import annotations

def main(argv: list[str] | None = None) -> int:
    from orbit_plugins.megatron_bridge.patches.conversion.convert_int4_checkpoint_direct import (
        main as patch_main,
    )

    return patch_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
