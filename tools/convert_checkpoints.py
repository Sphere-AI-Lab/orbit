"""Orbit wrapper around the patch-owned Megatron-Bridge checkpoint conversion."""

from __future__ import annotations

def main(argv: list[str] | None = None) -> int:
    from miles_plugins.megatron_bridge.patches.conversion.convert_checkpoints import main as patch_main

    return patch_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
