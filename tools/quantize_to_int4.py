"""Miles wrapper around the patch-owned Megatron-Bridge INT4 HF quantization."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from miles_plugins.megatron_bridge.patches.conversion.quantize_to_int4 import main as patch_main

    return patch_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
