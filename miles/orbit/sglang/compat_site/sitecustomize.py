import os
import warnings


if os.environ.get("ORBIT_SGLANG_FORCE_NATIVE_OPS") == "1":
    try:
        from miles.orbit.sglang.native_ops import patch_sglang_native_ops

        patch_sglang_native_ops()
    except Exception as exc:
        warnings.warn(f"Failed to apply ORBIT_SGLANG_FORCE_NATIVE_OPS: {exc!r}", RuntimeWarning)
