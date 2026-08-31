# Tier-2 reuse slice: the <=2-unit files on orbit-main-isolated

Goal: return vendored files to pristine by moving orbit's code into orbit/,
using the mechanism each piece actually needs. Base = ef7481ae3.

## Mechanism decision rules (measured, not by feel)

* **LIFT** — the function does not exist upstream at all (orbit added it to a
  vendored file). Move it to orbit/ and import it back. Always correct.
* **DELEGATING PATCH** — upstream owns the function and orbit's ownership is
  LOW (<50% of lines changed). Register a patch in orbit/ that calls the saved
  original and only adds orbit's behaviour. Preferred: orbit carries ~10 lines
  instead of a 40-190 line body, and upstream's fixes keep running.
* **FULL-REPLACE PATCH** — ownership >=50%. Orbit carries the whole body. Costs
  real merge visibility, so only where delegation genuinely cannot express it.
* **MIXIN OVERRIDE** — a changed CLASS method. Move it to an orbit mixin and
  DELETE it from the vendored class body. Do NOT "revert it to upstream's body":
  Python resolves a class's OWN __dict__ before any base, so a retained body
  SHADOWS the mixin and silently disables the override.
* **LEAVE IT** — flag and skip. See carve-outs.

## Carve-outs (do not touch in this slice)

* `miles_plugins/models/glm4.py` — GLM-4 is knowingly broken on the current
  Megatron pin and deferred by decision. Leave it entirely.
* `miles/backends/megatron_utils/bridge_lora_helpers.py` (deletes 119 upstream
  lines) and `miles/backends/megatron_utils/model_provider.py` (deletes 26) —
  a patch cannot express "upstream code removed"; these need their own design.
* Anything whose diff DELETES upstream lines you cannot account for: stop and
  report rather than guessing.

## The patch layer, and the two ways it silently fails

Patches live in orbit/, are hash-pinned (tools/check_patch_pins.py verifies the
upstream body statically), and install via a sys.meta_path hook armed by
`import orbit`. Both failure modes below actually happened this session:

1. **Never installed.** A process that reaches a patched module without ever
   importing orbit runs upstream's function. Entrypoints must `import orbit`;
   tests/fast/test_hf_export_patches.py enforces it.
2. **Installed but bypassed.** If the target's PACKAGE re-exported the function
   (`from .mod import fn` in __init__.py) before the hook was armed, callers
   dispatching through the package binding keep the original.
   orbit/patch/runtime.py::_repoint_reexports handles this -- do not remove it.

Declare patch targets as PLAIN STRING LITERALS (an f-string is not statically
readable and the pin gate will reject it).

## Gates

* All 5 guards green: import-integrity, args-dest, call-signatures,
  path-anchors, patch-pins.
* Purity manifest regenerated; pristine count must RISE.
* Fast suite: failure set identical to the 5 known lora_regret failures.
* Every moved/patched function needs a test proving orbit's behaviour still
  happens AND (for delegation) that upstream's path still runs upstream's body.
