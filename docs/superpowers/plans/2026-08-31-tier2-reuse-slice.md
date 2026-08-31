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

---

# Mid-tier (3-7 units) batches

Same rules and same carve-out principle. Deletions are the discriminator: a
patch replaces a function, so it cannot express "orbit removed upstream code".
Any file whose diff deletes more than a handful of upstream lines needs its own
design and is NOT in these batches.

## Deferred — high deletion count, need their own design

  123  update_weight/update_weight_from_tensor.py      67  utils/arguments.py
   37  megatron_utils/checkpoint.py                    27  inference_rollout_eval.py
   24  update_weight/hf_weight_iterator_bridge.py      21  rollout/sglang_rollout.py
   20  external_utils/command_utils.py                 18  sglang_utils/arguments.py
   18  utils/distributed_utils.py                      12  generate_endpoint_utils.py
   12  training_utils/log_utils.py

## Batch A -- LIFT-heavy (orbit-added functions; move + import back)

  miles/ray/utils.py                          del=0   2 lifts + 8 toplevel
  miles/backends/megatron_utils/replay_utils.py del=1 1 lift + 1 mixin + 28 toplevel
  miles/rollout/rm_hub/__init__.py            del=1   1 lift + 1 tiny patch
  miles/utils/replay_base.py                  del=6   1 lift + 1 mixin
  miles/rollout/rm_hub/deepscaler.py          del=8   2 lifts + 1 patch(100%)
  miles/backends/megatron_utils/arguments.py  del=4   3 lifts + 1 patch(18%)

## Batch B -- MIXIN-heavy (changed/added CLASS methods)

  miles/utils/chat_template_utils/tito_tokenizer.py del=0  2 methods, both 100% orbit
  miles/ray/train_actor.py                          del=2  3 methods
  miles/router/router.py                            del=2  6 methods
  .../update_weight/hf_weight_iterator_base.py      del=4  2 methods
  miles/utils/test_utils/mock_sglang_server.py      del=3  3 methods
  miles_plugins/mbridge/qwen3_5.py                  del=7  4 methods + 31 toplevel

## Batch C -- PATCH-heavy (upstream functions, mostly LOW ownership)

  miles/utils/processing_utils.py                   del=3  2 patches @16%
  miles/rollout/generate_utils/openai_endpoint_utils.py del=3 1 lift + 1 patch @4%
  miles/backends/megatron_utils/initialize.py       del=4  2 patches @9%/25%
  .../update_weight/hf_weight_iterator_direct.py    del=4  2 patches @3%/10%
  miles/rollout/inference_rollout/inference_rollout_common.py del=5 2 patches @3%/34%
  miles/backends/megatron_utils/megatron_to_hf/__init__.py del=7 2 patches @77%/3%
  miles/utils/data.py                               del=5  2 lifts + 2 patches @12%/1%

A mixin method at 100% ownership is orbit-ADDED (no upstream counterpart): it
still moves to the mixin, but there is nothing to delete from the class body.
