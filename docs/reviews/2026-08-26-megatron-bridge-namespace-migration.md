# Review brief: migrate Orbit to the restructured Megatron-Bridge namespace

## Goal

Make Orbit consume only the canonical private namespace introduced by the
Megatron-Bridge restructure:

```text
megatron.bridge.orbit.*
```

This is intended to be a namespace-and-pin migration, not a behavioral change.
After it lands, tracked Orbit code should no longer depend on Megatron-Bridge's
temporary backward-compatibility shims. Those shims are deliberately not
removed by this commit.

The commit is based on `orbit-main@23b6215f7c105407c0706e93b0267a553d7e7ab1`
and targets Megatron-Bridge
`69a8e369e23f522c354f1cd33c2cfd21ef5768d6` from
`Sphere-AI-Lab/Megatron-Bridge:orbit-main-restructured`.

## Intended changes

1. Replace all 25 executable imports of relocated Bridge modules with their
   canonical paths:

   | Previous namespace | Canonical namespace |
   |:--|:--|
   | `megatron.bridge.models.conversion.low_precision.*` | `megatron.bridge.orbit.low_precision.*` |
   | Orbit-specific modules under `megatron.bridge.models.{deepseek,llama,qwen}.*` | `megatron.bridge.orbit.model_bridges.*` |
   | OFT modules under `megatron.bridge.peft.*` | `megatron.bridge.orbit.oft.*` |
   | Quantization helpers under `megatron.bridge.peft.*_utils` | `megatron.bridge.orbit.quant.*_utils` |

2. Update the two physical source-path assumptions:

   - the DeepSeek-V4 conversion shell guard;
   - the OFT source-parity test, which now takes the Bridge checkout from
     `MEGATRON_BRIDGE_ROOT` instead of a hard-coded cluster path.

3. Pin the restructured Bridge commit consistently in `pyproject.toml`,
   `uv.lock`, CUDA 12.8 and CUDA 13.0 generated pin manifests, installation
   documentation, and pin-contract tests.

4. Add `tests/fast/test_megatron_bridge_orbit_namespace.py`. It imports every
   canonical module consumed by Orbit and asserts that all consumed symbols are
   exported.

## Explicit non-goals

- Do not change OFT, FP8, INT4, or NVFP4 algorithms or checkpoint semantics.
- Do not move implementation code inside Orbit.
- Do not remove the compatibility package or module-identity shims from
  Megatron-Bridge.
- Do not change sglang; its `orbit-main` branch has no Bridge Python imports.
- Do not update unrelated dependencies or normalize unrelated `uv.lock`
  entries.
- Do not claim compatibility for historical pickles or configurations that
  may embed an old Python module name; that artifact audit remains separate.

## Reviewer checklist

Please review the commit against these invariants:

- Every changed import maps to the same relocated implementation and imports
  the same symbol names as before.
- Imports of upstream Bridge modules that were not relocated remain unchanged.
- No executable legacy path or pre-restructure Bridge pin remains in Orbit's
  code, plugins, scripts, tests, tools, installation guide, `pyproject.toml`,
  or `uv.lock`.
- `uv.lock` changes only the Megatron-Bridge version/source and Orbit's matching
  dependency metadata; unrelated lock entries are unchanged.
- Both generated pin manifests agree with `pyproject.toml` and contain the
  restructured Bridge commit.
- The DeepSeek-V4 shell guard checks the new physical path.
- The OFT parity test has no hard-coded cluster path, reports an explicit skip
  when its configured Bridge source is unavailable, and reads
  `orbit/oft/oft_layers.py` when available.
- The new namespace contract checks both importability and symbol presence.
- No backward-compatibility shim removal is accidentally included.

## Verification evidence

- Test-first negative check: Slurm job `2720` against the old Bridge checkout
  failed with `ModuleNotFoundError: No module named 'megatron.bridge.orbit'`.
- Focused CPU/static gate: 35 tests passed; Python compilation, `bash -n`, and
  `git diff --check` passed.
- GPU import/symbol contract: Slurm job `2735` passed all 15 cases against
  Bridge `69a8e369` and Megatron-LM `00eb75b0`.
- A scoped live-reference audit found zero executable legacy paths and zero
  live references to the old Bridge pin.

The repository-wide test suite was not run. A broader run of
`test_lora_regret_arms_coverage.py` reached 114 passes and one unrelated
failure because the shared environment's installed sglang declares FlashInfer
`0.6.14`, while current `orbit-main` overrides `0.6.15.post1`. The
Bridge-specific contract in that file passes.

## Expected review outcome

Accept the commit if it is a behavior-preserving consumer migration to the
canonical Bridge namespace and the pin artifacts are internally consistent.
Request changes for any missing live reference, incorrect symbol mapping,
unrelated dependency change, or weakened compatibility test.
