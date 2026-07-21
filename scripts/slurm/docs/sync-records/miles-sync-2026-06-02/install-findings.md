# Install test findings — v0.5.12 / torch 2.11.0 bundle on slinky (Ubuntu 22.04, cu129)

> **STATUS — INSTALL fixed, but the sglang ENGINE does NOT run on this host (2026-06-03).**
> The env *builds* and `verify_env.py` reports **35/35 pass** (Issues 1–6 fixed in
> `install_env.sh` / `verify_env.py` / `extract_pins.py` / `pins.env`). BUT a real training run
> (geo3k multi-turn) showed the 35/35 was misleading — `verify_env`'s shallow `import sglang`
> never loaded the inference engine's native kernels. **The sglang engine hits a hard
> CUDA-13-kernel wall**: the synced `v0.5.12-23` pins torch 2.11, whose `sgl-kernel`/`deep_gemm`
> are cu13-only, and this host is CUDA 12.8 (can't run cu13). **`v0.5.12-23 / torch 2.11 is NOT
> bare-metal-viable on cu129`** — see the **`2026-06-03 — runtime validation`** section at the
> very end (Issues 7–9 + the strategic conclusion). Issues 1–6 + the `2026-06-02 follow-up` are
> the install-time trail; the runtime section is the current bottom line.

against branch `sync-upstream-20260602` (submodule `c74db48da`, pins `cu129-x86_64-v0.5.12`).
Host: 8× H200 (SM9.0), GLIBC 2.35, CUDA driver 12.8, nvcc 12.8.

**Result (FIRST run, since fixed): install_env.sh exited 2.** Everything up to the sglang_router
step installed (torch 2.11.0, sglang v0.5.12 editable, mbridge, modelopt, Megatron-Bridge,
torch_memory_saver, TE 2.10.0 compiled, FA2, FA3, apex) — but the env was broken (Issue 1) and
the router step hard-failed (Issue 2). The **sync mechanics are correct**; these were
install_env.sh / bundle-on-this-host issues, all surfaced by the torch bump and now fixed.

---

## Issue 1 [RESOLVED] — sglang v0.5.12 pyproject hard-requires CUDA 13; cu129 bare-metal needs a `cuda-python<13` override

**First symptom (run 1):** final env had `torch==2.11.0+cu130` (CUDA 13) while torchvision/
torchao/torchaudio were `+cu129` → `import torchvision` RuntimeError (CUDA major 13≠12) →
sglang can't import.

**Misdiagnosis (run 2):** I added a uv constraint pinning `torch==2.11.0+cu129` to the sglang
step. uv then reported the resolution **unsatisfiable**:
```
sglang==0.5.12...dev24 depends on cuda-python>=13.0
torch==2.11.0+cu129 depends on cuda-bindings>=12.9.4,<13
→ sglang[all] is unsatisfiable
```

**Actual root cause:** `thirdparty/sglang/python/pyproject.toml:25` declares
**`"cuda-python>=13.0"`** — a *direct* dependency. cuda-python>=13 → cuda-bindings 13.x
(CUDA 13). torch+cu129 → cuda-bindings 12.9.x. **Mutually exclusive.** So:
- The run-1 torch→cu130 swap was uv *correctly* satisfying sglang's CUDA-13 requirement
  (PyPI's plain torch 2.11.0 = cu130). The torchvision mismatch was the only visible symptom.
- sglang v0.5.12 genuinely wants CUDA 13 in its pip metadata; the `cu129-x86_64-v0.5.12`
  bundle (cu129 prebuilt FA/apex/TE/torch, CUDA 12.9) cannot satisfy it via full-tree pip
  resolution.

**Why the docker `v0.5.12-cu129` build works but bare-metal doesn't:** docker starts FROM
`lmsysorg/sglang:v0.5.12-cu129` (sglang + a cu12-compatible cuda-python already installed)
and never pip-reinstalls sglang, so `cuda-python>=13.0` is never enforced. Our bare-metal
install drops `--no-deps` for sglang (by design — no base image) → pip enforces the pin → wall.

**Host constraint that rules out "just go CUDA 13":** slinky driver is **570.195.03 (CUDA
12.8)**. CUDA 13.0 needs driver ≥ 580. So a cu130 / CUDA-13 bundle would not run here anyway —
the env MUST be cu129 on this host, which means sglang's `cuda-python>=13.0` must be overridden
down to a cu12 line.

**Fix options (need a decision — runtime-safety tradeoffs):**
- **(A) uv `--override cuda-python<13`** on the sglang editable install (force the cu12.9
  cuda-python that torch+cu129 wants). Surgical; mirrors what the cu129 docker base effectively
  ships. A plain `--constraint` does NOT work (can't satisfy `>=13` AND `<13`); must be an
  override. Risk: if sglang v0.5.12 calls cuda-python-13-only APIs at runtime it breaks — but
  the cu129 docker variant argues it's fine on cuda-python 12.x. **Untested.**
- **(B) sglang `--no-deps` + manual cu129-consistent runtime deps** (closest to docker;
  high-maintenance — must track sglang's full runtime tree).
- **(C) Local pyproject patch** relaxing `cuda-python>=13.0` → `>=12.9` for the cu129 line
  (another vendored mirror patch that travels forward each bump — like the mrope fix).

cu129 DOES ship torch 2.11.0 (`torch-2.11.0+cu129-…manylinux_2_28_x86_64.whl`), so torch
itself is fine; the blocker is sglang's cuda-python floor. The kept verify_env.py cuda-build
assertion still guards against an unnoticed cu130/cu129 split.

---

## Issue 2 — sglang_router release wheel needs GLIBC 2.39 > host 2.35

**Symptom:** `error: Failed to determine installation plan / A path dependency is
incompatible: sglang_router-0.3.2-cp38-abi3-manylinux_2_39_x86_64.whl ... you're on
manylinux_2_35`. install_env.sh L432–443 installs the release router wheel
**unconditionally** (no GLIBC guard).

**Root cause:** the miles-wheels release ships the router built for GLIBC 2.39 (Ubuntu 24.04
docker base). slinky compute nodes are Ubuntu 22.04 (GLIBC 2.35). The gateway has a
GLIBC<2.38 skip (L457–475); the router (added by #12's "install router from release, FATAL on
mismatch" change) has none.

**Not a v0.5.12 regression:** the old `cu129-x86_64` release ships the *same* manylinux_2_39
router wheel. The regression is purely #12's source change (PyPI → release). PyPI's
`sglang-router==0.3.2` ships `manylinux_2_17_x86_64` (GLIBC 2.17 floor) → installs fine on 2.35.

**Fix options (install_env.sh), mirror the gateway's GLIBC guard:**
- **(provenance-preserving) Build the patched router from source** on GLIBC < wheel-floor —
  `radixark/sgl-router-for-miles` (the Dockerfile's `SGL_ROUTER_USE_WHEELS=0` path); rustup is
  already installed for the sglang-grpc ext.
- **(simple) PyPI fallback** — on GLIBC < floor, `uv pip install sglang-router==$SGLANG_ROUTER_VERSION`
  from PyPI (manylinux_2_17). Caveat: PyPI is the upstream build, not the radixark-patched one;
  note it (miles version-gates on `sglang_router.__version__`, both 0.3.2).
- Parse the wheel's `manylinux_2_NN` tag and compare to host GLIBC (don't hardcode 2.38/2.39).

---

## Issue 3 — cudnn pin (9.16.0.29) wrong for torch 2.11.0

install_env.sh L552 force-installs `nvidia-cudnn-cu12==9.16.0.29` (a torch-2.9.x #168167
workaround; the Dockerfile pins the same, but its torch comes from the cu129 base image).
**torch 2.11.0+cu129 requires `nvidia-cudnn-cu12==9.17.1.4`** (exact, in torch metadata).
Downgrading to 9.16 breaks cudnn. Fix: honor torch's declared cudnn pin (derive it) instead
of hardcoding 9.16.0.29; keep CUDNN_CU12_VERSION as an override escape hatch. **Necessary but
not sufficient** — see Issue 4.

## Issue 4 — system cudnn 9.7.0 shadows the pip cudnn via LD_LIBRARY_PATH

Even at cudnn 9.17.1.4, torch fails: `cuDNN version incompatibility: compiled against (9,17,1)
but found runtime (9,7,0)`. The host's default `LD_LIBRARY_PATH` includes
`/usr/lib/x86_64-linux-gnu`, which has `libcudnn.so.9 -> 9.7.0`, shadowing the env's pip cudnn.
With `LD_LIBRARY_PATH` unset/cudnn-prioritized, torch cudnn = 9.17.1 and `transformer_engine`
imports. **torch 2.9.1 tolerated the system 9.7.0; torch 2.11.0 (needs 9.17.1) does not.**
Fix: env-activation / launcher (launch_miles.sbatch) must prepend the env's
`nvidia/cudnn/lib` (or strip the system cudnn) so the pip cudnn wins. Launch-time hygiene,
not a bundle change — but a real deployment blocker.

## Issue 5 [RESOLVED] — PyPI prebuilt TE 2.10.0 torch extension is ABI-incompatible with torch 2.11.0 (source-build fixes it)

**2026-06-02 update:** superseded/refined by the follow-up below. The blocker was the
prebuilt PyPI `transformer_engine_torch==2.10.0` wheel, not TE 2.10.0 itself; source-building
only the torch extension against the active torch 2.11.0+cu129 env works.

After fixing cudnn, `megatron.core` / `megatron.bridge` / `mbridge` still fail:
`transformer_engine/wheel_lib/transformer_engine_torch.cpython-312…so: undefined symbol:
_ZN3c104cuda29c10_cuda_check_implementation…` (a torch `c10::cuda` ABI symbol). install_env.sh
does `uv --no-build-isolation "transformer_engine[pytorch]==2.10.0"` expecting a ~10-min
SOURCE compile, but uv pulls the **prebuilt** `transformer-engine-torch==2.10.0` wheel from
PyPI, whose `.so` is bound to an older torch ABI. **TE 2.10.0 predates / doesn't match torch
2.11.0.** This is exactly the torch-ABI-bound class the pin-model guards for the miles-wheels
set — but TE comes from PyPI, not the release, so it's unguarded. Fix options: force a TE
source build against torch 2.11.0 (TE's actual no-binary path), or use a TE version/wheel
built for torch 2.11.0, or ship the torch-bound TE in the miles-wheels release.

## Issue 6 [RESOLVED] — `kernels>=0.15` breaks transformers `LayerRepository` (NOT sglang setuptools_scm, as first guessed)

**2026-06-02 update:** superseded. The observed `ValueError` was later traced to
`transformers.integrations.hub_kernels` constructing `kernels.LayerRepository(...)` with
`kernels>=0.15`, not to `setuptools_scm` or the sglang editable checkout.

`import sglang → ValueError: Either a revision or a version must be specified.`
setuptools_scm can't derive a version for the editable sglang at a detached-HEAD submodule
with no reachable tag in the shallow state. Cosmetic-ish but breaks `import sglang`; likely
needs `SETUPTOOLS_SCM_PRETEND_VERSION` or a tag reachable in the submodule checkout.

## CONCLUSION (superseded by follow-up below)

The **sync (git/pins/submodule) is correct**, but the `cu129-x86_64-v0.5.12` bundle does
**NOT install into a working env on this cu129 / torch-2.11.0 bare-metal host.** The torch
2.9.1 → 2.11.0 jump cascades into ≥6 distinct breakages (cuda-python≥13, router GLIBC, cudnn
pin, cudnn LD_LIBRARY_PATH shadow, TE-torch ABI, sglang scm) because install_env.sh's
assumptions were all tuned for torch 2.9.1. Issues 1–2 are fixed & validated; 3–4 have clear
fixes; **Issue 5 (TE ABI) is the hard blocker** and likely needs a TE source build or a
torch-2.11-built TE wheel. This is beyond incremental patching — it's a decision about
whether to (a) invest in a torch-2.11 install overhaul, (b) hold the sglang bump until the
bundle/wheels provide torch-2.11-consistent TE, or (c) treat the bundle as docker-only and
not support v0.5.12 bare-metal on cu129.

## Takeaways for the #12 update
- install_env.sh needs **torch-build pinning** through the sglang editable resolution (Issue 1)
  and a **router GLIBC guard + fallback** (Issue 2). Both are torch-2.11-bump-exposed.
- Add a post-install `torch.version.cuda` major-vs-cu-tag assertion to `verify_env.py`.
- The sync itself (merge, conflict resolutions, pins, submodule + re-applied mrope patch) is
  validated through the build up to these two install_env.sh gaps.

---

## 2026-06-02 follow-up — TE is fixable bare-metal; kernels is a separate resolver hazard

The earlier conclusion that TE was the hard blocker is now refined: **the prebuilt PyPI
`transformer_engine_torch==2.10.0` wheel is the blocker, not TE 2.10.0 itself.** A focused
spike in the existing `miles_v0512_test` env proved that TE 2.10.0 can work with
`torch==2.11.0+cu129` if the torch extension is source-built against the active torch.

### Docker TE path audit

`docker/Dockerfile` does **not** have a special fix for cu129 TE. On the cu129 path it still
runs:

```bash
pip -v install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
```

That is effectively the same shape as the previous bare-metal script and does not force a
source build. `--no-build-isolation` only controls the build environment; it does **not**
prevent pip/uv from choosing a prebuilt wheel. If PyPI offers
`transformer_engine_torch==2.10.0`, the resolver can install that ABI-bound `.so` instead of
compiling locally.

The more useful Docker pattern is the CUDA-13 branch, which decomposes TE:

```bash
pip install --no-deps transformer_engine==2.12.0
pip install transformer_engine_cu13==2.12.0
pip install /tmp/wheels/transformer_engine_torch-*.whl \
  || pip -v install --no-build-isolation transformer_engine_torch==2.12.0
```

The bare-metal design now mirrors that shape for cu12: install the pure/runtime TE packages,
then force `transformer_engine_torch` to compile against the already-installed torch.

### TE source-build spike result

In `miles_v0512_test`:

```bash
python -m pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-torch
python -m pip install --no-deps transformer_engine==2.10.0 transformer_engine_cu12==2.10.0

CUDA_HOME=/usr/local/cuda \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib \
NVTE_FRAMEWORK=pytorch \
MAX_JOBS=16 \
python -m pip install --no-deps --no-build-isolation --no-cache-dir \
  --no-binary=:all: transformer_engine_torch==2.10.0
```

This built a local wheel:

```text
transformer_engine_torch-2.10.0-cp312-cp312-linux_x86_64.whl
```

GPU-visible validation then passed:

```text
torch 2.11.0+cu129 12.9
cuda_available True count 8
cudnn 91701
cuda tensor 1.0
te pytorch ok
megatron core ok
```

So the install-script fix is:

```bash
$UV --no-deps "transformer_engine==$TE_VERSION" "transformer_engine_cu12==$TE_VERSION"

TE_BUILD_MAX_JOBS=${TE_BUILD_MAX_JOBS:-${MAX_JOBS:-16}}
MAX_JOBS="$TE_BUILD_MAX_JOBS" NVTE_FRAMEWORK=pytorch \
  $UV --no-deps --no-build-isolation --no-cache --no-binary :all: \
    --reinstall-package transformer_engine_torch \
    "transformer_engine_torch==$TE_VERSION"
```

Rationale:
- Avoid the PyPI prebuilt `transformer_engine_torch` wheel, which was compiled against an
  older torch ABI and imports with an undefined `c10::cuda` symbol.
- Keep `TE_VERSION` aligned with upstream Dockerfile (`2.10.0` on cu129).
- Rebuild only the torch-bound extension; do not drift the TE package version yet.

### `megatron.bridge` / `mbridge` follow-up

After TE was fixed, `megatron.bridge` and `mbridge` still failed, but on a different issue:

```text
ValueError: Either a revision or a version must be specified.
```

The stack is:

```text
megatron.bridge / mbridge
  -> transformers
  -> transformers.integrations.hub_kernels
  -> kernels.LayerRepository(...)
```

Root cause: `sglang` declares unpinned `Requires-Dist: kernels`, so the resolver installed
`kernels==0.15.1`. `transformers==5.6.0` still constructs several `LayerRepository` objects
without `version=` or `revision=`, while `kernels 0.15+` made one of those fields mandatory.

This is why the failure looked new: older working envs did not have the
`transformers==5.6.0 + kernels==0.15.x` combination. It is not a TE failure and not a
Megatron-Bridge code regression.

Tested workaround:

```bash
python -m pip install 'kernels<0.15'
```

After this, both imports passed:

```text
mbridge ok
megatron.bridge ok
```

Install-script design: use a compatibility range, **not** an exact pin:

```bash
KERNELS_SPEC=${KERNELS_SPEC:-"kernels>=0.12,<0.15"}
$UV "$KERNELS_SPEC"
```

Rationale:
- Exact `kernels==0.14.1` would encode an arbitrary patch version as truth.
- `kernels<0.15` encodes the actual API compatibility boundary.
- `kernels>=0.12` matches the range declared by `transformers` extras metadata
  (`kernels>=0.12.0,<0.13` for some extras), while still allowing the known-good pre-0.15
  line selected by the resolver.
- `USE_HUB_KERNELS=NO` does not solve this: `hub_kernels.py` constructs `_KERNEL_MAPPING`
  at import time before the disabled path can avoid the incompatible `LayerRepository`
  construction.
- Uninstalling `kernels` is worse because `sglang` directly depends on it and the resolver
  will bring it back.
- Patching `transformers` locally is more brittle than constraining the transitive dep.

### Container path note

Docker daemon was unavailable on this host (`/var/run/docker.sock` missing). An enroot import
of `lmsysorg/sglang:v0.5.12-cu129` was started as a possible container-runtime probe, but the
first `/tmp` import hit whiteout permission errors. Retrying with user-owned enroot temp/cache
progressed through download/extract but was stopped before producing a `.sqsh` because the
container direction was deprioritized. No runtime conclusion should be drawn from that aborted
probe.

Practical conclusion: for the cu129 bare-metal path, do **not** rely on Docker to have solved
TE. The maintainable design is:

1. Keep the cu129/torch ABI guards.
2. Override sglang's CUDA-13 `cuda-python>=13` floor for cu12 installs.
3. Stop treating Dockerfile `nvidia-cudnn-cu12` as bare-metal truth. Derive the expected cuDNN
   package from installed torch metadata at install time, while preserving `CUDNN_CU12_VERSION`
   as a manual override for future bad-torch-cuDNN escape hatches such as pytorch/pytorch#168167.
4. Reassert that effective cuDNN package after torch install, before the TE source build, and
   before final verification, because later resolver steps such as modelopt can downgrade it.
5. Prepend the env's cuDNN path at build/verify/runtime so torch 2.11 sees cuDNN 9.17.1, not
   host 9.7.0.
6. Source-build `transformer_engine_torch` against the active torch.
7. Constrain `kernels` to the pre-0.15 API line until `transformers==5.6.0` is replaced or fixed.

With steps 3-5 applied in `miles_v0512_test`, `verify_env.py --imports-only` reported:

```text
=== 19/19 pass, 0 fail ===
```

After adding the full cuDNN/kernels assertions to `verify_env.py`, the existing hand-patched
`miles_v0512_test` env also passed full verification under the env cuDNN prepend:

```text
=== 33/33 pass, 0 fail ===
```

That validates the target end-state, not a clean `install_env.sh` ordering. A fresh full run is
still required to prove the script's torch → modelopt → cuDNN reassert → TE build → requirements
ordering end to end.

### Fresh `install_env.sh` run follow-up

A clean env run with `MILES_ENV_NAME=miles_v0512_fresh_20260602` validated the new ordering
through torch, sglang editable resolution, kernels capping, modelopt, cuDNN reassert,
TE source build, miles-wheels flash-attn/apex, and the GLIBC router source fallback.

Key observations:
- Torch stayed on `2.11.0+cu129`; sglang's CUDA-13 `cuda-python` floor was overridden to
  the cu12 line (`cuda-python==12.9.0`).
- cuDNN was derived from torch metadata as `nvidia-cudnn-cu12==9.17.1.4`, then reasserted
  after torch, before TE, and before final verification.
- `transformer_engine_torch==2.10.0` source-built successfully against the active torch.
- `kernels` was first pulled to `0.15.1` by sglang, then explicitly capped to `0.14.1`.
- The router release wheel was rejected on this host (`manylinux_2_39` vs host GLIBC 2.35);
  the source fallback built and installed a local `sglang_router-0.3.2` wheel.

The fresh run then failed only at final imports:

```text
=== 30/33 pass, 3 fail ===
ModuleNotFoundError: No module named 'onnxscript'
```

Root cause: the TE source-build path uses `--no-deps` to avoid PyPI's prebuilt
torch-ABI-bound `transformer_engine_torch` wheel. That also skips the pure/runtime deps declared
by `transformer_engine_torch` metadata (`onnx`, `onnxscript`). Installing
`onnx>=1.21.0` and `onnxscript` after the TE source build fixed the remaining import failures.

After applying that local suffix in the fresh env, full verification passed:

```text
=== 35/35 pass, 0 fail ===
onnx==1.21.0
onnxscript==0.7.0
```

The final latest-script-from-empty-env proof was then run with
`MILES_ENV_NAME=miles_v0512_fresh2_20260602`. It completed end to end:

```text
=== 35/35 pass, 0 fail ===
[done] miles env ready: /data/shared/conda/miniconda3/envs/miles_v0512_fresh2_20260602
```

The only remaining output noise was warning-level/non-fatal: torchao checkpoint-object warning,
modelopt's experimental `transformers>=5.0` warning, apex SyntaxWarnings, and Megatron-Bridge
Qwen3-ASR docstring warnings about `cache_position`.

---

# 2026-06-03 — runtime validation (geo3k multi-turn): the sglang ENGINE hits a CUDA-13-kernel wall

`verify_env.py` 35/35 only proves the env *imports*. To check it actually *runs*, launched the
geo3k VLM multi-turn recipe smoke-sized on `miles_v0512_fresh3_20260603`:

```bash
MILES_ENV_NAME=miles_v0512_fresh3_20260603 JOB_NAME=geo3k-mt-smoke-fresh3 TIME=1:00:00 \
MILES_SCRIPT_NUM_ROLLOUT=2 MILES_SCRIPT_ROLLOUT_BATCH_SIZE=8 MILES_SCRIPT_N_SAMPLES_PER_PROMPT=2 \
bash scripts/slurm/submit.sh geo3k-vlm-multi-turn-colocate-1node
```

Two slurm jobs (13562, then 13563 after the deep_gemm unblock), both 1×8 H200 on slinky-3,
`env=miles_v0512_fresh3_20260603` confirmed. Both crashed before the first train step. Three
new issues, all the **same root class as Issue 1 (cuda-python≥13): sglang v0.5.12's native
kernels for torch 2.11 are CUDA-13 builds, and this host is CUDA 12.8 (driver 570.195.03; no
`libnvrtc.so.13`/`libcudart.so.13`; CUDA 13 needs driver ≥580).**

## Issue 7 — `deep_gemm` (`sgl-deep-gemm 0.1.0`) is a cu13 build (train.py import)

`train.py` → `miles.ray…rollout_manager` → `sglang.srt.debug_utils.dumper` → `…moe…deep_gemm`
→ `import deep_gemm` →
```
RuntimeError: Failed to load .../deep_gemm/_C.so: libcudart.so.13: cannot open shared object file
```
`sgl-deep-gemm 0.1.0`'s `_C.so` links cu13 (`ldd`: libcudart.so.13, libnvrtc.so.13, libcublas*.so.13).
PyPI ships only one `py3-none-manylinux2014` wheel per version — all cu13. **No cu12 wheel exists.**

**Handling (not a real fix, just unblock):** uninstall `sgl-deep-gemm`. sglang's
`_compute_enable_deep_gemm()` does `try: import deep_gemm; except ImportError: return False`, so
*absence* → `ModuleNotFoundError` (an ImportError) → caught → `ENABLE_JIT_DEEPGEMM=False`, graceful.
(The cu13 `.so` raised `RuntimeError`, which is NOT caught — hence the crash.) deep_gemm is the
FP8-MoE path; geo3k Qwen3-VL-2B is dense, so disabling it is fine here. For MoE models it matters.

## Issue 8 (THE WALL) — `sgl-kernel` for torch 2.11 is cu13-only; the cu12 build is torch-2.9-ABI

After removing deep_gemm, job 13563 reached `SGLangEngine.init()` (the inference server), which
died with `RayTaskError(ImportError)`. Root:
```
[sgl_kernel] CRITICAL: Could not load any common_ops library!
  found: .../sgl_kernel/sm90/common_ops.abi3.so   (CUDA version: 12.9, SM90)
  - ImportError: libnvrtc.so.13: cannot open shared object file
  - ModuleNotFoundError: No module named 'common_ops'
```
`sgl-kernel` is **core** (fused attention/quant kernels — not optional like deep_gemm; cannot be
disabled). Installed `sglang-kernel 0.4.2.post2` = cu13 (`ldd sm90/common_ops.abi3.so`:
libnvrtc.so.13, libcudart.so.13, libcublas*.so.13).

Tried pinning the cu12 build `sglang-kernel==0.4.1` (what the working `miles` env uses). It IS
cu12 (`ldd`: libnvrtc.so.12 ✓) — but then fails with the **torch c10 ABI symbol**:
```
undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```
i.e. `0.4.1` was built against torch **2.9.x**, not 2.11. So:

| sgl-kernel | CUDA | torch ABI | on this host |
|---|---|---|---|
| `0.4.2.post2` (PyPI latest, torch-2.11 era) | **cu13** ✗ | 2.11 ✓ | can't load (no libnvrtc.so.13) |
| `0.4.1` (cu12 build, `miles` env) | cu12 ✓ | **2.9.x** ✗ | c10 undefined symbol vs torch 2.11 |

**No prebuilt `sgl-kernel` is BOTH cu12 AND torch-2.11.** The public cu129 index
(`docs.sglang.ai/whl/cu129/sgl-kernel/`) is stale — tops out at `0.3.21`, no 0.4.x.

## Issue 9 — `verify_env.py` 35/35 is misleading (shallow import coverage)

`verify_env`'s `import sglang` imports the package `__init__`, which does NOT transitively load
`sgl_kernel` / the engine path (`sglang.srt.layers.quantization.fp8_kernel → from sgl_kernel
import …`, lazy under the server). So it passed despite the engine being unloadable. **Fix:** add
an engine-path import to `verify_env` (e.g. `from sgl_kernel import sgl_per_token_quant_fp8`, and
the `sglang.srt.configs.model_config`/`server_args` chain) so the kernel ABI is checked.

## Smoking gun — the WORKING `miles` env proves the bare-metal-viable point is torch 2.9.1

| env | torch | sglang | sgl-kernel | flashinfer | runs? |
|---|---|---|---|---|---|
| **`miles` (existing, working)** | **2.9.1** | 0.5.12.**post1** | 0.4.1 (cu12 / torch-2.9) | 0.6.7.post2 | ✅ |
| **fresh3 (our sync target)** | **2.11.0** | 0.5.12.post2.dev24 (**-23**) | 0.4.2.post2 cu13 / 0.4.1 ABI | 0.6.11.post1 | ❌ engine |

(The `miles` env's submodule checkout now *pins* torch==2.11.0 too — but its installed torch is
2.9.1, i.e. it was built BEFORE our sync bumped the submodule to v0.5.12-23. It's the last
bare-metal-built point and is the proof v0.5.12 runs here on torch 2.9.1.)

## Root cause + the strategic conclusion

`/sglang-sync` advanced `thirdparty/sglang` to **`3102015ca` = v0.5.12-23** (the current
sgl-project `sglang-miles` tip) + the re-applied mrope patch = `c74db48da`. v0.5.12-23's pyproject
pins **torch==2.11.0** (matching upstream radixark's Dockerfile bump: `SGLANG_IMAGE_TAG=v0.5.12-cu129`,
`WHEELS_TAG=cu129-x86_64-v0.5.12` = torch 2.11). **But torch 2.11 is AHEAD of the cu12 prebuilt
native-kernel ecosystem** — sgl-kernel/deep_gemm for torch 2.11 are published cu13-only.

Why upstream is fine and we're not: **upstream radixark is docker-first.** Their torch-2.11 setup
works because the `lmsysorg/sglang:v0.5.12-cu129` base image bakes in cu12-built, torch-2.11
sgl-kernel/deep_gemm that are NOT published as public wheels. Bare-metal cu129 re-resolves from
public indexes and only finds cu13 (torch-2.11) or cu12 (torch-2.9). **So bare-metal cu129
structurally LAGS upstream's torch bumps** until those cu12 kernels are published, or we
source-build them, or move to docker / a CUDA-13 driver. This is exactly what the ACTIVE vs
UPSTREAM_TARGET pin-model represents: `ACTIVE` (bare-metal install) sits behind `UPSTREAM_TARGET`
(upstream's docker torch-2.11) with a standing `[sglang-sync pending]`.

The torch 2.9.1→2.11.0 jump (flagged as "heavyweight" from the start) is the real culprit — it
also drove Issues 1 (cuda-python≥13), 3/4 (cuDNN), 5 (TE prebuilt), 7 (deep_gemm), 8 (sgl-kernel).

## Decision (PENDING user) — divergence question

"Would pinning sglang to torch 2.9.1 diverge from upstream?" → Yes for the *bundle* (torch +
sgl-kernel version), tracked as `[sglang-sync pending]`; NO for the miles application code (all
merged upstream commits kept). And it's **divergence the `miles` env already has** — the sync
*tried to erase* it (jump to torch 2.11) and that broke bare-metal. Options:
1. **Pin sglang to the torch-2.9.1 v0.5.12.post1 point** (what `miles` runs): bare-metal-viable;
   re-apply mrope onto that base; revert pins.env to torch 2.9.1 + the matching wheels tag. Holds
   ACTIVE behind upstream by design. Likely-compatible with merged miles code (same v0.5.12 minor).
2. **Treat v0.5.12-23 / torch 2.11 as docker-only** for bare-metal; hold the submodule bump.
3. **Source-build cu12 + torch-2.11 sgl-kernel (+ deep_gemm)**: heavy, uncertain, may cascade (flashinfer).
4. **CUDA-13 host** (driver ≥580): run upstream's cu13 kernels as-is — infra change.

## State at pause (2026-06-03, pre-compaction checkpoint)

**Git (miles-imp).** Branch `sync-upstream-20260602`:
- `34ddc522d` — #12 base (pin-model + skills; this is PR #12's branch `install-tooling-sglang-pin-model`)
- `fa8fa8d25` — Merge upstream/main (100 commits, 5 conflicts resolved)
- `830db4a6b` — fold-in: sglang bundle → v0.5.12 (pins.env + gitlink)
- **Uncommitted (5 files):** `scripts/slurm/setup/{extract_pins.py,pins.env,install_env.sh,verify_env.py}`
  + `.claude/skills/sglang-sync/SKILL.md`. Untracked: `examples/vagen/docs/plan-notes/` (pre-existing).
- Nothing pushed/PR'd in miles-imp beyond the local branch; PR #12 is the open draft (stale vs these).

**verify_env.py** — user-improved (closes Issue 9): now imports the engine path in `check_runtime`
(`from sgl_kernel import sgl_per_token_quant_fp8`, `sglang…fp8_kernel`) + cuda-build/runtime-cuDNN/
kernels/onnx checks. Uncommitted.

**sglang submodule.** `thirdparty/sglang` @ `c74db48da` (v0.5.12-23 + mrope), now on branch
`sync-v0.5.12-20260603` which is **pushed to `impossible-inc/sglang`**; `sglang-miles` mirror
untouched at `4d795356c`. The sglang-sync SKILL was rewritten to the branch + merge-PR model
(no force-overwrite) — see its diff.

**Envs.** Only `miles_v0512_fresh3_20260603` remains (HAND-PATCHED/BROKEN: sgl-kernel forced to
0.4.1 [torch-ABI-broken vs torch 2.11], sgl-deep-gemm removed — do NOT trust; rebuild after the
target is decided). The existing **`miles` env is the working reference** (sglang 0.5.12.post1 +
torch 2.9.1 + cu12 sgl-kernel 0.4.1). Slurm jobs 13562/13563 scancelled.

**#12 vs sync-upstream split — Option A APPROVED, NOT yet executed.** Agreed: the install
hardening (the 4 uncommitted `scripts/slurm/setup/*` files) + the v0.5.12 bundle ride with the
**v0.5.12 sync (`sync-upstream`)**; **#12 keeps just the framework** + takes the generic
`sglang-sync/SKILL.md` mirror-model update. (Deferred — env-independent; doesn't block the test.)

**ACTIVE NEXT STEP — container route.** Decided to try running v0.5.12-23/torch-2.11 in the
official `lmsysorg/sglang:v0.5.12-cu129` base image (cu12+torch-2.11 kernels) via pyxis/enroot
(docker daemon down; enroot 4.0.1 + pyxis available). Full plan + compatibility rationale +
Phase 0/1/2: **`container-install-plan.md`** (this folder). Start with Phase 0 (decisive gate:
does our sglang engine import on the base kernels).

**Artifacts (this folder, gitignored):** `prs.md` (upstream PR report), `pr-body.md` (draft sync
PR body), `divergence.{patch,stat}`, `install*.log` (4 runs), `container-install-plan.md`. Design
rationale (ACTIVE/UPSTREAM model, sglang topology): `../upstream-sync-design.md`.

---

## 2026-06-04 — container route PAUSED + old-env (torch-2.9.1) test → the version-gate finding

**Container route paused (cost, not correctness).** Phase 0 (`phase0_probe.sh`) ran via pyxis on
`lmsysorg/sglang:v0.5.12-cu129`. The pyxis import is just SLOW: ~35-min registry pull (~40G into
`/data/home/xiuyul/.enroot/tmp`, ENROOT_TEMP_PATH override needed — default TMPDIR pointed at a
harness path absent on compute nodes) + a long squashfs-assembly tail. First two attempts died at
a 45-min `--time` limit during "Creating squashfs filesystem"; bumped to 2.5h, then user called it
(too long for the iteration loop). The approach itself never got disproven — the engine-import
gate never ran. Parked. (To resume cheaply: do ONE import job with `--container-save=…/base.sqsh`
on `/data` and a 3h limit, then mount the local `.sqsh` for fast iteration.)

**Pivot — test the working `miles` env (torch 2.9.1) against the current branch SOURCE.** Premise
(user's plan): editable installs make Python read the current checkout; old binaries stay. CONFIRMED
the `miles` env's editables point at THIS repo: `sglang.__file__` → `thirdparty/sglang/python/...`,
`miles.__file__` → `miles/...`, submodule @ `c74db48da`. `verify_env.py --imports-only` = 24/24.

**THE FINDING (decisive, job 13760).** `verify_env` 24/24 is STILL misleading — the geo3k smoke
crashed at sglang ENGINE LAUNCH, but NOT on the cu13/ABI wall. It's a **hard version assert** in
the synced sglang code:
```
Exception: sglang-kernel is installed with version 0.4.1, which is less than the
minimum required version 0.4.2.post2. ...
```
Source: `sglang/srt/entrypoints/engine.py:_set_envs_and_config` (~L1299) →
`assert_pkg_version("sglang-kernel", "0.4.2.post2", …)` and, for the flashinfer backend,
`assert_pkg_version("flashinfer_python", "0.6.11.post1", …)`. The `miles` env has `sglang-kernel
0.4.1` and `flashinfer-python 0.6.7.post2` — BOTH below the synced code's floors. The assert fires
at engine launch (in all 8 SGLangEngine subprocesses), NOT at import — which is precisely why
`verify_env`'s import checks pass while the engine dies. (`sglang` pkg metadata in the env reads
`0.5.12.post1`, but `sglang.__file__` is our editable `c74db48da` source — metadata lag, as the
plan warned.)

**Why this matters / refines the whole picture.** The torch-2.11 bump didn't only change the
native ABI — the synced sglang code also RAISED its required kernel/flashinfer floors to the
cu13/torch-2.11-era versions. So "new code + old binaries" is gated by a version check before we
even learn whether 0.4.1's *symbols* suffice.

**Bypass exists (built-in):** the whole block is guarded by
`if not get_bool_env_var("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK")` (accepts `true`/`1`). Setting it
skips BOTH asserts. This is in-spirit with the plan (keep 0.4.1, do NOT `pip install -U
sglang-kernel` → that pulls the cu13 kernel). `--export=ALL` chains the env var submit.sh → sbatch
→ ray-start `srun` → `ray start` → engine subprocess, so NO launcher edit is needed.

**Open question being tested now (job 13761, `geo3k-smoke-skipverchk`):** with the assert bypassed,
does sglang-kernel 0.4.1 ACTUALLY run the synced v0.5.12-24 engine, or does it crash deeper on a
kernel symbol/API added in 0.4.2.post2 (and/or a flashinfer API added after 0.6.7.post2)? The
import-level fp8 symbols are present (verify). Result pending — see below when known.

### RESULT (job 13763, `geo3k-smoke-skipver2`) — ✅ FULL SUCCESS, end-to-end

**The synced v0.5.12-24 code RUNS on the old torch-2.9.1 `miles` binaries.** The version assert was
a pure GUARD, not a real incompatibility. sglang-kernel 0.4.1 + flashinfer 0.6.7.post2 ran the
whole GRPO loop:
- Engine init fully: `Load weight end` (Qwen3VLForConditionalGeneration), KV cache alloc,
  **`Capture cuda graph end` (35.9s)** — this executes the real attention/decode kernels, the most
  likely mismatch point, and it PASSED — `The server is fired up and ready to roll!` (all 8).
- Rollout: real multi-turn decode w/ cuda graphs (`gen throughput ~4000 tok/s`), `Finish rollout`.
- `step 0` AND `step 1`: `train/grad_norm=0.503` (healthy non-zero; loss≈0 is normal for miles GRPO,
  per [[feedback_miles_grpo_loss_magnitude]]). Colocate weight sync (`update_weights_from_tensor`
  200, `release/resume_memory_occupation` 200) worked. **Ray job `succeeded`** (both iters).
- At-exit noise (all AFTER success, cosmetic): wandb `teardown_atexit` `BrokenPipeError`,
  `torch_memory_saver … CUresult error 1 … func=free`, `destroy_process_group() not called` NCCL warn.

**Two knobs were required (both env, NO file edits):**
1. `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true` — skip the sglang-kernel≥0.4.2.post2 + flashinfer≥
   0.6.11.post1 asserts (pure guards; chained to the engine subprocess via `--export=ALL`).
2. `TMPDIR=/tmp` — WITHOUT this, the harness/session `TMPDIR=/tmp/claude-…` leaked through
   submit.sh→sbatch→ray-start srun and broke `ray start` (`ray_client_server [exit code=1]`, head
   died, launcher hung in its bring-up poll). This is a CLAUDE-SESSION artifact, not a real-user
   issue — a human shell has `TMPDIR=/tmp`. (Job 13760 happened to get past it; 13761 didn't —
   borderline AF_UNIX socket-path length under the long temp dir.)

**This RESOLVES the bare-metal question a third way (now the leading option):** keep the WORKING
torch-2.9.1 stack (sglang-kernel 0.4.1) and run the synced v0.5.12-24 source on top, with
`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK`. No container, no torch-2.11/cu13 kernels, no pinning sglang
to an older revision. Implication for `install_env.sh`: the validated path installs the OLD wheel
stack (torch 2.9.1 + 0.4.1 kernel) against the NEW source — i.e. the WHEELS_STACK/torch pins should
target the 2.9.1 bundle, NOT 2.11, until cu13 kernels are runnable on this host. The launcher
(`launch_miles.sbatch` NODE_PREAMBLE) should export `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK` for this
path (decision: bake it in vs. per-run env). — DECISIONS PENDING, but the RUNTIME is proven.

**Separately found — launcher post-success hang (real bug, `lib/ray_lifecycle.sh`):** when the ray
job finishes FAST/cleanly, `ray_submit_and_wait`'s `ray job logs --follow` exits rc=0 and
reconnects in a loop, re-streaming the finished job's logs forever (`Job '…' succeeded` re-printed
every ~11s; run.log → 37k+ lines) instead of detecting the terminal SUCCEEDED state and returning.
The node is held until the wall-clock TIME limit. Had to `scancel` after success. Worth fixing
(treat terminal job status as exit, or cap log-follow reconnects).

### CONFIRMED #2 (job 13771, `vagen-fl-smoke`) — ✅ vagen FrozenLake + launcher fix, end-to-end

Ran the vagen FrozenLake multi-turn agent task on the CURRENT (synced) branch with the same two
knobs (`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true`, `TMPDIR=/tmp`), PLUS the `ray_lifecycle.sh` fix
ported from the `vagen-mvp` worktree. Brought the recipe (`scripts/experiments/
vagen-frozenlake-smoke-colocate-1node.sh`) + a FrozenLake-only env spec
(`examples/vagen/train_envs_frozenlake.yaml`) into the current branch (examples/vagen CODE was
already identical to vagen-mvp via merge-base c9ac71f89). vagen pkg = editable
`/data/home/xiuyul/workspace/VAGEN`.

- ✅ `VagenEnvSpecDataSource: materialized 64 samples` (in-process FrozenLake gen — vagen data path
  works on synced branch). sglang chose `attention_backend='fa3'`.
- ✅ Engine ready (`fired up and ready to roll`), 2× `Finish rollout` (multi-turn), train `step 0`
  AND `step 1`. NOTE `grad_norm=0.0` / `rollout/rewards=0.0` here — EXPECTED: batch=1×2 samples both
  failed the maze with identical reward → GRPO intra-group std=0 → zero advantage → zero gradient.
  The LOOP is mechanically correct (rollout→reward→advantage→train→weight-sync); just no signal
  from a 2-sample identical-reward group. (Contrast geo3k 13763: grad_norm 0.503 — had reward var.)
- ✅✅ **Launcher fix CONFIRMED:** ray-job-succeeded printed ONCE (not dozens), `[submit] ray job
  logs observed terminal state SUCCEEDED; not reconnecting`, `train.py terminal state: SUCCEEDED
  job_rc=0`, `[teardown] …`. Job **COMPLETED on its own** (sacct COMPLETED 0:0, MANIFEST
  state=SUCCEEDED job_rc=0, 20:13 wall) — did NOT need a scancel, unlike 13763 which hung.

**Launcher fix provenance:** the fix is purely in `scripts/slurm/lib/ray_lifecycle.sh` (the
`launch_miles.sbatch` delta in vagen-mvp is only NCCL-env passthrough). It (a) pipes `ray job logs
--follow` through awk that writes a terminal-state marker and breaks the reconnect loop on terminal
marker OR clean rc=0, and (b) only counts an unreadable ray-dashboard probe toward CLUSTER_DEAD
when `squeue` says the job is NOT RUNNING (+ probe_timeout 10→30, fail_grace 24→40). Applied to the
current branch as an UNCOMMITTED working-tree change (backup at `ray_lifecycle.sh.pre-vagen-bak`).

**Net:** synced v0.5.12-24 + old torch-2.9.1 binaries is validated on BOTH geo3k (VLM math) and
vagen (FrozenLake multi-turn agent). Bare-metal path is solid. Remaining: land the install_env.sh
2.9.1-pin + the version-check-skip wiring + the ray_lifecycle.sh fix (all decisions pending).

### CONFIRMED #3 (job 13778, `vagen-fl-e2e`) — ✅ real e2e training, 30 steps, CLEAN AUTO-EXIT

Bounded e2e of the MAIN vagen recipe (`vagen-frozenlake-main-qwen3vl2b-colocate-1node.sh`, ported
to current branch; reused the prebuilt `data/frozenlake-main/{train,eval}/samples.jsonl` from the
vagen-mvp worktree — 10k train / 256 eval envs). Realistic sizing: batch 32 × n_samples 8 (global
256), NUM_ROLLOUT=30, eval-interval 20, wandb on, VAGEN_THINK_TAG=thinking. Same knobs
(SKIP_VERSION_CHECK + TMPDIR=/tmp) + the ray_lifecycle.sh fix.

- ✅ **30 train steps (0..29), COMPLETED, sacct=COMPLETED 0:0, MANIFEST SUCCEEDED job_rc=0, 1:26:33
  wall — exited on its OWN, no scancel.**
- ✅ **Launcher fix confirmed over a 90-min run:** `ray job logs observed terminal state SUCCEEDED;
  not reconnecting` + `train.py terminal state: SUCCEEDED` + `[teardown]`; "succeeded" printed
  ONCE (vs the infinite loop on the old code).
- ✅ **Non-zero grad_norm throughout** (~1.3–2.7, normal spikes to 12.0 @ step16 / 5.6 @ step20) —
  real gradients from n=8 reward variance.
- ✅ **Eval path works** (new vs smoke): step-0 baseline `eval/frozenlake_val=0.229`
  (traj_success 0.18), step-20 `=0.207`.
- ◻ No reward IMPROVEMENT over 30 steps (raw_reward stable ~0.17–0.29, eval flat/slightly down) —
  EXPECTED: lr=1e-6, 30 steps (full recipe = 400). This was a STABILITY/MECHANICS test, not a
  convergence run.
- ✅ No OOM/crash/NaN. At-exit tracebacks (wandb teardown BrokenPipe + torch_memory_saver CUresult
  at free) are the same benign post-success shutdown noise seen on geo3k.

**FINAL:** synced v0.5.12-24 + old torch-2.9.1 `miles` binaries validated on geo3k (VLM math) AND
vagen FrozenLake (multi-turn agent) at BOTH smoke and 30-step e2e scale. ray_lifecycle.sh launcher
fix validated on a short AND a 90-min run. Bare-metal path is production-shaped; remaining work is
landing decisions (install_env.sh 2.9.1-pin, where to wire SKIP_VERSION_CHECK, land the launcher
fix + #12 split).

### CONFIRMED #4 (job 13854, `vagen-sokoban-e2e`) — ✅ Qwen2.5-VL-3B + Sokoban, 160 steps, LEARNED

Existing recipe VERBATIM (`vagen-sokoban-main-qwen25vl3b-colocate-1node-global_bsz32.sh` overlay +
base, copied to current branch; reused prebuilt `data/sokoban-main/{train,eval}`). Only override:
MILES_SCRIPT_NUM_ROLLOUT=20 (+ runtime knobs). NO hand-authored config (per user: don't invent
settings I can't validate). global_batch 32 → 8 updates/rollout → 160 train steps.

- ✅ **2nd model arch** (Qwen2.5-VL-3B, not Qwen3-VL) — engine init clean on old binaries + version skip.
- ✅ **2nd env** (Sokoban) multi-turn agent rollout works (board re-render per turn, push actions).
- ✅ **LEARNED:** eval/sokoban_val 0.478 → 0.699, traj_success 0.156 → 0.281 over 20 rollouts.
  (Contrast frozenlake e2e flat — that was 30 single-update steps; this is 160 updates.)
- ✅ grad_norm max 5.63, mostly ~2–3 (some 0.0 = mini-batch slices w/ no reward variance, normal).
- ✅ COMPLETED on its own, sacct COMPLETED 0:0, MANIFEST SUCCEEDED job_rc=0, 1:05:13; launcher fix
  3rd confirmation ("succeeded" once, "not reconnecting"). No OOM/crash (at-exit wandb BrokenPipe noise only).

**Validation complete across 2 model archs × 3 tasks (geo3k/frozenlake/sokoban), smoke→160-step
e2e with demonstrated learning.** PR-scope decision (pending user): sync-only PR (upstream merge +
sglang submodule c74db48da + a "run on torch-2.9.1 miles env w/ SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"
doc); KEEP LOCAL the torch-2.11 dep-script changes; pins.env must NOT ship at 2.11 (revert to main's
2.9.1). Launcher ray_lifecycle.sh fix + sglang-sync SKILL.md = separate PRs.
