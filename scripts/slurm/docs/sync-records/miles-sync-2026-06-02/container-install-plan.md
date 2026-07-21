# Container-route plan — run v0.5.12-23/torch-2.11 in the official sglang base image

**Why this route.** v0.5.12-23 pins torch 2.11, whose sglang native kernels (`sgl-kernel`,
`deep_gemm`) are published **cu13-only**; this host is CUDA 12.8 (can't run cu13). The **only**
place the matching **cu12 + torch-2.11** kernels exist is the `lmsysorg/sglang:v0.5.12-cu129`
base image. Bare-metal conda re-resolves and pulls cu13 → engine won't load (see
`install-findings.md` Issues 7–9). The official `docker/Dockerfile` avoids this by running ON
that base and installing sglang/miles `-e --no-deps` (no re-resolution).

**Compatibility is by design (confirmed from the Dockerfile).** `docker/Dockerfile`:
- `FROM lmsysorg/sglang:v0.5.12-cu129` (cu12 + torch-2.11 `sgl-kernel`/`deep_gemm`/flashinfer baked in)
- L137–144: `cd /sgl-workspace/sglang && git fetch origin sglang-miles && git checkout FETCH_HEAD
  && pip install -e "python[all]" --no-deps`  ← the sglang-miles branch TIP, editable, no deps
- L149–152: `pip install -e . --no-deps` for miles

Our submodule `thirdparty/sglang` @ `c74db48da` = **sglang-miles tip `3102015ca` (v0.5.12-23) + a
single python-only commit (the mrope patch)** — i.e. exactly what the Dockerfile checks out, plus
our one local patch. So the base image's kernels are the *matched* set for our sglang code
(torch 2.11 ↔ submodule `torch==2.11.0`; mrope touches no kernels). Only caveat: base is a fixed
tag, `sglang-miles` is a moving branch — but we synced contemporaneously with upstream's bump to
this exact base tag, so it's upstream's current intended pairing.

**Runtime available.** docker daemon is DOWN, but `enroot 4.0.1` + **pyxis** are present
(`srun --container-image=…`). pyxis pulls/imports the image itself (cached after first use) and
supports `--container-mounts`, `--container-workdir`, `--container-writable`,
`--container-name`/`--container-save` (provision-once-reuse). No manual `enroot import` needed.

Repo: `/data/home/xiuyul/workspace/miles-imp` (branch `sync-upstream-20260602`, submodule sglang
@ `c74db48da`). HF assets already cached at `/data/shared/hf_cache` (Qwen3-VL-2B + geo3k). Secrets
at `~/.config/secrets.env` (HF_TOKEN, WANDB_API_KEY).

---

## Phase 0 — DECISIVE GATE: does our sglang engine load on the base's kernels? (~1 GPU, cheap)

One `srun` into the base image, repo bind-mounted:
```bash
srun -p slinky --gres=gpu:1 \
  --container-image=lmsysorg/sglang:v0.5.12-cu129 \
  --container-mounts=/data/home/xiuyul/workspace/miles-imp:/data/home/xiuyul/workspace/miles-imp \
  --container-workdir=/data/home/xiuyul/workspace/miles-imp --container-writable \
  bash -lc '
    python -c "import torch;print(\"torch\",torch.__version__,torch.version.cuda)"
    python - <<PY
import importlib.metadata as m
for k in ["sgl-kernel","sglang-kernel","sgl-deep-gemm","flashinfer-python"]:
    try: print(k, m.version(k))
    except Exception as e: print(k, "?", e)
PY
    ldd $(python -c "import sgl_kernel,glob,os;print(glob.glob(os.path.dirname(sgl_kernel.__file__)+\"/sm90/common_ops*.so\")[0])") | grep -iE "nvrtc|cudart"
    pip install -e thirdparty/sglang/python[all] --no-deps
    python -c "from sgl_kernel import sgl_per_token_quant_fp8; import sglang.srt.layers.quantization.fp8_kernel; from sglang.srt.server_args import ServerArgs; print(\"ENGINE IMPORT OK\")"
  '
```
**Expected if approach is sound:** base torch == 2.11 (cu12), sgl-kernel sm90 → `libnvrtc.so.12`,
and `ENGINE IMPORT OK` after installing our submodule sglang editable. **This is the core test —
it proves the engine runs our code on the base kernels. STOP and assess before Phase 1.**
(Also finally answers the open "what torch is in the base image" question.)

## Phase 1 — provision a reusable container (only if Phase 0 passes)
`srun --container-image=lmsysorg/sglang:v0.5.12-cu129 … --container-save=…/miles-v0512.sqsh` that
runs the Dockerfile's post-base steps the base lacks: our sglang + miles `-e --no-deps`, prebuilt
FA/apex/TE/router/fake_int4 wheels (from `yueming-yuan/miles-wheels@cu129-x86_64-v0.5.12`),
Megatron-LM/-Bridge editables, mbridge/modelopt, requirements.txt, cudnn, mooncake. = "build the
radixark/miles image via pyxis" (docker build unavailable). Output: `miles-v0512.sqsh`.
- TE note: in the base, TE 2.10.0 prebuilt torch-ext may already match (the base's torch ABI is
  what it was built for) — Phase 0's version dump decides whether we need a source build here.

## Phase 2 — containerized geo3k multi-turn smoke → first train step
`sbatch`/`srun --container-image=miles-v0512.sqsh` for 1-node colocate geo3k smoke, Ray local
in-container, with the smoke overrides
(`MILES_SCRIPT_NUM_ROLLOUT=2 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=2`), mounts: repo +
`/data/shared/hf_cache` + secrets. Bar: reach first train step, then `scancel`.
- Launcher: `launch_miles.sbatch` is conda-based. For 1-node colocate, a thin pyxis submit running
  `train.py` (Ray local, recipe args) is simplest; a full pyxis port of `launch_miles.sbatch`
  (multi-node Ray + healthcheck + fault-tolerance) is a later/separate task.

## Open items / risks
- One-time base-image pull is tens of GB (pyxis handles it; first Phase-0 run is slow).
- Phase 0 must confirm base torch == 2.11 + cu12 kernels; if base is torch 2.9.x, re-think (our
  submodule pins 2.11 — would need a 2.9.x sglang pin instead, per install-findings option 1).
- `--container-writable` + editable installs are ephemeral per-run unless saved (Phase 1 saves).
- Whether the in-container cudnn needs any LD_LIBRARY_PATH handling (likely not — image is
  self-consistent, unlike the bare-metal host's system cudnn 9.7.0 shadow).

## Decision context (from install-findings.md)
This container route is the "follow upstream to torch 2.11" path (doesn't diverge from upstream).
The alternative is pinning sglang to the torch-2.9.1 v0.5.12.post1 point (the working `miles` env)
— bare-metal-viable but ACTIVE held behind UPSTREAM by design. User chose to try the container
route first.
