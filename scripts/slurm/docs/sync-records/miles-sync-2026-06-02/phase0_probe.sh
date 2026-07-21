#!/usr/bin/env bash
# Phase 0 — decisive gate: does our sglang engine load on the base image's cu12+torch-2.11 kernels?
# Run INSIDE lmsysorg/sglang:v0.5.12-cu129 via pyxis. Read-only intent (one editable install of our
# submodule). No HF downloads, no model load — pure import test.
set -uo pipefail
REPO=/data/home/xiuyul/workspace/miles-imp

echo "########## A. interpreter / torch / cuda ##########"
which python; python --version
python -c 'import torch; print("torch", torch.__version__, "torch.version.cuda", torch.version.cuda)'

echo; echo "########## B. kernel package versions ##########"
python - <<'PY'
import importlib.metadata as m
for k in ["sgl-kernel","sglang","sgl-deep-gemm","deep-gemm","flashinfer-python",
          "flashinfer","cuda-python","transformer-engine","transformer-engine-torch"]:
    try: print(f"{k:26s} {m.version(k)}")
    except Exception as e: print(f"{k:26s} ?  {e}")
PY

echo; echo "########## C. sgl_kernel native linkage (expect libnvrtc.so.12 / libcudart.so.12) ##########"
python - <<'PY'
import sgl_kernel, os, glob
d = os.path.dirname(sgl_kernel.__file__); print("sgl_kernel dir:", d)
for so in glob.glob(d+"/**/*.so", recursive=True)[:6]: print(" so:", so)
PY
SO=$(python -c 'import sgl_kernel,os,glob;c=glob.glob(os.path.dirname(sgl_kernel.__file__)+"/**/*.so",recursive=True);print(c[0] if c else "")')
if [ -n "$SO" ]; then echo "ldd $SO:"; ldd "$SO" 2>/dev/null | grep -iE "nvrtc|cudart|libcuda" || echo "  (no cuda libs in ldd)"; else echo "  (no sgl_kernel .so found)"; fi

echo; echo "########## D. BASE image's OWN sglang engine import ##########"
python - <<'PY'
try:
    from sgl_kernel import sgl_per_token_quant_fp8
    import sglang.srt.layers.quantization.fp8_kernel
    from sglang.srt.server_args import ServerArgs
    import sglang; print("BASE ENGINE IMPORT OK ; sglang at", sglang.__file__)
except Exception as e:
    import traceback; traceback.print_exc(); print("BASE ENGINE IMPORT FAILED:", repr(e))
PY

echo; echo "########## E. install OUR submodule sglang -e --no-deps ##########"
pip install -e "$REPO/thirdparty/sglang/python[all]" --no-deps 2>&1 | tail -15

echo; echo "########## F. OUR sglang engine import (fresh process) ##########"
python - <<'PY'
try:
    from sgl_kernel import sgl_per_token_quant_fp8
    import sglang.srt.layers.quantization.fp8_kernel
    from sglang.srt.server_args import ServerArgs
    import sglang; print("OUR ENGINE IMPORT OK ; sglang at", sglang.__file__)
    print("our-submodule?", "/miles-imp/thirdparty/sglang/" in sglang.__file__)
except Exception as e:
    import traceback; traceback.print_exc(); print("OUR ENGINE IMPORT FAILED:", repr(e))
PY
echo; echo "########## PHASE 0 PROBE DONE ##########"
