#!/usr/bin/env bash
#
# The environment the lora_regret campaign runs in on the v0.5.16 sglang line.
#
#   source scripts/lora_regret/env_v0516.sh
#   bash scripts/lora_regret/run_e4_math_lora_verify_8gpu.sh
#
# NOT `orbit_env`, which every other doc in this tree still names. That venv
# carries sglang b52394d22 -- the v0.5.9 line -- and orbit's own
# `sglang_utils/arguments.py` says its mirror assumption is "v0.5.14-only".
# Under b52394d22 the mirror yields `sglang_data_parallel_size`, so
# `validate_args` dies on `args.sglang_dp_size` before a single rollout runs
# (measured 2026-08-17, i206). `orbit_env_v2` has the right sglang but its
# editable orbit points at `orbit-merged`, a different checkout.
#
# This borrows the stack that the PPO-critic benchmark was actually built and
# run against, then puts THIS checkout in front of it. `activate.sh` already
# exports PYTHONPATH pointing at its own orbit; the re-export below is what
# makes the orbit-iclr tree win, and it is load-bearing rather than cosmetic:
# Ray workers import orbit by path (SETUP.md), so without it the rollout actors
# would run a different checkout's orbit than the campaign driver does -- and
# `e4oftverify` lives only here.
#
# The provenance any report using these runs must state: sglang resolves from
# `orbit_env_v2`'s installed build at 05cd76b4d, NOT from this repo's sibling
# `orbit-iclr/sglang` checkout at a5def08d0. Same OFT kernels (the three
# intervening commits are tests and a chore), different commit id.

ORBIT_ICLR_ROOT="${ORBIT_ICLR_ROOT:-/fast/zqiu/orbit-iclr/orbit}"

# `orbit_env_v2`, NOT `clthegoat-orbit/uv_env_build`, and the difference is the
# whole OFT ladder.
#
# Both carry the v0.5.16 line and both clear `args.sglang_dp_size`, so the LoRA
# arms do not care which one they get. The OFT arms do:
#
#   clthegoat-orbit/sglang  33022a130  2026-08-08
#   orbit_env_v2  sglang    05cd76b4d  "feat(oft): port tiny-block OFT kernels"
#
# 33022a130 predates that port, and its fused kernel hard-asserts
# `BS >= 16, "Triton tl.dot requires BS >= 16"`. The b8 rung is BS=8, so it
# cannot launch at all there -- measured 2026-08-17 on i407, engine init dead
# in `_validate_inputs`. 05cd76b4d replaces that assert with a real BS<16
# branch, which is exactly the `orbit-main-oft-tiny` work this repo merged.
#
# 05cd76b4d is three commits behind orbit-iclr/sglang's own a5def08d0, and all
# three are tests and a chore -- no kernel change -- so this env is equivalent
# to the merged tip for everything the campaign touches.
#
# shellcheck disable=SC1091
source /fast/zqiu/orbit-iclr/orbit_env_v2/bin/activate

# CUDA_HOME explicitly: env.sh's `module load` is a no-op non-interactively and
# its fallback list does not include this cluster's path, so megatron.core's
# deep_ep import would assert with no message. See INSTALL.md.
export CUDA_HOME="${CUDA_HOME:-/is/software/nvidia/cuda-13.2}"

# PYTHONPATH last, and load-bearing: orbit_env_v2's editable install resolves
# orbit to the `orbit-merged` worktree, a DIFFERENT checkout that does not carry
# `e4oftverify`. Ray workers import orbit by path (SETUP.md), so without this
# the rollout actors would run other code than the campaign driver selected.
export PYTHONPATH="${ORBIT_ICLR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Disable the prefill CUDA graph. Two independent failures on this sglang line
# converge on the same flag:
#
#   * `orbit/ray/rollout.py` injects SGLANG_MEMORY_SAVER_CUDA_GRAPH=true into
#     every engine, and v0.5.16's BreakableCudaGraphBackend refuses it outright
#     -- "Breakable CUDA graph is not compatible with memory saver mode"
#     (measured 2026-08-17: killed every LoRA and OFT arm at engine init).
#   * f4112d5 hit it from the other side for OFT: the prefill graph captures a
#     warmup forward outside the normal batch-prep path, the OFT triton backend
#     has no batch_info there, and init dies in sgemm_oft_r_fwd.
#
# Decode graphs stay on. `ppo_critic_compare_common.sh` carries the identical
# flag, so this is the established setting for this stack rather than a new one.
#
# Appended rather than assigned: `e4_protocol.sh` puts
# `--disable-grpo-std-normalization` here, and that flag is the advantage
# definition the whole campaign rests on. Dropping it would silently change the
# experiment.
export RL_EXTRA_ARGS="${RL_EXTRA_ARGS:---disable-grpo-std-normalization} --sglang-cuda-graph-backend-prefill disabled"
