# Orbit Conversion Scripts

This directory is reserved for orbit-owned shell entrypoints for checkpoint
conversion workflows.

Current orbit-owned conversion entrypoints:
- `convert_fp8_checkpoint_direct.sh`: Orbit shell wrapper for direct-write HF
  FP8 -> Megatron conversion via `tools/convert_fp8_checkpoint_direct.py`.
- `convert_int4_checkpoint_direct.sh`: Orbit shell wrapper for direct-write HF
  INT4 -> Megatron conversion via `tools/convert_int4_checkpoint_direct.py`.
- `convert_nvfp4_checkpoint_direct.sh`: Orbit shell wrapper for direct-write HF
  NVFP4 -> Megatron conversion via `tools/convert_nvfp4_checkpoint_direct.py`.
- `convert_dsv4_hf_to_megatron.sh`: Orbit shell wrapper for DeepSeek V4
  Flash/Pro. By default it stages with DeepSeek's official `inference/convert.py`
  at mp1, keeps routed experts in FP4, patches the staged HF config with
  `configuration_deepseek_v4.py`/`auto_map` plus the native DSV4 aliases SGLang
  reads directly, patches the tokenizer config with a chat template for Orbit
  rollout, and writes a Megatron `torch_dist`.

Shared conventions:
- scripts resolve the current Orbit repo root automatically
- default output is `${REPO_ROOT}/checkpoints/<model_name>`
- if `megatron.bridge` is not importable from the active Python environment,
  set `MEGATRON_BRIDGE_ROOT` to a Megatron-Bridge checkout
- low-precision FP8 and INT4 conversions intentionally expose only the direct
  conversion paths, which retain the metadata needed for QOFT and parity checks

DeepSeek V4 Orbit flow:
- Keep the official HF checkpoint unchanged. Copy it to node-local/NVMe storage
  if desired, then run `convert_dsv4_hf_to_megatron.sh` on the copy. A staged
  input under `DSV4_OFFICIAL_STAGE_PATH` writes the official mp1 HF output beside
  that copy by default, for example `DeepSeek-V4-Flash-debug-inference-mp1`.
- With no arguments, the script uses the local Flash-debug copy and writes the
  matching local debug `torch_dist`; pass explicit paths for Flash/Pro.
- The script prints `HF_CKPT`, `MEGATRON_LOAD`, and `LOAD_CKPT` exports at the
  end. Use those values with the V4 Orbit launchers so SGLang reads the staged
  HF/tokenizer files while Megatron loads the pre-converted `torch_dist`.
- `DSV4_PATCH_CHAT_TEMPLATE=1` is the default because Orbit launchers pass
  `--apply-chat-template`; disable it only when the checkpoint already carries a
  verified `chat_template`.
- `DSV4_PATCH_HF_CONFIG=1` is the default because the official V4 configs do not
  carry the local `AutoConfig` shim or top-level native aliases such as
  `dim`, `n_layers`, `n_heads`, `window_size`, `head_dim`, `rope_head_dim`,
  `kv_lora_rank`, `moe_inter_dim`, `n_hash_layers`, `beta_fast`, `beta_slow`,
  `original_seq_len`, `rope_factor`, and `max_seq_len` needed by current
  Orbit/SGLang loading. The aliases are copied or derived from the official
  config fields; `max_seq_len` defaults to
  `rope_scaling.original_max_position_embeddings` unless
  `DSV4_DEFAULT_MAX_SEQ_LEN` is explicitly set.
- `DSV4_DROP_MTP=1` is the default for Orbit RL: the staged metadata disables
  `num_nextn_predict_layers`/`n_mtp_layers` and trims `compress_ratios` to
  `num_hidden_layers`. Set `DSV4_DROP_MTP=0` only when the downstream DSV4
  path supports MTP.

Progress logging:
- `ORBIT_CONVERSION_PROGRESS=1` enables periodic conversion progress logs and
  defaults to on for the Orbit wrappers
- `ORBIT_CONVERSION_PROGRESS=0` disables those logs
- `ORBIT_CONVERSION_PROGRESS_INTERVAL=10` controls the progress interval in
  seconds
- `convert_nvfp4_checkpoint_direct.sh` forwards those generic settings to
  `MEGATRON_BRIDGE_DIRECT_SAVE_PROGRESS` and
  `MEGATRON_BRIDGE_DIRECT_SAVE_PROGRESS_INTERVAL` unless they are already set
