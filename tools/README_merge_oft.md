# Merge OFT Adapters

Merge OFT adapters from multiple Orbit runs with compatible base model and OFT
configuration.

```bash
python tools/merge_oft_adapters.py \
  --adapters RUN_A/iter_0000800/adapter RUN_B/iter_0000800/adapter \
  --output /path/merged_oft \
  --method oft \
  --weights 0.5 0.5 \
  --save-megatron \
  --save-hf \
  --base /path/to/hf/base
```

`--method oft` is Orbit's native OrthoMerge strategy: magnitude-corrected
Lie-algebra averaging on OFT skew parameters. It supports optional per-adapter
weights. `--method oft-naive` is a plain average baseline.

`--method oft-original` reproduces the equal-weight adapter-only formula used by
the original OrthoMerge OFT script. It rejects `--weights` by design and is most
useful for parity checks and reproductions.

## Outputs

- `<output>/merged_adapter/`: HF PEFT OFT adapter, useful for re-merge/export.
- `<output>/merged_megatron/`: Megatron-native adapter shards, written with
  `--save-megatron`, resumable/servable in Orbit via `--peft-adapter-path`.
- `<output>/merged_model_hf/`: dense HF model with the merged rotation baked in,
  written with `--save-hf`.

All input adapters must share `oft_type`, `oft_block_size`, `target_modules`,
and `base_model_name_or_path`.

## Single-Adapter Bake

To bake one OFT adapter into a standalone dense HF model:

```bash
python tools/bake_oft_to_hf.py \
  --base /path/to/hf/base \
  --adapter RUN_A/iter_0000800/adapter \
  --output /path/baked_hf \
  --device cuda:0
```

This is useful for evaluation stacks that expect dense HF weights.

## Reference Comparison

`tools/orthomerge_bridge.py` contains small utilities to run the original
OrthoMerge OFT script, run Orbit's merge path, summarize adapters, and compare
the resulting tensors.
