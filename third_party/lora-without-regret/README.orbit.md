# lora-without-regret (vendored oracle)

Snapshot of https://github.com/michaelbzhu/lora-without-regret at commit `1c7bef8a9a8049f62120707033313eedd46c49a9`.

## Why this is here

This is a **throwaway validation oracle**, not part of Orbit's supported surface.
It exists to answer one question: does Orbit's `sft_loss` path produce the same
test NLL as a known-good HF/PEFT trainer? See
`docs/superpowers/specs/2026-07-27-lora-without-regret-repro-design.md` §7 (gates G1/G2).

Once G2 passes, nothing in the reproduction depends on this directory.

## Running it

It has its own uv environment, deliberately isolated from Orbit's:

    uv sync --directory third_party/lora-without-regret
    CUDA_VISIBLE_DEVICES=0 uv run --directory third_party/lora-without-regret \
        sft_lora.py --lr 2.5e-4 --lora-rank 256 --lora-type all --no-wandb

## Local patches

One, applied 2026-07-28 to `sft_lora.py` and `sft_full.py`, both inside `eval()`:

- **Added** a second reported metric, `val_loss_token_weighted`, alongside the existing
  `val_loss`. Upstream computes `total_loss / len(val_dataloader)` — an unweighted mean
  over batches of HF's per-batch token-mean — while Orbit reports a global token-weighted
  NLL. Gate G2 compares those two numbers directly, so without this the gate would fail on
  a reduction mismatch rather than on a training difference. The patch accumulates
  `loss * n_scored_tokens` and divides once, using HF's own denominator
  (`labels[:, 1:] != -100`, since HF shifts labels internally so position 0 never scores).
  It also prints the token and batch counts and logs the new metric to W&B.

This is instrumentation, not a fix: training is untouched, `val_loss` is computed and
returned exactly as upstream does, and `wandb.summary["final_val_loss"]` is unchanged, so
G1 still measures the published-style number.

## Do not

- Import from this directory in Orbit code.
- "Fix" it to match Orbit. If they disagree, that is the signal we are looking for.
  (Adding a *second* metric so both sides can be read in the same units is not a fix —
  changing the existing one would be.)
