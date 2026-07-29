"""Gate G4: score the held-out set through HuggingFace and compare against Orbit.

Orbit's step-0 reference, on the untrained Qwen3-4B base model over the 100 No Robots test
rows (produced by `examples/sft/run-qwen3-4b-norobots-sft.sh` with `LR=0.0`):

    nll = 3.589597   sample_mean = 4.232735   tokens = 18472   samples = 100

This script recomputes the same quantity with HuggingFace's own forward pass. Agreement
validates, in one shot, four things that Orbit's evaluator could otherwise only be argued
to get right: the log-prob/token index alignment, the packed-logits layout, the
token-weighted reduction, and the temperature bypass.

Why this script exists instead of the vendored oracle (`third_party/lora-without-regret/
sft_full.py`): that script's real flag surface is exactly `--model-id --lr --wandb-project
--wandb-run-name --no-wandb --batch-size --gradient-accumulation-steps --num-epochs
--output-dir --seed` (verified by reading it). There is no `--max-steps` and no `--dtype`.
Nothing in that list stops the run after step 0 -- `--lr 0.0` alone still runs a full epoch
of forward+backward over all 6400 training rows as a training-shaped no-op, purely to reach
the same step-0 print this script produces directly. That is not a reasonable way to get one
forward pass over 100 held-out rows, so G4's HF side is scored here instead.

Deliberate choices:

* The loss mask comes from Orbit's own `MultiTurnLossMaskGenerator` (tokenizer_type
  "qwen3"), which gate G3 already verified token-for-token against an independent HF
  reference on all 100 of these rows. Re-deriving the mask here would test the mask twice
  and the loss computation not at all.
* Orbit's mask is TARGET-indexed: `loss_mask[j] == 1` means token `j` is scored. Under a
  causal LM the logit at position `j-1` predicts token `j`, hence the standard shift
  `logits[:-1]` against `labels[1:]`.
* `--dtype` selects the HF-side forward precision. The cross-entropy itself always
  accumulates in float32 regardless of `--dtype`, so the sum over ~18.5k tokens does not
  lose precision to bf16 accumulation on top of a bf16 forward. G4's pass condition (design
  doc §7.2) needs this run twice -- once at `bfloat16` (matches Orbit's compute; this is the
  actual gate comparison) and once at `float32` (establishes the measured bf16-vs-fp32
  spread that the gate's tolerance is defined against, since a fixed sub-1e-3-nat bar is not
  achievable here).
* Both the token-weighted mean (what the study reports) and the sample mean are printed,
  because the two differ by ~0.64 nats here and confusing them is a ~70x error against a
  target table spanning 0.009 nats.

This script only prints the HF-side numbers and the delta against Orbit's reference; it does
not bake in a pass/fail threshold. Whether G4 passes depends on comparing the delta against
the *measured* bf16-vs-fp32 spread (run this script at both `--dtype` values), per design
doc §7.2 -- not a fixed constant.

Usage (see docs/superpowers/plans/2026-07-27-lora-without-regret-repro.md Task 11 Step 2):

    python tools/lora_regret/g4_hf_nll.py --dtype bfloat16   # -> logs/lora_regret/g4_hf.log
    python tools/lora_regret/g4_hf_nll.py --dtype float32    # -> logs/lora_regret/g4_hf_fp32.log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MODEL = "/lustre/fast/fast/zqiu/hf_models/Qwen3-4B"
DEFAULT_DATA = "/lustre/fast/fast/groups/ei-slm/data/lora_regret/no_robots_test.jsonl"

# Orbit's step-0 reference numbers (design doc §7.2 / gate log), measured once via
# examples/sft/run-qwen3-4b-norobots-sft.sh with LR=0.0.
ORBIT_NLL = 3.589597
ORBIT_SAMPLE_MEAN = 4.232735
ORBIT_TOKENS = 18472
ORBIT_SAMPLES = 100


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model path or hub id.")
    parser.add_argument(
        "--data", default=DEFAULT_DATA, help="Held-out JSONL (Orbit {'prompt': [...]} rows)."
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="HF forward-pass precision (default: bfloat16, matching Orbit's compute).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Deferred: keeps --help (and CPU-only invocation of this module) independent of torch,
    # transformers, and the orbit package actually being importable/GPU-visible.
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mask_gen = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")

    rows = [json.loads(line)["prompt"] for line in Path(args.data).read_text().splitlines()]
    print(f"loaded {len(rows)} rows from {args.data}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    per_sample_means = []
    unscorable_first_token = 0

    with torch.no_grad():
        for i, messages in enumerate(rows):
            token_ids, loss_mask = mask_gen.get_loss_mask(messages)
            assert len(token_ids) == len(loss_mask), f"row {i}: len mismatch"

            # Token 0 has no preceding logit, so it can never be scored.
            if loss_mask[0] == 1:
                unscorable_first_token += 1

            ids = torch.tensor(token_ids, device="cuda:0").unsqueeze(0)
            mask = torch.tensor(loss_mask, device="cuda:0", dtype=torch.bool)

            logits = model(ids).logits  # [1, T, V]

            # Causal shift: logits[j-1] predicts token j.
            shift_logits = logits[0, :-1, :].float()
            shift_targets = ids[0, 1:]
            shift_mask = mask[1:]

            nll_per_token = F.cross_entropy(shift_logits, shift_targets, reduction="none")
            scored = nll_per_token[shift_mask]

            total_nll += float(scored.sum())
            total_tokens += int(shift_mask.sum())
            if scored.numel():
                per_sample_means.append(float(scored.mean()))

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(rows)} rows, {total_tokens} scored tokens", flush=True)

    token_weighted = total_nll / total_tokens
    sample_mean = sum(per_sample_means) / len(per_sample_means)

    print()
    print("=" * 64)
    print(f"GATE G4 -- HuggingFace ({args.dtype}) vs Orbit (bf16), untrained {Path(args.model).name}")
    print("=" * 64)
    print(f"{'':22s} {'HF':>14s} {'Orbit':>14s} {'delta':>12s}")
    print(f"{'token-weighted NLL':22s} {token_weighted:14.6f} {ORBIT_NLL:14.6f} "
          f"{token_weighted - ORBIT_NLL:12.6f}")
    print(f"{'sample-mean NLL':22s} {sample_mean:14.6f} {ORBIT_SAMPLE_MEAN:14.6f} "
          f"{sample_mean - ORBIT_SAMPLE_MEAN:12.6f}")
    print(f"{'scored tokens':22s} {total_tokens:14d} {ORBIT_TOKENS:14d} "
          f"{total_tokens - ORBIT_TOKENS:12d}")
    print(f"{'samples':22s} {len(rows):14d} {ORBIT_SAMPLES:14d} "
          f"{len(rows) - ORBIT_SAMPLES:12d}")
    if unscorable_first_token:
        print(f"WARNING: {unscorable_first_token} row(s) scored token 0, which has no "
              f"predicting logit and is necessarily dropped by the shift")
    print("=" * 64)

    # The token count is the sharpest diagnostic: it must match exactly, since it is pure
    # integer bookkeeping with no floating-point involved. Everything else -- whether the
    # NLL delta is within tolerance -- is a judgment against the *measured* bf16-vs-fp32
    # spread (design doc §7.2), not something this script decides on its own.
    if total_tokens != ORBIT_TOKENS:
        print(f"scored-token count differs from Orbit's by {total_tokens - ORBIT_TOKENS}. "
              f"This is a masking/alignment disagreement, not a numerical one -- G4 fails "
              f"regardless of the NLL delta.")
        return 1

    print(f"delta vs Orbit = {token_weighted - ORBIT_NLL:+.6f} nats at --dtype={args.dtype}. "
          f"Run this script at the other --dtype value too, then compare the delta against "
          f"that measured bf16-vs-fp32 spread to decide G4 pass/fail (design doc §7.2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
