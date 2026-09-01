#!/usr/bin/env python
"""Rollout determinism harness (true-on-policy Phase 2).

Certifies that an SGLang server produces *byte-identical* prefill log-probs
regardless of batch composition — the rollout half of the true-on-policy
parity ladder (design doc: docs/plans/2026-07-06-true-on-policy-design.md).

Method:
1. Take (or generate) a fixed set of token sequences.
2. Score every sequence via prefill-only requests (``max_new_tokens=0,
   return_logprob=True, logprob_start_len=0``), flushing the radix/KV cache
   before each batch, under several *different* batch compositions
   (one big batch / reversed uneven triples / one-by-one).
3. Assert the per-token log-probs are exactly equal across compositions.

Run against a server started with ``--enable-deterministic-inference`` (and a
deterministic attention backend: fa3/triton/flashinfer) — expected PASS.
Against a default server this harness is the negative control: batch-variant
kernels should produce visible differences.

Usage:
    python tools/rollout_determinism_harness.py \\
        --base-url http://127.0.0.1:30700 \\
        --hf-checkpoint /path/to/Qwen2.5-0.5B-Instruct \\
        --prompts /path/to/train.jsonl --num-sequences 16 --gen-tokens 64

Exits 0 on PASS, 1 on FAIL; prints a greppable ``### DETERMINISM`` line.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

GROUPING_SCHEMES = ("single-batch", "reversed-triples", "singletons")


def _num_sequences(value: str) -> int:
    count = int(value)
    if count < 2:
        raise argparse.ArgumentTypeError("num-sequences must be at least 2 to vary batch composition")
    return count


def make_groupings(n: int, scheme: str) -> list[list[int]]:
    """Partition indices 0..n-1 into ordered batches per the named scheme."""
    if scheme == "single-batch":
        return [list(range(n))]
    if scheme == "reversed-triples":
        rev = list(reversed(range(n)))
        return [rev[i : i + 3] for i in range(0, n, 3)]
    if scheme == "singletons":
        return [[i] for i in range(n)]
    raise ValueError(f"Unknown grouping scheme: {scheme!r}")


def build_scoring_payload(input_ids: list[list[int]]) -> dict[str, Any]:
    """Prefill-only scoring payload covering the full sequence (start_len=0)."""
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": 0,
            "temperature": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }


def compare_logprob_sets(a: list[list[float]], b: list[list[float]]) -> tuple[bool, float, int]:
    """Exact comparison of two per-sequence log-prob sets.

    Returns (identical, max_abs_diff, n_mismatching_tokens). Raises on shape
    mismatch — a scoring bug, not a determinism finding.
    """
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b, strict=True)):
        raise ValueError(f"logprob set shape mismatch: {[len(x) for x in a]} vs {[len(y) for y in b]}")
    max_diff = 0.0
    n_mismatch = 0
    for seq_a, seq_b in zip(a, b, strict=True):
        for va, vb in zip(seq_a, seq_b, strict=True):
            if va != vb:
                n_mismatch += 1
                max_diff = max(max_diff, abs(va - vb))
    return n_mismatch == 0, max_diff, n_mismatch


# ---------------------------------------------------------------------------
# I/O half (requests-based; imported lazily so unit tests stay dependency-free)
# ---------------------------------------------------------------------------


def _session():
    import requests

    s = requests.Session()
    s.trust_env = False  # never route localhost scoring through a proxy
    return s


def _post(session, base_url: str, path: str, payload: dict[str, Any], expect_json: bool = True) -> Any:
    resp = session.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=600)
    resp.raise_for_status()
    # /flush_cache returns plain text ("Cache flushed. ..."), not JSON
    return resp.json() if expect_json else resp.text


def _load_prompt_token_ids(args) -> list[list[int]]:
    from miles.utils.processing_utils import load_tokenizer

    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    prompts: list[list[int]] = []
    with open(args.prompts) as f:
        for line in f:
            if len(prompts) >= args.num_sequences:
                break
            row = json.loads(line)
            prompt = row[args.prompt_key]
            if isinstance(prompt, str) and prompt.startswith("["):
                try:
                    prompt = json.loads(prompt.replace("'", '"'))
                except Exception:
                    pass
            if isinstance(prompt, list):
                # return_dict=False: transformers 5 defaults it to True, which
                # would make list(ids) below a list of dict keys, not token ids.
                ids = tokenizer.apply_chat_template(
                    prompt, tokenize=True, return_dict=False, add_generation_prompt=True
                )
            else:
                ids = tokenizer.encode(str(prompt))
            prompts.append(list(ids))
    if len(prompts) < args.num_sequences:
        raise ValueError(f"only {len(prompts)} prompts available, need {args.num_sequences}")
    return prompts


def _generate_sequences(session, args, prompt_ids: list[list[int]]) -> list[list[int]]:
    """Generate a response tail once so scored sequences look like rollouts.

    Generation settings don't matter for the assertion — we only need fixed
    token sequences; the determinism claim is about *scoring* them.
    """
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "max_new_tokens": args.gen_tokens,
            "temperature": 1.0,
            "skip_special_tokens": False,
        },
        "return_logprob": False,
    }
    outputs = _post(session, args.base_url, "/generate", payload)
    if not isinstance(outputs, list):
        outputs = [outputs]
    sequences = []
    for ids, out in zip(prompt_ids, outputs, strict=True):
        out_ids = out.get("output_ids") or []
        if not out_ids and "meta_info" in out and "output_token_logprobs" in out["meta_info"]:
            out_ids = [item[1] for item in out["meta_info"]["output_token_logprobs"]]
        if not out_ids:
            raise ValueError("generation returned no output token ids; cannot build sequences")
        sequences.append(list(ids) + list(out_ids))
    return sequences


def _score_grouping(session, args, sequences: list[list[int]], groups: list[list[int]]) -> list[list[float]]:
    results: dict[int, list[float]] = {}
    for group in groups:
        _post(session, args.base_url, "/flush_cache", {}, expect_json=False)
        payload = build_scoring_payload([sequences[i] for i in group])
        outputs = _post(session, args.base_url, "/generate", payload)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for idx, out in zip(group, outputs, strict=True):
            items = out["meta_info"]["input_token_logprobs"]
            # first entry is the placeholder (no logprob for the first token)
            results[idx] = [item[0] for item in items[1:]]
    return [results[i] for i in range(len(sequences))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--prompts", required=True, help="JSONL with a prompt column")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--num-sequences", type=_num_sequences, default=16)
    parser.add_argument("--gen-tokens", type=int, default=64)
    args = parser.parse_args()

    session = _session()
    prompt_ids = _load_prompt_token_ids(args)
    sequences = _generate_sequences(session, args, prompt_ids)
    print(f"scoring {len(sequences)} sequences (lengths {min(map(len, sequences))}-{max(map(len, sequences))})")

    reference = None
    ok = True
    for scheme in GROUPING_SCHEMES:
        scored = _score_grouping(session, args, sequences, make_groupings(len(sequences), scheme))
        if reference is None:
            reference = scored
            continue
        identical, max_diff, n_mismatch = compare_logprob_sets(reference, scored)
        status = "identical" if identical else f"MISMATCH max_abs_diff={max_diff:.3e} tokens={n_mismatch}"
        print(f"scheme {scheme} vs {GROUPING_SCHEMES[0]}: {status}")
        ok = ok and identical

    print(f"### DETERMINISM {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
