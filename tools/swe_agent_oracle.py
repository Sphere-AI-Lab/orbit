#!/usr/bin/env python
"""Golden-episode oracle for the agentic SWE loop (rung 2b acceptance gate).

Drives the REAL episode machinery — real Qwen tokenizer/chat template (the
append-only prefix assertion runs against the true template), real Apptainer
container session, real SWE-bench verification — with a scripted "model":

- golden episode: turn 1 applies the instance's golden patch via run_shell
  (base64-decoded, no quoting hazards), turn 2 submits -> reward MUST be 1.0;
- lazy episode: submits immediately -> reward MUST be 0.0.

Only the /generate HTTP call is faked (scripted turn texts, token ids from
the real tokenizer). Exits 0 iff both verdicts are correct.

Usage:
    python tools/swe_agent_oracle.py \\
        --swe-jsonl .../splits/swe.train.jsonl --sif-cache .../sif_cache \\
        --hf-checkpoint /path/to/Qwen2.5-0.5B-Instruct \\
        [--instance-id python-markdown__markdown-1529]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from types import SimpleNamespace

import orbit.rewards.swe_agent.episode as episode_mod
from miles.utils.types import Sample


def _load_instance(path: str, instance_id: str | None) -> dict:
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            md = row["responses_create_params"]["metadata"]
            if instance_id is None or md.get("instance_id") == instance_id:
                return json.loads(md["instance_dict"])
    raise SystemExit(f"instance {instance_id!r} not found")


def _scripted_post(tokenizer, turns):
    queue = list(turns)

    async def fake_post(url, payload):
        text = queue.pop(0)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return {
            "text": text,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [(-0.1, i) for i in ids],
            },
        }

    return fake_post


def _run(inst: dict, args_ns, tokenizer, turns) -> Sample:
    episode_mod.post = _scripted_post(tokenizer, turns)
    sample = Sample(
        prompt=[{"role": "user", "content": inst["problem_statement"][:4000]}],
        metadata={
            "swe": {
                "image_name": inst["image_name"],
                "test_patch": inst.get("test_patch") or "",
                "fail_to_pass": inst.get("FAIL_TO_PASS") or [],
                "pass_to_pass": inst.get("PASS_TO_PASS") or [],
            }
        },
    )
    return asyncio.run(episode_mod.generate(args_ns, sample, {"max_new_tokens": 1024}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swe-jsonl", required=True)
    ap.add_argument("--sif-cache", required=True)
    ap.add_argument("--hf-checkpoint", required=True)
    ap.add_argument("--instance-id", default="python-markdown__markdown-1529")
    args = ap.parse_args()

    inst = _load_instance(args.swe_jsonl, args.instance_id)
    print(f"instance: {inst['instance_id']}")

    args_ns = SimpleNamespace(
        sglang_router_ip="scripted",
        sglang_router_port=0,
        swe_rm_sif_cache=args.sif_cache,
        swe_rm_timeout_secs=600,
        swe_agent_max_turns=6,
        swe_agent_cmd_timeout_secs=60,
        rollout_max_response_len=8192,
        hf_checkpoint=args.hf_checkpoint,
        chat_template_path=None,
    )
    # GenerateState is the full rollout singleton (needs many args); the
    # episode only uses .tokenizer — substitute a light state with the REAL
    # tokenizer so the true chat template exercises the prefix assertion.
    from miles.utils.processing_utils import load_tokenizer

    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=None, trust_remote_code=True)

    class _LightState:
        def __init__(self, _args):
            self.tokenizer = tokenizer

    episode_mod.GenerateState = _LightState

    b64 = base64.b64encode(inst["patch"].encode()).decode()
    apply_cmd = f"echo {b64} | base64 -d > /orbit_scratch/golden.patch && git apply --whitespace=nowarn /orbit_scratch/golden.patch && echo APPLIED"

    def tc(name, **arguments):
        return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'

    golden = _run(inst, args_ns, tokenizer, [tc("run_shell", command=apply_cmd), tc("submit")])
    print(f"golden episode: reward={golden.reward} status={golden.status.name} "
          f"resp_len={golden.response_length} masked={golden.loss_mask.count(0)}/{len(golden.loss_mask)}")

    lazy = _run(inst, args_ns, tokenizer, [tc("submit")])
    print(f"lazy episode:   reward={lazy.reward} status={lazy.status.name}")

    ok = golden.reward == 1.0 and lazy.reward == 0.0 and golden.loss_mask.count(0) > 0
    print(f"### SWE_AGENT_ORACLE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
