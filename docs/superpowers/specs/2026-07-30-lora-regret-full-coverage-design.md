# LoRA-Without-Regret — full-coverage design

What the campaign is missing against the post, and how to wire it. The existing
campaign (`docs/superpowers/plans/2026-07-28-lora-without-regret-experiments.md`,
178 runs across E1-E5) covers the post's core SFT and RL claims on one model. This
design adds **130 arms in four phases** so every experiment in the post has a
script behind it, plus the three audit blockers that gate what already exists.

Companions:
- Claims, arms, acceptance criteria: `docs/superpowers/plans/2026-07-28-lora-without-regret-experiments.md`
- Operator's guide: `docs/superpowers/plans/2026-07-30-lora-without-regret-runbook.md`
- Port status: `docs/superpowers/plans/2026-07-29-lora-without-regret-gap.md`

## The gap this closes

Audited 2026-07-30 against `b94a9ea`. The tooling is ready — 593 tests pass,
`preflight --stage e1-lora` passes 23/23, every matrix dry-runs at its documented
count — but coverage against the post is partial:

| Post experiment | Before | After |
|---|---|---|
| LR-vs-loss by rank, Tulu3 (10x rule) | E1-1 | unchanged |
| Rank-vs-curve, Tulu3 | E1-2 | unchanged |
| Batch-size sensitivity, OpenThoughts3 | E2 | unchanged |
| Layer placement, dense | E3 | unchanged |
| RL on MATH+GSM8K | E4 (never executed) | Phase 0 smokes it |
| **Rank curves on OpenThoughts3** | absent | **`e1ot`** |
| **~100-step 15x multiplier** | absent | **`e1short`** |
| **Layer placement under RL** | absent | **`e4place`** |
| **LR scaling law across models** | absent | **`e6`** |
| **DeepMath RL + AIME** | absent | **`e7`** |
| **Layer placement, MoE** | declared-skipped | **`e3moe`** |
| **Figures** | absent | **`plot.py`** |

## Phasing

| Phase | Contents | New arms | New infrastructure |
|---|---|---|---|
| **0** | P3 DP check; smoke one `e4` arm; smoke one `e5scout` arm | 0 | none |
| **1** | `e1ot`, `e1short`, `e4place`, `plot.py` | 68 | none |
| **2** | `e6` scaling law | 40 | fetch + convert Qwen3-0.6B, Qwen3-8B; Qwen control-token scan |
| **3** | `e7` DeepMath | 2 | DeepMath-103K + AIME24/25 prep; avg@16 eval; response-length parser |
| **4** | `e3moe` MoE placement | 20 | convert Qwen3-30B-A3B; expert-module mapping; per-expert matched rank |

Phase 0 is not new work — it closes the three blockers the audit found, and every
number downstream of a FullFT arm depends on P3. Phases 2 and 3 share the
Qwen3-8B conversion, which is why the scaling law precedes DeepMath: Phase 3 gets
its policy for free. After Phase 0 the phases are independent; Phase 1 needs no
downloads and can start immediately while Phases 2-4 convert checkpoints.

### Phase 0 — the audit blockers

1. **P3, the DP>1 held-out NLL reduction.** Never executed. Runbook §4. Gates
   every FullFT number in the campaign, old and new.
2. **One `e4` arm, smoked.** The RL launcher has *never run* — `logs/` holds two
   SFT smokes and nothing else. `parse_final_accuracy` scrapes a Python dict repr
   out of `rollout.py`'s `logger.info(f"eval {rollout_id}: {log_dict}")`; unit
   tests pin the shape of that line and cannot prove it is reached. A wrong key
   shape fails all 16 `e4` arms identically at 8 GPUs each.
3. **One `e5scout` arm, smoked.** `--peft-method oft --oft-type canonical_oft` has
   never run on Llama-3.1-8B in this repo. 55 arms depend on it.

Each is the same argument the 2026-07-30 SFT smoke made and won: only a real run
proves a log line is reached.

## Architecture

No new execution mode. `sweep.py` is unchanged in shape; the work is six matrices,
one registry, one plotting tool, and four data/checkpoint prerequisites.

Two new modules:

- `tools/lora_regret/models.py` — the base-model registry.
- `tools/lora_regret/plot.py` — figures from ledgers.

`arms.py` grows the six matrices and one optional field. **The registry is split
out and the matrices are not**, because the split is by concern rather than by
size: the registry is paths and dimensions with no dependency on grid logic, while
the matrices share `Arm`, `lr_grid`, `ALL_MODULES` and the LR centres, and exist to
be enumerated together. A per-matrix package would make `common.py` carry that
shared vocabulary and leave eight files holding one function each.

### `Arm.model`

`Arm` gains `model: str = "llama3.1-8b"`. The default is load-bearing: every
existing matrix then serializes byte-identically and every existing ledger stays
valid, so this change cannot invalidate work already done.

## The model registry

```
Model(key, hf_checkpoint, megatron_load, model_args_plugin,
      hidden_size, ffn_size, num_layers, qkv_output_size,
      loss_mask_type, chat_template_path | None,
      moe: MoE(num_experts, moe_ffn_size, topk) | None)
```

Contents, read from `orbit_plugins/model_args/*.sh` and the HF configs on
2026-07-30 — measured, not assumed:

| key | hidden | ffn | layers | qkv_out | mask | chat template |
|---|---|---|---|---|---|---|
| `llama3.1-8b` | 4096 | 14336 | 32 | 6144 | llama3 | `llama3.1_pinned.jinja` |
| `qwen3-0.6b` | 1024 | 3072 | 28 | 4096 | qwen | model's own |
| `qwen3-1.7b` | 2048 | 6144 | 28 | 4096 | qwen | model's own |
| `qwen3-4b` | 2560 | 9728 | 36 | 6144 | qwen | model's own |
| `qwen3-8b` | 4096 | 12288 | 36 | 6144 | qwen | model's own |
| `qwen3-30b-a3b` | 2048 | 6144 | 48 | 5120 | qwen | model's own |

Qwen3-30B-A3B additionally carries `MoE(num_experts=128, moe_ffn_size=768, topk=8)`.

`qkv_output_size` is `(num_attention_heads + 2*num_query_groups) * kv_channels`,
with `kv_channels = 128` for every model here. It is a field rather than a
derivation from `hidden_size` because GQA makes the two differ, and E3/E5/E3moe's
matched-parameter arithmetic is wrong without it. Llama-3.1-8B's 6144 is the value
already pinned as `LLAMA31_8B_QKV_OUTPUT` in `arms.py`; the registry subsumes it.

Llama-3.1-8B base ships **no** chat template, which is why the campaign pins a
jinja file. Every Qwen3 base model here ships one, so `chat_template_path` is
`None` for them and the launcher omits `--chat-template-path`.

### Launcher parameterization

Both launchers gain `MODEL_KEY`, defaulting to `llama3.1-8b`, and source the
plugin the registry names. Three consequences:

**`--hidden-size/--ffn-size/--num-layers` leave the sweep command line.** Today
they are three CLI arguments an operator can get wrong independently of the model
being run, which is precisely the failure the registry removes. The sweep derives
them from the arm's model and **rejects a contradicting value rather than
preferring one**.

**The FullFT GPU guard becomes a formula.** Per-GPU cost under Megatron's
distributed optimizer is `4*P + 12*P/N` GB for `P` billion parameters — bf16
weights and grads replicated (`4*P`), fp32 master and Adam moments sharded across
`N` DP ranks (`12*P/N`). At 8.03B that is `32 + 96/N`, reproducing the launcher's
current hardcoded arithmetic exactly. It also permits Qwen3-0.6B FullFT on one
card (9.6 GB), which the hardcoded `>= 4` would wrongly refuse, and correctly
predicts that **Qwen3-30B-A3B FullFT does not fit on 8x80 GB** (~168 GB/GPU at
30.5B parameters). That
last is not a problem: `e3moe` is an attention-vs-expert comparison within LoRA
and has no FullFT arm.

**`LOSS_MASK_TYPE=qwen` runs in a sweep for the first time.** That generator was
rewritten in G1 to fix phantom-`<think>` injection and its parity gate passes, but
every Qwen3 arm in Phases 2-4 rests on that fix. Phase 2 therefore opens with a
loss-mask parity check against the real Qwen3 tokenizer before any arm runs.

## The six matrices

### `e1ot` — rank curves on OpenThoughts3 (40 arms + 2 sigma replicates)

FullFT plus LoRA r in {1, 4, 16, 64, 128, 256, 512}, five 0.3-decade LRs each,
centred on 2.5e-5 / 2.5e-4 exactly as `e1`. Dataset is the 10,000-row
OpenThoughts3 subset already materialized for E2.

**One epoch here is 312 optimizer steps at batch 32**, against Tulu3's 29,323. So
a single matrix yields both the argmins (the post's LR-vs-loss figure, second
dataset) and the full-epoch curves (the rank-vs-curve figure, second dataset).
There is no `e1otlong`: that split exists for Tulu3 only because a full epoch
there is unaffordable at 40 arms. `EVAL_NLL_INTERVAL=3`, about 1% of the epoch.

**Two sigma replicates are mandatory, not a refinement.** The OpenThoughts3
held-out split is **100 rows against Tulu3's 1,000**. Its noise floor is a
different number, and every claim in this campaign is a difference quoted in units
of sigma. Seeds 1 and 2 of `lora-r256-all-lr0.00025` land in
`results/e1ot_0_sigma.jsonl`; seed 0 is already a grid arm.

### `e1short` — the short-run LR multiplier (14 arms)

FullFT and LoRA r256 at `NUM_ROLLOUT=100` on Tulu3, **seven LRs each at
0.15-decade spacing**, centred on 2.5e-5 and 2.5e-4 — the same centres as `e1`, so
the two stages' ratios are read off grids that agree at their midpoints.

The spacing is forced by the claim. The post says short runs (~100 steps) want a
~15x multiplier where long runs converge to ~10x. Distinguishing those means
resolving a factor of 1.5, which is 0.176 decades — on the campaign's standard
0.3-decade grid, adjacent points differ by 2x and the effect is invisible. Both
arms need the fine grid, because the claim is a ratio of two argmins and a coarse
denominator ruins it as surely as a coarse numerator.

`EVAL_NLL_INTERVAL=10`. At interval 1 a 100-step arm would spend ~113 min
evaluating against ~14 min training; the trace is not what this stage is for.

Reuses Tulu3's sigma from E1-0 — same dataset, same 1,000-row split.

### `e4place` — layer placement under RL (12 arms)

LoRA attention-only r256, MLP-only r92, all-modules r256; four half-decade LRs
each centred on 1e-5. Trains on `math_gsm8k_train.jsonl` and evaluates
`math_test` + `gsm8k_test` through the RL launcher — the same data and the same
half-decade grid as `e4`, so the placement result and the rank result are read off
comparable arms. Scored by accuracy.

r92 is E3's solved match for attention r256 in Orbit's fused layout
(`matched_mlp_rank`), reused deliberately so the RL result is comparable to the
SFT one arm-for-arm. No FullFT arm: the post's RL placement panel is a comparison
within LoRA.

### `e6` — the LR scaling law (40 arms)

Per model, FullFT x 5 LRs and LoRA r256 x 5 LRs, on Tulu3 at `NUM_ROLLOUT=2000` —
the same budget as E1-1, so the Llama point is comparable without adjustment.

Four models: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B. **Llama-3.1-8B is
excluded and asserted excluded**, because its ten arms already exist as `e1`'s
FullFT and r256 rows; the analysis joins the two ledgers rather than paying for
them twice.

Hidden sizes 1024 / 2048 / 2560 / 4096 (Qwen3) plus 4096 (Llama, cross-family) —
four distinct sizes spanning 4x, with one within-family and one cross-family point
at 4096.

Per-model GPU tiering follows the registry formula: 0.6B and 1.7B FullFT on one
card, 4B on two, 8B on four.

### `e7` — DeepMath RL (2 arms)

Qwen3-8B-base, FullFT and LoRA r256, **500 rollouts each at a fixed budget** so
the trajectories are comparable. DeepMath-103K, 32 samples per problem.

Eval is AIME24 + AIME25 at `N_SAMPLES_PER_EVAL_PROMPT=16`. avg@16 rather than a
single sample because each set is 30 problems: at k=1 a single problem flipping
moves the score by 3.3 points, which swamps the effect being measured.

**LR provenance is recorded, not assumed.** The LR comes from `e4`'s argmins if
that stage has completed, else the RL launcher's centres (1e-6 FullFT, 1e-5 LoRA).
Either way the arm's ledger records which, because carrying an argmin across both
a model change and a dataset change is an assumption, and an unlabelled assumption
is the thing that makes a result unquotable six weeks later.

The CoT-growth half of the claim needs no new instrumentation: `rollout.py:1194`
emits `eval/<dataset>/response_len/*` and `rollout.py:1258` emits `response_len/*`
per training rollout. A new parser reads them. `parse_final_accuracy` already
excludes `eval/<name>/<metric>` sub-metric keys by design, so this is purely
additive.

### `e3moe` — layer placement on the MoE (20 arms)

Qwen3-30B-A3B, four LoRA configs x five LRs. No FullFT arm (it does not fit, and
the claim does not need one).

**Four configs, because the post's rank convention and matched-parameter counting
disagree on this model, and that disagreement is worth measuring.** Per layer:

```
attention   qkv  2048 -> 5120 :  r*(2048 + 5120) =  7168r
            proj 4096 -> 2048 :  r*(4096 + 2048) =  6144r
                                          total    13312r

experts     128 x [ fc1 2048 -> 1536 :  r*(2048 + 1536) = 3584r
                    fc2  768 -> 2048 :  r*( 768 + 2048) = 2816r ]
                                          total    128 * 6400r = 819200r

matched to attention r256 :  13312*256 / 819200  =  r_expert 4.16  ->  4
post's convention (rank / active experts, topk=8) :  256/8  ->  32
```

So: attention r256, expert r4 (matched parameters), expert r32 (the post's
convention), all-modules r256. This is E3's design one model over — run both pairs
so a disagreement lands on parameter accounting rather than on physics, and print
realized adapter parameter counts next to every arm.

`fc1`'s 1536 is the fused gate+up of `moe_ffn_hidden_size=768`.

## Data and checkpoints

### Fetches and conversions

`tools/convert_hf_to_torch_dist.py --hf-checkpoint X --save Y` already exists and
is the conversion path. The wrapper the old repo's plan referenced,
`scripts/conversion/convert_checkpoint.sh`, **was not ported** and is not in
`scripts/conversion/`; the design calls the Python tool directly rather than
resurrecting a second recipe that can drift.

`scripts/lora_regret/fetch_models.sh` gains `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-8B`;
those two are the only missing HF weights. Three conversions are needed —
Qwen3-0.6B, Qwen3-8B and Qwen3-30B-A3B — because `torch_dist` checkpoints exist
today only for Llama-3.1-8B, Qwen3-1.7B and Qwen3-4B. Qwen3-30B-A3B's HF weights
are already on disk, so it needs conversion only.

Acceptance for each conversion is what `preflight` already checks for
Llama-3.1-8B: the output directory contains `latest_checkpointed_iteration.txt`
and an `iter_*` subdirectory.

### New datasets

Both go through `prepare_data.py` with the guarantees the existing nine splits
have — asserted source row count, schema check, empty-label filter, re-read after
write, previous file untouched on mismatch:

| Output | For | Notes |
|---|---|---|
| `deepmath_train.jsonl` | `e7` | `zwhe99/DeepMath-103K`, `{"prompt", "label"}`, boxed-answer instruction appended |
| `aime24_test.jsonl` | `e7` | 30 rows |
| `aime25_test.jsonl` | `e7` | 30 rows |

The boxed-answer instruction is not optional. `--rm-type boxed_math` strips
`\boxed{...}` before grading, and Qwen3-8B **base** no more boxes unprompted than
Llama-3.1-8B base does — without it every rollout scores 0 and both arms look
identical.

### The Qwen control-token scan — gates all of Phase 2

Tulu3's scan came back `filtered=0` across 939,343 rows, and that result is real —
but it scanned for **Llama's** literals only: `ASSISTANT_HEADER_LITERAL` and
`EOT_LITERAL` at `tools/lora_regret/prepare_data.py:59-60`. Qwen3 arms tokenize
the same rows with `<|im_start|>` / `<|im_end|>`, and G1's bug was specifically
about `<think>` handling in the Qwen3 mask generator.

A Tulu3 row carrying a literal `<|im_end|>` would silently truncate a scored span
for every Qwen3 arm — no error, corrupted spans — while being completely invisible
to the existing scan. So `_llama_control_token_hazards` generalizes to a
per-family token set, and **Tulu3 is re-scanned under the Qwen set before any
Qwen3 arm runs**. The answer may well be `filtered=0` again; "probably clean" is
not the standard the existing scan set.

## Analysis and figures

`analyze.py` gains three claim subcommands and two existing ones gain a dataset
dimension. All follow the existing contract: seed-0 filter built in, sigma units,
edge-of-grid refusal with exit 3, `--json` emitting one document and nothing else.

**C7 — the scaling law.** Per model, `argmin_LR(LoRA r256) / argmin_LR(FullFT)`
against `hidden_size`, joining `e6`'s ledger with `e1`'s for the Llama point.
Upheld if the ratio is ~10 **and flat in hidden size**. Flatness is the real
content: it says both methods share the d_model dependence, so it cancels in the
ratio. A ratio that is 10 at one size and 3 at another refutes the post's rule
while still averaging to something near 10, so the average is not the reading.

**C8 — the short-run multiplier.** The same ratio at 100 steps against at 2,000
steps. Upheld if the short-run ratio is materially larger (the post: ~15 against
~10). Exits 3 if either grid's argmin sits on an edge.

**C9 — DeepMath.** Both arms' AIME avg@16 trajectories and their mean response
length over rollouts. Trajectory identity is the pointwise gap staying inside a
band; CoT growth is a positive response-length slope for both arms. Sigma for AIME
accuracy has never been measured — if the curves sit close, measuring it becomes a
prerequisite exactly as E1-0 was for NLL, and the tool says so rather than
quoting an unresolved difference.

**`c1` and `c4` gain `--dataset`**, so the OpenThoughts3 curves and the MoE
placement read through the same code as their Tulu3/dense originals rather than
through a parallel copy that can drift.

**The sigma ledger records its dataset, and a mismatch is refused.** Quoting
Tulu3's 1,000-row noise floor against OpenThoughts3's 100-row held-out set is the
specific error this check exists to prevent, and it is an easy one to make because
both files are called `*_sigma.jsonl`.

### `plot.py`

Reads `analyze --json` plus the NLL and accuracy traces; writes one PNG per post
panel into `results/figures/`. Pure function of the ledgers — no network, no
state, no side channel. Where the post's own figure exists under
`third_party/lora-without-regret/figures/`, the corresponding output is written
side-by-side with it so the comparison is visual rather than asserted.

Panels: LR-vs-loss by rank (Tulu3, OpenThoughts3); loss-vs-log-steps by rank
(Tulu3, OpenThoughts3); batch-size; placement (dense SFT, dense RL, MoE); RL
accuracy by rank; the scaling law; the short-run multiplier; DeepMath trajectory
and CoT length.

## Testing

Each test pins a fact a plausible edit would otherwise break silently.

**Registry drift.** For every model, parse the `model_args` plugin it names and
assert the registry's `hidden_size`, `ffn_size`, `num_layers` and
`qkv_output_size` equal the plugin's own flags, with `qkv_output_size` checked as
`(num_attention_heads + 2*num_query_groups) * kv_channels`. A registry that can
disagree with the plugin it points at is worse than none, because the wrong number
is then recorded twice and neither copy looks suspicious.

**Matrix builds.** Each new matrix builds at its documented count: `e1ot` 40,
`e1short` 14, `e4place` 12, `e6` 40, `e7` 2, `e3moe` 20.

**`e6` excludes Llama.** Asserted, so it cannot silently re-run `e1`'s ten arms
and produce a second, differently-seeded copy of the same measurement.

**`e1short` spacing.** Asserted <= 0.15 decades. The spacing is a requirement of
the claim, not a preference, and a future tidy-up that unified it with the 0.3
grid would destroy the stage without failing anything.

**`e3moe` expert rank** is asserted computed from the registry's shapes, not a
literal 4 — a hardcoded rank would survive a model change and quietly compare
unmatched adapters.

**FullFT GPU formula** yields 4 at 8.03B (reproducing today's guard), 1 at 0.6B,
and > 8 at 30B-A3B.

**Qwen control-token scan** catches a fixture row carrying a planted `<|im_end|>`.

**Analysis** — `c7`/`c8`/`c9` against synthetic ledgers with known answers;
`c1` refuses a sigma ledger whose dataset differs from the arms'.

**`plot.py`** writes the expected file count from a synthetic JSON document with
no network access.

**`preflight`** gains stage entries (`e6`, `e7`, `e3moe`), the three new
checkpoints, and the three new data files at their row counts.

## Fail-closed rules

New gates, in the style of `--argmins-from` and `--oft-lr-centre`:

| Gate | Exit | What it prevents |
|---|---|---|
| `--hidden-size` etc. contradicting the arm's model | 2 | the three-independent-CLI-args bug class the registry exists to remove |
| sigma ledger's dataset != the claim's arms' dataset | 3 | quoting a 1,000-row noise floor against a 100-row split |
| `e7` without recorded LR provenance | 2 | a cross-model, cross-dataset LR transfer becoming invisible |
| `e3moe` when the checkpoint's layer count != the registry's | 2 | a wrong or truncated conversion reading as a valid model |
| Qwen3 arm on a Tulu3 split not scanned under the Qwen token set | 2 | silently truncated scored spans across all of Phase 2 |

Existing gates are unchanged and still apply: `--argmins-from` refusing a partial
E1-1 ledger or an edge-of-grid argmin, `--oft-lr-centre` required for `e5` and
rejected elsewhere, and every claim subcommand exiting 3 on an edge-of-grid argmin
unless `--allow-edge-argmin`.

## Cost

Estimates below are **inferred** by scaling the runbook's measured Llama-3.1-8B
rates (8.5 s/step training, 67.7 s per 1,000-row eval) by parameter ratio. They
are not measured for any model other than Llama-3.1-8B.

| Stage | Arms | GPUs/arm | Estimate |
|---|---|---|---|
| `e1ot` + sigma | 42 | 1 | ~45 GPU-h — 312 steps and a 100-row eval make this the cheapest SFT stage |
| `e1short` | 14 | 1 or 4 | ~8 GPU-h |
| `e6` | 40 | 1-4 | ~280 GPU-h, dominated by the 8B FullFT arms |
| `e4place`, `e7`, `e3moe` | 34 | 4-8 | **not estimable** |

The last row is honest rather than lazy: **no RL arm has ever run in this repo**,
so there is no measured per-rollout time to scale from, and 30B on this cluster
has no measured step time either. Phase 0's `e4` smoke produces the first RL
number; `e3moe`'s smoke produces the first 30B number. Both estimates should be
written down when they exist, and no allocation for those three stages should be
requested before then.

For reference the existing campaign is ~265 GPU-h for E1-1 and, inferred from the
same rates, ~569 GPU-h for E1-2's eight full-epoch arms.

## Out of scope, deliberately

The post sweeps **14** Llama and Qwen variants for the scaling law; this design
uses five. Four distinct hidden sizes spanning 4x is enough to fit a slope and to
test that LoRA and FullFT share it, which is the claim; fourteen points would test
it more finely at roughly triple the cost and require nine more conversions. State
the model count next to the C7 reading.

The post's DeepMath run is a large-scale campaign; `e7` is two arms at a fixed
500-rollout budget. It tests the trajectory-identity claim at the smallest honest
size. A trajectory that has not diverged by rollout 500 and one that never diverges
look identical, so quote the budget next to the result — the same rule E1-2's
departure points carry.
