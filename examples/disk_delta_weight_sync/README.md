# Disk-Delta Weight Sync

Testing recipes for `--update-weight-transfer-mode disk-delta`: ship only the bytes that changed
between two syncs instead of the whole actor.

After one RL step almost every bf16 element is byte-identical to the previous step, so the trainer
diffs each gathered HF tensor against a CPU snapshot, publishes the changed bytes to a shared
directory, and each engine's `/pull_weights` patches a host-local checkpoint in place and reloads
from it through the ordinary `update_weights_from_disk` path. miles only ever talks to one endpoint
per engine — the engine fans the apply out to every host it spans — so multi-node serving and
external rollout engines need nothing extra.

Compare with [`../p2p_weight_transfer/`](../p2p_weight_transfer/), which moves the *full* weights
over RDMA. disk-delta trades interconnect bandwidth for CPU: it wins where the fabric is the
constraint (cross-rack, cross-datacenter, plain Ethernet) and loses where it isn't.

## Files

```
examples/disk_delta_weight_sync/
├── check_delta_roundtrip.py                  # CPU-only format check — run this first
├── _common.sh                                # shared workload; wrappers vary one axis each
├── _model-qwen3-30B-A3B.sh                   # MoE model block (arch, parallelism, layout)
│
├── 01-qwen3-4B-1node-delta-smoke.sh          # dense, 1 node 4+4, --check-weight-update-equal
├── 02-qwen3-4B-2node-delta.sh                # dense, deltas cross a real host boundary
├── 03-qwen3-4B-2node-broadcast.sh            # control arm for 02
│
├── 04-qwen3-30B-A3B-4node-delta-smoke.sh     # MoE correctness — first run of the expert path
├── 05-qwen3-30B-A3B-4node-delta.sh           # MoE measurement arm
├── 06-qwen3-30B-A3B-4node-broadcast.sh       # control arm for 05
└── 07-qwen3-30B-A3B-1node-delta-smoke.sh     # same expert coverage, one node
```

`_common.sh` fixes the dataset and the RL hyperparameters; a model block overrides the
architecture, parallelism, and layout. Within one model, the delta and broadcast arms differ in
the transport block and nothing else — `diff` the resolved `MILES_ARGS` if you want to confirm it.

## Sharding config

Every recipe is disaggregated — disk-delta asserts `not colocate` — so the training and rollout
GPUs below are disjoint sets, and `EXPERIMENT_NODES` covers both.

### Training parallelism

| Recipe | TP | PP | CP | EP | ETP | DP | tokens/GPU | nodes | train GPUs |
|---|---|---|---|---|---|---|---|---|---|
| `01` qwen3-4B 1-node | 2 | 1 | 1 | 1 | 1 | 2 | 9216 | 1 | 1 × 4 |
| `02` qwen3-4B 2-node delta | 2 | 1 | 1 | 1 | 1 | 4 | 9216 | 2 | 1 × 8 |
| `03` qwen3-4B 2-node broadcast | 2 | 1 | 1 | 1 | 1 | 4 | 9216 | 2 | 1 × 8 |
| `04` 30B-A3B 4-node smoke | 4 | 1 | 1 | **8** | 1 | 4 | 2048 | 4 | 2 × 8 |
| `05` 30B-A3B 4-node delta | 4 | 1 | 1 | **8** | 1 | 4 | 2048 | 4 | 2 × 8 |
| `06` 30B-A3B 4-node broadcast | 4 | 1 | 1 | **8** | 1 | 4 | 2048 | 4 | 2 × 8 |
| `07` 30B-A3B 1-node smoke | 4 | 1 | 1 | **4** | 1 | 1 | 2048 | 1 | 1 × 4 |

DP is derived, not set: `DP = train_GPUs / (TP × PP × CP)`. EP must divide `train_GPUs / ETP`,
which is the constraint that forces `07` down to EP4 — eight-way expert parallelism does not fit
across 4 training GPUs.

These are the configurations the recipes *specify*. Only `01` and `04` have been run — see
[which of these configs has actually run](#which-of-these-configs-has-actually-run) before
relying on any row.

### Rollout engine sharding

| Recipe | rollout GPUs | GPUs/engine | engines | `--sglang-ep-size` | mem fraction |
|---|---|---|---|---|---|
| `01` | 4 | 2 | 2 | — | 0.85 |
| `02` / `03` | 8 | 2 | 4 | — | 0.85 |
| `04` / `05` / `06` | 16 | 8 | 2 | 8 | 0.80 |
| `07` | 4 | 4 | 1 | 4 | 0.80 |

The MoE arms add `--sglang-enable-dp-attention` and `--sglang-enable-dp-lm-head`; the dense arms
use neither. Engine-side sharding is independent of the trainer's — the delta is published as
whole HF tensors and each engine re-shards through its ordinary loader on reload.

### Which of these configs has actually run

All seven, on slinky (H200s, InfiniBand), in the `miles_zeju` env.

| Recipe | Sharding | Job | Sync time (steady) | Density |
|---|---|---|---|---|
| `01` qwen3-4B 1-node | TP2 EP1, 4+4 | 38154 ✅ 45m | — | 0.44–0.58% |
| `02` qwen3-4B 2-node delta | TP2 EP1, 8+8 | 39268 ✅ | **4.9s** (4.6–5.2) | 0.39–0.60% |
| `03` qwen3-4B 2-node broadcast | TP2 EP1, 8+8 | 39274 ✅ | **0.3s** (0.3–0.5) | n/a |
| `04` 30B-A3B 4-node smoke | TP4 **EP8**, 16+16 | 38420 ✅ 1h03m | — | 0.28–0.36% |
| `05` 30B-A3B 4-node delta | TP4 **EP8**, 16+16 | 39277 ✅ 1h47m | **~91s** (54–179) | 0.24–0.35% |
| `06` 30B-A3B 4-node broadcast | TP4 **EP8**, 16+16 | 39385 ✅ 1h26m | **3.7s** (3.5–4.1) | n/a |
| `07` 30B-A3B 1-node | TP4 **EP4**, 4+4 | 39386 ✅ 1h30m | ~54–91s | 0.32–0.40% |

Every arm ran clean: zero checksum mismatches, zero out-of-order deltas, and
`--check-weight-update-equal` passed on every run that enabled it (`01`, `04`, `07`).

Caveats worth carrying:

- **`05` and `07` show large timing variance** (3.3× spread on `05`). disk-delta's cost depends on
  a *shared* filesystem, so it inherits cluster-wide I/O load from other tenants. Broadcast's NCCL
  path does not — hence its 0.5s total spread across ten syncs.
- **The first sync of every delta run is a warmup outlier** (31.7s vs 4.9s on `02`; 109.6s vs ~91s
  on `05`), and the *last* sync is sometimes a much larger one (46.7s on `02`, 731.8s on `07`)
  that coincides with teardown and which we did not diagnose.
- **`07` failed twice before succeeding.** Once on a config error of ours (concurrency left at the
  4-node value; see the file header), and once on a baseline `--check-weight-update-equal` failure
  — 292 tensors, uniform 1-ULP error — that never reproduced across the other two attempts on
  identical config. Unexplained.
- Jobs 38414/38417 died inside 80s on cluster faults (an InfiniBand probe failure; a node holding
  448 GB of leaked GPU memory while Slurm reported it idle). Neither reached training.

### What the sharding costs per recipe

| | qwen3-4B | 30B-A3B (EP8) | 30B-A3B (EP4) |
|---|---|---|---|
| Weights, bf16 | 8.04 GB | 57 GB | 57 GB |
| Experts | — | 128 | 128 |
| Experts per EP rank | — | 16 | 32 |
| Weight share per train GPU (÷TP) | ~4 GB | ~14 GB | ~14 GB |
| Adam state | on GPU | CPU-offloaded (~360 GB) | CPU-offloaded (~360 GB) |
| **Host-local checkpoint, per rollout host** | **~8 GB** | **~57 GB** | **~57 GB** |

That last row is the one that scales badly and is worth watching: `_reset_checkpoint` copies the
**entire** checkpoint to every rollout host — it is not sharded across them. 57 GB per host here;
a 355B-class model would want ~700 GB of node-local disk on each. Nothing in these recipes tests
that regime.

The trainer side is not sharded either, in the sense that matters: `_for_each_hf_bucket` *gathers*
the TP/EP-sharded parameters into full HF tensors before diffing, so the delta is computed in HF
space on whole tensors and never on Megatron shards. Published delta *files* are split one per
source rank (`model-NNNNN-of-NNNNN.safetensors`) purely for parallel I/O.

## Verified on slinky — and the answer is: use broadcast here

disk-delta works correctly and moves far less data. On this cluster it is also **substantially
slower**, at both model scales, and the gap widens with model size.

| | qwen3-4B (8.04 GB) | 30B-A3B (57 GB) |
|---|---|---|
| Data per sync, delta | 0.13 GB | ~0.5 GB |
| Data per sync, broadcast | 8.04 GB | 57 GB |
| Volume reduction | ~62× | ~114× |
| **Sync time, delta** | **4.9s** | **~91s** |
| **Sync time, broadcast** | **0.3s** | **3.7s** |
| **Broadcast is faster by** | **16×** | **25×** |

The reason is throughput, not volume. NCCL moves 8.04 GB in 0.3s (~27 GB/s) and 57 GB in 3.7s
(~15 GB/s). disk-delta must *scan* the entire checkpoint to diff it — memory-bandwidth-bound CPU
work at roughly 4 GB/s — then compress, write, and have every rollout host reload a full
checkpoint. Shrinking the payload 114× buys nothing when the fabric was never the bottleneck.

**This is the documented tradeoff, measured rather than assumed.** disk-delta trades interconnect
bandwidth for CPU; it is built for links where shipping 57 GB per sync genuinely hurts —
cross-rack, cross-datacenter, plain Ethernet. slinky's InfiniBand is not that. Read this as a
result about *this cluster*, not about the feature.

What the runs establish about correctness:

| | qwen3-4B | 30B-A3B |
|---|---|---|
| Tensors in baseline | 398 | 18,867 |
| Expert path (EP gather, dup-drop) | not reached (EP1) | exercised at **EP8 and EP4** |
| Integrity failures | none | none |
| `--check-weight-update-equal` | passed | passed |

MoE density (0.24–0.40%) runs *lower* than dense (0.39–0.60%): with top-8-of-128 routing a smaller
slice of the weights sees gradient per step. But `05`'s v5 carried 18,621 of 18,867 tensors, so
nearly every expert changed a little rather than a few changing a lot. Density stayed flat or
declined across every run — had the trainer's snapshot drifted from the engines' base, it would
have climbed toward 100% as the diff stopped finding reuse.

Off-GPU, `check_delta_roundtrip.py` also passed against real Qwen3-4B (jobs 38111, 38121) driving
sglang's real receiver, for both encodings and all three checksums — and `--corrupt` confirmed a
falsified checksum is refused rather than applied.

### Still untested

- **No engine has ever spanned more than one host.** Every recipe uses 8 GPUs per engine, which is
  exactly one node, so `/pull_weights`' multi-host fan-out — the design's central claim — has never
  executed. It needs `--rollout-num-gpus-per-engine 16`.
- **`overwrite` encoding and the `blake3`/`adler32` checksums** have only CPU round-trip coverage;
  every GPU run used `xor` + `xxh3-128`.
- **The object-store hooks** (`--custom-update-weight-post-write-path`,
  `--sglang-custom-pull-weights-pre-read-hook`) have never been invoked.
- **Late-joining hosts / chain reset**, engine restart mid-stream, and the per-host disk ceiling at
  355B scale (~700 GB/host).

## Start here: the format check

No GPU, no cluster, no Ray. It transcribes miles' encoder, then hands the published directory to
sglang's *real* receiver (`sglang.srt.weight_sync.local_checkpoint.pull`, the code `/pull_weights`
runs) and verifies the patched checkpoint byte-for-byte:

```bash
python examples/disk_delta_weight_sync/check_delta_roundtrip.py
```

Run it **in the environment the training job uses**. Its main job is catching the three things that
break a first bring-up — a codec or checksum dependency missing from the runtime env
(`blake3`, `xxhash`, `zstandard` are pinned in `requirements.txt` but easy to miss in an env built
before disk-delta landed), an encoding the two sides disagree about, and a shared directory the
engine can't read. A login-node pass says nothing about the compute nodes.

Useful variants:

```bash
# both encodings, all three checksums, longer delta chains
python .../check_delta_roundtrip.py --encoding overwrite --checksum blake3 --versions 5

# against real weights instead of a synthesized checkpoint
python .../check_delta_roundtrip.py --hf-checkpoint /data/shared/hf_cache/models/Qwen3-4B

# prove the integrity layer actually fires: falsifies a checksum, passes only if it is refused
python .../check_delta_roundtrip.py --corrupt

# coarser simulated step: --nudge bounds how far a changed element's bit pattern moves
python .../check_delta_roundtrip.py --density 0.01 --nudge 64
```

**What the compression ratio does and does not tell you.** It is governed by the mutation model,
not by the base weights: against real Qwen3-4B and against the synthesized checkpoint, the same
`--density` and `--nudge` produce the same ratio. The script perturbs low mantissa bits to imitate
an optimizer step — that shape matters, and randomizing elements instead understates the win by
roughly an order of magnitude — but the changed *positions* here are uniformly random where a real
step's are not. Read the ratio as a format and throughput check. The number worth quoting comes
from `perf/update_weights_wire_bytes` on an actual run.

## Cluster runs

The wrappers satisfy the `scripts/slurm/submit.sh` recipe contract, but `submit.sh` resolves
recipes by name under `scripts/experiments/`. Add a three-line shim:

```bash
# scripts/experiments/dd-01-smoke.sh
source "$MILES_REPO/examples/disk_delta_weight_sync/01-qwen3-4B-1node-delta-smoke.sh"
```

```bash
bash scripts/slurm/submit.sh dd-01-smoke
```

Run `01` first. It is short and turns on `--check-weight-update-equal`, which compares engine
tensors against the trainer's after each sync — the only direct proof that an applied delta
reconstructed the right bytes on real weights. Then run `02` and `03` back to back; `_common.sh`
holds the workload fixed so the difference between them is the transport and nothing else.

### The MoE arm

`04`–`06` run Qwen3-30B-A3B: 48 layers, 128 experts, 8 active per token, 57 GB in bf16. Both
checkpoints are already on slinky, outside `HF_CACHE_DIR`, so `submit.sh` skips the download:

| | |
|---|---|
| `--hf-checkpoint` | `/data/shared/models/Qwen3-30B-A3B` (16 shards, 57 GB) |
| `--ref-load` | `/data/home/zeju/models/Qwen3-30B-A3B_torch_dist` — a user-specific path; override `DD_HF_TORCHDIST_DIR` elsewhere |

Two reasons the MoE arm matters, and the second is the one that justifies running it before any
measurement:

1. **It is where the mechanism should pay.** Broadcast moves all 57 GB every sync no matter how
   little the step changed, against ~3B active parameters per token.
2. **It covers code the dense arm cannot reach.** `_for_each_hf_bucket` runs a TP pass and then a
   separate EP pass, and `_drop_duplicate_names` exists because an expert tensor can be gathered
   by more than one rank. At `--expert-model-parallel-size 1` the Qwen3-4B run entered neither.
   Run `04` first — it is the first exercise of the expert path, and it fails loudly.

Parallelism (TP4, EP8, 2 training + 2 rollout nodes, sglang `--sglang-ep-size 8` with DP
attention) mirrors [`../p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh`](../p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh),
the validated shape for this model on 4 nodes. Note the host-local checkpoint is now ~57 GB per
rollout host.

**`07` gets the same expert coverage on one node.** The nodes are H200s (141 GB), so 8 GPUs hold
1128 GB: 57 GB of weights shards to ~14 GB per GPU on both sides of a 4+4 disaggregated split,
and the 360 GB of Adam state lives in the node's 1.8 TB of host RAM via `--optimizer-cpu-offload`.
The forced change is EP: eight-way expert parallelism does not fit across 4 training GPUs, so `07`
runs EP4 — 32 experts per rank instead of 16. The EP gather and the duplicate-name drop still run,
so it covers the same code. What it does *not* cover is the fabric: with one node the publish dir
and the host-local checkpoint sit on the same box, so use `04`/`05` when the cross-host path is
the question.

### Adding another model

Write a `_model-<name>.sh` block setting `DD_MODEL_CONFIG` (a file in `scripts/models/`),
the checkpoint paths, parallelism, and layout, then a wrapper per transport. Other MoE
checkpoints already on the cluster: `Qwen3-VL-30B-A3B-Thinking` (58 GB) and
`Qwen3-Omni-30B-A3B-Instruct` (66 GB) under `/data/shared/models/`, though neither has a
converted `torch_dist` yet.

### The two directories

| Flag | What it is |
|---|---|
| `--update-weight-disk-dir` | Shared filesystem. Trainer writes one `weight_v{N:06d}/` per sync; every rollout host reads it. |
| `--update-weight-local-checkpoint-dir` | **Host-local** (NVMe if you have it). A full HF checkpoint each engine patches in place. |

Two things to get right:

- **The publish dir is wiped.** The trainer `rmtree`s it when it captures the baseline — a stale
  version stream from a previous run would apply against the wrong base. Point it at per-run
  scratch, never at anything you want to keep. `_common.sh` defaults to
  `$MILES_REPO/checkpoints/$RUN_NAME/delta-updates`.
- **The local dir holds a whole checkpoint per host**, ~8 GB for Qwen3-4B in bf16. `_common.sh`
  defaults to `/tmp/miles-delta-local-ckpt/$RUN_NAME`; override `DD_LOCAL_CKPT_DIR` if `/tmp` is
  small or is not node-local on your nodes. Putting it on the *shared* filesystem defeats the
  entire mechanism.

### Knobs

All read by `_common.sh` from the environment or the wrapper:

| Variable | Default | Notes |
|---|---|---|
| `DD_TRANSFER_MODE` | `disk-delta` | `broadcast` for the control arm |
| `DD_ENCODING` | `xor` | `new ^ old`; smallest and fastest, but an involution — it must land exactly once on the right base. `overwrite` stores positions + absolute values: larger, idempotent. |
| `DD_CHECKSUM` | `xxh3-128` | `blake3` for untrusted storage, `adler32` for interop |
| `DD_DISK_DIR` | `$MILES_REPO/checkpoints/$RUN_NAME/delta-updates` | shared; wiped at baseline |
| `DD_LOCAL_CKPT_DIR` | `/tmp/miles-delta-local-ckpt/$RUN_NAME` | host-local |
| `DD_CHECK_EQUAL` | `0` | `1` adds `--check-weight-update-equal`; on for `01` and `04` |
| `DD_NUM_ROLLOUT` | `3000` | `01` and `04` use 5 |
| `DD_MODEL_CONFIG` | `qwen3-4B` | names a file in `scripts/models/` |
| `DD_HF_MODEL_DIR` / `DD_HF_TORCHDIST_DIR` | Qwen3-4B under `HF_CACHE_DIR` | checkpoint paths |
| `DD_TP` / `DD_PP` / `DD_CP` / `DD_EP` / `DD_ETP` | `2/1/1/1/1` | training parallelism |
| `DD_GPUS_PER_ENGINE`, `DD_MEM_FRACTION` | `2`, `0.85` | rollout engine shape |
| `DD_EXTRA_SGLANG_ARGS`, `DD_EXTRA_OPTIMIZER_ARGS` | unset | arrays appended verbatim |

## Reading the results

The trainer logs per sync and records two metrics the actor drains onto the step log:

```
[disk delta v=3] density=2.87% wire=0.19 GB
```

| Metric | Meaning |
|---|---|
| `perf/update_weights_density` | fraction of weight bytes that changed — the premise. ~1–3% is the expected operating point. |
| `perf/update_weights_wire_bytes` | compressed bytes actually written |

A density near 100% means the diff is finding no reuse and something is wrong with the baseline —
most likely the snapshot and the engine base disagree. The snapshot is seeded from
`--hf-checkpoint` rather than from current GPU weights precisely so they agree even where the
Megatron→HF round trip trims vocab padding on `embed`/`lm_head`.

**Sync #1 publishes nothing.** The first `update_weights()` call only captures the baseline; deltas
start at v1, the second sync. A run of fewer than three syncs measures almost nothing.

## Constraints

Asserted at startup in [`miles/utils/arguments.py`](../../miles/utils/arguments.py):

- **No `--colocate`.** Colocated weights cross via CUDA IPC — only a handle moves — so snapshot +
  diff + encode would be pure overhead. Every recipe here is disaggregated.
- **No PD disaggregation** (`--prefill-num-servers`) — untested.
- **No LoRA** (`--lora-rank > 0`).
- **`--hf-checkpoint` must be a local directory**; the baseline is seeded from its safetensors bytes.

## Object-store filesystems

If your shared storage has no cross-host read-after-write consistency, the writes are not visible to
the engines without an explicit refresh. Two hooks, loaded by import path, keep that vendor logic
out of miles:

- `--custom-update-weight-post-write-path` (trainer) — after a version's files are written, before
  the engines are told to read them.
- `--sglang-custom-pull-weights-pre-read-hook` (engine) — on each host before it reads the version
  directory.

POSIX shared filesystems (NFS, Lustre, Weka) need neither.

## Upstream documentation

slime's [`docs/en/advanced/delta-weight-sync.md`](https://github.com/THUDM/slime/blob/main/docs/en/advanced/delta-weight-sync.md).
Flag names differ — slime splits mode and transport (`--update-weight-mode delta`
`--update-weight-transport disk`) where miles folds both into
`--update-weight-transfer-mode disk-delta` — but the on-disk format and the encodings are the same.
