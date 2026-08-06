# PEFT Allocator OOM — Symptoms, Mechanism, Fix

**Status:** resolved. Fix committed `1de85f1` (2026-08-06), verified on B200 (2026-08-07).

This is the resolution of [2026-08-06-peft-allocator-memory-diagnostic-design.md](2026-08-06-peft-allocator-memory-diagnostic-design.md),
whose §9 deferred the fix until the diagnostic counters were read. They were read;
hypothesis H2 (fragmentation) is confirmed and H1 is dead. Read this document first if you
are looking at a failing PEFT arm; read the design document if you want the hypotheses that
were considered and rejected.

---

## 1. One paragraph

A LoRA or OFT training step frees nearly all its GPU memory, but PyTorch's caching allocator
cannot return it to the driver, because each of its ~15 segments still holds a few MB of
live adapter state and `empty_cache()` only releases a segment that is *entirely* free. The
colocated SGLang engine then has no physical memory to resume into and dies. The fix is one
environment variable — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set on PEFT train
actors only — which makes the allocator map physical pages on demand, so a live block pins
pages instead of a whole segment.

---

## 2. Symptoms

The failure surfaces in three places at three levels of usefulness. You will almost always
see the least useful one first.

### 2.1 In the ledger

Arms land as `status: "failed"` with `steps: 0` and an accuracy identical across every rank,
because that number is the rollout-0 eval of the *untrained* base model:

```
lora-r1-all-gsm8k-lr7e-05-s0     failed  steps=0  accuracy=0.032600454890068235
lora-r16-all-gsm8k-lr7e-05-s0    failed  steps=0  accuracy=0.032600454890068235
lora-r256-all-gsm8k-lr7e-05-s0   failed  steps=0  accuracy=0.032600454890068235
```

An identical accuracy across three ranks is the tell. Ranks that actually trained cannot
agree to sixteen decimal places.

Failures later in the run look different — `steps: 24`, `49`, `99` with plausible
accuracies — because the arm died partway. Those rows are **not usable as data points**: the
step counts differ, so any comparison between them is confounded with training length.

### 2.2 In the driver log — the corpse report

The training driver talks to the engine over HTTP, so when the engine process dies the
driver reports a *network* error. Any of these, all on `/resume_memory_occupation` or
`flush_cache`, mean the same thing:

```
requests.exceptions.ConnectionError: ('Connection aborted.',
    RemoteDisconnected('Remote end closed connection without response'))
requests.exceptions.ConnectionError: ('Connection aborted.',
    ConnectionResetError(104, 'Connection reset by peer'))
requests.exceptions.ConnectionError: HTTPConnectionPool(host=..., port=15000):
    Max retries exceeded with url: /resume_memory_occupation
    (Caused by NewConnectionError(... [Errno 111] Connection refused))
RuntimeError: SGLang server process exited before flush_cache completed (172.22.8.10:15006).
```

**These are not network faults.** Do not go looking at ports, Ray, or the fabric. The engine
is a separate OS process that has already exited; the connection error is what the driver
notices a moment later.

### 2.3 In the engine log — the actual cause

The real error is printed by the engine's memory saver, and is the only message that names
the problem:

```
[torch_memory_saver.cpp] cuMemCreate CUDA_ERROR_OUT_OF_MEMORY
    (may not be an issue e.g. torch allocator will free cache and retry)
[torch_memory_saver.cpp] cudaError error: 2 (out of memory)
    file=csrc/core.cpp func=resume line=182
[TP1] Scheduler hit an exception: Traceback (most recent call last):
SIGQUIT received.
```

`func=resume` is the diagnostic word. The engine is trying to re-map the physical pages it
released before the training step, and the driver says there are none.

Occasionally the OOM is raised on the PyTorch side instead and prints the full accounting,
which is worth reading because it names the fix in its own error text:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.02 GiB.
GPU 4 has a total capacity of 79.18 GiB of which 835.81 MiB is free.
Process 3736537 has 522.00 MiB memory in use.
Process 3955547 has 49.48 GiB memory in use.
... If reserved but unallocated memory is large try setting
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
```

Two PIDs on one GPU is the colocation. 49.48 GiB in the other process is the hoard.

---

## 3. Mechanism

### 3.1 The cycle

Training and inference share the same cards. Each rollout runs:

```
engine generates -> engine offloads (kv_cache, weights, cuda_graph)
  -> train actor wakes up, onloads frozen base, takes a step, offloads
    -> engine resume_memory_occupation()      <-- fails here
```

### 3.2 What a segment is, and why partial occupancy is fatal

PyTorch does not call `cudaMalloc` per tensor — that is far too slow and it synchronizes.
The caching allocator takes large contiguous chunks from CUDA and sub-allocates tensors out
of them. `c10/cuda/CUDACachingAllocator.h` names the two levels exactly:

```c
// Struct containing info of an allocation block (i.e. a fractional part of a cudaMalloc)..
struct BlockInfo { ... };

// Struct containing info of a memory segment (i.e. one contiguous cudaMalloc).
struct SegmentInfo { ... };
```

A **segment is one `cudaMalloc`**; a **block is a slice of one**. Blocks split when you
allocate and coalesce with their neighbours when you free — but only *within* their own
segment. Two adjacent segments never merge; they came from separate `cudaMalloc` calls. In
the failing r1 run there were 15 segments totalling 34.25 GB, averaging ~2.3 GB apiece,
consistent with the large tensors (base weight shards, activations) that created them.

**`cudaFree` frees the whole `cudaMalloc`.** There is no API to release part of an
allocation — the same constraint as C's `free()`, where you cannot free the second half of a
buffer. So you cannot return 2.29 GB of a 2.3 GB segment. That is the hard limit
`empty_cache()` runs into; its only tool is, again from the header:

```c
SEGMENT_FREE,  // a call to cudaFree to return memory to the OS (e.g. to
               // defragment or empty_caches)
```

For that call to be legal the segment must be one single free block end to end. One
surviving 5 MB adapter block anywhere inside makes the whole 2.3 GB unreturnable. **The
effect is binary, not proportional** — the size of the survivor is irrelevant, only its
existence. A one-page block and a one-gigabyte block have identical holding power.

### 3.3 Two different things are called "freeing"

This distinction is the whole bug.

1. **PyTorch free.** The block goes back on the caching allocator's free list. The segment
   containing it stays mapped in this process.
2. **Driver release.** `empty_cache()` walks the segment list and `cudaFree`s segments —
   but **only those that are entirely free**, per §3.2. This is the step another process
   depends on.

Offloading the base does (1) and not (2). The memory is simultaneously "free" from PyTorch's
point of view and "in use" from the driver's. Within a single process nothing is wrong: the
next rollout reuses those segments with no waste. It is only visible because the engine is a
*different process*, and a process can only take memory the driver considers unallocated.

### 3.4 The counters

Rank 0, LoRA r1, 8×B200, one rollout, before the fix:

| phase | allocated | reserved | inactive_split | segments |
|---|---|---|---|---|
| after wake_up frozen_base | 15.04 GB | 34.25 GB | 19.22 GB | 15 |
| after offload frozen_base | 0.08 GB | 34.25 GB | 34.17 GB | 15 |
| before update_weights | 0.08 GB | 34.25 GB | 34.17 GB | 15 |

The 15 GB frozen base is genuinely freed — `allocated` drops to 0.08 GB. `reserved` does not
move. The freed memory migrates into `inactive_split`, the stat for free memory trapped
inside a segment that still has other content. At the end, 34.17 of 34.25 GB — 99.8% — is
unreleasable across 15 segments, so roughly 5 MB of straggler blocks is pinning ~2.3 GB
apiece. `active − allocated` measured 0.00, which rules out the other candidate: nothing was
merely awaiting stream release. It is fragmentation alone.

On the 80 GB H100 the pool ratchets until resume has nowhere to go:

| rollout | allocated | reserved | free at resume | resume |
|---|---|---|---|---|
| 0 | 0.11 GB | 29.76 GB | 33.79 GB | ok |
| 1 | 0.12 GB | 35.45 GB | 28.10 GB | ok |
| 2 | 0.12 GB | 50.01 GB | 13.54 GB | **fails** |

SGLang's static pool is `mem_fraction_static=0.75 × 79.18 = 59.4 GB`; 13.54 GB is not enough.

### 3.5 Why something so small has so large an effect

The adapter is tiny and that is exactly the problem. LoRA r1 has 2,228,224 params — bf16
weights, fp32 master, grads and two Adam moments come to roughly 40 MB, and the measured
live residual across 54 samples is 0.07–0.10 GB. Nothing about LoRA is 34 GB.

What *is* 34–66 GB is the pool those crumbs sit in, and it was sized by something else: the
frozen base (15 GB) plus the training step's activations and logits. Peak `reserved` reached
65.71 GB while the largest *sampled* `allocated` was 15.05 GB — the memory logs fire at
phase boundaries, never inside the forward/backward, so the extra ~50 GB is a transient that
lives and dies between two samples.

Combine that with §3.2's binary release rule and the leverage is roughly 650×: ~100 MB live
pinning 65.71 GB reserved.

The adapter's tensors end up spread across segments rather than packed into one because they
are allocated interleaved with the big transients and outlive them. (That the distribution
is one-or-more per segment is inferred from the counters — 15 segments, all retained, 100 MB
live. No per-segment snapshot was taken.)

### 3.6 Why full fine-tuning is immune

The mirror image: what stays resident is the same order of magnitude as what leaves, so
segments refill densely instead of leaving crumbs. `_finalize_train_offload_args` forces the
grad-buffer and optimizer offloads on for full FT, and those return genuinely live state
every step. Measured on the same node, its `reserved` tracks `allocated` to within 0.07 GB.

The pathology needs a large transient and a *tiny* persistent — the shape PEFT has by
construction. Note that a bigger adapter does not help: r256 has 256× the params and died
the same way, because 570M params is still small next to 65 GB of transient.

---

## 4. The fix

[`orbit/ray/actor_group.py`](../../../orbit/ray/actor_group.py), in `_build_train_actor_env`:

```python
if getattr(args, "peft_method", "none") != "none":
    env_vars.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    )
```

`expandable_segments:True` splits the single `cudaMalloc` of §3.2 into two independent
things: **virtual address space**, reserved once and generously (`cuMemAddressReserve` —
virtual space is not scarce on 64-bit, so over-reserving costs nothing), and **physical
pages**, created and mapped into that range on demand (`cuMemCreate` + `cuMemMap`).

That gives the allocator a release action it did not previously have. The two enum entries
following `SEGMENT_FREE` in the same header state the whole fix in four words:

```c
SEGMENT_MAP,    // a call to cuMemMap (used with expandable_segments)
SEGMENT_UNMAP,  // unmap part of a segment (used with expandable segments)
```

`unmap part of a segment` is precisely what `cudaFree` cannot do. When the 15 GB base is
freed its pages are unmapped and returned to the driver, while the adapter's pages stay
mapped wherever they happen to sit. The granularity of release drops from a ~2.3 GB segment
to a page, `reserved` tracks `allocated` instead of ratcheting to a high-water mark, and
"mostly empty" finally translates into "mostly returned".

**Note what this is not.** Nothing is moved, grouped, or given a dedicated region. The
adapter's tensors remain exactly as scattered as before; their location simply stops
mattering once every page is independently releasable.

The alternative — packing long-lived small tensors into their own pool, via
`torch.cuda.MemPool` or indirectly via `max_split_size_mb`, so they never land inside
segments carved for transients — was rejected. It requires knowing at every allocation site
which tensors are long-lived, which means changes inside Megatron's optimizer and the PEFT
wrappers, and it breaks silently the moment a new persistent allocation is added somewhere
nobody tagged. Expandable segments solve the same problem at the allocator, with one
environment variable, no change to any allocation site, and automatic coverage of OFT. The
cost is a less battle-tested allocator path, which is why the change is scoped to PEFT
actors rather than applied globally.

Four decisions in those four lines, each of which would be a bug if made the other way:

- **An environment variable, not a runtime call.** `PYTORCH_CUDA_ALLOC_CONF` is read when the
  caching allocator initializes; setting it after the first allocation is too late. It
  reaches the actor process at birth via `runtime_env={"env_vars": env_vars}` where the
  actor class is decorated.
- **Train actor, not engine.** The engine was the victim, not the hoarder. Giving *it*
  expandable segments would not conjure free physical memory.
- **`peft_method != "none"`, not `== "lora"`.** OFT takes the identical frozen-base offload
  path and was measured hitting the same resume OOM. A LoRA-only predicate would have
  silently excluded it. There is a test pinning this.
- **`setdefault` wrapping `os.environ.get`.** Precedence is `train_env_vars` > shell > default,
  so the old behaviour can still be pinned from the environment to A/B against.

Full fine-tuning is deliberately left alone. It has no gap to close, and its arms were
already producing completed 149/149 runs whose allocator behaviour there was no reason to
perturb mid-campaign.

---

## 5. Verifying it

### 5.1 Unit

```
python -m pytest tests/fast/utils/test_train_actor_allocator_env.py -q
```

Five tests: PEFT gets the variable, OFT counts as PEFT, full FT does not get it, an explicit
shell setting wins, an explicit `train_env_vars` setting wins.

### 5.2 On hardware where the crash cannot reproduce

This is the case that matters in practice: on a 180 GB B200 there is enough headroom that the
engine never fails to resume, so **absence of a crash proves nothing**. Check the mechanism
instead. The fragmentation gap is printed every rollout and is independent of card size.

Pull rank-0 `Memory-Usage` lines from the arm's log and compare `reserved` against
`allocated`:

```bash
grep "Memory-Usage.*'gpu': '0'" logs/lora_regret/<arm>.log | tail -20
```

The cleanest comparison is `after wake_up frozen_base`, because `allocated` is identical on
both sides — the same 15 GB base is resident — so the entire difference is the gap:

| | allocated | reserved | gap | inactive_split | segments |
|---|---|---|---|---|---|
| before fix (2026-08-06 04:46) | 15.04 GB | 34.25 GB | **19.21 GB** | 19.22 GB | 15 |
| after fix (2026-08-07 00:47) | 15.04 GB | 15.11 GB | **0.07 GB** | 0.00 GB | 0 |

Same arm (LoRA r1), same rank, same phase, same node class, same log file. Post-fix the pool
tracks live memory to 0.07 GB *while the model is resident*, which is also the check that you
are not merely measuring an idle actor.

At the offload phases, where only adapter state remains, the same holds with everything
scaled down:

| | allocated | reserved | gap | inactive_split | segments |
|---|---|---|---|---|---|
| before fix | 0.08 GB | 34.25 GB | **34.17 GB** | 34.17 GB | 15 |
| after fix | 0.01 GB | 0.02 GB | **0.01 GB** | 0.00 GB | 0 |

**Pass criterion:** `reserved − allocated` under ~1 GB at the offload phases, and
`inactive_split` at or near 0. A gap of tens of GB means the variable did not reach the actor
process — check `peft_method` is set and that nothing in `train_env_vars` or the shell is
overriding it.

### 5.3 End to end

Not yet complete at the time of writing. The pre-fix arms died at rollout 2 (H100) and
rollouts 24–99 (B200); the longest post-fix observation is a 10-rollout smoke plus rollout
20 of 149 on a live r1 column, where the gap is holding at 0.07 GB. The fragmentation this
addresses was fully present at rollout 1 before the fix, so there is no known reason for it
to re-emerge later, but sustained behaviour over a full 150-rollout arm is unproven.

The thing to watch is that `reserved` stays flat across rollouts rather than ratcheting. The
pre-fix signature was monotonic growth (29.76 → 35.45 → 50.01 GB over three rollouts on
H100); a post-fix run should show the same number every rollout.

---

## 6. If it comes back

Work the list in this order; the first two are far more likely than the rest.

1. **The variable did not reach the actor.** Check the `Memory-Usage` gap first (§5.2). A
   large gap means the fix is not active, which is a different problem from the fix not
   working.
2. **A shell or config override.** `setdefault` means an explicit `PYTORCH_CUDA_ALLOC_CONF`
   in `train_env_vars` or the environment wins by design. Verify nothing sets it.
3. **The gap is small but the engine still OOMs.** Then the pool is genuinely too large for
   the card, not fragmented, and this document does not apply — look at
   `mem_fraction_static`, the rollout batch size, or TP degree.
4. **A new PEFT method that does not set `peft_method`.** The predicate keys on that
   attribute; a method that leaves it `"none"` inherits full FT's treatment and this bug.

---

## 7. Provenance

| | |
|---|---|
| Investigation | [2026-08-06-peft-allocator-memory-diagnostic-design.md](2026-08-06-peft-allocator-memory-diagnostic-design.md) |
| Implementation plan | [../plans/2026-08-06-peft-allocator-memory-diagnostic.md](../plans/2026-08-06-peft-allocator-memory-diagnostic.md) |
| Counter instrumentation | `b40f077`, `6993e5a` |
| Fix | `1de85f1` |
| Affected arms | `results/e4_gsm8k_lr4.jsonl` — LoRA r1/r16/r256 at 7e-05 |
| Pre-fix evidence | `logs/lora_regret/lora-r{1,16,256}-all-gsm8k-lr7e-05-s0.log`, 2026-08-05 / 2026-08-06 |
| Post-fix evidence | same files, 2026-08-07; `results/smoke/e4_smoke_expandable.jsonl` |
