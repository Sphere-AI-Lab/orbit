# PEFT Allocator Memory Diagnostic — Design

- **Date:** 2026-08-06
- **Status:** approved, pending implementation plan
- **Repo:** this one (`orbit`, branch `feat/lora-without-regret`)
- **Trigger:** the three LoRA arms of E4 gsm8k column 4 die at rollout 2 on 8xH100
  while the FullFT arm of the same column completes 149/149 rollouts on the same node.

## 1. Purpose

Make one number legible: the ~49.9 GB that the Megatron train actor holds as *reserved but
not allocated* under PEFT at the moment SGLang tries to resume.

This design adds **instrumentation only**. It applies no fix. Two hypotheses currently
explain the failure equally well, they imply different fixes, and the counter that
separates them is already computed by PyTorch and simply not read. Choosing a fix before
reading it would be guessing.

## 2. The failure

`results/e4_gsm8k_lr4.jsonl` rows 8-10 — LoRA r1, r16 and r256 at lr 7e-05, all
`status: "failed"`, all `steps: 0`, all `accuracy: 0.0326`. That accuracy is identical
across all three ranks because it is the rollout-0 eval of the *untrained* base model; no
arm survived to produce a trained measurement. Wall-clock 868 s, 813 s and 951 s.

Each dies the same way. SGLang's memory saver cannot re-acquire the pages it released for
the training step:

```
[torch_memory_saver.cpp] cuMemCreate CUDA_ERROR_OUT_OF_MEMORY
[torch_memory_saver.cpp] cudaError error: 2 (out of memory)  file=csrc/core.cpp func=resume line=182
[TP1] Scheduler hit an exception ... SIGQUIT received.
```

The scheduler dies, the server process exits, and the driver surfaces it as
`RemoteDisconnected` / `SGLang server process exited before flush_cache completed`.

The cause is visible in rank 0's memory line across the rollouts it reaches. The allocator's
reserved pool ratchets monotonically while `allocated` stays at ~0.1 GB:

| rollout | allocated | reserved | free at resume | resume |
|---|---|---|---|---|
| 0 | 0.11 GB | 29.76 GB | 33.79 GB | ok |
| 1 | 0.12 GB | 35.45 GB | 28.10 GB | ok |
| 2 | 0.12 GB | 50.01 GB | 13.54 GB | **fails** |

At the failure the training process is holding 50.01 GB of which 0.12 GB is live. SGLang's
static pool is `mem_fraction_static=0.75 x 79.18 = 59.4 GB` and it has 13.54 GB to resume
into.

## 3. Explanations ruled out

Each of these was checked against the logs and rejected. They are recorded because three of
them are the obvious first guesses.

### 3.1 Not the GPU size

The natural reading is that the campaign moved from B200 (178.35 GB) to H100 (79.18 GB) and
the memory budget did not follow. That is true of the hardware and false as the cause:
**the FullFT arm ran 149/149 rollouts to exit code 0 on the same 79.18 GB node**, with free
memory never dropping below 39.84 GB and reserved never above 29.20 GB, measured across all
eight ranks for the whole run. An 80 GB card is sufficient for this workload.

### 3.2 Not the batch-size change

`global_batch_size 1024` is in the argument dump of *both* the B200 run that reached
rollout 113 and the H100 run that died at rollout 2. Commit `9a2a2ad` raised it from the
launcher's 256, and it plausibly moved the fragmentation threshold (see §4), but it is not
itself the discriminator between a run that works and one that does not.

### 3.3 Not the offload-flag pair, despite that being the visible difference

The two arms genuinely take different offload routes, forced by
`_finalize_train_offload_args` (`orbit/utils/arguments.py:2277`), which sets
`offload_train_grad_buffers` and `offload_train_optimizer` to `True` only when PEFT is
disabled:

```
FullFT  H100:  offload_train_grad_buffers=True   offload_train_optimizer=True
LoRA    H100:  offload_train_grad_buffers=False  offload_train_optimizer=False
```

Enabling them for PEFT would nonetheless be the wrong fix, because under PEFT there is
nothing for them to move. Measured on FullFT, rank 0, one mid-run step:

```
before offload model        alloc=23.02  reserved=23.09  free=45.94
after offload grad_buffers  alloc=15.54  reserved=15.61  free=56.01
```

FullFT's `reserved` tracks `allocated` to within 0.07 GB — the pool is tight, and the flag
earns its keep by returning 7.5 GB of genuinely live gradient buffers. Under LoRA
`allocated` is already 0.12 GB; r1's adapter is 2,228,224 parameters, so its Adam state is
~18 MB. The flag would move megabytes against a 49.9 GB problem.

This is worth stating precisely because the same code comment records the *inverse* bug:
FullFT used to die in `torch_memory_saver ... func=resume` on 8xH100 while the LoRA arms
resumed fine, and forcing these two flags on is what fixed FullFT. The polarity has flipped
but the flags are not the mechanism this time.

### 3.4 Not a missing `empty_cache()` call

`offload_megatron_frozen_base_to_cpu` ends with `gc.collect()` followed by
`torch.cuda.empty_cache()` (`orbit/backends/megatron_utils/peft_offload.py:726-727`). It
runs on the PEFT path every rollout, and the 50.01 GB survives it. The call is present and
ineffective, so adding another one is not obviously a fix.

## 4. The two surviving hypotheses

**H1 — Fragmentation.** `empty_cache()` can only release a segment that is *entirely* free.
If the 0.12 GB of live blocks is scattered across many segments, each partially-occupied
segment is pinned in full, and 0.12 GB of live data can strand 49.9 GB of empty pool.
Variable-length rollouts at batch 1024 are an efficient generator of exactly this, which
would also explain the timing relative to `9a2a2ad`.

**H2 — Blocks pending stream release.** The caching allocator defers freeing blocks that
still carry recorded stream uses. `_empty_cuda_cache_if_available`
(`peft_offload.py:96-98`) calls `torch.cuda.empty_cache()` with no preceding
`torch.cuda.synchronize()`, whereas `clear_memory` (`orbit/utils/memory_utils.py:11-15`)
synchronizes first. If the freed blocks are still pending, `empty_cache()` legitimately
skips them and the segments are fully free but unreleased.

The two imply different fixes — `expandable_segments:True` for H1, a synchronizing clear
for H2 — and are distinguished by a single counter.

## 5. The change

`available_memory()` in `orbit/utils/memory_utils.py` returns three additional fields, read
from `torch.cuda.memory_stats(device)`:

| field | source key | role |
|---|---|---|
| `inactive_split_GB` | `inactive_split_bytes.all.current` | load-bearing: decides H1 vs H2 |
| `segments` | `segment.all.current` | context: how far the pool is split |
| `alloc_retries` | `num_alloc_retries` | context: allocator distress |

`inactive_split_bytes` is *defined* as bytes that are free but sit inside a segment still
holding at least one live block. It is H1 stated numerically.

Byte-valued fields go through the existing `_byte_to_gb` and carry the existing `_GB`
suffix; the two counts are dimensionless and take no suffix.

**Every lookup uses `.get(key, 0)`.** `torch.cuda.memory_stats()` returns `{}` for a device
with no allocator activity, and `print_memory` is called during setup before the first
allocation, so a direct subscript raises `KeyError` at startup.

**No new call sites and no new flag.** Every existing `print_memory` picks the fields up,
including the three in `MegatronTrainRayActor.sleep()` that bracket the offload and those in
`wake_up()`. This is deliberate: the B200 runs then report the same counter, which answers
the separate question of whether they carry the same fragmentation harmlessly below
threshold — i.e. whether this is a latent bug on every node or genuinely an 80 GB one.
`memory_stats()` is a host-side read of allocator counters with no device synchronisation,
so attaching it to a log line that is already emitted costs nothing measurable.

## 6. How the result is read

At the `before update_weights` probe of the failing rollout, where
`reserved - allocated ~ 49.9 GB`:

- `inactive_split_GB` ≈ 49 → **H1 confirmed.** Fix is
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the Megatron train actors.
- `inactive_split_GB` ≈ 0 → **H2 indicated.** The bytes are in fully free segments that
  `empty_cache()` declined to release; fix is a synchronising clear on the PEFT offload
  path.
- Anything in between → the pool is mixed; `segments` and `alloc_retries` inform whether to
  escalate to a full `torch.cuda.memory_snapshot()` capture with per-segment occupancy.

## 7. Validation

**CPU, run before the change is called done.** A new `tests/fast/` test — there is no
existing test for `memory_utils.py` — asserting that `available_memory()` returns the three
new keys and that it survives an empty `memory_stats()` dict. Executed locally with pytest.

**GPU, user-launched.** Re-run the failing LoRA arm; it reaches the failure at rollout 2 in
roughly 14 minutes, so no full campaign is needed. The exact command is delivered with the
implementation, not run automatically.

## 8. Non-goals

No fix is applied here. Specifically unchanged: `SGLANG_MEM_FRACTION_STATIC`,
`SGLANG_MAX_TOTAL_TOKENS`, `SGLANG_MAX_RUNNING_REQUESTS`, `GLOBAL_BATCH_SIZE`, the
`offload_train_*` flag defaults, and every protocol constant in `e4_protocol.sh`.

`GLOBAL_BATCH_SIZE` in particular is protocol rather than ops — `e4_protocol.sh` derives it
from `ROLLOUT_BATCH_SIZE x N_SAMPLES_PER_PROMPT` to make the update exactly on-policy, and
the campaign is one comparison across fourteen columns. It must not become node-dependent.

## 9. Follow-up, gated on the measurement

The fix is deliberately left unspecified. Once `inactive_split_GB` is known, §6 selects
between the two candidate fixes, and that fix is a separate change with its own smoke run.
Whichever it is, it is expected to be hardware-independent and to benefit the B200 arms
as well, since nothing in either hypothesis is specific to an 80 GB card.
