- [ ] launcher watchdog 改读 upstream FT 事件流后,train.py/train_async.py 的 sentinel 两行才可退役 (2026-07-24 merge 时确认保留)
- [ ] 统一 fully-async 并发旋钮: theirs --async-max-concurrent-samples(绝对数上限) vs 我们 --fully-async-prefetch-batches(流水线深度,联动 staleness) 是不同层概念但喂同一变量; 长期二选一或显式组合; 顺手缩短 --fully-async-max-completed-queue-groups 的 help (2026-07-24)

## Baseline speed references (for the post-sync validation reruns, 2026-07-24)

Stable window = rollout steps 100-200, mean(median):

| ref run | recipe | rollout_time | tok/gpu/s | effective tok/gpu/s |
|---|---|---|---|---|
| M3TRL/baseline/21354 (06-29, pre-sync) | async/geo3k-vlm-mt-fully-async-prefetch2-3node | 40.3s (39.2) | 706 | 617 |
| M3TRL/OPD/27456 (07-19, m10b = frozen baseline source) | OPD/multimodal/baseline 200step | 14.7s (14.5) | 1698 | 1685 |

New runs land in M3TRL/baseline; same perf/* keys (verified present in merged
metrics.py). Compare on the same window once the reruns pass step 200.

## Trainer perf deltas root-caused (2026-07-25, see baseline-validation.html §4)

- actor_train_time +32% / tok_s & tflops ×0.75 = `--recompute-granularity full` finally
  HONORED on the bridge path (#1483; pre-sync silently dropped). Hardware parity confirmed
  (×4/3-corrected tflops 270.6 vs 265.0). log_probs_time invariance = the env is clean.
- KNOB OPEN: dropping --recompute-* from the two baseline recipes reclaims ~25% trainer
  speed (recipes ran without it for months at 4k response len) but costs memory headroom;
  end-to-end benefit ≈0 for async (trainer not bottleneck), ≈ +13% for the OPD pipeline.
  Decide when the OPD cadence starts to matter.
- tflops across the sync boundary is NON-comparable (factor-3 convention doesn't credit
  the recompute forward). Compare ×4/3-corrected values instead.
- async data_preprocess 7.0→0.6s = June-era reference env artifact (torch 2.9.1 /
  transformers 4.x / old ray shard transfer); OPD pair shows parity (1.03≈1.02s). Closed.
