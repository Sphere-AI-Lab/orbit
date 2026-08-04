#!/usr/bin/env bash
#
# The E4 protocol, in one file, sourced by every `run_e4_*_lr*_8gpu.sh`.
#
# Fourteen columns run on fourteen separate node bookings, and a learning-rate
# sweep is only a sweep if every arm differs in the learning rate and nothing
# else. Copying these lines into fourteen scripts would make a silent drift
# between two of them indistinguishable, in the results, from a real effect.
#
# Every value is a DEFAULT, not a lock: `NUM_ROLLOUT=234 bash
# scripts/lora_regret/run_e4_gsm8k_lr3_8gpu.sh` works, because `: "${VAR=x}"`
# assigns only when the variable is unset. Whatever you override, override it
# for all fourteen -- the campaign is one comparison.

# --- the update, as the post describes it -----------------------------------
#
# Plain policy gradient with importance sampling and GRPO-style centring: the
# advantage is the reward minus the group mean, and nothing else.
#
# `--disable-grpo-std-normalization` is not a detail. Orbit divides the centred
# reward by the group's standard deviation by default, which for binary rewards
# with group mean p is a factor of 1/sqrt(p(1-p)): about 4x at the measured
# starting reward of 0.02-0.03, falling to 2x as p approaches 0.5. That is a
# schedule on the effective step size, driven by how hard the current problems
# are, laid on top of the constant learning rate that is Figure 6's x-axis.
: "${RL_EXTRA_ARGS=--disable-grpo-std-normalization}"

# Clipping off. With GLOBAL_BATCH_SIZE=256 against 1,024 rollouts the learner
# takes four updates per rollout, so minibatches 2-4 are off-policy and the
# ratio does real work. At small learning rates the drift is tiny and a 0.2 clip
# never binds; at large ones it binds hard -- truncating exactly the updates
# that are supposed to blow the run up, and widening the apparent stable band
# for whichever method takes bigger steps. That band is the claim under test.
#
# 1e9 disables it exactly: the loss is max(-rho*A, -clamp(rho,1-eps,1+eps)*A),
# and a clamp that wide makes the second term identical to the first.
: "${EPS_CLIP=1e9}"
: "${EPS_CLIP_HIGH=1e9}"

# One update per rollout batch: GLOBAL_BATCH_SIZE equals ROLLOUT_BATCH_SIZE=32
# times N_SAMPLES_PER_PROMPT=32, so the learner consumes each iteration's 1,024
# rollouts in a single optimizer step and the importance ratio is identically 1.
#
# This is what makes the clipping-off choice above coherent rather than merely
# aggressive. At the launcher's 256 the learner took FOUR updates per rollout,
# minibatches 2-4 were off-policy, and with EPS_CLIP=1e9 the -rho*A term on
# those minibatches was unbounded. The 2026-08-04 gsm8k column 4 measured the
# consequence: FullFT at 7e-07 -- the BOTTOM of the grid -- ran healthy for 130
# rollouts (raw_reward 0.65, held-out 0.68) and then collapsed in one iteration
# (0.649 at rollout 130, 0.060 at 131, ~0 after; ppo_kl 9.2e-2 at the cliff),
# and LoRA r1 at 7e-05 died the same way at ~97. The collapse is one-way:
# every response runs to the 2,048 cap, truncation loses the \boxed{}, every
# reward in the group is 0, so the centred advantage is 0 and no gradient can
# ever pull the policy back.
#
# On-policy, the ratio never leaves 1 and no clip is needed -- which is the
# post's actual setting, not a softening of it. The cost: 150 optimizer steps
# per arm instead of 600, at 4x the batch. Same tokens, same exposures; the
# LR axis now measures the step size of an on-policy update.
: "${GLOBAL_BATCH_SIZE=1024}"

# --- where the metrics go ------------------------------------------------------
#
# OFFLINE, and synced afterwards from a node that has egress. Not a preference:
# on 2026-08-02 seven arms ran for 90 minutes writing 4-7 MB .wandb files each
# and NOTHING reached the server -- `wandb.Api()` could not find the project at
# all. No retry, no error, no warning in any training log; the uploader simply
# never ran. The compute node had already said why, in Ray's own startup line:
#
#   Failed to determine local IP via external connectivity to: 8.8.8.8:53
#
# The same `mode="shared", x_primary=True` init that logged nothing there logs
# fine from the login node (5/5 history rows, measured both ways), so the
# configuration was never the problem -- the egress was.
#
# What makes offline the answer rather than a workaround: `wandb sync` REPLAYS
# an offline directory into a real run, name and history intact (verified: 5/5
# rows, correct run name). It does NOT do that for a shared-mode directory --
# those replay into a run with config and summary and **zero history rows**,
# which is an empty dashboard, which is exactly what tonight's first attempt
# produced. Offline is the only local format that can be recovered.
#
# Sync with `bash scripts/lora_regret/sync_wandb.sh` from the login node, during
# the run for a near-live view and once more at the end.
: "${WANDB_MODE=offline}"

# --- schedule ----------------------------------------------------------------
#
# 150 rollouts is 4,800 problem-group exposures against the post's ~7,500, so
# the peaks will land below its 0.75 and the absolute numbers are not comparable
# to the published figure. What survives a shortened budget is the SHAPE --
# matched peaks across ranks, a wider LoRA band -- and that is the claim.
#
# It is also the least-measured number in the campaign. One column at
# NUM_ROLLOUT=234 (one full epoch of either dataset, since gsm8k has 7,473
# training problems and math 7,498) would show where the reward curve flattens
# and let the rest be cut with evidence instead of a guess.
: "${NUM_ROLLOUT=150}"

# No checkpoints. `SAVE_INTERVAL=` EMPTY, not a large number: orbit's
# `should_run_periodic_action` short-circuits on `interval is None` and only
# then checks the final rollout, so any non-None interval still writes one
# checkpoint -- 616 s and 15 GB for a FullFT arm, measured. Only the absent flag
# writes none, and the launcher drops the flag when this is empty.
#
# What that costs: nothing downstream reads checkpoints (`analyze` reads
# ledgers, `e5rl` recovers its OFT centre from ledger argmins) and resume does
# not depend on them either, since `--load` points at the base checkpoint and
# `campaign.sh` resumes per arm. The one real loss is re-evaluating a trained
# policy later -- a different grader, pass@k on a held-out set. Set
# SAVE_INTERVAL=999999 for a single end-of-run save if that matters; the LoRA
# adapters cost almost nothing to write (4.5 MB at rank 1) and only FullFT's
# 15 GB is expensive.
: "${SAVE_INTERVAL=}"

# Eval every 25 rollouts: seven passes per arm (before training, then rollouts
# 24/49/74/99/124/149). The headline number is unchanged -- `sweep.py` takes the
# eval with the HIGHEST rollout id, and the final-rollout branch guarantees one
# at 149 regardless of divisibility -- so the intermediate passes are a curve on
# top, not a different measurement.
#
# This was 100000 -- "once, at the end" -- when an eval meant all 6,319 held-out
# prompts of both datasets, ~12% of the campaign. Both inputs to that trade have
# since moved. The per-dataset arms eval only their own split (`arm_env` sets
# EVAL_DATASETS), and the smoke measured that split at ~27 s for gsm8k_test's
# 1,319 prompts against 2.5-5.5 h arms: seven passes cost ~1% on the gsm8k
# panel and a few percent on math's 5,000. And the 2026-08-03 campaign showed
# what end-only measurement cannot see: its collapses (1e-06 at rollout 84,
# 3e-06 at 70) exist only as TRAINING reward traces, with no held-out number
# anywhere in the fall -- and the r16 arm that died at 620 s left nothing at
# all, where an every-25 schedule would have salvaged a partial curve.
: "${EVAL_INTERVAL=25}"

# Response length is deliberately NOT overridden here: the launcher's 2,048
# stands. The 2026-08-02 probe measured 7.1% (gsm8k) and 10.6% (math) truncation
# at 1,024 under this exact rendering, against the plan's 2% gate -- and a
# truncated response has lost its \boxed{...}, so it grades 0 however well it
# argued. Mean response is only ~250-280 tokens, so the cost of the headroom is
# small and the cost of clipping the tail is a reward of zero.

export RL_EXTRA_ARGS EPS_CLIP EPS_CLIP_HIGH GLOBAL_BATCH_SIZE NUM_ROLLOUT SAVE_INTERVAL EVAL_INTERVAL WANDB_MODE
