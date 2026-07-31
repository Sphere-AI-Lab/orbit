"""The LoRA-without-regret experiment matrices.

Two LR grids, because the LoRA and FullFT optima sit a decade apart and one
shared grid would spend most of its points where nothing happens.

Six matrices, selected by ``sweep.py --matrix``:

* ``sft82`` (:func:`sft_arms`) -- the original 82-arm LoRA/OFT matrix, on 7-point
  grids that *bracket* the published optima. Kept byte-for-byte because the gate
  log records its dry run. Its OFT arms are superseded by ``e5``: they solve the
  block size from the square attention shape (all-modules lands at parameter ratio
  0.75) and put 35 of 40 on LoRA's LR grid, which the plan says is not justified
  for a rotation parameterization.
* ``e1`` / ``e2`` / ``e3`` -- the campaign matrices of
  ``docs/superpowers/plans/2026-07-28-lora-without-regret-experiments.md``, on
  5-point 0.3-decade grids *centred* on the post's own predictions. Centring is
  the point: a confirmation is then a hit rather than a fit.
* ``e4`` -- RL (C5). Scored by accuracy, not NLL, and driven through the RL
  launcher; half-decade spacing because the post gives no LR multiplier for policy
  gradient and C5's second half is about the *width* of the performant band.
* ``e5scout`` / ``e5`` -- matched-parameter OFT. The scout comes first and the
  refinement grid is centred on its argmin; see :func:`e5_arms` for why the match
  is solved by inverting to a LoRA rank rather than by choosing a block size.

The two grid styles are deliberately not unified. Bracketing answers "where is
the optimum", centring answers "is the optimum where the post says".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from orbit.utils.peft_param_match import (
    ATTENTION_MODULES,
    lora_param_count_for_modules,
    matched_mlp_rank,
    matched_oft_block_size,
    megatron_module_shapes,
    oft_block_size_matching_params,
    oft_lora_match_report,
    oft_param_count_for_modules,
)
from orbit.utils.peft_param_match import MLP_MODULES as PEFT_MLP_MODULES

ALL_MODULES = "linear_qkv,linear_proj,linear_fc1,linear_fc2"
ATTN_MODULES = "linear_qkv,linear_proj"
MLP_MODULES = "linear_fc1,linear_fc2"

# Brackets every published LoRA optimum (1.2e-4 .. 3.5e-4) with >=2 points a side.
LORA_LR_GRID = [5e-5, 8e-5, 1.2e-4, 2e-4, 3e-4, 5e-4, 8e-4]
# Same shape, one decade down; brackets the FullFT optimum 2.5e-5.
FULL_LR_GRID = [5e-6, 8e-6, 1.2e-5, 2e-5, 3e-5, 5e-5, 8e-5]
# OFT's natural LR scale is unknown a priori: it parameterizes a rotation, not
# an additive update. Scout wide, then refine around the argmin.
OFT_SCOUT_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

LORA_ALPHA = 32
# LORA_A_INIT_METHOD is fixed at "kaiming" for the whole sweep, never
# "uniform" -- orbit/utils/arguments.py registers
# choices=["xavier","normal","kaiming","zero"], so "uniform" is rejected by
# argparse outright. Orbit's own default is "xavier"; PEFT-compatible init is
# "kaiming", and the two differ by ~2.4x in std (see the launcher's comment),
# which shifts the measured optimal LR, so this is pinned rather than left to
# the launcher's own default.
LORA_A_INIT_METHOD = "kaiming"

# Centres for the campaign grids (E1-E3). FULL_LR_CENTRE is the post's own
# FullFT prediction; LORA_LR_CENTRE is exactly 10x it, which is C2's claim built
# into the grid instead of fitted out of it.
FULL_LR_CENTRE = 2.5e-5
LORA_LR_CENTRE = 2.5e-4
# RL runs an order of magnitude lower than SFT, and the post gives no multiplier
# for policy gradient -- these are the RL launcher's own documented defaults,
# used as scout centres rather than as predictions.
RL_FULL_LR_CENTRE = 1e-6
RL_LORA_LR_CENTRE = 1e-5
# One Tulu3 epoch is (939,343 - 1,000 held out) / 32 = 29,323 optimizer steps,
# and ~1% of that is 293 -- about 100 trace points, which is what C1's departure
# detector needs, for ~1.9 h of eval against ~70 h of training. At the
# launcher's default of 10 the same arm would spend ~55 h evaluating.
E1LONG_EVAL_INTERVAL = 293
E1LONG_RANKS = (1, 4, 16, 64, 128, 256, 512)
# OpenThoughts3's 10,000-row subset is 312 optimizer steps at batch 32, and ~1%
# of that is 3 -- about 100 trace points. (The launcher ceilings: (10000+31)//32
# = 313.) The contrast with E1LONG_EVAL_INTERVAL (293) is the whole reason e1ot
# needs no separate long matrix: one epoch here is affordable at all 40 arms,
# and one epoch on Tulu3 is not.
E1OT_EVAL_INTERVAL = 3
# Llama-3.1-8B's fused QKV width: (32 query + 8 key + 8 value heads) * 128 head
# dim = 6144. Needed for the matched-parameter attention/MLP pair in E3, and not
# derivable from hidden_size alone under GQA.
LLAMA31_8B_QKV_OUTPUT = 6144
# Defaults for the builders that now need shapes to solve an OFT block size.
# Every matrix in this module is single-model; `sweep.py` passes the registry's
# values explicitly, and these keep the builders callable bare from tests and
# from a REPL. Pinned against tools/lora_regret/models.py by
# test_the_arms_module_defaults_match_the_registry.
LLAMA31_8B_HIDDEN = 4096
LLAMA31_8B_FFN = 14336
# Where tools/lora_regret/prepare_data.py writes its splits.
DATA_DIR = "/lustre/fast/fast/groups/ei-slm/data/lora_regret"
# E4's training file is the MATH+GSM8K concatenation (`--dataset rl_mix`), which
# has no matching single test split: the RL launcher evaluates math_test and
# gsm8k_test separately so per-dataset accuracy stays visible instead of being
# averaged away. So `arm_env` must not export a TEST_JSONL for it.
RL_MIX_DATASET = "math_gsm8k"
DATASETS_WITHOUT_TEST_SPLIT = frozenset({RL_MIX_DATASET})


# Where an unscouted OFT cell looks for its optimum. OFT parameterizes a
# rotation rather than an additive update, so nothing about LoRA's optimal LR
# transfers to it -- not the value, not the decade. Until `e5scout` has run,
# every OFT cell in every matrix is a *scout* across these two decades and its
# arms are named `oftscout` so no reader can mistake one for a measurement.
# `sft82` put 35 of its 40 OFT arms on LoRA's own grid; the module docstring
# above calls that unjustified, and this is what replaces it.
OFT_SCOUT_SPAN = (1e-5, 1e-3)
# RL runs about a decade below SFT for both FullFT and LoRA, and OFT has never
# been scouted in either regime -- so the RL scout is the SFT span shifted by
# that decade. That shift is an assumption, which is exactly why these arms are
# named `oftscout` and not `oft`.
RL_OFT_SCOUT_SPAN = (1e-6, 1e-4)


def lr_grid(centre: float, n: int = 5, step_decades: float = 0.3) -> list[float]:
    """`n` learning rates spaced `step_decades` apart, with `centre` inside.

    For odd `n` the centre sits in the middle; for even `n` it sits at index 1,
    so a 4-point grid still has a point below the prediction. Values are rounded
    to three significant figures, which keeps arm names stable and readable at
    the cost of the spacing being 0.3 decades to within ~0.3%.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    low = -(n // 2) if n % 2 else -1
    exponents = [low + i for i in range(n)]
    grid = [centre * 10 ** (step_decades * k) for k in exponents]
    return [float(f"{lr:.3g}") for lr in grid]


@dataclass(frozen=True)
class Arm:
    name: str
    method: str  # "full" | "lora" | "oft"
    rank: int | None
    oft_block_size: int | None
    target_modules: str
    lr: float
    seed: int
    # E2 varies the batch size; E1/E3 leave it at the launcher's own default.
    global_batch_size: int | None = None
    # Realized OFT-to-LoRA parameter ratio for a matched pair (E5 only). Carried
    # on the arm so the ledger records how well the "matched" claim actually held
    # for the arm that ran, rather than for the arm that was intended.
    matched_ratio: float | None = None
    # Which prepare_data.py split pair to train on. None means "whatever the
    # launcher defaults to" (tulu3).
    dataset: str | None = None
    # E1-2 only. The long curves must run a full Tulu3 epoch, and the launcher
    # derives that itself -- but only if NUM_ROLLOUT is unset or empty.
    full_epoch: bool = False
    # Explicit so the long curves get ~100 trace points instead of the
    # launcher's default of 10, which would cost 37 h of eval per arm.
    eval_nll_interval: int | None = None
    # Which base model this arm runs on. Defaults to the campaign's original
    # anchor so every pre-registry matrix serializes byte-identically and every
    # ledger written before the registry existed stays valid.
    model: str = "llama3.1-8b"
    # An explicit rollout cap. `full_epoch` is the opposite request and wins if
    # both are set: E1-2's arms must re-derive the epoch even if a stale
    # NUM_ROLLOUT is exported in the operator's shell.
    num_rollout: int | None = None


def oft_lr_values(
    centre: float | None,
    n: int,
    step_decades: float = 0.3,
    scale: float = 1.0,
    span: tuple[float, float] = OFT_SCOUT_SPAN,
) -> tuple[list[float], bool]:
    """An OFT cell's learning rates, and whether they are a scout.

    With a centre from `e5scout`, this mirrors the LoRA cell it sits beside --
    same width, same spacing -- so the two are compared on equal grids. Without
    one it is `n` log-spaced points across `span`, which is a search rather than
    a measurement; the caller names those arms `oftscout`.

    `n` always mirrors the LoRA cell so an OFT cell cannot be quietly cheaper
    (fewer points, so a worse argmin) or finer than what it is compared against.
    """
    if centre is not None:
        return [lr * scale for lr in lr_grid(centre, n=n, step_decades=step_decades)], False
    low, high = span
    step = (math.log10(high) - math.log10(low)) / (n - 1)
    return [float(f"{low * 10 ** (step * i) * scale:.3g}") for i in range(n)], True


# Block sizes an OFT cell may use. Powers of two because Megatron-Bridge's
# `OFTRotationModule` snaps whatever it is given to a divisor of each layer's own
# `d_in`, and every shape here is a power of two times a small factor.
OFT_BLOCK_CANDIDATES = tuple(2**k for k in range(3, 14))  # 8 .. 8192

# The largest OFT block SGLang's rotation kernels can launch.
#
# Was 128. Every rotation kernel staged the whole BS x BS block in shared
# memory, against sm_90's 232,448 B limit, so it could not launch at all above
# 128 -- which is how every RL OFT arm died in the 2026-07-31 coverage probe.
#
# It took TWO commits, and the first one alone is not enough:
#   893f329a2  the fused QKV / gate_up kernel (fused_rotate_project)
#   166041d28  the un-fused pair (gemm_oft_r, sgemm_oft_r), which o_proj and
#              down_proj take because they have nothing to fuse into
# After the first, a --target all arm still died on every layer at
# `Required: 2228224`. Raising this constant is only valid against a package
# containing both.
#
# Verified on an H100 through the installed package: BS 16..1024 all launch.
# The un-fused pair is BIT-IDENTICAL to the untiled original on all 40
# configurations the original could run; the fused kernel matches to 1.2e-04
# against a 2e-3 bar.
#
# 1024 rather than "unbounded": that is the largest block the campaign's
# matched-parameter arithmetic ever asks for (LoRA r256 all-modules on
# Llama-3.1-8B), and a cap that has been measured is worth more than one that
# has not.
OFT_MAX_BLOCK_SGLANG = 1024


def matched_oft_block(
    rank: int,
    modules: str,
    hidden_size: int,
    ffn_size: int,
    qkv_output_size: int = LLAMA31_8B_QKV_OUTPUT,
    max_block: int | None = None,
) -> tuple[int, dict]:
    """The OFT block nearest LoRA `rank` here, and the match it actually achieves.

    Solved in E5's direction -- **fix the block, solve for the rank** -- and not
    the other way round, because the block lattice is provably too coarse to
    invert: on Llama-3.1-8B all-modules, block 1024 carries 0.764 of LoRA r256's
    parameters and the next block up carries 1.529. There is no block that
    matches r256, so asking for one and taking the nearest silently produces a
    24%-undersized adapter and calls it matched.

    So this picks the block whose *implied* LoRA rank is closest to `rank` in log
    space, and hands back `oft_lora_match_report`'s own accounting. The report's
    `ratio` is near 1 by construction (block against its own implied rank); the
    caller stores it on the arm, so the ledger records the pairing that actually
    ran rather than the one that was intended.

    Solved against *this cell's own module shapes*, not the square attention
    shape: OFT's parameter count follows `d_in`, and the MLP's `d_in` sum is
    larger -- reusing attention's block would compare method and capacity at
    once, the confound E3 exists to avoid, one method over.
    """
    shapes = megatron_module_shapes(hidden_size, ffn_size, qkv_output_size)
    selected = {name: shape for name, shape in shapes.items()
                if name in [m.strip() for m in modules.split(",") if m.strip()]}
    if not selected:
        raise ValueError(f"no known module in {modules!r} (known: {sorted(shapes)})")
    candidates = OFT_BLOCK_CANDIDATES
    if max_block is not None:
        candidates = tuple(b for b in candidates if b <= max_block)
        if not candidates:
            raise ValueError(
                f"no OFT block size at or below {max_block}; "
                f"candidates are {OFT_BLOCK_CANDIDATES}"
            )
    best: tuple[float, int, dict] | None = None
    for block in candidates:
        report = oft_lora_match_report(block, selected)
        implied = report["lora_rank"]
        if implied < 1:
            continue
        error = abs(math.log(implied / rank))
        if best is None or error < best[0]:
            best = (error, block, report)
    if best is None:
        raise ValueError(
            f"no OFT block size reaches LoRA rank {rank} on {modules!r}; "
            f"tried {candidates}"
        )
    return best[1], best[2]


def _oft_cell(
    rank: int,
    modules: str,
    hidden_size: int,
    ffn_size: int,
    seed: int,
    dataset: str,
    centre: float | None,
    n: int,
    *,
    step_decades: float = 0.3,
    scale: float = 1.0,
    span: tuple[float, float] = OFT_SCOUT_SPAN,
    qkv_output_size: int = LLAMA31_8B_QKV_OUTPUT,
    extra: str = "",
    max_block: int | None = None,
    **arm_kwargs,
) -> list[Arm]:
    """One OFT cell near `rank` on `modules`, at this matrix's cell width.

    Every arm carries the realized `matched_ratio`, so a reader can see how well
    the pairing held for the arm that ran rather than for the one intended.
    """
    block, report = matched_oft_block(
        rank, modules, hidden_size, ffn_size, qkv_output_size, max_block=max_block
    )
    lrs, scouting = oft_lr_values(centre, n, step_decades=step_decades, scale=scale, span=span)
    label = "oftscout" if scouting else "oft"
    return [
        Arm(_name(label, f"b{block}", modules, lr, seed, extra=extra),
            "oft", None, block, modules, lr, seed, dataset=dataset,
            matched_ratio=report["ratio"], **arm_kwargs)
        for lr in lrs
    ]


def _name(method: str, tag: str, modules: str, lr: float, seed: int, extra: str = "") -> str:
    short = {ALL_MODULES: "all", ATTN_MODULES: "attn", MLP_MODULES: "mlp"}.get(modules, "na")
    parts = [method, tag, short]
    if extra:
        parts.append(extra)
    parts += [f"lr{lr:g}", f"s{seed}"]
    return "-".join(parts)


def sft_arms(hidden_size: int, ffn_size: int, seed: int = 0) -> list[Arm]:
    """The 82-arm SFT matrix: 42 LoRA/FullFT plus 40 OFT (5 scout + 5x7).

    `ffn_size` is accepted (not just `hidden_size`) to keep the signature
    stable for a future per-module OFT match -- MLP's `linear_fc2` has
    `d_in == ffn_size`, not `hidden_size` -- but today's matched block size is
    deliberately solved against the square attention shape only (one shared
    `OFT_BLOCK_SIZE` knob per arm; Megatron-Bridge's `OFTRotationModule`
    silently snaps it to a divisor of each layer's own `d_in`, so the MLP
    layers still end up with a valid, if not perfectly matched, block size).
    See `orbit.utils.peft_param_match`'s module docstring for the accounting.
    """
    if hidden_size <= 0 or ffn_size <= 0:
        raise ValueError(f"hidden_size and ffn_size must be positive, got {hidden_size}, {ffn_size}")

    arms: list[Arm] = []

    for lr in FULL_LR_GRID:
        arms.append(Arm(_name("full", "na", "", lr, seed), "full", None, None, "", lr, seed))

    lora_configs = [
        (256, ALL_MODULES),
        (256, ATTN_MODULES),
        (256, MLP_MODULES),
        (16, ALL_MODULES),
        (1, ALL_MODULES),
    ]
    for rank, modules in lora_configs:
        for lr in LORA_LR_GRID:
            arms.append(
                Arm(_name("lora", f"r{rank}", modules, lr, seed), "lora", rank, None, modules, lr, seed)
            )

    # Matched OFT. Block size is solved against the square (attention) shape so
    # all arms share one OFT_BLOCK_SIZE; per-layer snapping handles the rest.
    oft_configs = [
        (1, ALL_MODULES),
        (16, ALL_MODULES),
        (256, ALL_MODULES),
        (16, ATTN_MODULES),
        (16, MLP_MODULES),
    ]
    scout_block = matched_oft_block_size(16, hidden_size, hidden_size)
    for lr in OFT_SCOUT_GRID:
        arms.append(
            Arm(_name("oftscout", f"b{scout_block}", ALL_MODULES, lr, seed),
                "oft", None, scout_block, ALL_MODULES, lr, seed)
        )
    for rank, modules in oft_configs:
        block = matched_oft_block_size(rank, hidden_size, hidden_size)
        for lr in LORA_LR_GRID:
            arms.append(
                Arm(_name("oft", f"b{block}", modules, lr, seed), "oft", None, block, modules, lr, seed)
            )

    return arms


def e1_arms(
    seed: int = 0,
    hidden_size: int = LLAMA31_8B_HIDDEN,
    ffn_size: int = LLAMA31_8B_FFN,
    oft_lr_centre: float | None = None,
) -> list[Arm]:
    """E1: capacity, rank, and the 10x LR rule -- decides C1 and C2.

    8 arms x 5 LRs = 40 runs, plus one OFT cell matched to the r256 anchor
    (45 total). C1 and C2 are LoRA-vs-FullFT claims and the OFT cell decides
    neither; it is here so the task's dashboard carries all three methods and
    so E5's OFT result has a same-grid companion on the rank ladder.
    """
    arms: list[Arm] = []
    for lr in lr_grid(FULL_LR_CENTRE):
        arms.append(Arm(_name("full", "na", "", lr, seed), "full", None, None, "", lr, seed, dataset="tulu3"))
    for rank in (1, 4, 16, 64, 128, 256, 512):
        for lr in lr_grid(LORA_LR_CENTRE):
            arms.append(
                Arm(
                    _name("lora", f"r{rank}", ALL_MODULES, lr, seed),
                    "lora",
                    rank,
                    None,
                    ALL_MODULES,
                    lr,
                    seed,
                    dataset="tulu3",
                )
            )
    arms += _oft_cell(256, ALL_MODULES, hidden_size, ffn_size, seed, "tulu3",
                      oft_lr_centre, n=5)
    return arms


def e1long_arms(
    argmins: dict[tuple[str, int | None], float],
    seed: int = 0,
) -> list[Arm]:
    """E1-2: the long learning curves that decide C1.

    Eight runs -- one per E1 arm, each at *that arm's own* argmin LR from E1-1,
    each a full Tulu3 epoch. Eight rather than forty precisely because E1-1 has
    already located the learning rates: run at a shared LR instead and a rank
    that departs early is indistinguishable from a rank whose LR was too high.

    `argmins` maps `(method, rank)` to a learning rate. A missing key raises
    rather than being skipped: eight arms silently becoming five would look like
    a completed stage.
    """
    wanted: list[tuple[str, int | None]] = [("full", None)] + [("lora", r) for r in E1LONG_RANKS]
    missing = [key for key in wanted if key not in argmins]
    if missing:
        raise ValueError(
            f"e1long is missing an argmin for {missing}; run E1-1 to completion first "
            "(runbook section 8)"
        )
    arms: list[Arm] = []
    for method, rank in wanted:
        lr = argmins[(method, rank)]
        modules = "" if method == "full" else ALL_MODULES
        tag = "na" if method == "full" else f"r{rank}"
        arms.append(
            Arm(
                _name(method, tag, modules, lr, seed, extra="long"),
                method,
                rank,
                None,
                modules,
                lr,
                seed,
                dataset="tulu3",
                full_epoch=True,
                eval_nll_interval=E1LONG_EVAL_INTERVAL,
            )
        )
    return arms


def e1ot_arms(
    seed: int = 0,
    hidden_size: int = LLAMA31_8B_HIDDEN,
    ffn_size: int = LLAMA31_8B_FFN,
    oft_lr_centre: float | None = None,
) -> list[Arm]:
    """E1-OT: the rank ladder on OpenThoughts3 -- the post's second SFT dataset.

    Identical in shape to :func:`e1_arms` and deliberately so: the post's claim
    is that the rank/capacity behaviour is a property of LoRA rather than of
    Tulu3, and a differently-shaped grid would make a difference in the result
    indistinguishable from a difference in the design.

    One epoch here is 312 optimizer steps against Tulu3's 29,323, so these arms
    run to completion and yield the argmins *and* the learning curves. There is
    no `e1otlong`; the E1-1/E1-2 split exists only because a Tulu3 epoch at 40
    arms is unaffordable.

    The held-out split is 100 rows against Tulu3's 1,000, so its noise floor is
    a different number: run seeds 1 and 2 of one arm into a separate sigma
    ledger before quoting anything against it (runbook section 7).
    """
    arms: list[Arm] = []
    for lr in lr_grid(FULL_LR_CENTRE):
        arms.append(
            Arm(_name("full", "na", "", lr, seed), "full", None, None, "", lr, seed,
                dataset="openthoughts3", full_epoch=True,
                eval_nll_interval=E1OT_EVAL_INTERVAL)
        )
    for rank in E1LONG_RANKS:
        for lr in lr_grid(LORA_LR_CENTRE):
            arms.append(
                Arm(_name("lora", f"r{rank}", ALL_MODULES, lr, seed), "lora", rank, None,
                    ALL_MODULES, lr, seed, dataset="openthoughts3", full_epoch=True,
                    eval_nll_interval=E1OT_EVAL_INTERVAL)
            )
    arms += _oft_cell(256, ALL_MODULES, hidden_size, ffn_size, seed, "openthoughts3",
                      oft_lr_centre, n=5, full_epoch=True,
                      eval_nll_interval=E1OT_EVAL_INTERVAL)
    return arms


# The post: short runs (~100 steps) want a ~15x multiplier where long runs
# converge to ~10x, because B's zero initialization acts as an implicit warmup
# that has not finished in 100 steps.
E1SHORT_ROLLOUTS = 100
# 0.15 rather than the campaign's 0.3. Resolving 15x from 10x means resolving a
# factor of 1.5, which is log10(1.5) = 0.176 decades; on a 0.3-decade grid the
# adjacent points differ by 2x and the effect cannot appear. This is a
# requirement of the claim -- do not unify it with lr_grid's default.
E1SHORT_STEP_DECADES = 0.15
E1SHORT_POINTS = 7
# 100/10 = 10 measurements, ~11 min of eval against ~14 min of training. At the
# campaign's usual 1% the arm would spend 8x longer evaluating than training.
E1SHORT_EVAL_INTERVAL = 10


def e1short_arms(
    seed: int = 0,
    hidden_size: int = LLAMA31_8B_HIDDEN,
    ffn_size: int = LLAMA31_8B_FFN,
    oft_lr_centre: float | None = None,
) -> list[Arm]:
    """E1-short: the ~100-step learning-rate multiplier (the second half of C2).

    FullFT and LoRA r256 only. The claim is about the *ratio* of two argmins at
    a short horizon, so the rank ladder adds nothing and every extra rank would
    dilute the resolution budget that the 0.15-decade grid is spending.

    Centred on the same 2.5e-5 / 2.5e-4 as `e1`, so the short-run and long-run
    ratios are read off grids that agree at their midpoints and any difference
    between them is a difference in the optimum rather than in the grid.
    """
    grid = lambda centre: lr_grid(  # noqa: E731 -- local alias, three uses
        centre, n=E1SHORT_POINTS, step_decades=E1SHORT_STEP_DECADES
    )
    arms: list[Arm] = []
    for lr in grid(FULL_LR_CENTRE):
        arms.append(
            Arm(_name("full", "na", "", lr, seed, extra="short"), "full", None, None, "",
                lr, seed, dataset="tulu3", num_rollout=E1SHORT_ROLLOUTS,
                eval_nll_interval=E1SHORT_EVAL_INTERVAL)
        )
    for lr in grid(LORA_LR_CENTRE):
        arms.append(
            Arm(_name("lora", "r256", ALL_MODULES, lr, seed, extra="short"), "lora", 256,
                None, ALL_MODULES, lr, seed, dataset="tulu3",
                num_rollout=E1SHORT_ROLLOUTS, eval_nll_interval=E1SHORT_EVAL_INTERVAL)
        )
    # C8 is a LoRA/FullFT ratio, so the OFT cell decides nothing here -- but it
    # inherits the 0.15-decade spacing anyway, because an OFT cell read against
    # these two on a coarser grid would be a different measurement.
    arms += _oft_cell(256, ALL_MODULES, hidden_size, ffn_size, seed, "tulu3",
                      oft_lr_centre, n=E1SHORT_POINTS,
                      step_decades=E1SHORT_STEP_DECADES, extra="short",
                      num_rollout=E1SHORT_ROLLOUTS,
                      eval_nll_interval=E1SHORT_EVAL_INTERVAL)
    return arms


def e2_arms(
    seed: int = 0,
    hidden_size: int = LLAMA31_8B_HIDDEN,
    ffn_size: int = LLAMA31_8B_FFN,
    oft_lr_centre: float | None = None,
) -> list[Arm]:
    """E2: batch-size sensitivity -- decides C3.

    3 cells x 3 batch sizes x 4 LRs = 36 runs, on the post's own 10,000-example
    OpenThoughts3 subset. LoRA r16 is in here for E2-2 specifically: the post
    blames the product-of-matrices parametrization rather than capacity, so the
    gap has to survive a change of rank. If it shrinks, the mechanism is wrong.

    Each cell's grid is re-centred by sqrt(batch/32) -- with gradient noise
    falling as 1/sqrt(batch), that is the scaling that holds the update-to-weight
    ratio roughly fixed. It is a starting point, not a claim: the acceptance rule
    is unchanged, and any argmin landing on a grid edge is re-run on a re-centred
    grid before its number is quoted.
    """
    cells = [("full", None, FULL_LR_CENTRE), ("lora", 256, LORA_LR_CENTRE), ("lora", 16, LORA_LR_CENTRE)]
    arms: list[Arm] = []
    for batch in (32, 128, 512):
        scale = (batch / 32) ** 0.5
        for method, rank, centre in cells:
            tag = "na" if rank is None else f"r{rank}"
            modules = "" if method == "full" else ALL_MODULES
            for lr in lr_grid(centre * scale, n=4):
                arms.append(
                    Arm(
                        _name(method, tag, modules, lr, seed, extra=f"b{batch}"),
                        method,
                        rank,
                        None,
                        modules,
                        lr,
                        seed,
                        global_batch_size=batch,
                        dataset="openthoughts3",
                    )
                )
        # One OFT cell per batch, never one pooled across batches: C3 compares
        # within a batch size, and an OFT arm measured at 32 could not be
        # differenced against a FullFT arm at 512. The same sqrt(batch/32)
        # re-centring applies -- gradient noise falls the same way whatever
        # parameterizes the update.
        arms += _oft_cell(256, ALL_MODULES, hidden_size, ffn_size, seed,
                          "openthoughts3", oft_lr_centre, n=4, scale=scale,
                          extra=f"b{batch}", global_batch_size=batch)
    return arms


def e3_arms(
    hidden_size: int,
    ffn_size: int,
    seed: int = 0,
    qkv_output_size: int = LLAMA31_8B_QKV_OUTPUT,
    oft_lr_centre: float | None = None,
) -> list[Arm]:
    """E3: layer placement at matched parameter count -- decides C4.

    4 arms x 5 LRs = 20 runs, plus a FullFT baseline (5) and an OFT cell at each
    placement (10) -- 35 total. The FullFT arms decide nothing about placement
    (there is no adapter to place) and are a reference line; the OFT cells ask
    whether the placement finding is a property of low-rank updates or of PEFT
    in general, which is C6's question restricted to one axis.

    The matched MLP rank is *solved*, not assumed:
    Orbit's fused ``linear_qkv``/``linear_fc1`` bundle projections that HF keeps
    separate, so the post's own attention-r256/MLP-r128 pair is not matched in
    this layout. Both pairs are run -- ours and the post's -- so a disagreement
    can be pinned on parameter accounting rather than on physics.
    """
    matched_rank = matched_mlp_rank(256, hidden_size, ffn_size, qkv_output_size)
    configs = [
        (256, ATTN_MODULES),
        (matched_rank, MLP_MODULES),
        (128, MLP_MODULES),
        (256, ALL_MODULES),
    ]
    arms: list[Arm] = []
    for rank, modules in configs:
        for lr in lr_grid(LORA_LR_CENTRE):
            arms.append(
                Arm(
                    _name("lora", f"r{rank}", modules, lr, seed),
                    "lora",
                    rank,
                    None,
                    modules,
                    lr,
                    seed,
                    dataset="tulu3",
                )
            )
    # Tagged `place` for the same reason E4-place's are: E1 runs FullFT on this
    # exact Tulu3 grid, and an untagged name here would collide with it.
    for lr in lr_grid(FULL_LR_CENTRE):
        arms.append(
            Arm(_name("full", "na", "", lr, seed, extra="place"), "full", None, None,
                "", lr, seed, dataset="tulu3")
        )
    for rank, modules in ((256, ATTN_MODULES), (matched_rank, MLP_MODULES)):
        arms += _oft_cell(rank, modules, hidden_size, ffn_size, seed, "tulu3",
                          oft_lr_centre, n=5, qkv_output_size=qkv_output_size)
    return arms


def e4_arms(
    seed: int = 0,
    hidden_size: int = LLAMA31_8B_HIDDEN,
    ffn_size: int = LLAMA31_8B_FFN,
    oft_lr_centre: float | None = None,
) -> list[Arm]:
    """E4: RL parity at low rank -- decides C5.

    4 arms x 4 LRs = 16 runs on MATH + GSM8K. Rank 1 is in here because it is
    the claim ("LoRA matches FullFT under policy gradient **even at rank 1**"),
    not because it is cheap; it is the last arm to drop, not the first.

    The grid is half-decade rather than E1's 0.3-decade, and that is deliberate.
    C5's second half is about the *width* of the performant LR band, which needs
    coverage across a wide range more than resolution near one point -- and the
    RL optimum is less well predicted to begin with, since the post gives a
    multiplier for SFT and not for policy gradient. LoRA is still centred a
    decade above FullFT, which is C2's rule carried over as a prior; if the
    argmins say otherwise for RL, that is a finding rather than a grid error.
    """
    arms: list[Arm] = []
    for lr in lr_grid(RL_FULL_LR_CENTRE, n=4, step_decades=0.5):
        arms.append(
            Arm(_name("full", "na", "", lr, seed), "full", None, None, "", lr, seed, dataset=RL_MIX_DATASET)
        )
    for rank in (1, 16, 256):
        for lr in lr_grid(RL_LORA_LR_CENTRE, n=4, step_decades=0.5):
            arms.append(
                Arm(
                    _name("lora", f"r{rank}", ALL_MODULES, lr, seed),
                    "lora",
                    rank,
                    None,
                    ALL_MODULES,
                    lr,
                    seed,
                    dataset=RL_MIX_DATASET,
                )
            )
    # RL's own OFT scout: the SFT span shifted down a decade, matching how RL's
    # FullFT and LoRA centres sit a decade below their SFT counterparts. Nothing
    # has ever measured OFT under policy gradient, so these are `oftscout` arms
    # until one of them wins.
    # Capped: an RL arm's rotation runs inside SGLang, which cannot launch a
    # block above OFT_MAX_BLOCK_SGLANG. b128 matches LoRA r24 all-modules, which
    # sits beside this matrix's own r16 arm rather than the r256 the SFT cells
    # match -- a smaller adapter, but one that runs.
    arms += _oft_cell(256, ALL_MODULES, hidden_size, ffn_size, seed, RL_MIX_DATASET,
                      oft_lr_centre, n=4, step_decades=0.5, span=RL_OFT_SCOUT_SPAN,
                      max_block=OFT_MAX_BLOCK_SGLANG)
    return arms


def e4place_arms(
    hidden_size: int,
    ffn_size: int,
    seed: int = 0,
    qkv_output_size: int = LLAMA31_8B_QKV_OUTPUT,
    oft_lr_centre: float | None = None,
) -> list[Arm]:
    """E4-place: does the attention-vs-MLP finding survive policy gradient?

    2 placements x 4 LRs = 8 LoRA runs, plus a FullFT reference (4) and an OFT
    cell at each placement (8) -- 20 total, all on E4's own data and E4's own
    half-decade grid so the placement result and the rank result are comparable
    arm for arm.

    **All-modules is deliberately absent.** E4 already runs LoRA r256
    all-modules on this exact grid, so including it here would produce four
    byte-identical arm names -- four re-run RL arms at 8 GPUs each, and a
    duplicate key the moment both ledgers are globbed into `analyze` together,
    where the better of two independent runs of one configuration would win.
    Read the all-modules cell from E4's ledger; the same reasoning excludes
    Llama from `e6`.

    The MLP rank is E3's *solved* match (r92 on Llama-3.1-8B), not the post's
    r128: Orbit fuses qkv and gate+up, so the post's pair is not matched in this
    layout, and an unmatched pair compares placement and capacity at once.

    The FullFT arms answer no placement question -- there is no adapter to
    place -- and duplicate what E4 measures on the same grid. They are here as
    the reference line the placement cells are read against inside this task's
    own dashboard, at 4 runs on 8 GPUs; drop them first under budget pressure
    and read E4's instead.
    """
    matched_rank = matched_mlp_rank(256, hidden_size, ffn_size, qkv_output_size)
    configs = [
        (256, ATTN_MODULES),
        (matched_rank, MLP_MODULES),
    ]
    arms: list[Arm] = []
    for rank, modules in configs:
        for lr in lr_grid(RL_LORA_LR_CENTRE, n=4, step_decades=0.5):
            arms.append(
                Arm(_name("lora", f"r{rank}", modules, lr, seed), "lora", rank, None,
                    modules, lr, seed, dataset=RL_MIX_DATASET)
            )
    # Tagged `place`, because E4 runs FullFT on this exact grid: untagged, all
    # four names would be byte-identical to E4's, which is a duplicate key the
    # moment both ledgers are globbed into `analyze` together -- the same hazard
    # the missing all-modules cell above avoids.
    for lr in lr_grid(RL_FULL_LR_CENTRE, n=4, step_decades=0.5):
        arms.append(
            Arm(_name("full", "na", "", lr, seed, extra="place"), "full", None, None,
                "", lr, seed, dataset=RL_MIX_DATASET)
        )
    for rank, modules in configs:
        # Capped for SGLang -- see OFT_MAX_BLOCK_SGLANG.
        arms += _oft_cell(rank, modules, hidden_size, ffn_size, seed, RL_MIX_DATASET,
                          oft_lr_centre, n=4, step_decades=0.5,
                          span=RL_OFT_SCOUT_SPAN, qkv_output_size=qkv_output_size,
                          max_block=OFT_MAX_BLOCK_SGLANG)
    return arms


# The OFT capacity ladder for E5, on all four projections. Block sizes rather
# than ranks, because the block size is what Megatron takes -- and these three
# are where inverting the match works: b=8 lands on LoRA rank 1, where the rank
# lattice is too coarse to match (ratio 1.34), so it is left out.
E5_BLOCK_LADDER = (32, 64, 256)
# Which of those the scout uses. Scouting at a block size the refinement never
# runs would locate the learning rate for a model that is not then measured.
E5_SCOUT_BLOCK = 64

# E5-RL's capacity ladder. Three rungs a factor of 4 apart -- a 16x span in
# trainable parameters (0.41M / 1.69M / 6.80M all-modules on Llama-3.1-8B).
#
# Why these three and not E5's (32, 64, 256):
#   * they must each launch inside SGLang, so every rung is <= the kernel's
#     block ceiling. E5's ladder satisfies that too, but its rungs are 32/64/256
#     -- a 8x span concentrated at the bottom, chosen for an SFT run where the
#     interesting regime was small adapters.
#   * spreading to 32/128/512 costs nothing extra (the arm count is the same)
#     and widens the lever arm, which is what a "does OFT track LoRA as capacity
#     grows" claim is actually resting on.
#   * the bottom rung stops at 32 rather than going lower because the rank
#     lattice cannot follow below 16: block 8 matches rank 1 at ratio 1.34, and
#     a 34% capacity mismatch would confound exactly what this matrix isolates.
#
# Realized match against the solved LoRA ranks: 0.988 (b32, r6), 1.012 (b128,
# r24), 0.997 (b512, r98). Every arm carries its own ratio into the ledger.
E5RL_BLOCK_LADDER = (32, 128, 512)

# Matrices whose arms cannot be built without a measured OFT learning-rate
# centre, because OFT parameterizes a rotation and no LoRA learning rate
# transfers to it. Declared once here rather than tested for by name at each
# call site: `probe.py` used to special-case `matrix == "e5"`, and that literal
# silently excluded the second such matrix the moment one existed -- the plan
# simply raised instead of skipping.
#
# Where each one's centre comes from:
#   e5    -- the e5scout matrix's argmin
#   e5rl  -- E4's `oftscout` arms, which exist for exactly this purpose
MATRICES_REQUIRING_OFT_CENTRE = frozenset({"e5", "e5rl"})


def _e5_shapes(hidden_size: int, ffn_size: int, qkv_output_size: int) -> dict[str, tuple[int, int]]:
    return megatron_module_shapes(hidden_size, ffn_size, qkv_output_size)


def e5_scout_arms(
    hidden_size: int,
    ffn_size: int,
    seed: int = 0,
    qkv_output_size: int = LLAMA31_8B_QKV_OUTPUT,
) -> list[Arm]:
    """E5's LR scout: 5 arms, half a decade apart, one block size.

    OFT parameterizes a *rotation* rather than an additive update, so nothing
    about LoRA's optimal LR transfers to it -- not the value, not even the decade.
    Hence a wide scout before any refinement grid, exactly as the campaign plan
    requires.
    """
    shapes = _e5_shapes(hidden_size, ffn_size, qkv_output_size)
    report = oft_lora_match_report(E5_SCOUT_BLOCK, shapes)
    return [
        Arm(
            _name("oftscout", f"b{E5_SCOUT_BLOCK}", ALL_MODULES, lr, seed),
            "oft",
            None,
            E5_SCOUT_BLOCK,
            ALL_MODULES,
            lr,
            seed,
            dataset="tulu3",
            matched_ratio=report["ratio"],
        )
        for lr in OFT_SCOUT_GRID
    ]


def e5rl_arms(
    hidden_size: int,
    ffn_size: int,
    seed: int = 0,
    qkv_output_size: int = LLAMA31_8B_QKV_OUTPUT,
    oft_lr_centre: float | None = None,
) -> list[Arm]:
    """E5-RL: does matched-parameter OFT track LoRA under policy gradient?

    24 arms -- three matched capacities x {OFT, LoRA} x 4 learning rates -- on
    E4's data, E4's half-decade grid and E4's accuracy metric, so its cells are
    comparable arm for arm with the rank ladder E4 measures.

    **This is the only place OFT and LoRA meet across a range of matched
    capacities.** E4 has an OFT cell and E4-place has two, but each is a single
    block size: one point per placement. A single point can show that OFT works;
    it cannot show that OFT *tracks* LoRA as capacity varies, which is the
    claim this matrix exists to test.

    The pairing fixes the block size and solves for the rank, the same direction
    E5 uses and for the same reason: LoRA's rank is a fine lattice while an OFT
    block must divide the input dimension, so inverting gets within ~1% where
    the forward direction lands 24-53% off. Each arm carries its realized ratio.

    **Capacity only -- no placement axis.** E4-place already runs OFT against
    LoRA at attention-only and MLP-only on this grid; repeating it would be
    eight more 8-GPU arms answering a question already asked, and duplicate arm
    names the moment both ledgers are globbed into `analyze`. That is the same
    rule that keeps all-modules out of E4-place.

    `oft_lr_centre` is required and comes from E4's `oftscout` argmin -- E4's
    OFT cell is built from RL_OFT_SCOUT_SPAN precisely so this matrix does not
    need a scout of its own. No default: a made-up centre would be an invented
    answer to the question those arms exist to ask, and it would be invisible,
    since the arms would still run and still report accuracies.
    """
    if oft_lr_centre is None:
        raise ValueError(
            "oft_lr_centre is required; take it from E4's oftscout argmin "
            "(--argmins-from results/e4*.jsonl, or --oft-lr-centre)"
        )

    shapes = _e5_shapes(hidden_size, ffn_size, qkv_output_size)
    oft_grid = lr_grid(oft_lr_centre, n=4, step_decades=0.5)
    lora_grid = lr_grid(RL_LORA_LR_CENTRE, n=4, step_decades=0.5)
    arms: list[Arm] = []

    for block_size in E5RL_BLOCK_LADDER:
        report = oft_lora_match_report(block_size, shapes)
        rank = report["lora_rank"]
        ratio = report["ratio"]
        for lr in oft_grid:
            arms.append(
                Arm(
                    _name("oft", f"b{block_size}", ALL_MODULES, lr, seed),
                    "oft",
                    None,
                    block_size,
                    ALL_MODULES,
                    lr,
                    seed,
                    dataset=RL_MIX_DATASET,
                    matched_ratio=ratio,
                )
            )
        for lr in lora_grid:
            arms.append(
                Arm(
                    _name("lora", f"r{rank}", ALL_MODULES, lr, seed),
                    "lora",
                    rank,
                    None,
                    ALL_MODULES,
                    lr,
                    seed,
                    dataset=RL_MIX_DATASET,
                    matched_ratio=ratio,
                )
            )
    return arms


def e5_arms(
    hidden_size: int,
    ffn_size: int,
    seed: int = 0,
    qkv_output_size: int = LLAMA31_8B_QKV_OUTPUT,
    oft_lr_centre: float | None = None,
) -> list[Arm]:
    """E5: does matched-parameter OFT behave like LoRA on C1/C2/C4?

    50 arms in two axes, every OFT arm paired with a LoRA arm at the **same
    realized parameter count**:

    * capacity (C1/C2) -- all four projections at OFT block sizes 32/64/256,
      against LoRA at the ranks those block sizes match (6/12/49 on Llama-3.1-8B).
    * placement (C4) -- a 2x2 of {OFT, LoRA} x {attention-only, MLP-only}, all
      four at one capacity. The MLP block size is *solved* to match attention's
      realized count instead of being reused, because OFT's parameter count
      follows `d_in` and the MLP's `d_in` sum is larger: the same rank-vs-
      parameters confound E3 exists to avoid, one method over.

    The pairing runs this direction -- fix the block size, solve for the rank --
    because a single global block size provably cannot match LoRA across mixed
    shapes (see `orbit.utils.peft_param_match`'s module docstring: the best
    all-modules ratio is 0.764). Rank is the finer lattice, so inverting gets
    within a few percent, and each arm carries the realized ratio it achieved.

    `oft_lr_centre` is required and comes from `e5_scout_arms`' argmin. There is
    deliberately no default: a made-up centre would be an invented answer to the
    question the scout exists to ask.
    """
    if oft_lr_centre is None:
        raise ValueError("oft_lr_centre is required; run the e5scout matrix first and pass its argmin")

    shapes = _e5_shapes(hidden_size, ffn_size, qkv_output_size)
    attn_shapes = {name: shapes[name] for name in ATTENTION_MODULES}
    mlp_shapes = {name: shapes[name] for name in PEFT_MLP_MODULES}
    oft_grid = lr_grid(oft_lr_centre)
    lora_grid = lr_grid(LORA_LR_CENTRE)
    arms: list[Arm] = []

    def _add_pair(block_size: int, modules: str, module_shapes: dict[str, tuple[int, int]]) -> None:
        report = oft_lora_match_report(block_size, module_shapes)
        for lr in oft_grid:
            arms.append(
                Arm(
                    _name("oft", f"b{block_size}", modules, lr, seed),
                    "oft",
                    None,
                    block_size,
                    modules,
                    lr,
                    seed,
                    dataset="tulu3",
                    matched_ratio=report["ratio"],
                )
            )
        for lr in lora_grid:
            arms.append(
                Arm(
                    _name("lora", f"r{report['lora_rank']}", modules, lr, seed),
                    "lora",
                    report["lora_rank"],
                    None,
                    modules,
                    lr,
                    seed,
                    dataset="tulu3",
                    matched_ratio=report["ratio"],
                )
            )

    for block_size in E5_BLOCK_LADDER:
        _add_pair(block_size, ALL_MODULES, shapes)

    # Placement axis, all four cells at attention's realized capacity.
    attn_block = E5_SCOUT_BLOCK
    attn_params = oft_param_count_for_modules(attn_block, attn_shapes)
    mlp_block = oft_block_size_matching_params(attn_params, mlp_shapes)
    _add_pair(attn_block, ATTN_MODULES, attn_shapes)
    _add_pair(mlp_block, MLP_MODULES, mlp_shapes)
    return arms


MATRICES = {
    "sft82": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: sft_arms(hidden, ffn, seed=seed),
    "e1": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e1_arms(
        seed=seed, hidden_size=hidden, ffn_size=ffn, oft_lr_centre=oft_lr_centre
    ),
    "e1long": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e1long_arms(argmins, seed=seed),
    "e1ot": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e1ot_arms(
        seed=seed, hidden_size=hidden, ffn_size=ffn, oft_lr_centre=oft_lr_centre
    ),
    "e1short": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e1short_arms(
        seed=seed, hidden_size=hidden, ffn_size=ffn, oft_lr_centre=oft_lr_centre
    ),
    "e2": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e2_arms(
        seed=seed, hidden_size=hidden, ffn_size=ffn, oft_lr_centre=oft_lr_centre
    ),
    "e3": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e3_arms(
        hidden, ffn, seed=seed, oft_lr_centre=oft_lr_centre
    ),
    "e4": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e4_arms(
        seed=seed, hidden_size=hidden, ffn_size=ffn, oft_lr_centre=oft_lr_centre
    ),
    "e4place": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e4place_arms(
        hidden, ffn, seed=seed, oft_lr_centre=oft_lr_centre
    ),
    "e5rl": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e5rl_arms(
        hidden, ffn, seed=seed, oft_lr_centre=oft_lr_centre
    ),
    "e5scout": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e5_scout_arms(hidden, ffn, seed=seed),
    "e5": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e5_arms(
        hidden, ffn, seed=seed, oft_lr_centre=oft_lr_centre
    ),
}


def arm_env(arm: Arm, data_dir: str = DATA_DIR) -> dict[str, str]:
    """Environment overrides for one launcher invocation.

    Deliberately does not set ROLLOUT_SEED: the launcher ties it to SEED
    itself (scripts/lib/train.sh + scripts/lib/rollout.sh), which is exactly
    what makes a seed sweep vary training data order along with init -- an
    override here would silently defeat that.
    """
    env = {"LR": f"{arm.lr:g}", "SEED": str(arm.seed)}
    if arm.full_epoch:
        # The EMPTY STRING, not an omitted key. The launcher spells it
        # ${NUM_ROLLOUT:-$((...))} -- the colon form re-derives on an empty
        # value, so this both requests the full epoch and immunises the arm
        # against a NUM_ROLLOUT=2000 left exported in the shell from E1-1.
        env["NUM_ROLLOUT"] = ""
    elif arm.num_rollout is not None:
        env["NUM_ROLLOUT"] = str(arm.num_rollout)
    if arm.eval_nll_interval is not None:
        env["EVAL_NLL_INTERVAL"] = str(arm.eval_nll_interval)
    if arm.dataset is not None:
        env["TRAIN_JSONL"] = f"{data_dir}/{arm.dataset}_train.jsonl"
        if arm.dataset not in DATASETS_WITHOUT_TEST_SPLIT:
            # The launcher derives EVAL_NLL_DATA from TEST_JSONL at its own
            # default, but only if TEST_JSONL was exported before it ran --
            # which it is here.
            env["TEST_JSONL"] = f"{data_dir}/{arm.dataset}_test.jsonl"
    if arm.global_batch_size is not None:
        # Both knobs, always together. --global-batch-size alone would leave
        # --rollout-batch-size at 32, so a "batch 512" arm would still draw 32
        # prompts per rollout and never assemble a 512-sample step.
        env["GLOBAL_BATCH_SIZE"] = str(arm.global_batch_size)
        env["ROLLOUT_BATCH_SIZE"] = str(arm.global_batch_size)
    if arm.method == "full":
        env["PEFT_METHOD"] = "none"
        return env
    env["TARGET_MODULES"] = arm.target_modules
    if arm.method == "lora":
        env["PEFT_METHOD"] = "lora"
        env["LORA_RANK"] = str(arm.rank)
        env["LORA_ALPHA"] = str(LORA_ALPHA)
        env["LORA_A_INIT_METHOD"] = LORA_A_INIT_METHOD
    elif arm.method == "oft":
        env["PEFT_METHOD"] = "oft"
        env["OFT_BLOCK_SIZE"] = str(arm.oft_block_size)
    else:
        raise ValueError(f"unknown method {arm.method!r}")
    return env


def adapter_param_count(
    arm: Arm,
    hidden_size: int,
    ffn_size: int,
    num_layers: int,
    qkv_output_size: int = LLAMA31_8B_QKV_OUTPUT,
) -> int | None:
    """Trainable adapter parameters for this arm, or None for full fine-tuning.

    Analytic rather than read back from a written checkpoint, so it is available
    at dry-run time -- before compute is spent -- and so E3's and E5's
    matched-parameter claims can be checked against the arm that is *about* to
    run. Verified exact against the real 2026-07-30 r256 adapter
    (570,425,344 parameters); see the plan's Task 4.

    `None` for `full` arms is meaningful, not missing: full fine-tuning has no
    adapter, and recording 0 would read as "an adapter with no parameters".
    """
    if arm.method == "full":
        return None
    shapes = megatron_module_shapes(hidden_size, ffn_size, qkv_output_size)
    wanted = [name.strip() for name in arm.target_modules.split(",") if name.strip()]
    selected = {name: shape for name, shape in shapes.items() if name in wanted}
    if not selected:
        raise ValueError(
            f"arm {arm.name!r} targets no known module: {arm.target_modules!r} "
            f"(known: {sorted(shapes)})"
        )
    if arm.method == "lora":
        per_layer = lora_param_count_for_modules(arm.rank, selected)
    elif arm.method == "oft":
        per_layer = oft_param_count_for_modules(arm.oft_block_size, selected)
    else:
        raise ValueError(f"unknown method {arm.method!r}")
    return per_layer * num_layers
