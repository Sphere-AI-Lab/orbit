# ORBIT-SEAM: logging backs the logger below (phase-timing + eval-NLL log lines)
import logging
import asyncio
# ORBIT-SEAM: contextlib/time back the _timed_phase/_timed_block instrumentation helpers below
import contextlib
import os
import time

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

# ORBIT-SEAM: tqdm backs the rollout progress bar added below
from tqdm.auto import tqdm

# ORBIT-SEAM: tracking module import backs startup/progress metric logging below (init_tracking
# import further down is base's, kept as-is); re-anchored onto upstream's tracking_utils package
from miles.utils.tracking_utils import tracking
from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils import object_store

# ORBIT-SEAM: uses_rollout_engines/uses_separate_critic replace base's raw args.use_critic checks and
# gate the rollout-engine-optional (SFT-mode) code paths added throughout train() below
from miles.utils.arguments import parse_args, uses_rollout_engines, uses_separate_critic
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.ft_utils.control_server.server import start_control_server
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller

# ORBIT-SEAM: held-out NLL eval feature (gate G4) - metric builder for the eval-NLL blocks below
from orbit.utils.eval_nll import build_eval_nll_metrics
from miles.utils.logging_utils import configure_logger
# ORBIT-SEAM: computes the tracking-log step number for eval-NLL and progress metrics below
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

# ORBIT-SEAM: ETA tracking + duration formatting for the progress bar / progress metrics below
from orbit.utils.training_eta import TrainingETA, format_duration

logger = logging.getLogger(__name__)


# ORBIT-SEAM: new top-level helpers - held-out NLL eval feature (gate G4: config check +
# metric/log emission) and two phase-timing context managers (_timed_phase/_timed_block below),
# used throughout train() to wrap every startup/loop phase with elapsed-time logging + accumulation
def _eval_nll_enabled(args) -> bool:
    return bool(args.eval_nll_data) and args.eval_nll_interval > 0


def _log_eval_nll(args, rollout_id: int, stats: dict, *, before_train: bool = False) -> None:
    """Record one held-out NLL measurement.

    ``eval/test_nll`` is the token-weighted mean the study reports.
    ``eval/test_nll_before_train`` duplicates the pre-training measurement under
    its own key so gate G4 (step-0 NLL of the unmodified base model) can be read
    back unambiguously -- the loop logs both the pre-train and the post-rollout-0
    values at ``rollout/step == 0``, exactly as the generation eval already does.
    """
    step = compute_rollout_step(args, rollout_id)
    metrics = build_eval_nll_metrics(stats, step, before_train=before_train)
    logger.info(
        "eval/test_nll rollout_id=%d step=%d phase=%s nll=%.6f sample_mean=%.6f tokens=%d samples=%d",
        rollout_id,
        step,
        "before_train" if before_train else "after_train",
        stats["nll"],
        stats["sample_mean_nll"],
        stats["num_tokens"],
        stats["num_samples"],
    )
    tracking.log(args, metrics, step_key="rollout/step")


@contextlib.asynccontextmanager
async def _timed_phase(prefix: str, name: str, *, timing_raw: dict | None = None, start_extra: str = ""):
    if start_extra:
        logger.info("%s: %s start %s", prefix, name, start_extra)
    else:
        logger.info("%s: %s start", prefix, name)
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        logger.info("%s: %s done elapsed=%.2fs", prefix, name, elapsed)
        if timing_raw is not None:
            timing_raw[name] = timing_raw.get(name, 0.0) + elapsed


@contextlib.contextmanager
def _timed_block(prefix: str, name: str, *, timing_raw: dict | None = None, start_extra: str = ""):
    if start_extra:
        logger.info("%s: %s start %s", prefix, name, start_extra)
    else:
        logger.info("%s: %s start", prefix, name)
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        logger.info("%s: %s done elapsed=%.2fs", prefix, name, elapsed)
        if timing_raw is not None:
            timing_raw[name] = timing_raw.get(name, 0.0) + elapsed


async def train(args):
    assert not args.fully_async, "--fully-async requires the async driver: run train_async.py"
    configure_logger(args, source=MainProcessIdentity())
    maybe_start_periodic_pyspy_dump()
    # ORBIT-SEAM: startup_timing accumulates the _timed_block/_timed_phase elapsed times below into
    # the startup_metrics log emitted further down; rollout_engines_enabled gates every rollout-engine
    # call in this function (SFT/no-rollout-engine mode support), replacing base's unconditional calls
    startup_timing: dict[str, float] = {}
    rollout_engines_enabled = uses_rollout_engines(args)

    # allocate the GPUs
    with _timed_block("startup", "placement groups", timing_raw=startup_timing):
        pgs = create_placement_groups(args)
    object_store.init_instance(args, contribute_segment=False)
    with _timed_block("startup", "init tracking", timing_raw=startup_timing):
        init_tracking(args)

    # ORBIT-SEAM: startup phases below wrapped in _timed_block/_timed_phase (elapsed-time
    # instrumentation, see note above); rollout-engine calls gated by rollout_engines_enabled
    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    with _timed_block("startup", "create rollout manager", timing_raw=startup_timing):
        rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    async with _timed_phase("startup", "create training models", timing_raw=startup_timing):
        actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    if args.control_server_port:
        start_control_server(
            actor_model=actor_model,
            rollout_manager=rollout_manager,
            port=args.control_server_port,
            ft_components=args.ft_components,
        )

    maybe_start_mini_ft_controller(args)

    # ORBIT-SEAM: guard + timing wrap (see note above)
    if rollout_engines_enabled and args.offload_rollout:
        async with _timed_phase("startup", "onload rollout weights", timing_raw=startup_timing):
            await rollout_manager.onload_weights.remote()

    # always update weight first so that sglang has the loaded weights from training.
    # ORBIT-SEAM: guard + timing wrap (see note above)
    if rollout_engines_enabled:
        async with _timed_phase("startup", "actor update_weights", timing_raw=startup_timing):
            await actor_model.update_weights()

    # ORBIT-SEAM: rollout_engines_enabled guard added (see note above)
    if rollout_engines_enabled and args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    if rollout_engines_enabled and args.offload_rollout:
        async with _timed_phase("startup", "onload rollout kv", timing_raw=startup_timing):
            await rollout_manager.onload_kv.remote()

    # ORBIT-SEAM: emits the accumulated startup_timing as tracking metrics (new; base had no
    # startup-phase timing at all)
    if startup_timing:
        startup_metrics = {f"timing_s_startup/{k}": v for k, v in startup_timing.items()}
        startup_metrics["timing_s_startup/total"] = sum(startup_timing.values())
        startup_metrics["rollout/step"] = compute_rollout_step(args, args.start_rollout_id)
        tracking.log(args, startup_metrics, step_key="rollout/step")

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        async with _timed_phase("startup", "eval-only"):
            await rollout_manager.eval.remote(rollout_id=0)

    # Eval-only held-out NLL: --num-rollout 0 with --eval-nll-data measures the
    # loaded checkpoint and exits. This is how gate G4 (step-0 NLL must match
    # HF's) is run without training anything.
    if args.num_rollout == 0 and _eval_nll_enabled(args):
        async with _timed_phase("startup", "eval nll"):
            nll_stats = await actor_model.compute_eval_nll(args.start_rollout_id)
        _log_eval_nll(args, args.start_rollout_id, nll_stats, before_train=True)

    async def offload_train():
        # ORBIT-SEAM: uses_separate_critic(args) replaces base's raw args.use_critic check here and
        # in save() below. Upstream now offloads critic/actor inline in the train dispatch, so this
        # returns early on the separate-critic path instead of doing the offload dance itself.
        if uses_separate_critic(args):
            return
        if args.offload_train:
            await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def save(rollout_id, force_sync=False):
        force_sync = force_sync or rollout_id == args.num_rollout - 1

        async def save_training_model(model):
            # ORBIT-SEAM: uses_separate_critic(args) replaces args.use_critic (see note above)
            if uses_separate_critic(args) and args.offload_train:
                await model.onload()
            await model.save_model(rollout_id, force_sync=force_sync)
            if uses_separate_critic(args) and args.offload_train:
                await model.offload()

        # ORBIT-SEAM: uses_separate_critic(args) replaces args.use_critic here too
        if (not uses_separate_critic(args)) or (rollout_id >= args.num_critic_only_steps):
            await save_training_model(actor_model)
        if uses_separate_critic(args):
            await save_training_model(critic_model)
        await rollout_manager.save.remote(rollout_id)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    # ORBIT-SEAM: new ETA tracker + tqdm progress bar, driven by eta.mark_rollout_start/done below
    # and rendered in the progress-metrics block at the end of each loop iteration
    eta = TrainingETA(start_rollout_id=args.start_rollout_id, num_rollout=args.num_rollout)
    rollout_pbar = tqdm(
        total=max(args.num_rollout - args.start_rollout_id, 0),
        desc="RL training",
        unit="rollout",
        initial=0,
        dynamic_ncols=True,
        smoothing=0.0,
        mininterval=1.0,
    )
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # ORBIT-SEAM: per-rollout ETA/timing bookkeeping (timing_raw feeds the _timed_phase calls
        # below and the progress-metrics timing_s/* keys at the end of the loop body)
        eta.mark_rollout_start(rollout_id)
        timing_raw: dict[str, float] = {}
        prefix = f"rollout {rollout_id}"

        if (
            args.eval_interval is not None
            and rollout_id == args.start_rollout_id
            and not args.skip_eval_before_train
        ):
            async with _timed_phase(prefix, "eval-before-train", timing_raw=timing_raw):
                await rollout_manager.eval.remote(rollout_id)

        # ORBIT-SEAM: held-out NLL before-train block (new) + generate wrap (timing instrumentation)
        # Held-out NLL of the untouched starting weights. Gate G4 compares this
        # against HF's step-0 number, so it has to be measured before any
        # optimizer step -- the periodic block below only ever sees post-update
        # weights.
        if _eval_nll_enabled(args) and rollout_id == 0 and not args.skip_eval_before_train:
            async with _timed_phase(prefix, "eval nll before-train", timing_raw=timing_raw):
                nll_stats = await actor_model.compute_eval_nll(rollout_id)
            _log_eval_nll(args, rollout_id, nll_stats, before_train=True)

        async with _timed_phase(prefix, "generate", timing_raw=timing_raw):
            # ORBIT-SEAM: timing wrap; upstream renamed rollout_data_ref -> rollout_data_pack
            rollout_data_pack = await rollout_manager.generate.remote(rollout_id)

        # ORBIT-SEAM: rollout_engines_enabled guard + timing wrap; new prefetch_train_state block below
        if rollout_engines_enabled and args.offload_rollout:
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            async with _timed_phase(
                prefix, "offload rollout", timing_raw=timing_raw, start_extra=f"tags={offload_tags}"
            ):
                await rollout_manager.offload.remote(tags=offload_tags)

        if args.offload_train and args.offload_train_async:
            async with _timed_phase(prefix, "prefetch train state", timing_raw=timing_raw):
                await actor_model.prefetch_train_state(rollout_id)

        # ORBIT-SEAM: uses_separate_critic(args) replaces args.use_critic; both branches' actor
        # train call wrapped in _timed_phase below (timing instrumentation)
        if uses_separate_critic(args):
            values = await critic_model.train(rollout_id, rollout_data_pack)
            if args.offload_train:
                await critic_model.offload()
            if rollout_id >= args.num_critic_only_steps:
                async with _timed_phase(prefix, "actor train", timing_raw=timing_raw):
                    await actor_model.train(rollout_id, rollout_data_pack, external_data=values)
        else:
            async with _timed_phase(prefix, "actor train", timing_raw=timing_raw):
                await actor_model.train(rollout_id, rollout_data_pack)
        remove_rollout_data_refs(args, rollout_data_pack)

        # ORBIT-SEAM: periodic held-out NLL block (new; num_rollout arg documented below); save wrap
        # further down adds _timed_phase instrumentation to base's plain save(rollout_id) call
        # Must sit between `actor train` and `offload_train()`: held-out NLL is a
        # forward pass through the TRAINING model, so unlike the generation eval
        # further down (which goes through the SGLang rollout engine) it cannot
        # run after the weights have left the GPU. This is also why it is placed
        # after BOTH `actor train` call sites -- the critic branch and the plain
        # branch -- rather than inside either: the measurement is of the actor's
        # post-update weights regardless of which branch produced them.
        # num_rollout is passed so the final rollout always produces a
        # measurement -- the study's headline number per arm is the last
        # held-out NLL, and without this an arm whose num_rollout is not a
        # multiple of the interval would never report one.
        if _eval_nll_enabled(args) and should_run_periodic_action(
            rollout_id, args.eval_nll_interval, num_rollout_per_epoch, args.num_rollout
        ):
            async with _timed_phase(prefix, "eval nll", timing_raw=timing_raw):
                nll_stats = await actor_model.compute_eval_nll(rollout_id)
            _log_eval_nll(args, rollout_id, nll_stats)

        # ORBIT-SEAM: upstream offloads the actor inline immediately after the critic-branch
        # actor train; deferred to here so the held-out NLL forward above still sees the
        # post-update weights on GPU (the whole point of the eval-NLL placement note above)
        if uses_separate_critic(args) and args.offload_train and rollout_id >= args.num_critic_only_steps:
            await actor_model.offload()

        # ORBIT-SEAM: timing wrap around the save call
        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        if external_save or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            async with _timed_phase(prefix, "save", timing_raw=timing_raw):
                await save(rollout_id, force_sync=external_save)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        # ORBIT-SEAM: timing wraps + rollout_engines_enabled guards, same pattern as startup above
        async with _timed_phase(prefix, "offload/clear train", timing_raw=timing_raw):
            await offload_train()
        if rollout_engines_enabled and args.offload_rollout:
            async with _timed_phase(prefix, "onload rollout weights", timing_raw=timing_raw):
                await rollout_manager.onload_weights.remote()
        if rollout_engines_enabled:
            async with _timed_phase(prefix, "actor update_weights", timing_raw=timing_raw):
                await actor_model.update_weights(rollout_id=rollout_id)
        if rollout_engines_enabled and args.offload_rollout:
            async with _timed_phase(prefix, "onload rollout kv", timing_raw=timing_raw):
                await rollout_manager.onload_kv.remote()

        # ORBIT-SEAM: num_rollout arg + timing wrap added to the base periodic-eval call (comment
        # below explains why num_rollout is passed)
        # num_rollout is passed for the same reason the held-out NLL eval above
        # passes it: the final rollout must always produce a measurement. An RL
        # arm's headline number is its accuracy after the last update, and
        # without this a run whose num_rollout is not a multiple of the interval
        # ends having evaluated only the UNTRAINED policy, from the
        # eval-before-train branch. That is not a hypothetical -- the E4 gsm8k
        # columns ran 150 rollouts at --eval-interval 100000, chosen precisely
        # to mean "once, at the end", and produced zero post-training evals.
        if should_run_periodic_action(
            rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout
        ):
            async with _timed_phase(prefix, "eval", timing_raw=timing_raw):
                await rollout_manager.eval.remote(rollout_id)

        # ORBIT-SEAM: new progress reporting - ETA/tqdm postfix update, plus per-phase timing_raw
        # entries merged into the tracking-log progress metrics below (base logged nothing here)
        eta_report = eta.mark_rollout_done(rollout_id)
        logger.info("progress %s", eta_report.to_log_message())
        rollout_pbar.set_postfix(
            last=format_duration(eta_report.last_rollout_seconds),
            avg=format_duration(eta_report.avg_rollout_seconds),
            eta=format_duration(eta_report.eta_seconds),
            refresh=False,
        )
        rollout_pbar.update(1)
        rollout_step = compute_rollout_step(args, rollout_id)
        progress_metrics = eta_report.to_metrics(step=rollout_step)
        for phase_name, elapsed in timing_raw.items():
            progress_metrics[f"timing_s/{phase_name.replace(' ', '_')}"] = elapsed
        tracking.log(args, progress_metrics, step_key="rollout/step")

        if (
            args.debug_exit_after_rollout is not None
            and (rollout_id - args.start_rollout_id + 1) >= args.debug_exit_after_rollout
        ):
            logger.info(
                "debug_exit_after_rollout=%d reached at rollout_id=%d, exiting",
                args.debug_exit_after_rollout,
                rollout_id,
            )
            break

    # ORBIT-SEAM: closes the progress bar opened above; dispose wrapped in _timed_phase
    rollout_pbar.close()
    async with _timed_phase("shutdown", "dispose rollout"):
        await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
