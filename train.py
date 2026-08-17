import logging
import asyncio
import contextlib
import time

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from tqdm.auto import tqdm

from orbit.utils import tracking_utils
from orbit.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from orbit.utils.arguments import parse_args, uses_rollout_engines, uses_separate_critic
from orbit.utils.async_utils import eager_create_task
from orbit.utils.eval_nll import build_eval_nll_metrics
from orbit.utils.logging_utils import configure_logger
from orbit.utils.metric_utils import compute_rollout_step
from orbit.utils.misc import should_run_periodic_action
from orbit.utils.training_eta import TrainingETA, format_duration
from orbit.utils.tracking_utils import init_tracking

logger = logging.getLogger(__name__)


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
    tracking_utils.log(args, metrics, step_key="rollout/step")


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
    configure_logger()
    startup_timing: dict[str, float] = {}
    rollout_engines_enabled = uses_rollout_engines(args)

    # allocate the GPUs
    with _timed_block("startup", "placement groups", timing_raw=startup_timing):
        pgs = create_placement_groups(args)
    with _timed_block("startup", "init tracking", timing_raw=startup_timing):
        init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    with _timed_block("startup", "create rollout manager", timing_raw=startup_timing):
        rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    async with _timed_phase("startup", "create training models", timing_raw=startup_timing):
        actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    if rollout_engines_enabled and args.offload_rollout:
        async with _timed_phase("startup", "onload rollout weights", timing_raw=startup_timing):
            await rollout_manager.onload_weights.remote()

    # always update weight first so that sglang has the loaded weights from training.
    if rollout_engines_enabled:
        async with _timed_phase("startup", "actor update_weights", timing_raw=startup_timing):
            await actor_model.update_weights()

    if rollout_engines_enabled and args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(action="compare")

    if rollout_engines_enabled and args.offload_rollout:
        async with _timed_phase("startup", "onload rollout kv", timing_raw=startup_timing):
            await rollout_manager.onload_kv.remote()

    if startup_timing:
        startup_metrics = {f"timing_s_startup/{k}": v for k, v in startup_timing.items()}
        startup_metrics["timing_s_startup/total"] = sum(startup_timing.values())
        startup_metrics["rollout/step"] = compute_rollout_step(args, args.start_rollout_id)
        tracking_utils.log(args, startup_metrics, step_key="rollout/step")

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
        if args.offload_train:
            if uses_separate_critic(args):
                await critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    await actor_model.offload()
            else:
                await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def save(rollout_id):
        if (not uses_separate_critic(args)) or (rollout_id >= args.num_critic_only_steps):
            await actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if uses_separate_critic(args):
            await critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            await rollout_manager.save.remote(rollout_id)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
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
        eta.mark_rollout_start(rollout_id)
        timing_raw: dict[str, float] = {}
        prefix = f"rollout {rollout_id}"

        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            async with _timed_phase(prefix, "eval-before-train", timing_raw=timing_raw):
                await rollout_manager.eval.remote(rollout_id)

        # Held-out NLL of the untouched starting weights. Gate G4 compares this
        # against HF's step-0 number, so it has to be measured before any
        # optimizer step -- the periodic block below only ever sees post-update
        # weights.
        if _eval_nll_enabled(args) and rollout_id == 0 and not args.skip_eval_before_train:
            async with _timed_phase(prefix, "eval nll before-train", timing_raw=timing_raw):
                nll_stats = await actor_model.compute_eval_nll(rollout_id)
            _log_eval_nll(args, rollout_id, nll_stats, before_train=True)

        async with _timed_phase(prefix, "generate", timing_raw=timing_raw):
            rollout_data_ref = await rollout_manager.generate.remote(rollout_id)

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

        if uses_separate_critic(args):
            critic_task = await eager_create_task(critic_model.train(rollout_id, rollout_data_ref))
            if rollout_id >= args.num_critic_only_steps:
                async with _timed_phase(prefix, "actor train", timing_raw=timing_raw):
                    await actor_model.train(rollout_id, rollout_data_ref)
            await critic_task
        else:
            async with _timed_phase(prefix, "actor train", timing_raw=timing_raw):
                await actor_model.train(rollout_id, rollout_data_ref)

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

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            async with _timed_phase(prefix, "save", timing_raw=timing_raw):
                await save(rollout_id)

        async with _timed_phase(prefix, "offload/clear train", timing_raw=timing_raw):
            await offload_train()
        if rollout_engines_enabled and args.offload_rollout:
            async with _timed_phase(prefix, "onload rollout weights", timing_raw=timing_raw):
                await rollout_manager.onload_weights.remote()
        if rollout_engines_enabled:
            async with _timed_phase(prefix, "actor update_weights", timing_raw=timing_raw):
                await actor_model.update_weights()
        if rollout_engines_enabled and args.offload_rollout:
            async with _timed_phase(prefix, "onload rollout kv", timing_raw=timing_raw):
                await rollout_manager.onload_kv.remote()

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
        tracking_utils.log(args, progress_metrics, step_key="rollout/step")

    rollout_pbar.close()
    async with _timed_phase("shutdown", "dispose rollout"):
        await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(train(args))
