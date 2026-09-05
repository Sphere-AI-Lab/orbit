import asyncio
import logging
import os

from orbit.ray.placement_group import (
    check_weight_update_equal_after_initial_sync,
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
)
from orbit.ray.rollout.eval_dispatch import EvalDispatcher
from orbit.utils import object_store
from orbit.utils.arguments import parse_args, validate_async_off_policy_correction
from orbit.utils.audit_utils.process_identity import MainProcessIdentity
from orbit.utils.data import remove_rollout_data_refs
from orbit.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from orbit.utils.ft_utils.control_server.server import start_control_server
from orbit.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from orbit.utils.logging_utils import configure_logger
from orbit.utils.misc import should_run_periodic_action
from orbit.utils.tracking_utils.tracking import finish_tracking, init_tracking
from orbit.utils.train_status import set_progress, start_heartbeat, write_train_status

logger = logging.getLogger(__name__)


# The framework supports other asynchronous approaches such as fully async (see orbit/rollout/fully_async_rollout.py).
async def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    validate_async_off_policy_correction(args)
    configure_logger(args, source=MainProcessIdentity())
    maybe_start_periodic_pyspy_dump()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    if args.control_server_port:
        start_control_server(
            actor_model=actor_model,
            rollout_manager=rollout_manager,
            port=args.control_server_port,
            ft_components=args.ft_components,
        )

    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await check_weight_update_equal_after_initial_sync(args, actor_model, rollout_manager)

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_manager)

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def save_training_model(model, rollout_id, force_sync):
        if args.use_critic and args.offload_train:
            await model.onload()
        await model.save_model(rollout_id, force_sync=force_sync)
        if args.use_critic and args.offload_train:
            await model.offload()

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Progress signal: record the step + refresh the sentinel each iteration. The
        # background heartbeat (start_heartbeat) covers process-liveness incl. warmup;
        # this bump proves steps are advancing (see orbit/utils/train_status.py).
        set_progress(rollout_id)
        write_train_status("running")
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = await rollout_data_next_future

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.offload_train and args.offload_train_async:
            await actor_model.prefetch_train_state(rollout_id)

        if args.use_critic:
            values = await critic_model.train(rollout_id, rollout_data_curr_ref)
            if args.offload_train:
                await critic_model.offload()
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_curr_ref, external_data=values)
                if args.offload_train:
                    await actor_model.offload()
        else:
            await actor_model.train(rollout_id, rollout_data_curr_ref)
        remove_rollout_data_refs(args, rollout_data_curr_ref)

        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        if external_save or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = external_save or rollout_id == args.num_rollout - 1
            await save_training_model(actor_model, rollout_id, force_sync)
            if args.use_critic:
                await save_training_model(critic_model, rollout_id, force_sync)
            await rollout_manager.save.remote(rollout_id)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            await actor_model.update_weights(rollout_id=rollout_id)

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout):
            await eval_dispatcher.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)

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

    await eval_dispatcher.drain()
    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    write_train_status("running")
    _stop_heartbeat = start_heartbeat()
    try:
        asyncio.run(train(args))
    except BaseException as exc:
        rc = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        state = "completed" if rc == 0 else "failed"
        _stop_heartbeat()
        write_train_status(state, rc, None if rc == 0 else exc)
        raise
    else:
        _stop_heartbeat()
        write_train_status("completed", 0)
    finally:
        _stop_heartbeat()
        finish_tracking()
