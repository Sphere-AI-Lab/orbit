import asyncio

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.async_utils import eager_create_task
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import finish_tracking, init_tracking
from miles.utils.train_status import set_progress, start_heartbeat, write_train_status


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
async def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(
            action="compare", allow_quant_error=args.check_weight_update_allow_quant_error
        )

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        await rollout_manager.eval.remote(0)

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Progress signal: record the step + refresh the sentinel each iteration. The
        # background heartbeat (start_heartbeat) covers process-liveness incl. warmup;
        # this bump proves steps are advancing (see miles/utils/train_status.py).
        set_progress(rollout_id)
        write_train_status("running")
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = await rollout_data_next_future

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            critic_task = await eager_create_task(critic_model.train(rollout_id, rollout_data_curr_ref))
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_curr_ref)
            await critic_task
        else:
            await actor_model.train(rollout_id, rollout_data_curr_ref)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            await actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
            if args.use_critic:
                await critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                await rollout_manager.save.remote(rollout_id)

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            await actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            await rollout_manager.eval.remote(rollout_id)

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
