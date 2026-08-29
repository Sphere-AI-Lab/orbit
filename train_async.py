import asyncio

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args, uses_separate_critic, validate_async_off_policy_correction
from miles.utils.async_utils import eager_create_task
# ORBIT-SEAM: eval-NLL entrypoint guard (see rejection note below)
from orbit.utils.eval_nll import reject_eval_nll_on_unsupported_entrypoint
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
async def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    # ORBIT-SEAM: entrypoint guards: SFT and --eval-nll-data belong to train.py
    assert args.training_mode != "sft", "SFT mode is supported by train.py; train_async.py is RL rollout-only."
    # --eval-nll-data is on the shared parser, so this entrypoint would otherwise
    # accept it and silently emit no metric. Wiring the hook here was considered
    # and rejected: this loop overlaps the next rollout's generation with the
    # current rollout's training (rollout_data_next_future), so "the weights at
    # the moment of measurement" needs its own design pass rather than a copy of
    # train.py's call sites -- and nothing under scripts/ points ORBIT_ENTRYPOINT
    # here, so the addition could not be exercised.
    reject_eval_nll_on_unsupported_entrypoint(args, "train_async.py")
    validate_async_off_policy_correction(args)
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    # ORBIT-SEAM: async engines stay resident; documented no-op offload
    # Note: unlike train.py there is deliberately no offload/onload dance here even
    # when --offload-rollout is passed. Actor and rollout GPUs are disjoint in async
    # mode, so start_rollout_servers marks every engine group needs_offload=False and
    # create_rollout_manager's initial offload() is a no-op: the engines stay resident
    # for the whole run (see tests/fast/test_async_offload_noop.py).
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(action="compare")

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = await rollout_data_next_future

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        # ORBIT-SEAM: one-trunk adapter critic: separate-critic dispatch
        if uses_separate_critic(args):
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
            # ORBIT-SEAM: one-trunk adapter critic: separate-critic dispatch
            if uses_separate_critic(args):
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
    asyncio.run(train(args))
