"""``TrainRayActor.compute_eval_nll`` after the move into orbit's home mixin.

The stub is the abstract end of the eval-NLL subsystem: an OPTIONAL backend hook
that train.py calls when ``--eval-nll-data`` is set. It is orbit-added, lives in
orbit/utils/train_actor_ext.py, and its whole job is to fail with a message that
names the backend rather than an AttributeError.

The failure this file exists to catch is the MRO one. ``MegatronTrainRayActor``
mixes BOTH orbit mixins:

    MegatronTrainRayActor(OrbitTrainActorExtensions, TrainRayActor)
    TrainRayActor(OrbitTrainRayActorExtensions, RayActor)

If the real implementation stopped preceding the stub, every ``--eval-nll-data``
run would raise NotImplementedError instead of scoring -- and would do so only
at the first eval, well into a job.
"""

from miles.ray.train_actor import TrainRayActor
from orbit.megatron.actor_ext import OrbitTrainActorExtensions
from orbit.utils.train_actor_ext import OrbitTrainRayActorExtensions

HOOK = "compute_eval_nll"


def test_stub_resolves_to_the_mixin_and_is_not_shadowed():
    assert HOOK not in TrainRayActor.__dict__, (
        "TrainRayActor defines compute_eval_nll in its own body; that copy would "
        "shadow the mixin (a class's __dict__ beats every base)"
    )
    assert getattr(TrainRayActor, HOOK).__qualname__ == f"{OrbitTrainRayActorExtensions.__name__}.{HOOK}"


def test_the_stub_is_not_abstract():
    """Unlike upstream's train/save_model/update_weights, this hook is optional:
    marking it abstract would stop every backend that ignores --eval-nll-data
    from instantiating."""
    assert not getattr(TrainRayActor.compute_eval_nll, "__isabstractmethod__", False)
    assert HOOK not in getattr(TrainRayActor, "__abstractmethods__", frozenset())
    assert getattr(TrainRayActor.train, "__isabstractmethod__", False), (
        "sanity: upstream's mandatory hooks really are marked abstract"
    )


def test_stub_raises_notimplementederror_naming_the_backend():
    class Backend(TrainRayActor):
        def __init__(self):  # skip the ray/dist bootstrap
            pass

        sleep = wake_up = train = save_model = update_weights = None
        connect_actor_critic = _get_parallel_config = None

    try:
        Backend().compute_eval_nll(0)
    except NotImplementedError as exc:
        assert "Backend" in str(exc)
        assert "--eval-nll-data" in str(exc)
    else:  # pragma: no cover - the stub must refuse
        raise AssertionError("compute_eval_nll must refuse, not return")


def test_the_megatron_implementation_still_wins_over_the_stub():
    """Both mixins are in MegatronTrainRayActor's MRO; the real one must be first."""
    from miles.backends.megatron_utils.actor import MegatronTrainRayActor

    mro = MegatronTrainRayActor.__mro__
    assert mro.index(OrbitTrainActorExtensions) < mro.index(OrbitTrainRayActorExtensions)
    assert getattr(MegatronTrainRayActor, HOOK).__qualname__ == f"{OrbitTrainActorExtensions.__name__}.{HOOK}"
    assert HOOK not in MegatronTrainRayActor.__dict__
