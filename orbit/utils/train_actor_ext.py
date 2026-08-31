"""Orbit's added ``TrainRayActor`` base-class API.

Home mixin for the method lifted out of miles/ray/train_actor.py: the optional
``compute_eval_nll`` hook that train.py calls when ``--eval-nll-data`` is set.

This is the ABSTRACT end of the eval-NLL subsystem -- the declaration that says
"a training backend may implement a forward-only held-out NLL, and one that does
not simply does not support the flag". The megatron implementation lives in
``orbit/megatron/actor_ext.py::OrbitTrainActorExtensions.compute_eval_nll`` and
the layout-free planning/reduction logic in ``orbit/utils/eval_nll.py``.

``TrainRayActor`` in the miles file lists this mixin as its FIRST base:

    class TrainRayActor(OrbitTrainRayActorExtensions, RayActor):

The stub deliberately does NOT get ``@abc.abstractmethod``: upstream's other
abstract methods are contracts every backend must satisfy, while this one is
optional, and marking it abstract would make every backend that ignores
``--eval-nll-data`` fail to instantiate.

MRO note, and it is the whole reason this file is small. ``MegatronTrainRayActor``
is declared ``(OrbitTrainActorExtensions, TrainRayActor)``, which linearizes to

    MegatronTrainRayActor -> OrbitTrainActorExtensions -> TrainRayActor
                          -> OrbitTrainRayActorExtensions -> RayActor -> object

so the real megatron ``compute_eval_nll`` precedes this stub and wins. If the
stub were left in ``TrainRayActor``'s own body it would still lose to
``OrbitTrainActorExtensions`` -- but any FUTURE backend that mixed the two in the
other order would silently get the NotImplementedError, which is exactly the
class of accident the mixin layout removes.

Plain mixin: no ``__init__``, no state, no ``super()`` call (from here ``super()``
is ``RayActor``, which has no ``compute_eval_nll``; there is nothing to delegate
to and the stub is the end of the chain).
"""

from __future__ import annotations


class OrbitTrainRayActorExtensions:
    def compute_eval_nll(self, rollout_id):
        """Forward-only held-out NLL. Returns the reduced statistics on exactly
        one rank and None on all others, so the caller can dedupe TP/PP replicas
        without knowing the parallel layout. Optional: backends that do not
        implement it simply do not support --eval-nll-data."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement compute_eval_nll; --eval-nll-data is unsupported."
        )


__all__ = ["OrbitTrainRayActorExtensions"]
