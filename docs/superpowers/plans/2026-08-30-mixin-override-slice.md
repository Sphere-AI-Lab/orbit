# Slice: move orbit-owned methods into the existing mixins + shadow-drift guard

Direction (user, 2026-08-30): reuse miles as much as possible; where reuse is
impossible, override via a mixin (or, as a last resort, a verified patch) rather
than editing the vendored file. This slice does that for the three classes that
ALREADY carry an orbit mixin, so it introduces no new machinery.

## The decision rule (measured, not by feel)

Moving a method into the mixin means orbit carries its WHOLE body. That is only
a win when orbit already owns most of it; for a 2-line tweak inside a 200-line
upstream method it makes things strictly worse (the delta stops being visible to
git's merge). So: move only methods whose changed-line fraction is >= 50%.

Measured ownership (changed lines / method size) over the three mixin classes:

MOVE (14 methods, 1057 lines):
  100%  SGLangEngine.post_process_weights (36)
  100%  SGLangEngine.unload_oft_adapter (5)
   87%  SGLangEngine.begin_weight_update (15)
   86%  SGLangEngine.end_weight_update (14)
   77%  UpdateWeightFromTensor.update_weights (245)
   75%  UpdateWeightFromTensor.is_rollout_engines_fresh (4)
   68%  MegatronTrainRayActor.wake_up (37)
   65%  UpdateWeightFromTensor._send_base_params (52)
   63%  UpdateWeightFromTensor.connect_rollout_engines (156)
   59%  MegatronTrainRayActor.update_weights (121)
   55%  MegatronTrainRayActor._switch_model (11)
   52%  MegatronTrainRayActor.train_actor (239)
   52%  UpdateWeightFromTensor.__init__ (85)
   51%  MegatronTrainRayActor.sleep (37)

KEEP as a stamped seam (13 methods) -- notably MegatronTrainRayActor.init (49%,
336 lines) and SGLangEngine.flush_cache (47%): copying these would import more
upstream code into orbit/ than it removes from the seam.

## Mechanism

Each class already inherits its orbit mixin FIRST
(`class MegatronTrainRayActor(OrbitTrainActorExtensions, TrainRayActor)`), so a
mixin method shadows the BASE class's method. It does NOT shadow the vendored
class's own body -- Python checks the class's own `__dict__` before any base:

    MRO: [MegatronTrainRayActor, OrbitTrainActorExtensions, TrainRayActor, object]

Verified empirically: with `class Own(Mixin, Base)`, a method defined in `Own`'s
body wins and the mixin's copy is DEAD; only when `Own` omits it does the mixin
win over `Base`. For each moved method:
1. Move orbit's version verbatim into the mixin module
   (orbit/sglang/engine_ext.py, orbit/megatron/actor_ext.py,
   orbit/transport/update_weight_ext.py).
2. **DELETE the method from the vendored class body.** (An earlier draft of this
   plan said "revert it to upstream's body" -- that was WRONG and would have
   silently disabled every override, since the retained body outranks the mixin.
   Deletion is also the better merge posture: an upstream edit to a deleted
   region conflicts loudly; an edit to a retained-but-shadowed body does not.)
3. `super().<name>(...)` is available only where the BASE defines that name --
   true for MegatronTrainRayActor (TrainRayActor) but not for
   UpdateWeightFromTensor, whose only declared base is the mixin itself. Inline
   upstream's behaviour where super() is unavailable.
4. Delete the now-dead ORBIT-SEAM stamp for that method.
5. Verify per method that it resolves to the mixin and that its name is NOT in
   the vendored class's own `__dict__`.

## The risk this creates, and the guard that answers it

A shadowed upstream method is now DEAD CODE that git will happily let upstream
change without any conflict -- orbit's override keeps winning and nobody
notices. That is the silent-drift failure mode. So this slice also adds:

  tools/check_shadow_drift.py + tests/fast/test_shadow_drift.py

For every method an orbit mixin shadows, record a hash of the UPSTREAM method's
source (from the vendored file, which is upstream's body once step 2 is done) in
tests/fast/shadow_manifest.json. The test fails when upstream's version of a
shadowed method changes, with a message naming the method and telling the reader
to review whether orbit's override needs the same change. Regenerating the
manifest is the deliberate act that records "reviewed".

This is the mixin analogue of a pinned-hash patch: it converts silent drift into
a loud, reviewable failure, and it is what makes the override architecture safe
to expand.

## Gates

- All 4 existing static guards green; new shadow-drift guard green.
- Purity manifest regenerated: budgeted count and delta must DROP.
- Args-surface golden unchanged (no argument touched by this slice).
- Fast suite: failure set identical to the 5 known.
- GPU reference smoke (this slice touches the training and weight-transfer
  paths, so it is REQUIRED, not optional).
