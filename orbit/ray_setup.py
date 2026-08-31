"""Arm orbit's patches inside every Ray worker process.

Orbit's behaviour installs through an import hook that ``import orbit`` arms, so
a process that reaches a patched miles module without importing orbit silently
runs upstream's function. Ray worker processes are the hard case: they are
started by the raylet, not by orbit, and what they import is decided by whatever
task or actor they are handed.

Most of orbit's actors are safe by accident -- their defining module happens to
import orbit -- but four vendored modules define ``@ray.remote`` callables and do
not (``miles/ray/utils.py``, ``miles/utils/misc.py``, ``http_utils.py``,
``prometheus_utils.py``). Those files are byte-pristine, which is the whole point
of the isolation, so an ``import orbit`` line cannot be added to them. The
nearest one is one call site from being wrong: ``Lock`` inherits
``RayActor._get_current_node_ip_and_free_port``, which calls the PATCHED
``get_free_port`` -- unarmed, it would allocate ports without orbit's flock and
collide silently.

Ray's ``worker_process_setup_hook`` runs a named callable in each worker as it
starts, before any task body. Passing the module path (rather than a callable)
keeps it a plain string in ``runtime_env``, which is what survives being written
into ``--runtime-env-json`` on the launch path, and Ray carries it to workers in
an env var that per-actor ``env_vars`` merge rather than replace.

This makes arming a property of the CLUSTER rather than of each actor's import
closure, so tools/check_arming.py can stop asking every ray module to arm itself.
"""

from __future__ import annotations

# The string form the launch paths pass to Ray. Kept here so the launcher, the
# guard and the hook itself cannot disagree about the name.
WORKER_SETUP_HOOK = "orbit.ray_setup.arm_ray_worker"


def arm_ray_worker() -> None:
    """Ray worker startup hook: arm orbit's patches.

    Deliberately not fail-soft. If orbit cannot be imported in a worker, every
    patched function in that worker is upstream's, which is the exact silent
    wrong-numbers failure this layer exists to prevent; Ray surfaces an
    exception here as a startup error naming this function.
    """
    import orbit  # noqa: F401  -- imported for the arming side effect only
