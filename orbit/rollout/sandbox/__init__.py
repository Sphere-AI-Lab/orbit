"""Execution-graded rewards: a sandboxed executor + code-test reward hooks.

This is the first rung of orbit's environment layer. Scope decision (kept
deliberately narrow): single-turn execution grading — the model emits a full
program, the sandbox runs it against unit tests, the reward is pass/fail.
That covers the competitive-coding slices of the Nemotron-RL-Ultra blends
(``code_gen_simple_agent`` rows). A formal reset/step environment protocol is
deferred until a multi-turn consumer exists (the SWE harness — repo checkout,
agent loop, Apptainer execution — is that consumer, and a different project).

Isolation model (documented threat model, not a security boundary): untrusted
model-generated code runs as the training user in a subprocess with rlimits
(address space, CPU, file size), a scratch working directory, python ``-I``
(isolated mode), a scrubbed environment, and — when ``unshare -rn`` is
available — an empty network namespace. This matches common RL-framework
practice (NeMo-Skills/verl-style local executors); use a container-backed
executor for anything stronger.
"""

from .executor import ExecResult, network_isolation_available, run_python

__all__ = ["ExecResult", "network_isolation_available", "run_python"]
