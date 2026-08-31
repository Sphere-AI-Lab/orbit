"""Arm orbit's patches for every pytest process in this repo.

Orbit's behaviour over the vendored `miles/` tree installs through an import
hook that `import orbit` arms (see orbit/patch/). A process that never imports
orbit gets upstream's unpatched functions -- no error, just orbit's behaviour
silently absent. That is fine for production, where the entrypoint imports
orbit; it was NOT true for tests.

Eight tests in tests/test_qwen2_true_on_policy_conversion.py assert against the
converter patches without importing orbit themselves. They passed only because
some earlier-collected file had imported orbit first and `_repoint_reexports`
rewrote their already-bound names: run that file alone and all eight failed,
asserting happily against UPSTREAM's converters. Collection order decided
whether the suite tested orbit or miles.

pytest imports the rootdir conftest before collecting anything, so arming here
makes that order irrelevant. Tests that need a genuinely unarmed interpreter
(import-order regressions) already spawn a subprocess, which this cannot reach.
"""

import orbit  # noqa: F401  -- imported for the arming side effect only
