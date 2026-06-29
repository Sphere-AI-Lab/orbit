from __future__ import annotations

import inspect
from collections.abc import Callable


def call_all_samples_process_fn(fn: Callable, args, samples, data_source, /, **kwargs):
    """Invoke an all-samples hook while preserving legacy signatures."""

    sig = inspect.signature(fn)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    accepted = kwargs if has_var_keyword else {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(args, samples, data_source, **accepted)
