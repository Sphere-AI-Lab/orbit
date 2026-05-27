import os

_printed_experimental_rollout_refactor = False


def enable_experimental_rollout_refactor() -> bool:
    result = bool(int(os.environ.get("ORBIT_EXPERIMENTAL_ROLLOUT_REFACTOR", "0")))

    global _printed_experimental_rollout_refactor
    if result and not _printed_experimental_rollout_refactor:
        print("ORBIT_EXPERIMENTAL_ROLLOUT_REFACTOR=1 is enabled (experimental feature)")
        _printed_experimental_rollout_refactor = True

    return result
