import logging

# Import-time side effects (deep_ep TMS patch, bridge plugin registration)
# moved to runtime_hooks.install_runtime_hooks(), called from initialize.init():
# they cost a ~75s megatron.bridge+transformers import that non-trainer
# processes (anything touching ft.types via the audit event models) must not pay.

logging.getLogger("megatron").setLevel(logging.WARNING)
