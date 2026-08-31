"""Orbit: the adapter-first extension layer over the vendored miles/ tree.

Importing ``orbit`` ARMS orbit's hash-pinned patches over miles (see
orbit/patch/) without applying them yet: each one installs when, and only when,
something actually imports the module it patches. That is what lets the vendored
files stay byte-pristine -- nothing in miles/ imports orbit to get this
behaviour, so there is nothing to merge-conflict.

Arming rather than applying is deliberate. Applying eagerly here means importing
every patch target, which drags torch and the whole vendored backend into
``import orbit``. The declarations below name their targets as strings, so this
module stays cheap.

Every process that runs orbit code imports ``orbit`` (ray starts each worker
fresh, and orbit/ray_setup.py arms the ones that do not), so the hook is armed
everywhere it is needed with no explicit call site. tools/check_arming.py is the
gate on that.
"""

from orbit.patch import install_hook as _install_hook

# Imported for the registration side effect only: the decorators in this module
# populate the patch registry. It names its targets as strings and imports no
# miles module, so this stays cheap.
from orbit.megatron import hf_export_patches as _hf_export_patches  # noqa: F401

# Same deal for the seams that replace a module-level `import` line a vendored
# file used to carry (orbit/patch/on_import.py): these modules only register, and
# do their own importing lazily inside the callback.
from orbit.megatron import bridge_plugins as _bridge_plugins  # noqa: F401

# The rest of the delegating patches, grouped by the vendored package they cover.
from orbit.megatron import megatron_utils_patches as _megatron_utils_patches  # noqa: F401
from orbit.megatron import training_utils_patches as _training_utils_patches  # noqa: F401
from orbit.utils import metric_utils_patches as _metric_utils_patches  # noqa: F401
from orbit.utils import miles_utils_patches as _miles_utils_patches  # noqa: F401
from orbit.utils import processing_utils_patches as _processing_utils_patches  # noqa: F401
from orbit.rewards import rm_hub_patches as _rm_hub_patches  # noqa: F401
from orbit.rollout import inference_rollout_patches as _inference_rollout_patches  # noqa: F401

_install_hook()
