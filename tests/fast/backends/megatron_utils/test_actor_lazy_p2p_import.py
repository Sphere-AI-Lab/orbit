"""Unused P2P transport must not load Mooncake's native engine into the actor."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ACTOR_PATH = Path(__file__).resolve().parents[4] / "orbit/backends/megatron_utils/actor.py"
P2P_IMPORT = "update_weight.update_weight_from_distributed.p2p"
DELTA_IMPORT = "update_weight.update_weight_from_distributed.delta"


def test_fresh_actor_import_does_not_load_p2p_or_mooncake_engine():
    script = (
        "import json, sys\n"
        "from orbit.backends.megatron_utils import actor\n"
        "prefixes = ('mooncake.engine', 'orbit.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p')\n"
        "print('IMPORT_AUDIT=' + json.dumps({\n"
        "    'actor_file': actor.__file__,\n"
        "    'optional_modules': sorted(name for name in sys.modules if name.startswith(prefixes)),\n"
        "    'ray_initialized': actor.ray.is_initialized(),\n"
        "}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    audit_lines = [line for line in result.stdout.splitlines() if line.startswith("IMPORT_AUDIT=")]
    assert len(audit_lines) == 1, result.stdout
    audit = json.loads(audit_lines[0].removeprefix("IMPORT_AUDIT="))
    assert Path(audit["actor_file"]).resolve() == ACTOR_PATH
    assert not audit["ray_initialized"]
    assert audit["optional_modules"] == [], audit


@pytest.mark.parametrize(
    ("peft_method", "colocate", "mode", "selected", "expected_imports"),
    [
        ("oft", False, "p2p", "tensor", []),
        ("none", True, "p2p", "tensor", []),
        ("none", False, "broadcast", "broadcast", []),
        ("none", False, "disk-delta", "delta", [DELTA_IMPORT]),
        ("none", False, "p2p", "p2p", [P2P_IMPORT]),
    ],
)
def test_transport_selection_imports_p2p_only_in_its_branch(peft_method, colocate, mode, selected, expected_imports):
    # Execute the actual selection block without constructing a distributed actor.
    module = ast.parse(ACTOR_PATH.read_text())
    actor = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
    )
    init = next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == "init")
    selection = next(
        node
        for node in init.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and "self.args.colocate" in ast.unparse(node.test)
    )
    classes = {name: object() for name in ("tensor", "broadcast", "delta", "p2p")}
    imports = []

    def optional_import(name, globals=None, locals=None, fromlist=(), level=0):
        assert level == 1
        imports.append(name)
        if name == DELTA_IMPORT:
            return SimpleNamespace(UpdateWeightFromDiskDelta=classes["delta"])
        assert name == P2P_IMPORT
        return SimpleNamespace(UpdateWeightP2P=classes["p2p"])

    namespace = {
        "__builtins__": {"__import__": optional_import},
        "self": SimpleNamespace(
            args=SimpleNamespace(peft_method=peft_method, colocate=colocate, update_weight_transfer_mode=mode)
        ),
        "get_peft_method": lambda args: args.peft_method,
        "UpdateWeightFromTensor": classes["tensor"],
        "UpdateWeightFromDistributed": classes["broadcast"],
        "UpdateWeightP2P": classes["p2p"],
    }
    exec(compile(ast.Module(body=[selection], type_ignores=[]), str(ACTOR_PATH), "exec"), namespace)

    assert namespace["update_weight_cls"] is classes[selected]
    assert imports == expected_imports
