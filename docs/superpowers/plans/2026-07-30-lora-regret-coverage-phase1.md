# LoRA-Regret Full Coverage — Phase 1 (foundation + Llama-only matrices)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the base-model registry and the three experiment matrices that need no new checkpoints (`e1ot`, `e1short`, `e4place`), plus the figure generator, so the campaign covers the post's OpenThoughts3 rank curves, its short-run LR multiplier, and its RL layer-placement panel.

**Architecture:** A new `tools/lora_regret/models.py` holds one frozen dataclass per base model and is the single source of truth for checkpoint paths, dimensions, loss-mask type and chat template; a drift test asserts it agrees with the `orbit_plugins/model_args/*.sh` plugin it names. `Arm` gains a `model` field defaulting to `llama3.1-8b`, so every existing matrix serializes byte-identically. Three new matrices join `MATRICES`; `sweep.py` derives dimensions from the arm's model instead of trusting three CLI arguments; `analyze.py` gains the short-run-multiplier claim and a guard against quoting a sigma measured on the wrong dataset; `plot.py` turns `analyze --json` into PNGs.

**Tech Stack:** Python 3.12, pytest, matplotlib (lazy import), bash launchers, Megatron-Core via Orbit.

**Spec:** `docs/superpowers/specs/2026-07-30-lora-regret-full-coverage-design.md`

**Out of scope for this plan** (each gets its own plan, in this order): Phase 2 `e6` scaling law and the Qwen fetch/convert/scan; Phase 3 `e7` DeepMath; Phase 4 `e3moe` MoE. They all consume the registry this plan builds, which is why they come after.

## Global Constraints

- **Environment for every command:** `source /fast/zqiu/orbit-iclr/orbit_env/bin/activate`, then `cd /lustre/fast/fast/zqiu/orbit-iclr/orbit`, `export CUDA_HOME=/is/software/nvidia/cuda-13.2`, `source env.sh`. Activate **before** sourcing `env.sh`. `env.sh` is required even for CPU-only work: `megatron.core` imports `deep_ep`, which asserts on an unset `CUDA_HOME`.
- **Run tests as `pytest tests`**, never `pytest tests/fast/` — `norecursedirs` matches `tools` and `scripts` at any depth and silently skips whole directories.
- **Baseline is 593 passed, 0 failed.** Every task's final run must show **0 failed** and a total strictly greater than the previous task's. Exact cumulative totals are deliberately not asserted — a parametrized test's count depends on the registry's size, and a brittle number would read as a regression when it is not.
- **No GPU runs in this plan.** Every task is CPU-verifiable. Arms are launched by the operator later.
- **`LORA_ALPHA = 32`, `LORA_A_INIT_METHOD = "kaiming"`, `WEIGHT_DECAY = 0.0`, constant LR, no warmup, bf16** — the post's conventions, already pinned in `arms.py`. New matrices inherit them by construction; do not restate them per arm.
- **Grid points are seed 0.** Replicates at seeds 1 and 2 are sigma measurements, not grid points.
- **Never `uv cache clean`** — uv installs in symlink mode here and it guts every env pointing into the cache.
- **Commit style:** one short conventional-commit line, no AI attribution trailer.

---

### Task 1: The base-model registry

**Files:**
- Create: `tools/lora_regret/models.py`
- Test: `tests/fast/utils/test_lora_regret_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MoE`, `Model`, `MODELS: dict[str, Model]`, `DEFAULT_MODEL: str`, `get(key: str) -> Model`, `model_env(model: Model, repo_root: Path) -> dict[str, str]`, `Model.min_gpus_fullft() -> int`, `Model.param_billions: float`.

- [ ] **Step 1: Write the failing test**

Create `tests/fast/utils/test_lora_regret_models.py`:

```python
"""The registry must agree with the model_args plugin it names.

A registry that can disagree with its plugin is worse than no registry: the
wrong number is then written down twice and neither copy looks suspicious.
"""

import re
from pathlib import Path

import pytest

from tools.lora_regret.models import DEFAULT_MODEL, MODELS, get, model_env

REPO_ROOT = Path(__file__).resolve().parents[3]


def _plugin_flags(plugin_name: str) -> dict[str, str]:
    """Every `--flag value` in a model_args plugin, as a dict."""
    text = (REPO_ROOT / "orbit_plugins" / "model_args" / plugin_name).read_text(encoding="utf-8")
    return dict(re.findall(r"--([a-z0-9-]+)\s+(\S+)", text))


@pytest.mark.parametrize("key", sorted(MODELS))
def test_registry_dimensions_match_the_plugin_it_names(key):
    model = MODELS[key]
    flags = _plugin_flags(model.model_args_plugin)
    assert int(flags["hidden-size"]) == model.hidden_size
    assert int(flags["ffn-hidden-size"]) == model.ffn_size


@pytest.mark.parametrize("key", sorted(MODELS))
def test_qkv_output_size_is_the_gqa_arithmetic_not_hidden_size(key):
    """(heads + 2*kv_groups) * kv_channels. Under GQA this differs from
    hidden_size, and E3/E5's matched-parameter arithmetic is wrong without it."""
    model = MODELS[key]
    flags = _plugin_flags(model.model_args_plugin)
    heads = int(flags["num-attention-heads"])
    groups = int(flags["num-query-groups"])
    channels = int(flags["kv-channels"])
    assert model.qkv_output_size == (heads + 2 * groups) * channels


@pytest.mark.parametrize("key", sorted(MODELS))
def test_every_named_plugin_exists(key):
    assert (REPO_ROOT / "orbit_plugins" / "model_args" / MODELS[key].model_args_plugin).is_file()


def test_llama_names_the_plugin_the_launcher_already_defaults_to():
    """The dimension test above passes for llama3-8B.sh too -- both plugins carry
    the same six numbers. They differ in --use-rope-scaling, which changes every
    NLL, so the registry must not silently switch which one runs."""
    launcher = (REPO_ROOT / "examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh").read_text(
        encoding="utf-8"
    )
    assert f"model_args/{get('llama3.1-8b').model_args_plugin}" in launcher


def test_llama_is_the_default_so_existing_matrices_are_unchanged():
    assert DEFAULT_MODEL == "llama3.1-8b"
    assert get(DEFAULT_MODEL).qkv_output_size == 6144


def test_min_gpus_fullft_reproduces_the_launchers_hardcoded_guard():
    """4*P + 12*P/N GB per GPU. At 8.03B that is 32+96/N, which is the
    arithmetic the SFT launcher currently hardcodes as `>= 4`."""
    assert get("llama3.1-8b").min_gpus_fullft() == 4


def test_min_gpus_fullft_permits_one_card_for_small_models():
    """The hardcoded guard would wrongly refuse a 0.6B FullFT arm at 9.6 GB."""
    assert get("qwen3-0.6b").min_gpus_fullft() == 1
    assert get("qwen3-1.7b").min_gpus_fullft() == 1
    assert get("qwen3-4b").min_gpus_fullft() == 2


def test_min_gpus_fullft_refuses_the_moe_outright():
    """Qwen3-30B-A3B FullFT is ~168 GB/GPU at N=8. e3moe has no FullFT arm."""
    with pytest.raises(ValueError, match="does not fit"):
        get("qwen3-30b-a3b").min_gpus_fullft()


def test_unknown_key_names_the_valid_ones():
    with pytest.raises(KeyError, match="qwen3-0.6b"):
        get("qwen3-0.7b")


def test_model_env_omits_the_chat_template_for_models_that_ship_one():
    """Llama-3.1-8B base ships none, so the campaign pins a jinja file. Every
    Qwen3 base here ships one, and passing the Llama template would be wrong."""
    llama = model_env(get("llama3.1-8b"), REPO_ROOT)
    qwen = model_env(get("qwen3-4b"), REPO_ROOT)
    assert llama["CHAT_TEMPLATE_PATH"].endswith("llama3.1_pinned.jinja")
    assert qwen["CHAT_TEMPLATE_PATH"] == ""


def test_model_env_carries_the_mask_type_and_the_gpu_floor():
    env = model_env(get("llama3.1-8b"), REPO_ROOT)
    assert env["LOSS_MASK_TYPE"] == "llama3"
    assert env["MIN_GPUS_FULLFT"] == "4"
    assert env["MODEL_ARGS_FILE"].endswith("orbit_plugins/model_args/llama3.1-8B-Instruct.sh")
    assert model_env(get("qwen3-4b"), REPO_ROOT)["LOSS_MASK_TYPE"] == "qwen"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fast/utils/test_lora_regret_models.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.lora_regret.models'`

- [ ] **Step 3: Write the registry**

Create `tools/lora_regret/models.py`:

```python
"""The base models the campaign runs on, and everything a launcher needs to know.

One source of truth. Before this existed, `--hidden-size`, `--ffn-size` and
`--num-layers` were three independent CLI arguments an operator could get wrong
without the model being run changing, and a wrong `--num-layers` makes every
`adapter_params` in the ledger wrong by a constant factor.

`qkv_output_size` is a field rather than a derivation from `hidden_size` because
GQA makes the two differ: Llama-3.1-8B fuses 32 query and 2x8 key/value heads at
128 channels into 6144, against a 4096 hidden size. E3's and E5's
matched-parameter arithmetic is wrong without it.

Every field is checked against the `orbit_plugins/model_args/*.sh` plugin it
names by `tests/fast/utils/test_lora_regret_models.py`, so the registry cannot
drift from the plugin that actually configures the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HF_MODELS_DIR = "/lustre/fast/fast/zqiu/hf_models"
# Still under the *old* repo's path -- a cross-repo dependency rather than a
# break, which is why preflight checks it rather than assuming it.
MEGATRON_CKPT_DIR = "/lustre/fast/fast/zqiu/orbit-infra/orbit/checkpoints"
PINNED_LLAMA_TEMPLATE = "orbit/utils/chat_template_utils/templates/llama3.1_pinned.jinja"

# One H100. `HEADROOM_GB` is what a FullFT arm needs for activations, the
# rollout engine's share and allocator fragmentation on top of optimizer state.
# 20 GB is chosen so the formula reproduces the SFT launcher's existing
# hardcoded `>= 4` for Llama-3.1-8B exactly; changing it moves every model's
# GPU floor, so it is a campaign-wide constant rather than a tuning knob.
GPU_GB = 80.0
HEADROOM_GB = 20.0
# Megatron shards optimizer state across DP but not weights or grads, so only
# the second term divides. Candidate DP sizes are powers of two because that is
# what the launcher's placement supports.
DP_CANDIDATES = (1, 2, 4, 8)


@dataclass(frozen=True)
class MoE:
    num_experts: int
    moe_ffn_size: int
    topk: int


@dataclass(frozen=True)
class Model:
    key: str
    hf_checkpoint: str
    megatron_load: str
    model_args_plugin: str
    hidden_size: int
    ffn_size: int
    num_layers: int
    qkv_output_size: int
    loss_mask_type: str
    param_billions: float
    # None means "the model ships its own and the launcher must not override
    # it". Llama-3.1-8B *base* ships none, which is why the campaign pins one.
    chat_template: str | None = None
    moe: MoE | None = None

    def per_gpu_fullft_gb(self, dp: int) -> float:
        """bf16 weights + bf16 grads replicated; fp32 master + Adam moments sharded.

        2 + 2 bytes/param replicated = 4*P GB per GPU for P in billions;
        4 + 8 bytes/param sharded = 12*P/N.
        """
        return 4.0 * self.param_billions + 12.0 * self.param_billions / dp

    def min_gpus_fullft(self) -> int:
        """Smallest DP size whose optimizer state leaves room for activations.

        Raises rather than returning a number that does not exist: for
        Qwen3-30B-A3B no supported DP size fits, and returning 8 would let an
        arm start and OOM twenty minutes into a reserved node.
        """
        budget = GPU_GB - HEADROOM_GB
        for dp in DP_CANDIDATES:
            if self.per_gpu_fullft_gb(dp) <= budget:
                return dp
        raise ValueError(
            f"{self.key} full fine-tuning does not fit: "
            f"{self.per_gpu_fullft_gb(max(DP_CANDIDATES)):.0f} GB/GPU at DP="
            f"{max(DP_CANDIDATES)} against a {budget:.0f} GB budget. "
            "Use a PEFT method, or more nodes than this formula models."
        )


MODELS: dict[str, Model] = {
    "llama3.1-8b": Model(
        key="llama3.1-8b",
        hf_checkpoint=f"{HF_MODELS_DIR}/Llama-3.1-8B",
        megatron_load=f"{MEGATRON_CKPT_DIR}/Llama-3.1-8B_torch_dist",
        # NOT llama3-8B.sh, which has the same six dimensions and differs in
        # --max-position-embeddings (8192 vs 131072) and, decisively,
        # --use-rope-scaling --rotary-scaling-factor 8.0. RoPE scaling changes
        # positional encoding, so it changes every NLL. This is the plugin the
        # SFT launcher has defaulted to all along; the registry records that
        # rather than quietly switching it. The "Instruct" in the filename names
        # the config, not the weights -- the architecture is identical.
        model_args_plugin="llama3.1-8B-Instruct.sh",
        hidden_size=4096, ffn_size=14336, num_layers=32, qkv_output_size=6144,
        loss_mask_type="llama3", param_billions=8.03,
        chat_template=PINNED_LLAMA_TEMPLATE,
    ),
    "qwen3-0.6b": Model(
        key="qwen3-0.6b",
        hf_checkpoint=f"{HF_MODELS_DIR}/Qwen3-0.6B",
        megatron_load=f"{MEGATRON_CKPT_DIR}/Qwen3-0.6B_torch_dist",
        model_args_plugin="qwen3-0.6B.sh",
        hidden_size=1024, ffn_size=3072, num_layers=28, qkv_output_size=4096,
        loss_mask_type="qwen", param_billions=0.752,
    ),
    "qwen3-1.7b": Model(
        key="qwen3-1.7b",
        hf_checkpoint=f"{HF_MODELS_DIR}/Qwen3-1.7B",
        megatron_load=f"{MEGATRON_CKPT_DIR}/Qwen3-1.7B_torch_dist",
        model_args_plugin="qwen3-1.7B.sh",
        hidden_size=2048, ffn_size=6144, num_layers=28, qkv_output_size=4096,
        loss_mask_type="qwen", param_billions=1.72,
    ),
    "qwen3-4b": Model(
        key="qwen3-4b",
        hf_checkpoint=f"{HF_MODELS_DIR}/Qwen3-4B",
        megatron_load=f"{MEGATRON_CKPT_DIR}/Qwen3-4B_torch_dist",
        model_args_plugin="qwen3-4B.sh",
        hidden_size=2560, ffn_size=9728, num_layers=36, qkv_output_size=6144,
        loss_mask_type="qwen", param_billions=4.02,
    ),
    "qwen3-8b": Model(
        key="qwen3-8b",
        hf_checkpoint=f"{HF_MODELS_DIR}/Qwen3-8B",
        megatron_load=f"{MEGATRON_CKPT_DIR}/Qwen3-8B_torch_dist",
        model_args_plugin="qwen3-8B.sh",
        hidden_size=4096, ffn_size=12288, num_layers=36, qkv_output_size=6144,
        loss_mask_type="qwen", param_billions=8.19,
    ),
    "qwen3-30b-a3b": Model(
        key="qwen3-30b-a3b",
        hf_checkpoint=f"{HF_MODELS_DIR}/Qwen3-30B-A3B",
        megatron_load=f"{MEGATRON_CKPT_DIR}/Qwen3-30B-A3B_torch_dist",
        model_args_plugin="qwen3-30B-A3B.sh",
        hidden_size=2048, ffn_size=6144, num_layers=48, qkv_output_size=5120,
        loss_mask_type="qwen", param_billions=30.5,
        moe=MoE(num_experts=128, moe_ffn_size=768, topk=8),
    ),
}

DEFAULT_MODEL = "llama3.1-8b"


def get(key: str) -> Model:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known: {sorted(MODELS)}") from None


def model_env(model: Model, repo_root: Path) -> dict[str, str]:
    """Environment overrides that point a launcher at this model.

    `CHAT_TEMPLATE_PATH` is the **empty string** for models that ship their own
    template, not an omitted key. The launcher reads it with the no-colon
    `${CHAT_TEMPLATE_PATH-default}` form, so empty means "omit the flag" while
    unset means "use the Llama default" -- the same distinction `LABEL_KEY`
    makes, and for the same reason: the colon form would re-default an
    intentionally empty value.
    """
    template = str(repo_root / model.chat_template) if model.chat_template else ""
    try:
        min_gpus = str(model.min_gpus_fullft())
    except ValueError:
        # A model with no viable FullFT DP size still runs PEFT arms. Passing a
        # floor larger than any node forces the launcher's own guard to refuse
        # a FullFT arm rather than letting it start.
        min_gpus = str(max(DP_CANDIDATES) + 1)
    return {
        "MODEL_KEY": model.key,
        "HF_CKPT": model.hf_checkpoint,
        "MEGATRON_LOAD": model.megatron_load,
        "MODEL_ARGS_FILE": str(repo_root / "orbit_plugins" / "model_args" / model.model_args_plugin),
        "LOSS_MASK_TYPE": model.loss_mask_type,
        "CHAT_TEMPLATE_PATH": template,
        "MIN_GPUS_FULLFT": min_gpus,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/fast/utils/test_lora_regret_models.py -q`
Expected: PASS, 25 tests — three parametrized over six models (18) plus seven scalar.

- [ ] **Step 5: Run the full suite for regressions**

Run: `pytest tests -q 2>&1 | tail -3`
Expected: **0 failed**, total 593 + 25.

- [ ] **Step 6: Commit**

```bash
git add tools/lora_regret/models.py tests/fast/utils/test_lora_regret_models.py
git commit -m "feat(lora_regret): add the base-model registry"
```

---

### Task 2: `Arm.model` and model-derived dimensions in the sweep

**Files:**
- Modify: `tools/lora_regret/arms.py` (the `Arm` dataclass; add `num_rollout`)
- Modify: `tools/lora_regret/sweep.py` (`run_arm`, `main`)
- Test: `tests/fast/utils/test_lora_regret_sweep.py`

**Interfaces:**
- Consumes: `models.get`, `models.model_env`, `models.DEFAULT_MODEL` from Task 1.
- Produces: `Arm.model: str = "llama3.1-8b"`, `Arm.num_rollout: int | None = None`. `run_arm` now merges `model_env(...)` into its overrides. `sweep.main` accepts `--hidden-size/--ffn-size/--num-layers` as optional and rejects contradicting values with exit 2.

- [ ] **Step 1: Write the failing test**

Append to `tests/fast/utils/test_lora_regret_sweep.py`:

```python
class TestModelRegistryWiring:
    """The three dimension flags are derived, and a contradicting value is a
    hard error rather than a silent preference for one of two sources."""

    def test_every_existing_arm_defaults_to_llama(self):
        from tools.lora_regret.arms import MATRICES

        for name in ("e1", "e2", "e3", "e4", "e5scout", "sft82"):
            built = MATRICES[name](4096, 14336, 0, 1e-4 if name == "e5" else None, None)
            assert {arm.model for arm in built} == {"llama3.1-8b"}, name

    def test_dry_run_exports_the_models_checkpoint_and_mask_type(self, tmp_path, capsys):
        from tools.lora_regret.arms import ALL_MODULES, Arm
        from tools.lora_regret.sweep import run_arm

        arm = Arm("probe", "lora", 16, None, ALL_MODULES, 2.5e-4, 0, dataset="tulu3")
        run_arm(arm, tmp_path, tmp_path / "r.jsonl", dry_run=True)
        printed = capsys.readouterr().out
        assert "LOSS_MASK_TYPE=llama3" in printed
        assert "MIN_GPUS_FULLFT=4" in printed
        assert "Llama-3.1-8B_torch_dist" in printed

    def test_num_rollout_reaches_the_launcher_environment(self):
        from tools.lora_regret.arms import ALL_MODULES, Arm, arm_env

        arm = Arm("probe", "lora", 256, None, ALL_MODULES, 2.5e-4, 0, num_rollout=100)
        assert arm_env(arm)["NUM_ROLLOUT"] == "100"

    def test_full_epoch_still_wins_over_num_rollout(self):
        """`full_epoch` sets NUM_ROLLOUT to the empty string so the launcher
        re-derives the epoch. A stale num_rollout must not resurrect a cap."""
        from tools.lora_regret.arms import ALL_MODULES, Arm, arm_env

        arm = Arm("probe", "lora", 256, None, ALL_MODULES, 2.5e-4, 0,
                  num_rollout=100, full_epoch=True)
        assert arm_env(arm)["NUM_ROLLOUT"] == ""

    def test_contradicting_hidden_size_exits_two(self, tmp_path):
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "tools.lora_regret.sweep", "--matrix", "e1",
             "--hidden-size", "9999", "--dry-run", "--results", str(tmp_path / "r.jsonl")],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert proc.returncode == 2
        assert "9999" in proc.stderr and "llama3.1-8b" in proc.stderr

    def test_dimension_flags_are_now_optional(self, tmp_path):
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "tools.lora_regret.sweep", "--matrix", "e1",
             "--dry-run", "--results", str(tmp_path / "r.jsonl")],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert proc.returncode == 0
        assert len(proc.stdout.strip().splitlines()) == 40
```

`REPO_ROOT` already exists in this test module; if it does not, add
`REPO_ROOT = Path(__file__).resolve().parents[3]` next to the other module-level constants.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fast/utils/test_lora_regret_sweep.py::TestModelRegistryWiring -q`
Expected: FAIL — `TypeError: Arm.__init__() got an unexpected keyword argument 'num_rollout'`

- [ ] **Step 3: Add the two `Arm` fields**

In `tools/lora_regret/arms.py`, inside the `Arm` dataclass, after `eval_nll_interval`:

```python
    # Which base model this arm runs on. Defaults to the campaign's original
    # anchor so every pre-registry matrix serializes byte-identically and every
    # ledger written before the registry existed stays valid.
    model: str = "llama3.1-8b"
    # An explicit rollout cap. `full_epoch` is the opposite request and wins if
    # both are set: E1-2's arms must re-derive the epoch even if a stale
    # NUM_ROLLOUT is exported in the operator's shell.
    num_rollout: int | None = None
```

In `arm_env`, replace the `full_epoch` block with:

```python
    if arm.full_epoch:
        # The EMPTY STRING, not an omitted key. The launcher spells it
        # ${NUM_ROLLOUT:-$((...))} -- the colon form re-derives on an empty
        # value, so this both requests the full epoch and immunises the arm
        # against a NUM_ROLLOUT=2000 left exported in your shell from E1-1.
        env["NUM_ROLLOUT"] = ""
    elif arm.num_rollout is not None:
        env["NUM_ROLLOUT"] = str(arm.num_rollout)
```

- [ ] **Step 4: Wire the registry into `run_arm` and `main`**

In `tools/lora_regret/sweep.py`, add to the imports:

```python
from tools.lora_regret.models import get as get_model, model_env
```

In `run_arm`, change the `overrides` construction so the model's environment is
applied *before* the per-arm names, and the arm's own settings win over the
model's defaults:

```python
    overrides = dict(model_env(get_model(arm.model), repo_root))
    overrides.update(arm_env(arm))
    overrides.update(
        {
            "LAUNCHER_NAME": arm.name,
            ...
        }
    )
```

In `main`, make the three flags optional and validate them:

```python
    parser.add_argument("--hidden-size", type=int, default=None,
                        help="Deprecated: derived from the arm's model. Kept so the "
                             "runbook's existing commands still work; a value that "
                             "contradicts the model is an error, not a preference.")
    parser.add_argument("--ffn-size", type=int, default=None, help="Deprecated; see --hidden-size.")
    parser.add_argument("--num-layers", type=int, default=None, help="Deprecated; see --hidden-size.")
```

After the existing `parser.error` guards, add:

```python
    default_model = get_model(DEFAULT_MODEL)
    for flag, given, derived in (
        ("--hidden-size", args.hidden_size, default_model.hidden_size),
        ("--ffn-size", args.ffn_size, default_model.ffn_size),
        ("--num-layers", args.num_layers, default_model.num_layers),
    ):
        if given is not None and given != derived:
            parser.error(
                f"{flag}={given} contradicts model {default_model.key!r}, which has "
                f"{derived}. These are derived from the arm's model now; drop the flag."
            )
```

Import `DEFAULT_MODEL` alongside `get_model`. Replace the matrix call and the
`adapter_param_count` call so both read the model:

```python
    arms = MATRICES[args.matrix](
        default_model.hidden_size, default_model.ffn_size,
        args.seed, args.oft_lr_centre, recovered,
    )
    ...
        model = get_model(arm.model)
        run_arm(
            arm, repo_root, args.results, args.dry_run,
            launcher=launcher, metric=metric,
            adapter_params=adapter_param_count(
                arm, model.hidden_size, model.ffn_size, model.num_layers,
                qkv_output_size=model.qkv_output_size,
            ),
        )
```

Replace `_oft_match_summary(args.hidden_size)` with `_oft_match_summary(default_model.hidden_size)`.

> **Note for the implementer:** the matrix builders still take `hidden`/`ffn`
> positionally. That signature is unchanged on purpose — every matrix in this
> plan is single-model, and threading a per-arm model through the builders is
> Phase 2's job, where `e6` actually needs it. Do not change it here.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/fast/utils/test_lora_regret_sweep.py -q`
Expected: PASS, including the six new tests.

- [ ] **Step 6: Confirm the dry-run output is unchanged in count for every existing matrix**

Run:
```bash
for m in e1 e2 e3 e4 e5scout; do
  echo -n "$m "
  python -m tools.lora_regret.sweep --matrix $m --dry-run 2>/dev/null | wc -l
done
```
Expected: `e1 40`, `e2 36`, `e3 20`, `e4 16`, `e5scout 5`.

- [ ] **Step 7: Run the full suite**

Run: `pytest tests -q 2>&1 | tail -3`
Expected: **0 failed**, six tests more than Task 1's total.

- [ ] **Step 8: Commit**

```bash
git add tools/lora_regret/arms.py tools/lora_regret/sweep.py tests/fast/utils/test_lora_regret_sweep.py
git commit -m "feat(lora_regret): derive model dimensions from the registry"
```

---

### Task 3: Launcher — GPU floor as a formula, chat template overridable

**Files:**
- Modify: `examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh:134` and `:247-252`
- Modify: `examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh:226` (same guard)
- Test: `tests/test_lora_regret_rl_launcher.py`, `tests/fast/utils/test_lora_regret_models.py`

**Interfaces:**
- Consumes: `MIN_GPUS_FULLFT` and `CHAT_TEMPLATE_PATH` from `model_env` (Task 1).
- Produces: no Python interface. Both launchers honour `MIN_GPUS_FULLFT` (default 4) and `CHAT_TEMPLATE_PATH` (no-colon default: the pinned Llama jinja).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lora_regret_rl_launcher.py`:

```python
SFT_LAUNCHER = REPO_ROOT / "examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh"


def _sft_text() -> str:
    return SFT_LAUNCHER.read_text(encoding="utf-8")


def test_both_launchers_read_the_gpu_floor_from_the_environment():
    """The hardcoded `< 4` is right for Llama-3.1-8B and wrong for every other
    model: Qwen3-0.6B FullFT is 9.6 GB and fits on one card. The registry
    computes the floor; the launcher must not second-guess it."""
    for text in (_text(), _sft_text()):
        assert "MIN_GPUS_FULLFT:-4" in text
        assert "GPUS_PER_NODE < 4" not in text
        assert 'GPUS_PER_NODE < MIN_GPUS_FULLFT' in text


def test_sft_launcher_uses_the_no_colon_form_for_the_chat_template():
    """Empty must mean "omit the flag" (Qwen3 ships its own template), while
    unset must mean "use the pinned Llama one". The colon form collapses those
    two into one, which is the LABEL_KEY bug, one flag over."""
    text = _sft_text()
    assert "${CHAT_TEMPLATE_PATH-" in text
    assert "${CHAT_TEMPLATE_PATH:-" not in text


def test_sft_launcher_still_defaults_to_the_pinned_llama_template():
    """Llama-3.1-8B base ships no chat template at all. A run with none applied
    would train on raw concatenated text."""
    assert "llama3.1_pinned.jinja" in _sft_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lora_regret_rl_launcher.py -q -k "gpu_floor or chat_template"`
Expected: FAIL — `assert 'MIN_GPUS_FULLFT:-4' in text`

- [ ] **Step 3: Edit the SFT launcher**

Replace line 134 of `examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh`
(inside `ROLLOUT_ARGS`, currently an unconditional `--chat-template-path ...`)
with nothing, and add this block immediately after the `ROLLOUT_ARGS=( ... )`
array closes:

```bash
# Llama-3.1-8B *base* ships no chat template, so the campaign pins one. Qwen3
# base models ship their own and must not be given Llama's. The no-colon form
# distinguishes "unset" (use the pinned default) from "set to empty" (the model
# has its own -- omit the flag); the colon form would collapse both.
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH-${ORBIT_ROOT}/orbit/utils/chat_template_utils/templates/llama3.1_pinned.jinja}
if [[ -n "${CHAT_TEMPLATE_PATH}" ]]; then
    ROLLOUT_ARGS+=( --chat-template-path "${CHAT_TEMPLATE_PATH}" )
fi
```

Replace the FullFT guard at lines 247-252 with:

```bash
        # Per-GPU optimizer state is 4*P + 12*P/N GB for P billion parameters:
        # bf16 weights and grads replicated, fp32 master and Adam moments
        # sharded by Megatron's distributed optimizer. At 8.03B that is
        # 32+96/N, so N=4 is 56 GB and N=2 is 80 GB with nothing left for
        # activations. tools/lora_regret/models.py computes the floor per model
        # and exports it; 4 is the Llama-3.1-8B value, kept as the default so a
        # hand-run arm behaves as before.
        MIN_GPUS_FULLFT=${MIN_GPUS_FULLFT:-4}
        if (( GPUS_PER_NODE < MIN_GPUS_FULLFT )) && ! is_true "${ALLOW_SMALL_FULLFT:-0}"; then
            echo "PEFT_METHOD=none (full fine-tuning) needs GPUS_PER_NODE>=${MIN_GPUS_FULLFT}; got ${GPUS_PER_NODE}." >&2
            echo "Per-GPU optimizer state is 4*P+12*P/N GB. Set ALLOW_SMALL_FULLFT=1 to override." >&2
            exit 2
        fi
```

- [ ] **Step 4: Apply the same guard to the RL launcher**

Make the identical `MIN_GPUS_FULLFT` substitution at
`examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh:226`. Do **not**
add the chat-template block there — the RL launcher takes plain-string prompts
and applies no template.

- [ ] **Step 5: Verify both guards still fire**

Run:
```bash
GPUS_PER_NODE=1 PEFT_METHOD=none bash examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh; echo "exit=$?"
GPUS_PER_NODE=1 MIN_GPUS_FULLFT=1 PEFT_METHOD=none TRAIN_ROWS=8 NUM_ROLLOUT=0 \
  bash examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh 2>&1 | grep -c "needs GPUS_PER_NODE"
```
Expected: first prints `needs GPUS_PER_NODE>=4` and `exit=2`; second prints `0`
(the guard did not fire at a floor of 1). The second command will fail later for
unrelated reasons — only the grep count matters.

- [ ] **Step 6: Run the tests**

Run: `pytest tests -q 2>&1 | tail -3`
Expected: **0 failed**, three tests more than Task 2's total.

- [ ] **Step 7: Commit**

```bash
git add examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh \
        examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh \
        tests/test_lora_regret_rl_launcher.py
git commit -m "feat(lora_regret): make the FullFT GPU floor and chat template per-model"
```

---

### Task 4: `e1ot` — rank curves on OpenThoughts3

**Files:**
- Modify: `tools/lora_regret/arms.py` (add `e1ot_arms`, `MATRICES["e1ot"]`)
- Modify: `tools/lora_regret/sweep.py` (`MATRIX_LAUNCHERS`, `MATRIX_METRICS`)
- Modify: `tools/lora_regret/preflight.py` (`EXPECTED_ARMS`)
- Test: `tests/fast/utils/test_lora_regret_arms_coverage.py` (create)

**Interfaces:**
- Consumes: `Arm`, `lr_grid`, `FULL_LR_CENTRE`, `LORA_LR_CENTRE`, `ALL_MODULES` from `arms.py`.
- Produces: `e1ot_arms(seed: int = 0) -> list[Arm]`, `E1OT_EVAL_INTERVAL: int`, `MATRICES["e1ot"]`.

- [ ] **Step 1: Write the failing test**

Create `tests/fast/utils/test_lora_regret_arms_coverage.py`:

```python
"""The three matrices that close the post's coverage gaps on Llama-3.1-8B."""

import pytest

from tools.lora_regret.arms import (
    ALL_MODULES,
    ATTN_MODULES,
    MATRICES,
    MLP_MODULES,
    e1ot_arms,
)

HIDDEN, FFN = 4096, 14336


class TestE1Ot:
    def test_forty_arms_matching_e1s_shape(self):
        arms = e1ot_arms()
        assert len(arms) == 40
        assert sum(1 for a in arms if a.method == "full") == 5
        assert {a.rank for a in arms if a.method == "lora"} == {1, 4, 16, 64, 128, 256, 512}

    def test_every_arm_reads_openthoughts3(self):
        """E1 is Tulu3; this matrix exists precisely to be the other dataset."""
        assert {a.dataset for a in e1ot_arms()} == {"openthoughts3"}

    def test_the_epoch_is_short_enough_that_no_second_long_matrix_is_needed(self):
        """10,000 rows at batch 32 is 312 steps, so these arms run a full epoch
        and yield both the argmins and the curves. `full_epoch` must be set, or
        the launcher caps them at its own NUM_ROLLOUT default."""
        assert all(a.full_epoch for a in e1ot_arms())

    def test_eval_interval_is_about_one_percent_of_the_epoch(self):
        """~100 trace points, which is what C1's departure detector needs."""
        assert {a.eval_nll_interval for a in e1ot_arms()} == {3}

    def test_it_is_registered(self):
        assert len(MATRICES["e1ot"](HIDDEN, FFN, 0, None, None)) == 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fast/utils/test_lora_regret_arms_coverage.py -q`
Expected: FAIL — `ImportError: cannot import name 'e1ot_arms'`

- [ ] **Step 3: Write the matrix**

In `tools/lora_regret/arms.py`, after `e1long_arms`, add the constant near
`E1LONG_EVAL_INTERVAL`:

```python
# OpenThoughts3's 10,000-row subset is 312 optimizer steps at batch 32, and ~1%
# of that is 3 -- about 100 trace points. (The launcher ceilings: (10000+31)//32
# = 313.) The contrast with E1LONG_EVAL_INTERVAL
# (293) is the whole reason e1ot needs no separate long matrix: one epoch here
# is affordable at all 40 arms, and one epoch on Tulu3 is not.
E1OT_EVAL_INTERVAL = 3
```

and the builder:

```python
def e1ot_arms(seed: int = 0) -> list[Arm]:
    """E1-OT: the rank ladder on OpenThoughts3 -- the post's second SFT dataset.

    Identical in shape to :func:`e1_arms` and deliberately so: the post's claim
    is that the rank/capacity behaviour is a property of LoRA rather than of
    Tulu3, and a differently-shaped grid would make a difference in the result
    indistinguishable from a difference in the design.

    One epoch here is 312 optimizer steps against Tulu3's 29,323, so these arms
    run to completion and yield the argmins *and* the learning curves. There is
    no `e1otlong`; the E1-1/E1-2 split exists only because a Tulu3 epoch at 40
    arms is unaffordable.

    The held-out split is 100 rows against Tulu3's 1,000, so its noise floor is
    a different number: run seeds 1 and 2 of one arm into a separate sigma
    ledger before quoting anything against it (runbook section 7).
    """
    arms: list[Arm] = []
    for lr in lr_grid(FULL_LR_CENTRE):
        arms.append(
            Arm(_name("full", "na", "", lr, seed), "full", None, None, "", lr, seed,
                dataset="openthoughts3", full_epoch=True,
                eval_nll_interval=E1OT_EVAL_INTERVAL)
        )
    for rank in E1LONG_RANKS:
        for lr in lr_grid(LORA_LR_CENTRE):
            arms.append(
                Arm(_name("lora", f"r{rank}", ALL_MODULES, lr, seed), "lora", rank, None,
                    ALL_MODULES, lr, seed, dataset="openthoughts3", full_epoch=True,
                    eval_nll_interval=E1OT_EVAL_INTERVAL)
            )
    return arms
```

Add to `MATRICES`:

```python
    "e1ot": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e1ot_arms(seed=seed),
```

In `tools/lora_regret/sweep.py`, add `"e1ot": LAUNCHER` to `MATRIX_LAUNCHERS` and
`"e1ot": "nll"` to `MATRIX_METRICS`. In `tools/lora_regret/preflight.py`, add
`"e1ot": 40` to `EXPECTED_ARMS`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/fast/utils/test_lora_regret_arms_coverage.py -q`
Expected: PASS, 5 tests.

- [ ] **Step 5: Verify the dry run**

Run: `python -m tools.lora_regret.sweep --matrix e1ot --dry-run 2>/dev/null | wc -l`
Expected: `40`

Run: `python -m tools.lora_regret.sweep --matrix e1ot --dry-run 2>/dev/null | head -1`
Expected: a line containing `EVAL_NLL_INTERVAL=3`, `NUM_ROLLOUT=` (empty), and
`openthoughts3_train.jsonl`.

- [ ] **Step 6: Commit**

```bash
git add tools/lora_regret/arms.py tools/lora_regret/sweep.py tools/lora_regret/preflight.py \
        tests/fast/utils/test_lora_regret_arms_coverage.py
git commit -m "feat(lora_regret): add the e1ot rank ladder on OpenThoughts3"
```

---

### Task 5: `e1short` — the short-run LR multiplier

**Files:**
- Modify: `tools/lora_regret/arms.py` (add `e1short_arms`, `MATRICES["e1short"]`)
- Modify: `tools/lora_regret/sweep.py`, `tools/lora_regret/preflight.py`
- Test: `tests/fast/utils/test_lora_regret_arms_coverage.py`

**Interfaces:**
- Consumes: `lr_grid`, `Arm.num_rollout` from Task 2.
- Produces: `e1short_arms(seed: int = 0) -> list[Arm]`, `E1SHORT_ROLLOUTS: int`, `E1SHORT_STEP_DECADES: float`, `MATRICES["e1short"]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/fast/utils/test_lora_regret_arms_coverage.py`:

```python
class TestE1Short:
    def test_fourteen_arms_two_methods_seven_lrs(self):
        from tools.lora_regret.arms import e1short_arms

        arms = e1short_arms()
        assert len(arms) == 14
        assert sum(1 for a in arms if a.method == "full") == 7
        assert {a.rank for a in arms if a.method == "lora"} == {256}

    def test_the_grid_resolves_fifteen_from_ten(self):
        """The claim is a 15x multiplier against a long-run 10x. That is a
        factor of 1.5 == 0.176 decades. On the campaign's standard 0.3-decade
        grid, adjacent points differ by 2x and the effect is invisible, so the
        spacing is a requirement of the claim, not a preference."""
        import math

        from tools.lora_regret.arms import e1short_arms

        lrs = sorted({a.lr for a in e1short_arms() if a.method == "full"})
        steps = [math.log10(b / a) for a, b in zip(lrs, lrs[1:])]
        assert max(steps) <= 0.155, steps
        assert math.log10(1.5) > max(steps), "grid cannot resolve 15x from 10x"

    def test_both_methods_get_the_fine_grid(self):
        """The claim is a ratio of two argmins; a coarse denominator ruins it
        as surely as a coarse numerator."""
        from tools.lora_regret.arms import e1short_arms

        arms = e1short_arms()
        assert len({a.lr for a in arms if a.method == "full"}) == 7
        assert len({a.lr for a in arms if a.method == "lora"}) == 7

    def test_one_hundred_rollouts_and_a_cheap_eval_interval(self):
        """At interval 1 a 100-step arm spends ~113 min evaluating against ~14
        min training. The trace is not what this stage measures."""
        from tools.lora_regret.arms import e1short_arms

        arms = e1short_arms()
        assert {a.num_rollout for a in arms} == {100}
        assert {a.eval_nll_interval for a in arms} == {10}
        assert not any(a.full_epoch for a in arms)

    def test_it_runs_on_tulu3_so_e1s_sigma_applies(self):
        from tools.lora_regret.arms import e1short_arms

        assert {a.dataset for a in e1short_arms()} == {"tulu3"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fast/utils/test_lora_regret_arms_coverage.py::TestE1Short -q`
Expected: FAIL — `ImportError: cannot import name 'e1short_arms'`

- [ ] **Step 3: Write the matrix**

In `tools/lora_regret/arms.py`, after `e1ot_arms`:

```python
# The post: short runs (~100 steps) want a ~15x multiplier where long runs
# converge to ~10x, because B's zero initialization acts as an implicit warmup
# that has not finished in 100 steps.
E1SHORT_ROLLOUTS = 100
# 0.15 rather than the campaign's 0.3. Resolving 15x from 10x means resolving a
# factor of 1.5, which is log10(1.5) = 0.176 decades; on a 0.3-decade grid the
# adjacent points differ by 2x and the effect cannot appear. This is a
# requirement of the claim -- do not unify it with lr_grid's default.
E1SHORT_STEP_DECADES = 0.15
E1SHORT_POINTS = 7
# 100/10 = 10 measurements, ~11 min of eval against ~14 min of training. At the
# campaign's usual 1% the arm would spend 8x longer evaluating than training.
E1SHORT_EVAL_INTERVAL = 10


def e1short_arms(seed: int = 0) -> list[Arm]:
    """E1-short: the ~100-step learning-rate multiplier (the second half of C2).

    FullFT and LoRA r256 only. The claim is about the *ratio* of two argmins at
    a short horizon, so the rank ladder adds nothing and every extra rank would
    dilute the resolution budget that the 0.15-decade grid is spending.

    Centred on the same 2.5e-5 / 2.5e-4 as `e1`, so the short-run and long-run
    ratios are read off grids that agree at their midpoints and any difference
    between them is a difference in the optimum rather than in the grid.
    """
    grid = lambda centre: lr_grid(  # noqa: E731 -- local alias, three uses
        centre, n=E1SHORT_POINTS, step_decades=E1SHORT_STEP_DECADES
    )
    arms: list[Arm] = []
    for lr in grid(FULL_LR_CENTRE):
        arms.append(
            Arm(_name("full", "na", "", lr, seed, extra="short"), "full", None, None, "",
                lr, seed, dataset="tulu3", num_rollout=E1SHORT_ROLLOUTS,
                eval_nll_interval=E1SHORT_EVAL_INTERVAL)
        )
    for lr in grid(LORA_LR_CENTRE):
        arms.append(
            Arm(_name("lora", "r256", ALL_MODULES, lr, seed, extra="short"), "lora", 256,
                None, ALL_MODULES, lr, seed, dataset="tulu3",
                num_rollout=E1SHORT_ROLLOUTS, eval_nll_interval=E1SHORT_EVAL_INTERVAL)
        )
    return arms
```

Register it in `MATRICES` (`"e1short"`), `MATRIX_LAUNCHERS` (`LAUNCHER`),
`MATRIX_METRICS` (`"nll"`) and `EXPECTED_ARMS` (`14`).

- [ ] **Step 4: Run the tests**

Run: `pytest tests/fast/utils/test_lora_regret_arms_coverage.py -q`
Expected: PASS, 10 tests.

- [ ] **Step 5: Verify the dry run and read the grid**

Run:
```bash
python -m tools.lora_regret.sweep --matrix e1short --dry-run 2>/dev/null | wc -l
python -c "
from tools.lora_regret.arms import e1short_arms
print(sorted({a.lr for a in e1short_arms() if a.method=='full'}))"
```
Expected: `14`, then a 7-point grid whose middle value is `2.5e-05` and whose
neighbours are within a factor of ~1.41.

- [ ] **Step 6: Commit**

```bash
git add tools/lora_regret/arms.py tools/lora_regret/sweep.py tools/lora_regret/preflight.py \
        tests/fast/utils/test_lora_regret_arms_coverage.py
git commit -m "feat(lora_regret): add the e1short matrix for the 100-step LR multiplier"
```

---

### Task 6: `e4place` — layer placement under RL (8 arms)

**Files:**
- Modify: `tools/lora_regret/arms.py` (add `e4place_arms`, `MATRICES["e4place"]`)
- Modify: `tools/lora_regret/sweep.py`, `tools/lora_regret/preflight.py`
- Test: `tests/fast/utils/test_lora_regret_arms_coverage.py`

**Interfaces:**
- Consumes: `matched_mlp_rank` (already imported in `arms.py`), `RL_LORA_LR_CENTRE`, `RL_MIX_DATASET`.
- Produces: `e4place_arms(hidden_size, ffn_size, seed=0, qkv_output_size=LLAMA31_8B_QKV_OUTPUT) -> list[Arm]`, `MATRICES["e4place"]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/fast/utils/test_lora_regret_arms_coverage.py`:

```python
class TestE4Place:
    def test_eight_arms_two_placements_four_lrs(self):
        from tools.lora_regret.arms import e4place_arms

        arms = e4place_arms(HIDDEN, FFN)
        assert len(arms) == 8
        assert {a.target_modules for a in arms} == {ATTN_MODULES, MLP_MODULES}

    def test_it_does_not_restate_any_arm_e4_already_runs(self):
        """e4's LoRA r256 all-modules cell uses this exact grid, so an
        all-modules cell here would be four byte-identical arm names -- four
        re-run RL arms at 8 GPUs each, and a duplicate key if both ledgers are
        ever globbed into analyze together."""
        from tools.lora_regret.arms import e4_arms, e4place_arms

        assert not ({a.name for a in e4_arms()} & {a.name for a in e4place_arms(HIDDEN, FFN)})

    def test_the_mlp_rank_is_e3s_solved_match_not_a_round_number(self):
        """Comparing attention r256 against MLP r256 would compare placement and
        capacity at once. Orbit fuses qkv and gate+up, so the post's own
        attention-256/MLP-128 pair is not matched in this layout either."""
        from orbit.utils.peft_param_match import matched_mlp_rank
        from tools.lora_regret.arms import LLAMA31_8B_QKV_OUTPUT, e4place_arms

        expected = matched_mlp_rank(256, HIDDEN, FFN, LLAMA31_8B_QKV_OUTPUT)
        mlp = {a.rank for a in e4place_arms(HIDDEN, FFN) if a.target_modules == MLP_MODULES}
        assert mlp == {expected}
        assert expected != 256 and expected != 128

    def test_no_fullft_arm(self):
        """The post's RL placement panel is a comparison within LoRA."""
        from tools.lora_regret.arms import e4place_arms

        assert all(a.method == "lora" for a in e4place_arms(HIDDEN, FFN))

    def test_it_shares_e4s_data_and_half_decade_grid(self):
        """So the placement result and the rank result are read off comparable
        arms rather than off two differently-shaped grids."""
        import math

        from tools.lora_regret.arms import RL_MIX_DATASET, e4_arms, e4place_arms

        place = e4place_arms(HIDDEN, FFN)
        assert {a.dataset for a in place} == {RL_MIX_DATASET}
        lrs = sorted({a.lr for a in place if a.target_modules == ATTN_MODULES})
        steps = [math.log10(b / a) for a, b in zip(lrs, lrs[1:])]
        assert all(abs(s - 0.5) < 0.01 for s in steps), steps
        e4_lora = sorted({a.lr for a in e4_arms() if a.method == "lora"})
        assert lrs == e4_lora

    def test_it_is_registered_and_scored_by_accuracy(self):
        from tools.lora_regret.sweep import MATRIX_LAUNCHERS, MATRIX_METRICS

        assert MATRIX_METRICS["e4place"] == "accuracy"
        assert "rl-math-gsm8k" in MATRIX_LAUNCHERS["e4place"]
        assert len(MATRICES["e4place"](HIDDEN, FFN, 0, None, None)) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fast/utils/test_lora_regret_arms_coverage.py::TestE4Place -q`
Expected: FAIL — `ImportError: cannot import name 'e4place_arms'`

- [ ] **Step 3: Write the matrix**

In `tools/lora_regret/arms.py`, after `e4_arms`:

```python
def e4place_arms(
    hidden_size: int,
    ffn_size: int,
    seed: int = 0,
    qkv_output_size: int = LLAMA31_8B_QKV_OUTPUT,
) -> list[Arm]:
    """E4-place: does the attention-vs-MLP finding survive policy gradient?

    2 placements x 4 LRs = 8 runs, on E4's own data and E4's own half-decade
    grid so the placement result and the rank result are comparable arm for arm.

    **All-modules is deliberately absent.** E4 already runs LoRA r256
    all-modules on this exact grid, so including it here would produce four
    byte-identical arm names -- four re-run RL arms at 8 GPUs each, and a
    duplicate key the moment both ledgers are globbed into `analyze` together,
    where the better of two independent runs of one configuration would win.
    Read the all-modules cell from E4's ledger; the same reasoning excludes
    Llama from `e6`.

    The MLP rank is E3's *solved* match (r92 on Llama-3.1-8B), not the post's
    r128: Orbit fuses qkv and gate+up, so the post's pair is not matched in this
    layout, and an unmatched pair compares placement and capacity at once.

    No FullFT arm. The post's RL placement panel is a comparison within LoRA,
    and a FullFT arm here would cost 8 GPUs to answer a question E4 already asks.
    """
    matched_rank = matched_mlp_rank(256, hidden_size, ffn_size, qkv_output_size)
    configs = [
        (256, ATTN_MODULES),
        (matched_rank, MLP_MODULES),
    ]
    arms: list[Arm] = []
    for rank, modules in configs:
        for lr in lr_grid(RL_LORA_LR_CENTRE, n=4, step_decades=0.5):
            arms.append(
                Arm(_name("lora", f"r{rank}", modules, lr, seed), "lora", rank, None,
                    modules, lr, seed, dataset=RL_MIX_DATASET)
            )
    return arms
```

Register: `MATRICES["e4place"]` (passing `hidden`/`ffn`), `MATRIX_LAUNCHERS["e4place"] = RL_LAUNCHER`,
`MATRIX_METRICS["e4place"] = "accuracy"`, `EXPECTED_ARMS["e4place"] = 8`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/fast/utils/test_lora_regret_arms_coverage.py -q`
Expected: PASS, 16 tests.

- [ ] **Step 5: Verify the dry run names the RL launcher**

Run: `python -m tools.lora_regret.sweep --matrix e4place --dry-run 2>&1 | head -2`
Expected: `8 arms selected, 0 already done, 8 to run` and
`launcher=examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh metric=accuracy`

- [ ] **Step 6: Run the full suite**

Run: `pytest tests -q 2>&1 | tail -3`
Expected: **0 failed**, sixteen tests more than Task 3's total (Tasks 4-6 together).

- [ ] **Step 7: Commit**

```bash
git add tools/lora_regret/arms.py tools/lora_regret/sweep.py tools/lora_regret/preflight.py \
        tests/fast/utils/test_lora_regret_arms_coverage.py
git commit -m "feat(lora_regret): add the e4place matrix for RL layer placement"
```

---

### Task 7: Refuse a sigma measured on the wrong dataset

**Files:**
- Modify: `tools/lora_regret/analyze.py` (`sigma`, `main`)
- Test: `tests/fast/utils/test_lora_regret_analyze.py`

**Interfaces:**
- Consumes: ledger records carrying `"dataset"` (already written by `run_arm`).
- Produces: `sigma_dataset(records: list[dict]) -> str | None`, and a `main` guard that exits 3 on a mismatch. New CLI flag `--allow-sigma-dataset-mismatch`.

- [ ] **Step 1: Write the failing test**

Append to `tests/fast/utils/test_lora_regret_analyze.py`:

```python
class TestSigmaDatasetGuard:
    """Tulu3's held-out split is 1,000 rows; OpenThoughts3's is 100. Their noise
    floors are different numbers, and both ledgers are called *_sigma.jsonl."""

    @staticmethod
    def _ledger(tmp_path, name, dataset, values, seeds=(0, 1, 2)):
        path = tmp_path / name
        with path.open("w", encoding="utf-8") as fh:
            for seed, value in zip(seeds, values):
                fh.write(json.dumps({
                    "arm": f"lora-r256-all-lr0.00025-s{seed}", "method": "lora", "rank": 256,
                    "target_modules": "linear_qkv,linear_proj,linear_fc1,linear_fc2",
                    "lr": 2.5e-4, "seed": seed, "metric": "nll", "test_nll": value,
                    "dataset": dataset, "status": "ok",
                }) + "\n")
        return path

    def test_sigma_dataset_reads_the_single_dataset_in_the_ledger(self, tmp_path):
        from tools.lora_regret.analyze import load_records, sigma_dataset

        path = self._ledger(tmp_path, "s.jsonl", "tulu3", [1.0, 1.001, 1.002])
        assert sigma_dataset(load_records([path], seed=None)) == "tulu3"

    def test_mixed_dataset_sigma_ledger_raises(self, tmp_path):
        from tools.lora_regret.analyze import load_records, sigma_dataset

        path = tmp_path / "s.jsonl"
        rows = [("tulu3", 1.0, 0), ("openthoughts3", 1.1, 1), ("tulu3", 1.002, 2)]
        with path.open("w", encoding="utf-8") as fh:
            for dataset, value, seed in rows:
                fh.write(json.dumps({
                    "arm": f"a-s{seed}", "method": "lora", "rank": 256, "target_modules": "x",
                    "lr": 2.5e-4, "seed": seed, "metric": "nll", "test_nll": value,
                    "dataset": dataset, "status": "ok",
                }) + "\n")
        with pytest.raises(ValueError, match="more than one dataset"):
            sigma_dataset(load_records([path], seed=None))

    def test_claim_exits_three_when_the_sigma_dataset_differs(self, tmp_path):
        import subprocess
        import sys

        sigma_path = self._ledger(tmp_path, "sig.jsonl", "tulu3", [1.0, 1.001, 1.002])
        arms_path = self._ledger(
            tmp_path, "arms.jsonl", "openthoughts3", [2.0, 2.1, 2.2], seeds=(0, 0, 0)
        )
        proc = subprocess.run(
            [sys.executable, "-m", "tools.lora_regret.analyze", "argmins",
             "--ledgers", str(arms_path), "--sigma-ledger", str(sigma_path)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert proc.returncode == 3
        assert "tulu3" in proc.stderr and "openthoughts3" in proc.stderr

    def test_the_override_exists_and_is_named_for_what_it_does(self, tmp_path):
        import subprocess
        import sys

        sigma_path = self._ledger(tmp_path, "sig.jsonl", "tulu3", [1.0, 1.001, 1.002])
        arms_path = self._ledger(
            tmp_path, "arms.jsonl", "openthoughts3", [2.0, 2.1, 2.2], seeds=(0, 0, 0)
        )
        proc = subprocess.run(
            [sys.executable, "-m", "tools.lora_regret.analyze", "argmins",
             "--ledgers", str(arms_path), "--sigma-ledger", str(sigma_path),
             "--allow-sigma-dataset-mismatch"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert proc.returncode != 3
```

Add `import json`, `import pytest` and `REPO_ROOT = Path(__file__).resolve().parents[3]`
to the module if they are not already there.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fast/utils/test_lora_regret_analyze.py::TestSigmaDatasetGuard -q`
Expected: FAIL — `ImportError: cannot import name 'sigma_dataset'`

- [ ] **Step 3: Implement the guard**

In `tools/lora_regret/analyze.py`, after `sigma`:

```python
def sigma_dataset(records: list[dict]) -> str | None:
    """Which dataset a sigma ledger's replicates were measured on.

    `None` means the ledger predates the `dataset` field, in which case the
    guard cannot fire and says so rather than guessing.
    """
    datasets = {r.get("dataset") for r in records if r.get("dataset")}
    if len(datasets) > 1:
        raise ValueError(
            f"sigma ledger holds more than one dataset ({sorted(datasets)}); "
            "a noise floor is a property of one held-out set, not of a mixture"
        )
    return datasets.pop() if datasets else None
```

In `main`, add the flag next to `--allow-edge-argmin`:

```python
    parser.add_argument(
        "--allow-sigma-dataset-mismatch",
        action="store_true",
        help="Quote a sigma measured on a different dataset than the arms. Off by "
             "default: Tulu3's held-out split is 1,000 rows and OpenThoughts3's is "
             "100, so their noise floors are different numbers.",
    )
```

and immediately after `sigma_value` is computed from the sigma ledger:

```python
    if sigma_value is not None and not args.allow_sigma_dataset_mismatch:
        measured_on = sigma_dataset(sigma_records)
        used_on = {r.get("dataset") for r in records if r.get("dataset")}
        if measured_on is not None and used_on and measured_on not in used_on:
            print(
                f"sigma was measured on {measured_on!r} but these arms ran on "
                f"{sorted(used_on)}. Held-out split sizes differ between datasets, "
                "so the noise floor does not transfer. Measure sigma on this dataset "
                "(runbook section 7), or pass --allow-sigma-dataset-mismatch.",
                file=sys.stderr,
            )
            return 3
```

The sigma ledger is currently loaded inline into the `sigma(...)` call. Hoist it
to a local so the guard can inspect the same records:

```python
    # before
    sigma_value = args.sigma if args.sigma is not None else sigma(
        load_records(args.sigma_ledger, seed=None)
    )
    # after
    sigma_records: list[dict] = []
    if args.sigma is not None:
        sigma_value = args.sigma
    else:
        sigma_records = load_records(args.sigma_ledger, seed=None)
        sigma_value = sigma(sigma_records)
```

`--sigma` bypasses the guard by construction: an explicitly supplied number
carries no dataset, so there is nothing to compare and nothing to refuse.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/fast/utils/test_lora_regret_analyze.py -q`
Expected: PASS, including the four new tests.

- [ ] **Step 5: Confirm the existing readings still work**

Run: `pytest tests -q 2>&1 | tail -3`
Expected: **0 failed**, four tests more than Task 6's total.

- [ ] **Step 6: Commit**

```bash
git add tools/lora_regret/analyze.py tests/fast/utils/test_lora_regret_analyze.py
git commit -m "feat(lora_regret): refuse a sigma measured on another dataset"
```

---

### Task 8: The claim layer — `c8`, and `c4` under an accuracy metric

**Files:**
- Modify: `tools/lora_regret/analyze.py` (`main`: `c8` in choices, `--short-ledgers`, `--metric`; `placement_deltas`, `all_modules_deltas`)
- Test: `tests/fast/utils/test_lora_regret_analyze.py`

**Interfaces:**
- Consumes: `argmins`, `edge_of_grid`, `load_records`, `_pairwise_deltas`.
- Produces: `short_run_multiplier(long_records, short_records) -> dict`, the `c8` subcommand with `--short-ledgers`, and a `--metric {nll,accuracy}` flag that `c4` honours so `e4place`'s ledger reads through the same code as `e3`'s.

**Why both in one task:** they are the same layer of `analyze.py` — the claim
dispatch in `main` plus the reducers it calls — and a reviewer would accept or
reject them together. Splitting would mean two tasks each touching the same
forty lines.

> **Note on `ALL_MODULES_KEY`:** `main` already binds a local `all_modules` for
> the `c2` block. `short_run_multiplier` is a module-level function and cannot
> see it, hence the module constant. Set the existing local to `ALL_MODULES_KEY`
> rather than leaving two spellings of the same string in one file.

- [ ] **Step 1: Write the failing test**

Append to `tests/fast/utils/test_lora_regret_analyze.py`:

```python
class TestC8ShortRunMultiplier:
    ALL = "linear_qkv,linear_proj,linear_fc1,linear_fc2"

    @staticmethod
    def _rows(path, entries):
        with path.open("w", encoding="utf-8") as fh:
            for method, rank, lr, nll in entries:
                fh.write(json.dumps({
                    "arm": f"{method}-r{rank}-{lr:g}", "method": method, "rank": rank,
                    "target_modules": TestC8ShortRunMultiplier.ALL if method == "lora" else "",
                    "lr": lr, "seed": 0, "metric": "nll", "test_nll": nll,
                    "dataset": "tulu3", "status": "ok",
                }) + "\n")
        return path

    def test_ratio_is_higher_at_one_hundred_steps(self, tmp_path):
        """Long run: FullFT argmin 2.5e-5, LoRA 2.5e-4 -> 10x.
        Short run: FullFT argmin 2.5e-5, LoRA 3.75e-4 -> 15x."""
        from tools.lora_regret.analyze import load_records, short_run_multiplier

        long_path = self._rows(tmp_path / "long.jsonl", [
            ("full", None, 1.5e-5, 1.10), ("full", None, 2.5e-5, 1.00), ("full", None, 4.0e-5, 1.09),
            ("lora", 256, 1.5e-4, 1.20), ("lora", 256, 2.5e-4, 1.05), ("lora", 256, 4.0e-4, 1.19),
        ])
        short_path = self._rows(tmp_path / "short.jsonl", [
            ("full", None, 1.5e-5, 1.40), ("full", None, 2.5e-5, 1.30), ("full", None, 4.0e-5, 1.39),
            ("lora", 256, 2.5e-4, 1.38), ("lora", 256, 3.75e-4, 1.32), ("lora", 256, 5.6e-4, 1.37),
        ])
        result = short_run_multiplier(load_records([long_path]), load_records([short_path]))
        assert result["long_ratio"] == pytest.approx(10.0, rel=1e-6)
        assert result["short_ratio"] == pytest.approx(15.0, rel=1e-6)
        assert result["upholds"] is True

    def test_it_does_not_uphold_when_the_short_ratio_is_not_larger(self, tmp_path):
        from tools.lora_regret.analyze import load_records, short_run_multiplier

        same = [
            ("full", None, 1.5e-5, 1.10), ("full", None, 2.5e-5, 1.00), ("full", None, 4.0e-5, 1.09),
            ("lora", 256, 1.5e-4, 1.20), ("lora", 256, 2.5e-4, 1.05), ("lora", 256, 4.0e-4, 1.19),
        ]
        long_path = self._rows(tmp_path / "long.jsonl", same)
        short_path = self._rows(tmp_path / "short.jsonl", same)
        result = short_run_multiplier(load_records([long_path]), load_records([short_path]))
        assert result["upholds"] is False

    def test_missing_arm_raises_rather_than_reporting_half_a_ratio(self, tmp_path):
        from tools.lora_regret.analyze import load_records, short_run_multiplier

        long_path = self._rows(tmp_path / "long.jsonl", [
            ("full", None, 2.5e-5, 1.00), ("lora", 256, 2.5e-4, 1.05),
        ])
        short_path = self._rows(tmp_path / "short.jsonl", [("full", None, 2.5e-5, 1.30)])
        with pytest.raises(ValueError, match="lora"):
            short_run_multiplier(load_records([long_path]), load_records([short_path]))

    def test_c8_requires_short_ledgers(self, tmp_path):
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "tools.lora_regret.analyze", "c8",
             "--ledgers", str(tmp_path / "nothing.jsonl"), "--sigma", "0.001"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert proc.returncode == 2
        assert "--short-ledgers" in proc.stderr


class TestC4UnderAccuracy:
    """e4place scores by accuracy, and higher is better. Reading it with the
    NLL comparator would invert every placement verdict."""

    ATTN = "linear_qkv,linear_proj"
    MLP = "linear_fc1,linear_fc2"

    @staticmethod
    def _ledger(tmp_path, rows):
        path = tmp_path / "e4place.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for modules, rank, lr, acc in rows:
                fh.write(json.dumps({
                    "arm": f"lora-r{rank}-{lr:g}", "method": "lora", "rank": rank,
                    "target_modules": modules, "lr": lr, "seed": 0,
                    "metric": "accuracy", "accuracy": acc, "test_nll": None,
                    "dataset": "math_gsm8k", "status": "ok",
                }) + "\n")
        return path

    def test_argmin_picks_the_highest_accuracy(self, tmp_path):
        from tools.lora_regret.analyze import argmins, load_records

        path = self._ledger(tmp_path, [
            (self.ATTN, 256, 1e-5, 0.31), (self.ATTN, 256, 3.16e-5, 0.44),
            (self.MLP, 92, 1e-5, 0.38), (self.MLP, 92, 3.16e-5, 0.52),
        ])
        records = load_records([path], metric="accuracy")
        best = argmins(records, metric="accuracy")
        assert best[("lora", 256, self.ATTN)]["accuracy"] == 0.44
        assert best[("lora", 92, self.MLP)]["accuracy"] == 0.52

    def test_c4_with_metric_accuracy_exits_zero_and_reports_a_delta(self, tmp_path):
        import subprocess
        import sys

        path = self._ledger(tmp_path, [
            (self.ATTN, 256, 1e-5, 0.31), (self.ATTN, 256, 3.16e-5, 0.44),
            (self.ATTN, 256, 1e-4, 0.29),
            (self.MLP, 92, 1e-5, 0.38), (self.MLP, 92, 3.16e-5, 0.52),
            (self.MLP, 92, 1e-4, 0.35),
        ])
        proc = subprocess.run(
            [sys.executable, "-m", "tools.lora_regret.analyze", "c4",
             "--ledgers", str(path), "--sigma", "0.01", "--metric", "accuracy", "--json"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert proc.returncode == 0, proc.stderr
        assert "c4" in json.loads(proc.stdout)

    def test_metric_accuracy_on_an_nll_ledger_finds_no_records(self, tmp_path):
        """load_records filters on the ledger's own `metric` field, so a
        mismatched --metric yields nothing rather than silently mixing units."""
        import subprocess
        import sys

        path = self._ledger(tmp_path, [(self.ATTN, 256, 1e-5, 0.31)])
        proc = subprocess.run(
            [sys.executable, "-m", "tools.lora_regret.analyze", "c4",
             "--ledgers", str(path), "--sigma", "0.01"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert "0 records" in (proc.stdout + proc.stderr) or proc.returncode != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fast/utils/test_lora_regret_analyze.py::TestC8ShortRunMultiplier -q`
Expected: FAIL — `ImportError: cannot import name 'short_run_multiplier'`

- [ ] **Step 3: Implement**

In `tools/lora_regret/analyze.py`, after `edge_of_grid`:

```python
def short_run_multiplier(long_records: list[dict], short_records: list[dict]) -> dict:
    """C8: the LoRA/FullFT LR ratio at ~100 steps against at the long horizon.

    The post reports ~15x for runs under about 100 steps against ~10x for long
    ones, attributing the difference to B's zero initialization acting as an
    implicit warmup that has not finished in 100 steps.

    Both ratios are computed from argmins of the *same* two arms (FullFT and
    LoRA r256 all-modules), so a missing arm raises rather than producing half
    a ratio: reporting the long ratio alone, labelled C8, would read as a
    measurement of a difference that was never measured.
    """
    all_modules = ALL_MODULES_KEY

    def ratio(records: list[dict], label: str) -> float:
        best = argmins(records)
        lora = best.get(("lora", 256, all_modules))
        full = best.get(("full", None, ""))
        missing = [n for n, v in (("lora r256", lora), ("full", full)) if v is None]
        if missing:
            raise ValueError(f"{label} ledger is missing {missing}; C8 needs both arms")
        return lora["lr"] / full["lr"]

    long_ratio = ratio(long_records, "long-run")
    short_ratio = ratio(short_records, "short-run")
    return {
        "long_ratio": long_ratio,
        "short_ratio": short_ratio,
        "predicted_long": 9.8,
        "predicted_short": 15.0,
        "upholds": short_ratio > long_ratio,
    }
```

Add the module-level constant next to `ArmKey` if one does not already exist:

```python
# The target_modules string every all-modules arm carries. Spelled once so the
# claim readers and arms.py cannot drift apart on module ordering.
ALL_MODULES_KEY = "linear_qkv,linear_proj,linear_fc1,linear_fc2"
```

In `main`: add `"c8"` to the `command` choices, add the flag

```python
    parser.add_argument(
        "--short-ledgers", nargs="+", default=None,
        help="e1short ledger paths or globs. Required by c8 and meaningless "
             "elsewhere: the claim is a comparison of two horizons, and one "
             "horizon is not a comparison.",
    )
```

the guards

```python
    if args.command == "c8" and args.short_ledgers is None:
        parser.error("c8 requires --short-ledgers; run --matrix e1short first")
    if args.command != "c8" and args.short_ledgers is not None:
        parser.error(f"--short-ledgers is only meaningful for c8, not {args.command}")
```

and the dispatch block, placed after the `c2` block so the two ratios read
adjacently:

```python
    if args.command == "c8":
        result = short_run_multiplier(records, load_records(args.short_ledgers))
        payload["c8"] = result
        say(f"\nC8: LR multiplier at ~100 steps = {result['short_ratio']:.2f}")
        say(f"    at the long horizon          = {result['long_ratio']:.2f}")
        say(f"    the post predicts ~{result['predicted_short']:g} against "
            f"~{result['predicted_long']:g}")
        say(f"    {'UPHOLDS' if result['upholds'] else 'CONTRADICTS'}: the short-run "
            "multiplier is " + ("larger" if result["upholds"] else "not larger"))
```

- [ ] **Step 4: Let `c4` read an accuracy ledger**

`e4place` scores by accuracy, where higher is better; `argmins` and
`edge_of_grid` already take a `metric` argument for exactly this, but `main`
hardcodes the default. Add the flag:

```python
    parser.add_argument(
        "--metric", choices=("nll", "accuracy"), default="nll",
        help="Which score the ledgers carry. e4/e4place ledgers are 'accuracy' "
             "and are compared in the opposite direction; load_records filters on "
             "the ledger's own metric field, so a mismatch yields no records "
             "rather than mixing nats with fractions.",
    )
```

Thread it through the three call sites in `main` — the `load_records` for
`--ledgers`, the `argmins` that produces `best`, and the `edge_of_grid` check:

```python
    records = load_records(args.ledgers, metric=args.metric)
    best = argmins(records, metric=args.metric)
    flagged = edge_of_grid(records, metric=args.metric)
```

`placement_deltas` and `all_modules_deltas` take `records` and a sigma and call
`argmins` internally at the default metric. Give both a `metric: str = "nll"`
parameter, pass it to their internal `argmins` call, and pass `args.metric` from
the `c4` dispatch block. **Do not flip the sign of the delta**: the claim is
still `attention − mlp`, and with accuracy a positive value still means
attention-only is worse, so the existing UPHOLDS/CONTRADICTS wording holds
without change.

Leave `c1`, `c2`, `c3` and `c6` on `nll` — they read SFT ledgers by
construction, and `load_records`' metric filter already makes a mistaken
`--metric accuracy` return nothing rather than a wrong number.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/fast/utils/test_lora_regret_analyze.py -q`
Expected: PASS, including the eight new tests (four for `c8`, four for `c4`).

- [ ] **Step 6: Confirm `c8` and `--metric` are in `--help`**

Run: `python -m tools.lora_regret.analyze --help 2>&1 | grep -E "c8|--metric"`
Expected: `c8` among the positional choices, and a `--metric {nll,accuracy}` option.

- [ ] **Step 7: Run the full suite**

Run: `pytest tests -q 2>&1 | tail -3`
Expected: **0 failed**, eight tests more than Task 7's total.

- [ ] **Step 8: Commit**

```bash
git add tools/lora_regret/analyze.py tests/fast/utils/test_lora_regret_analyze.py
git commit -m "feat(lora_regret): add the short-run multiplier and accuracy-metric placement readings"
```

---

### Task 9: `plot.py` — figures from the analysis JSON

**Files:**
- Create: `tools/lora_regret/plot.py`
- Modify: `pyproject.toml` (`[project.optional-dependencies]`)
- Test: `tests/fast/utils/test_lora_regret_plot.py`

**Interfaces:**
- Consumes: the `--json` payload shape emitted by `analyze` (`{"command": ..., "argmins": [...], "c1": [...], "c2": {...}, "c3": [...], "c8": {...}}`).
- Produces: `PANELS: dict[str, str]`, `available_panels(payload: dict) -> list[str]`, `render(payload: dict, out_dir: Path) -> list[Path]`, and a `main()` CLI.

- [ ] **Step 1: Write the failing test**

Create `tests/fast/utils/test_lora_regret_plot.py`:

```python
"""plot.py must be a pure function of the ledgers: no network, no state."""

import json
from pathlib import Path

import pytest

from tools.lora_regret.plot import available_panels, render

ALL = "linear_qkv,linear_proj,linear_fc1,linear_fc2"


def _payload():
    return {
        "command": "all",
        "sigma": 0.000992,
        "argmins": [
            {"arm": "full", "method": "full", "size": None, "target_modules": "",
             "lr": 2.5e-5, "test_nll": 1.00, "lr_grid": [1.5e-5, 2.5e-5, 4e-5], "edge_of_grid": False},
            {"arm": "lora r256", "method": "lora", "size": 256, "target_modules": ALL,
             "lr": 2.5e-4, "test_nll": 1.05, "lr_grid": [1.5e-4, 2.5e-4, 4e-4], "edge_of_grid": False},
            {"arm": "lora r16", "method": "lora", "size": 16, "target_modules": ALL,
             "lr": 2.5e-4, "test_nll": 1.12, "lr_grid": [1.5e-4, 2.5e-4, 4e-4], "edge_of_grid": False},
        ],
        "c2": {"lora_r256_argmin_lr": 2.5e-4, "fullft_argmin_lr": 2.5e-5, "ratio": 10.0},
        "c8": {"long_ratio": 10.0, "short_ratio": 15.0, "upholds": True,
               "predicted_long": 9.8, "predicted_short": 15.0},
        "c1": [
            {"arm": "lora-r1-all", "departure_step": 400, "step_budget": 2000},
            {"arm": "lora-r256-all", "departure_step": None, "step_budget": 2000},
        ],
    }


def test_the_fixture_matches_what_analyze_actually_emits():
    """The payload keys are copied from analyze.py's `payload["argmins"]` block.
    A fixture that invents a key lets plot.py pass its tests and KeyError on the
    real pipeline -- which is the whole failure this file exists to prevent."""
    row = _payload()["argmins"][0]
    assert set(row) >= {"arm", "method", "size", "target_modules", "lr", "test_nll",
                        "lr_grid", "edge_of_grid"}


def test_available_panels_reports_only_what_the_payload_supports():
    """A payload with no c3 must not produce an empty batch-size figure -- an
    axes with no data reads as 'measured, and flat'."""
    payload = _payload()
    panels = available_panels(payload)
    assert "lr_vs_loss" in panels
    assert "short_run_multiplier" in panels
    assert "batch_size" not in panels


def test_render_writes_one_png_per_available_panel(tmp_path):
    payload = _payload()
    written = render(payload, tmp_path)
    assert len(written) == len(available_panels(payload))
    assert all(p.exists() and p.suffix == ".png" and p.stat().st_size > 0 for p in written)


def test_render_is_idempotent(tmp_path):
    payload = _payload()
    first = render(payload, tmp_path)
    second = render(payload, tmp_path)
    assert sorted(first) == sorted(second)
    assert len(list(tmp_path.glob("*.png"))) == len(first)


def test_empty_payload_writes_nothing_and_does_not_raise(tmp_path):
    assert render({"command": "sigma"}, tmp_path) == []
    assert list(tmp_path.glob("*.png")) == []


def test_cli_reads_a_json_file(tmp_path):
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    payload_path = tmp_path / "analysis.json"
    payload_path.write_text(json.dumps(_payload()), encoding="utf-8")
    out = tmp_path / "figures"
    proc = subprocess.run(
        [sys.executable, "-m", "tools.lora_regret.plot",
         "--analysis", str(payload_path), "--out", str(out)],
        capture_output=True, text=True, cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    assert list(out.glob("*.png"))


def test_reference_figures_are_vendored_for_comparison():
    """The post's own figures are checked in so the comparison is visual rather
    than asserted. If they move, the comparison silently stops happening."""
    repo_root = Path(__file__).resolve().parents[3]
    figures = repo_root / "third_party/lora-without-regret/figures"
    assert (figures / "sft_lr_vs_nll_by_rank.png").is_file()
    assert (figures / "sft_training_curves.png").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fast/utils/test_lora_regret_plot.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.lora_regret.plot'`

- [ ] **Step 3: Write the plotter**

Create `tools/lora_regret/plot.py`:

```python
"""Figures from `analyze --json`, one PNG per panel of the post.

A pure function of the ledgers: no network, no state, no side channel. The
input is the JSON document `analyze --json` writes, so a figure can never show
a number the analysis declined to quote -- an edge-of-grid argmin is absent
from the payload and is therefore absent from the plot.

matplotlib is imported inside `render` rather than at module scope so
`available_panels` and the CLI's argument handling stay importable in an
environment that has not installed it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Panel key -> the payload key it needs. A panel is drawn only when its data is
# present: an empty axes reads as "measured, and flat", which is a lie the
# reader has no way to detect.
PANELS: dict[str, str] = {
    "lr_vs_loss": "argmins",
    "learning_curves": "c1",
    "batch_size": "c3",
    "placement": "c4",
    "rl_accuracy": "c5",
    "short_run_multiplier": "c8",
}

# The post's own figures, for side-by-side comparison where one exists.
REFERENCE_FIGURES = {
    "lr_vs_loss": "third_party/lora-without-regret/figures/sft_lr_vs_nll_by_rank.png",
    "learning_curves": "third_party/lora-without-regret/figures/sft_training_curves.png",
    "placement": "third_party/lora-without-regret/figures/sft_lr_vs_nll_by_type.png",
    "rl_accuracy": "third_party/lora-without-regret/figures/rl_lr_vs_acc.png",
}


def available_panels(payload: dict) -> list[str]:
    """Panels whose data the payload actually carries, in PANELS order."""
    return [name for name, key in PANELS.items() if payload.get(key)]


def _label(method: str, size) -> str:
    return "FullFT" if method == "full" else f"LoRA r{size}"


def _draw_lr_vs_loss(ax, payload: dict) -> None:
    rows = payload["argmins"]
    for row in sorted(rows, key=lambda r: (r["method"], r.get("size") or 0)):
        ax.scatter(row["lr"], row["test_nll"], label=_label(row["method"], row.get("size")))
    ax.set_xscale("log")
    ax.set_xlabel("argmin learning rate")
    ax.set_ylabel("held-out NLL (nats)")
    ax.set_title("Optimal LR by method and rank")
    ax.legend(fontsize="small")


def _draw_learning_curves(ax, payload: dict) -> None:
    rows = payload["c1"]
    names = [r["arm"] for r in rows]
    # `None` means "no departure within the budget", which is a different
    # statement from "departed at the last step" -- plot it at the budget and
    # mark it, rather than dropping the arm.
    values = [r["departure_step"] if r["departure_step"] is not None else r["step_budget"]
              for r in rows]
    colors = ["tab:blue" if r["departure_step"] is not None else "tab:grey" for r in rows]
    ax.barh(names, values, color=colors)
    ax.set_xlabel("departure step (grey = no departure within budget)")
    ax.set_title("Where each rank leaves the envelope")


def _draw_batch_size(ax, payload: dict) -> None:
    rows = payload["c3"]
    for row in rows:
        ax.plot(row["batch_size"], row["gap_sigma"], marker="o", label=row.get("arm", ""))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("global batch size")
    ax.set_ylabel("best LoRA - best FullFT (sigma)")
    ax.set_title("Batch-size penalty")
    ax.legend(fontsize="small")


def _draw_placement(ax, payload: dict) -> None:
    rows = payload["c4"]
    ax.bar(list(rows), [rows[k] for k in rows])
    ax.set_ylabel("delta (sigma)")
    ax.set_title("Layer placement at matched parameters")
    ax.tick_params(axis="x", labelrotation=20)


def _draw_rl_accuracy(ax, payload: dict) -> None:
    rows = payload["c5"]
    for row in rows:
        ax.scatter(row["lr"], row["accuracy"], label=row.get("arm", ""))
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("accuracy")
    ax.set_title("RL: peak parity and band width")
    ax.legend(fontsize="small")


def _draw_short_run_multiplier(ax, payload: dict) -> None:
    c8 = payload["c8"]
    ax.bar(["~100 steps", "long horizon"], [c8["short_ratio"], c8["long_ratio"]])
    ax.axhline(c8["predicted_short"], linestyle="--", label=f"post: {c8['predicted_short']:g}x")
    ax.axhline(c8["predicted_long"], linestyle=":", label=f"post: {c8['predicted_long']:g}x")
    ax.set_ylabel("argmin_LR(LoRA r256) / argmin_LR(FullFT)")
    ax.set_title("LR multiplier by horizon")
    ax.legend(fontsize="small")


_DRAW = {
    "lr_vs_loss": _draw_lr_vs_loss,
    "learning_curves": _draw_learning_curves,
    "batch_size": _draw_batch_size,
    "placement": _draw_placement,
    "rl_accuracy": _draw_rl_accuracy,
    "short_run_multiplier": _draw_short_run_multiplier,
}


def render(payload: dict, out_dir: Path) -> list[Path]:
    """Write one PNG per available panel. Returns the paths written."""
    panels = available_panels(payload)
    if not panels:
        return []
    import matplotlib

    matplotlib.use("Agg")  # no display on a compute node
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in panels:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        _DRAW[name](ax, payload)
        fig.tight_layout()
        path = out_dir / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True,
                        help="the JSON document `analyze --json` wrote")
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    args = parser.parse_args(argv)

    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    written = render(payload, args.out)
    if not written:
        print(f"no plottable claims in {args.analysis}; nothing written")
        return 0
    for path in written:
        reference = REFERENCE_FIGURES.get(path.stem)
        suffix = f"   (compare: {reference})" if reference else ""
        print(f"wrote {path}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Declare matplotlib as an extra**

In `pyproject.toml`, inside `[project.optional-dependencies]`, add:

```toml
# Only tools/lora_regret/plot.py needs this, and it imports lazily so the
# module stays importable without it.
plots = [
    "matplotlib>=3.8",
]
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/fast/utils/test_lora_regret_plot.py -q`
Expected: PASS, 6 tests.

- [ ] **Step 6: Run the full suite**

Run: `pytest tests -q 2>&1 | tail -3`
Expected: **0 failed**, six tests more than Task 8's total (plot.py's six).

- [ ] **Step 7: Commit**

```bash
git add tools/lora_regret/plot.py tests/fast/utils/test_lora_regret_plot.py pyproject.toml
git commit -m "feat(lora_regret): render the campaign figures from the analysis JSON"
```

---

### Task 10: Preflight stages and the runbook

**Files:**
- Modify: `tools/lora_regret/preflight.py` (`STAGE_GPU_REQUIREMENTS`)
- Modify: `docs/superpowers/plans/2026-07-30-lora-without-regret-runbook.md`
- Test: `tests/fast/utils/test_lora_regret_preflight.py`

**Interfaces:**
- Consumes: `EXPECTED_ARMS` entries added in Tasks 4-6; `models.get` from Task 1.
- Produces: stages `e1ot`, `e1short`, `e4place` in `STAGE_GPU_REQUIREMENTS`.

- [ ] **Step 1: Write the failing test**

Append to `tests/fast/utils/test_lora_regret_preflight.py`:

```python
def test_the_new_stages_carry_their_gpu_floors():
    from tools.lora_regret.preflight import STAGE_GPU_REQUIREMENTS

    assert STAGE_GPU_REQUIREMENTS["e1ot"] == 1
    assert STAGE_GPU_REQUIREMENTS["e1short"] == 1
    assert STAGE_GPU_REQUIREMENTS["e4place"] == 8


def test_every_matrix_is_expected_at_its_documented_count():
    from tools.lora_regret.arms import MATRICES
    from tools.lora_regret.preflight import EXPECTED_ARMS

    assert set(EXPECTED_ARMS) == set(MATRICES) - {"e1long"}
    assert EXPECTED_ARMS["e1ot"] == 40
    assert EXPECTED_ARMS["e1short"] == 14
    assert EXPECTED_ARMS["e4place"] == 8


def test_the_fullft_stages_agree_with_the_registrys_formula():
    """preflight's floor and the launcher's guard must not drift apart."""
    from tools.lora_regret.models import get
    from tools.lora_regret.preflight import STAGE_GPU_REQUIREMENTS

    assert STAGE_GPU_REQUIREMENTS["e1-full"] == get("llama3.1-8b").min_gpus_fullft()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fast/utils/test_lora_regret_preflight.py -q`
Expected: FAIL — `KeyError: 'e1ot'`

- [ ] **Step 3: Add the stages**

In `tools/lora_regret/preflight.py`, extend `STAGE_GPU_REQUIREMENTS`:

```python
    # e1ot and e1short are LoRA-and-FullFT matrices, but their FullFT arms are
    # selected with --only and run on the e1-full allocation; the stage floor
    # here is the LoRA one, which is what an operator checks before the bulk of
    # the arms.
    "e1ot": 1,
    "e1short": 1,
    "e4place": 8,
```

- [ ] **Step 4: Run preflight for one new stage**

Run:
```bash
export DATA_DIR=/lustre/fast/fast/groups/ei-slm/data/lora_regret
python -m tools.lora_regret.preflight --stage e1ot
```
Expected: exit 0, and the matrix lines now include `matrix:e1ot 40 arms`,
`matrix:e1short 14 arms`, `matrix:e4place 8 arms`.

- [ ] **Step 5: Update the runbook**

In `docs/superpowers/plans/2026-07-30-lora-without-regret-runbook.md`, add three
rows to the §5 execution-order table, after the E3 row:

```markdown
| §16 | E1-OT: OpenThoughts3 rank ladder | 42 | 1 / ≥4 | σ(OT3) | curves + argmins on the second dataset → **C1/C2** |
| §17 | E1-short: 100-step multiplier | 14 | 1 / ≥4 | σ | short-vs-long LR ratio → **C8** |
| §18 | E4-place: RL layer placement | 8 | 8 | data, P3 | attention-vs-MLP under policy gradient → **C4** |
```

and append three sections:

````markdown
## 16. E1-OT — the rank ladder on OpenThoughts3 (C1, C2 on the second dataset)

**Measure this dataset's σ first.** The held-out split is **100 rows against
Tulu3's 1,000**, so its noise floor is a different number and E1-0's σ does not
transfer. Two extra runs:

```bash
export DATA_DIR=/lustre/fast/fast/groups/ei-slm/data/lora_regret
for seed in 1 2; do
  python -m tools.lora_regret.sweep --matrix e1ot --seed $seed \
    --only 'lora-r256-all-lr0.00025' --results results/e1ot_0_sigma.jsonl
done
```

Seed 0 of that arm is already a grid point in the sweep below; point the σ
reading at both files. `analyze` **refuses** to quote a Tulu3 σ against these
arms — that guard is the reason to run this first rather than last.

```bash
python -m tools.lora_regret.sweep --matrix e1ot --only '^lora-r(1|4|16)-' --results results/e1ot_a.jsonl &
python -m tools.lora_regret.sweep --matrix e1ot --only '^lora-r(64|128)-'  --results results/e1ot_b.jsonl &
python -m tools.lora_regret.sweep --matrix e1ot --only '^lora-r(256|512)-' --results results/e1ot_c.jsonl &
wait
GPUS_PER_NODE=4 python -m tools.lora_regret.sweep --matrix e1ot --only '^full-' --results results/e1ot_full.jsonl
```

One epoch is **312 optimizer steps**, so these arms run to completion and give
both the argmins and the curves — there is no long-run counterpart to schedule.

```bash
python -m tools.lora_regret.analyze all --ledgers 'results/e1ot_*.jsonl' \
  --sigma-ledger results/e1ot_0_sigma.jsonl
```

## 17. E1-short — the ~100-step LR multiplier (C8)

14 arms, ~30 min each. The grid is **0.15-decade**, not the campaign's 0.3:
resolving 15x from 10x means resolving 0.176 decades, and on a 0.3-decade grid
adjacent points differ by 2x.

```bash
python -m tools.lora_regret.sweep --matrix e1short --only '^lora-' --results results/e1short_lora.jsonl
GPUS_PER_NODE=4 python -m tools.lora_regret.sweep --matrix e1short --only '^full-' --results results/e1short_full.jsonl

python -m tools.lora_regret.analyze c8 \
  --ledgers 'results/e1_*.jsonl' \
  --short-ledgers 'results/e1short_*.jsonl' \
  --sigma-ledger results/e1_0_sigma.jsonl
```

`--ledgers` is E1-1's long-horizon result and `--short-ledgers` is this stage's.
Passing one without the other exits 2: the claim is a comparison of two
horizons, and one horizon is not a comparison.

## 18. E4-place — layer placement under RL (C4 under policy gradient)

8 arms on 8 GPUs, on E4's own data and grid. The MLP arm is **r92** — E3's
solved match for attention r256 in Orbit's fused layout, not the post's r128.
There is no all-modules cell: E4 already ran it at these four learning rates,
so read it from `results/e4_lora.jsonl` and glob both files into `analyze`.

```bash
GPUS_PER_NODE=8 python -m tools.lora_regret.sweep --matrix e4place --results results/e4place.jsonl
python -m tools.lora_regret.analyze c4 --ledgers results/e4place.jsonl --sigma-ledger ...
```

σ for accuracy has never been measured. If the arms sit close, measuring it
becomes a prerequisite exactly as E1-0 was for NLL — say so rather than quoting
an unresolved difference.
````

- [ ] **Step 6: Run the full suite**

Run: `pytest tests -q 2>&1 | tail -3`
Expected: **0 failed**, three tests more than Task 9's total.

- [ ] **Step 7: Commit**

```bash
git add tools/lora_regret/preflight.py tests/fast/utils/test_lora_regret_preflight.py \
        docs/superpowers/plans/2026-07-30-lora-without-regret-runbook.md
git commit -m "docs(runbook): drive e1ot, e1short and e4place through the sweep"
```

---

## Verification

After Task 10, confirm the whole of Phase 1 from a clean shell:

```bash
source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
export CUDA_HOME=/is/software/nvidia/cuda-13.2 && source env.sh
export DATA_DIR=/lustre/fast/fast/groups/ei-slm/data/lora_regret

pytest tests -q 2>&1 | tail -3                      # 0 failed; ~655 passed
python -m tools.lora_regret.preflight --stage e1ot   # exit 0, 26 checks
for m in e1 e2 e3 e4 e5scout e1ot e1short e4place; do
  echo -n "$m "; python -m tools.lora_regret.sweep --matrix $m --dry-run 2>/dev/null | wc -l
done
# e1 40 / e2 36 / e3 20 / e4 16 / e5scout 5 / e1ot 40 / e1short 14 / e4place 8
```

The arm counts for `e1` through `e5scout` must be **unchanged** — that is the
check that `Arm.model`'s default did what it was supposed to.

## One deliberate deviation from the spec

The spec's §5 says `c1` and `c4` gain a `--dataset` flag so the OpenThoughts3
curves read through the same code as the Tulu3 ones. **`--dataset` is not
implemented, and should not be.** Each stage already writes its own ledger file
(`results/e1ot_*.jsonl` against `results/e1_*.jsonl`), so `--ledgers` already
selects the dataset, and a second selector over the same axis would let an
operator pass a pair that selects nothing and get a silent empty reading.

`c4` does gain `--metric`, which the spec also asked for, because that axis is
*not* already expressed by the file: an accuracy ledger and an NLL ledger are
distinguished by the comparator, not the path.

## What Phase 1 does not do

No GPU arm is launched. The three audit blockers (P3, the `e4` RL smoke, the
`e5scout` OFT smoke) are operator work and are unchanged by this plan; the RL
launcher having never run is why `e4place`'s 8 arms should not be scheduled
before the `e4` smoke produces a real accuracy line.

Phases 2-4 (`e6` scaling law, `e7` DeepMath, `e3moe` MoE) each get their own
plan and each begins with a checkpoint conversion. They consume `models.py`,
which is why they follow this plan rather than running beside it.
