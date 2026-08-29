"""The registry must agree with the model_args plugin it names.

A registry that can disagree with its plugin is worse than no registry: the
wrong number is then written down twice and neither copy looks suspicious.
"""

import re
from pathlib import Path

import pytest

from tools.lora_regret.models import DEFAULT_MODEL, MODELS, get, model_env

REPO_ROOT = Path(__file__).resolve().parents[3]

# The negative lookahead is load-bearing. A model_args plugin mixes valued flags
# with bare ones (`--group-query-attention`, `--swiglu`), and without it a bare
# flag consumes the NEXT flag's name as its value: in llama3.1-8B-Instruct.sh,
# `--group-query-attention` would swallow `--num-query-groups`, leaving the key
# absent and this file's GQA assertion raising KeyError instead of comparing
# anything.
_FLAG = re.compile(r"--([a-z0-9-]+)\s+(?!--)(\S+)")


def _plugin_flags(plugin_name: str) -> dict[str, str]:
    """Every `--flag value` in a model_args plugin, as a dict."""
    text = (REPO_ROOT / "miles_plugins" / "model_args" / plugin_name).read_text(encoding="utf-8")
    return dict(_FLAG.findall(text))


def test_the_flag_parser_does_not_let_a_bare_flag_eat_the_next_one():
    """Pins the lookahead above. Llama's plugin has `--group-query-attention`
    immediately before `--num-query-groups 8`, which is exactly the shape that
    breaks a naive `--(\\w+)\\s+(\\S+)`."""
    flags = _plugin_flags("llama3.1-8B-Instruct.sh")
    assert flags["num-query-groups"] == "8"
    assert flags["num-layers"] == "32"
    assert "group-query-attention" not in flags


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
    assert (REPO_ROOT / "miles_plugins" / "model_args" / MODELS[key].model_args_plugin).is_file()


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
    assert env["MODEL_ARGS_FILE"].endswith("miles_plugins/model_args/llama3.1-8B-Instruct.sh")
    assert model_env(get("qwen3-4b"), REPO_ROOT)["LOSS_MASK_TYPE"] == "qwen"
