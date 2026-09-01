"""Bake a merged OFT adapter into dense HF weights.

Replicates orbit's canonical-OFT runtime exactly so the baked dense model matches
what orbit serves: R via the 5-term Cayley-Neumann series (mirrors
megatron.bridge.orbit.oft.oft_layers._cayley_batch), applied as a block-diagonal INPUT
rotation (W' = W @ blockdiag(R^T), matching the forward einsum '...rk,rkc->...rc').
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from orbit.merge.oft_merge import _is_oft_key


def skew_from_vec(oft_r: torch.Tensor, block_size: int) -> torch.Tensor:
    """(num_blocks, P) strict-upper-triangle params -> (num_blocks, b, b) skew matrix.

    Mirrors oft_layers._pytorch_skew_symmetric: fill the strict upper triangle
    (row-major) then antisymmetrize S = U - U^T.
    """
    nb, p = oft_r.shape
    expected = block_size * (block_size - 1) // 2
    assert p == expected, f"P={p} != C(block_size={block_size}, 2)={expected}"
    idx = torch.triu_indices(block_size, block_size, 1, device=oft_r.device)
    S = torch.zeros(nb, block_size, block_size, dtype=oft_r.dtype, device=oft_r.device)
    S[:, idx[0], idx[1]] = oft_r
    return S - S.transpose(-1, -2)


def cayley_neumann(oft_r: torch.Tensor, block_size: int, num_terms: int = 5) -> torch.Tensor:
    """R = I + 2Q + 2Q^2 + ... + 2Q^(n-2) + Q^(n-1), the n-term Neumann series
    orbit's _cayley_batch uses (n=num_terms=5 by default). fp32.

    Every power except the last carries coefficient 2.0; the last power Q^(n-1)
    carries coefficient 1.0, matching megatron.bridge.orbit.oft.oft_layers._cayley_batch
    and its Triton kernel exactly.
    """
    Q = skew_from_vec(oft_r.float(), block_size)  # (nb, b, b)
    nb, b, _ = Q.shape
    R = torch.eye(b, device=Q.device, dtype=Q.dtype).expand(nb, b, b).clone()
    if num_terms > 1:
        R = R + 2.0 * Q
        q_power = Q
        for term in range(2, num_terms):
            q_power = torch.bmm(q_power, Q)
            coeff = 2.0 if term < num_terms - 1 else 1.0  # last term coefficient is 1
            R = R + coeff * q_power
    return R


def _expand_shared_rotation(rotation: torch.Tensor, in_features: int, block_share: bool) -> torch.Tensor:
    if not block_share:
        return rotation
    if rotation.shape[0] != 1:
        raise ValueError(f"block-shared OFT expects one rotation block, got {rotation.shape[0]}")
    block_size = rotation.shape[1]
    if in_features % block_size != 0:
        raise ValueError(f"input width {in_features} is not divisible by block size {block_size}")
    return rotation.repeat(in_features // block_size, 1, 1)


def bake_linear_weight(
    weight: torch.Tensor,
    rotation: torch.Tensor,
    *,
    block_share: bool = False,
) -> torch.Tensor:
    """W' = W @ blockdiag(R^T): the input-side block rotation baked into a linear's
    weight. weight:(out, in), rotation:(num_blocks, b, b), in == num_blocks*b.

    Derivation: runtime does y = W @ x_rot where x_rot block r = R_r^T @ x_r
    (from einsum '...rk,rkc->...rc'); so W' block-r columns = W_r @ R_r^T, i.e.
    W'[o, r, k] = sum_c W[o, r, c] * R[r, k, c].
    """
    out_f, in_f = weight.shape
    rotation = _expand_shared_rotation(rotation, in_f, block_share)
    nb, b, b2 = rotation.shape
    assert b == b2, f"non-square rotation block: {rotation.shape}"
    assert in_f == nb * b, f"in_features {in_f} != num_blocks*block_size {nb * b}"
    w_blocked = weight.float().reshape(out_f, nb, b)
    w_prime = torch.einsum("orc,rkc->ork", w_blocked, rotation.float())
    return w_prime.reshape(out_f, in_f).to(weight.dtype)


def bake_embedding_weight(
    weight: torch.Tensor,
    rotation: torch.Tensor,
    *,
    block_share: bool = False,
) -> torch.Tensor:
    """Bake an output-side OFT rotation into an embedding table as ``W @ R``."""
    vocab_size, hidden_size = weight.shape
    rotation = _expand_shared_rotation(rotation, hidden_size, block_share)
    num_blocks, block_size, block_size_2 = rotation.shape
    assert block_size == block_size_2, f"non-square rotation block: {rotation.shape}"
    assert (
        hidden_size == num_blocks * block_size
    ), f"hidden_size {hidden_size} != num_blocks*block_size {num_blocks * block_size}"
    blocked = weight.float().reshape(vocab_size, num_blocks, block_size)
    baked = torch.einsum("vrc,rck->vrk", blocked, rotation.float())
    return baked.reshape(vocab_size, hidden_size).to(weight.dtype)


def _is_embedding_weight_key(weight_key: str) -> bool:
    return weight_key.endswith(
        (
            ".embed_tokens.weight",
            ".word_embeddings.weight",
            ".tok_embeddings.weight",
            ".wte.weight",
        )
    )


def _hf_weight_key(oft_key: str) -> str:
    """Map a PEFT OFT key to the base HF weight key.

    'base_model.model.model.layers.N....q_proj.oft_R.weight'
      -> 'model.layers.N....q_proj.weight'
    """
    return oft_key.replace("base_model.model.", "", 1).replace(".oft_R.weight", ".weight")


def bake_hf_model(
    base_model_path: str,
    merged_adapter_dir: str,
    block_size: int,
    output_dir: str,
    device: str = "cpu",
    adapter: dict[str, torch.Tensor] | None = None,
) -> int:
    """Load the base HF model, bake the merged OFT rotation into each target linear's
    dense weight, and save a standalone HF model. Returns the number of linears baked.

    ``adapter`` may be passed in (the already-merged state dict) to skip re-reading
    ``merged_adapter_dir/adapter_model.safetensors`` from disk.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if adapter is None:
        from safetensors.torch import load_file

        adapter = load_file(str(Path(merged_adapter_dir) / "adapter_model.safetensors"))
    config = json.loads((Path(merged_adapter_dir) / "adapter_config.json").read_text())
    block_share = config.get("block_share", False)
    if type(block_share) is not bool:
        raise ValueError(f"adapter block_share must be a boolean, got {block_share!r}")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    state = model.state_dict()
    baked = 0
    with torch.no_grad():
        for oft_key, oft_r in adapter.items():
            if not _is_oft_key(oft_key):
                continue
            weight_key = _hf_weight_key(oft_key)
            if weight_key not in state:
                raise KeyError(f"no HF weight {weight_key!r} for adapter key {oft_key!r}")
            rotation = cayley_neumann(oft_r.to(device), block_size)
            if _is_embedding_weight_key(weight_key):
                baked_weight = bake_embedding_weight(state[weight_key], rotation, block_share=block_share)
            else:
                baked_weight = bake_linear_weight(state[weight_key], rotation, block_share=block_share)
            state[weight_key].copy_(baked_weight)
            baked += 1
    if baked == 0:
        raise ValueError("no .oft_ keys found in merged adapter; nothing baked")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(base_model_path).save_pretrained(output_dir)
    return baked
