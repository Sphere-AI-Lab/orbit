"""OrthoMerge OFT merge: magnitude-corrected Lie-algebra average on raw oft_R vectors."""

from __future__ import annotations

import logging
import math

import torch

from miles.merge.strategy import MergeStrategy, StateDict, StateKey, register

logger = logging.getLogger(__name__)

_DSV4_GROUPED_MOE_OFT_PARAM_NAMES = frozenset({"w1_oft_r", "w2_oft_r", "w3_oft_r"})


def magnitude_corrected_merge(
    vectors: list[torch.Tensor],
    weights: list[float] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Merge OFT skew-parameter tensors on the Lie algebra with magnitude correction.

    Each tensor has shape ``(num_blocks, P)`` where ``P = block_size*(block_size-1)/2``
    (the strict upper triangle of a per-block skew-symmetric generator). Because
    vectorization is linear and ``||S||_F = sqrt(2)*||vec||_2``, this reproduces
    OrthoMerge's ``merge_cayley_Q_list`` (computed on full skew matrices) exactly for
    equal weights, and generalizes it to a weighted manifold merge.

    Returns a tensor of shape ``(num_blocks, P)`` in the input dtype.
    """
    if len(vectors) == 0:
        raise ValueError("magnitude_corrected_merge requires >= 1 vector")
    n = len(vectors)
    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"got {len(weights)} weights for {n} vectors")
    out_dtype = vectors[0].dtype
    stacked = torch.stack([v.float() for v in vectors], dim=0)  # (N, B, P)
    w = torch.tensor(weights, dtype=torch.float32, device=stacked.device)
    if w.sum().abs().item() < eps:
        raise ValueError("weights sum to zero")
    w = w / w.sum()  # normalize to sum 1
    weighted_sum = (w.view(n, 1, 1) * stacked).sum(dim=0)  # (B, P)
    per_norms = stacked.flatten(1).norm(dim=1)  # (N,)
    target_mag = (w * per_norms).sum()  # weighted avg strength
    norm_of_sum = weighted_sum.norm()
    merged = target_mag * weighted_sum / (norm_of_sum + eps)
    return merged.to(out_dtype)


def _local_name(key: StateKey) -> str:
    if type(key) is str:
        return key
    if type(key) is tuple and len(key) == 2 and type(key[0]) is int and key[0] >= 0 and type(key[1]) is str and key[1]:
        return key[1]
    raise TypeError(f"invalid adapter state key {key!r}")


def _is_oft_key(name: StateKey) -> bool:
    # Mirrors miles.backends.megatron_utils.oft_utils.is_oft_weight_name without
    # importing the megatron-coupled module.
    return ".oft_" in _local_name(name)


def _is_original_oft_key(name: StateKey) -> bool:
    parts = _local_name(name).lower().replace("/", ".").split(".")
    if any("classifier" in part for part in parts):
        return False
    return any(part == "oft_r" or part in _DSV4_GROUPED_MOE_OFT_PARAM_NAMES for part in parts)


def infer_oft_block_size(num_params: int) -> int:
    """Infer OFT block size from strict upper-triangle parameter count."""
    if num_params < 1:
        raise ValueError(f"{num_params} is not a valid OFT upper-triangle width")
    disc = 1 + 8 * int(num_params)
    root = math.isqrt(disc)
    if root * root != disc:
        raise ValueError(f"{num_params} is not a valid OFT upper-triangle width")
    block_size = (1 + root) // 2
    if block_size * (block_size - 1) // 2 != num_params:
        raise ValueError(f"{num_params} is not a valid OFT upper-triangle width")
    return block_size


def oft_params_to_skew_matrix(oft_params: torch.Tensor, block_size: int | None = None) -> torch.Tensor:
    if oft_params.ndim < 2:
        raise ValueError(f"OFT params must have shape (..., P), got shape {tuple(oft_params.shape)}")
    num_params = oft_params.shape[-1]
    if block_size is None:
        block_size = infer_oft_block_size(num_params)
    expected = block_size * (block_size - 1) // 2
    if num_params != expected:
        raise ValueError(f"num_params_per_block={num_params} does not match block_size={block_size}")
    leading_shape = oft_params.shape[:-1]
    flat_params = oft_params.reshape(-1, num_params)
    indices = torch.triu_indices(block_size, block_size, offset=1, device=oft_params.device)
    rows, cols = indices[0], indices[1]
    skew = torch.zeros(
        flat_params.shape[0],
        block_size,
        block_size,
        dtype=oft_params.dtype,
        device=oft_params.device,
    )
    skew[:, rows, cols] = flat_params
    skew = skew - skew.transpose(-2, -1)
    return skew.reshape(*leading_shape, block_size, block_size)


def skew_matrix_to_oft_params(skew: torch.Tensor) -> torch.Tensor:
    if skew.ndim < 3 or skew.shape[-1] != skew.shape[-2]:
        raise ValueError(f"skew matrix must have shape (..., d, d), got {tuple(skew.shape)}")
    block_size = skew.shape[-1]
    indices = torch.triu_indices(block_size, block_size, offset=1, device=skew.device)
    return skew[..., indices[0], indices[1]]


def orthomerge_original_merge(
    vectors: list[torch.Tensor],
    block_size: int | None = None,
) -> torch.Tensor:
    """Mirror OrthoMerge_OFT_models.py merge_cayley_Q_list on full skew matrices."""
    if len(vectors) == 0:
        raise ValueError("orthomerge_original_merge requires >= 1 vector")
    if vectors[0].ndim < 2:
        raise ValueError(f"OFT params must have shape (..., P), got shape {tuple(vectors[0].shape)}")
    if block_size is None:
        block_size = infer_oft_block_size(vectors[0].shape[-1])
    out_dtype = vectors[0].dtype
    leading_shape = vectors[0].shape[:-1]
    num_params = vectors[0].shape[-1]
    flat_vectors = [v.float().reshape(-1, num_params) for v in vectors]
    skews = [oft_params_to_skew_matrix(v, block_size=block_size) for v in flat_vectors]
    stack = torch.stack(skews, dim=0)
    merged_sum = stack.sum(dim=0)
    flat = stack.reshape(stack.shape[0], -1)
    sum_of_norms = flat.norm(dim=1).sum()
    norm_of_sum = merged_sum.norm()
    if norm_of_sum.item() == 0:
        raise ValueError("cannot magnitude-correct OFT tensors with zero norm of summed generators")
    correction = sum_of_norms / norm_of_sum
    merged = (1.0 / len(vectors)) * correction * merged_sum
    merged = 0.5 * (merged - merged.transpose(-1, -2))
    return skew_matrix_to_oft_params(merged).reshape(*leading_shape, num_params).to(out_dtype)


class OFTLieAlgebraMerge(MergeStrategy):
    name = "oft"

    def merge(self, adapters: list[StateDict], weights: list[float] | None = None) -> StateDict:
        if len(adapters) < 2:
            raise ValueError("OFT merge requires >= 2 adapters")
        keys = list(adapters[0].keys())
        key_set = set(keys)
        for i, ad in enumerate(adapters[1:], start=1):
            if set(ad.keys()) != key_set:
                missing = key_set ^ set(ad.keys())
                raise ValueError(f"adapter {i} key set differs; symmetric diff: " f"{sorted(missing, key=repr)[:5]}")
        merged: StateDict = {}
        non_oft: list[StateKey] = []
        for key in keys:
            tensors = [ad[key] for ad in adapters]
            shapes = {tuple(t.shape) for t in tensors}
            if len(shapes) != 1:
                raise ValueError(f"shape mismatch for {key}: {shapes}")
            if _is_oft_key(key):
                merged[key] = magnitude_corrected_merge(tensors, weights)
            else:
                non_oft.append(key)
                merged[key] = torch.stack([t.float() for t in tensors]).mean(0).to(tensors[0].dtype)
        if non_oft:
            logger.warning(
                "OFT merge: %d non-oft keys plain-averaged: %s",
                len(non_oft),
                sorted(non_oft, key=repr),
            )
        return merged


register(OFTLieAlgebraMerge())


class OFTOriginalFormulaMerge(MergeStrategy):
    """Exact adapter merge formula from the original OrthoMerge OFT script."""

    name = "oft-original"

    def merge(self, adapters: list[StateDict], weights: list[float] | None = None) -> StateDict:
        if weights is not None:
            raise ValueError("oft-original reproduces the original equal-weight formula and does not accept weights")
        if len(adapters) < 2:
            raise ValueError("OFT original merge requires >= 2 adapters")
        keys = list(adapters[0].keys())
        key_set = set(keys)
        for i, ad in enumerate(adapters[1:], start=1):
            if set(ad.keys()) != key_set:
                missing = key_set ^ set(ad.keys())
                raise ValueError(f"adapter {i} key set differs; symmetric diff: " f"{sorted(missing, key=repr)[:5]}")
        merged: StateDict = {}
        non_oft: list[StateKey] = []
        for key in keys:
            tensors = [ad[key] for ad in adapters]
            shapes = {tuple(t.shape) for t in tensors}
            if len(shapes) != 1:
                raise ValueError(f"shape mismatch for {key}: {shapes}")
            if _is_original_oft_key(key):
                merged[key] = orthomerge_original_merge(tensors)
            else:
                non_oft.append(key)
                merged[key] = torch.stack([t.float() for t in tensors]).mean(0).to(tensors[0].dtype)
        if non_oft:
            logger.warning(
                "OFT original merge: %d non-oft keys plain-averaged: %s",
                len(non_oft),
                sorted(non_oft, key=repr),
            )
        return merged


register(OFTOriginalFormulaMerge())


class OFTNaiveMerge(MergeStrategy):
    """Baseline: plain (weighted) arithmetic mean of every tensor — NO magnitude
    correction, NO manifold. Exists only to quantify what OrthoMerge's
    magnitude-corrected Lie-algebra merge buys over a naive average."""

    name = "oft-naive"

    def merge(self, adapters: list[StateDict], weights: list[float] | None = None) -> StateDict:
        if len(adapters) < 2:
            raise ValueError("OFT merge requires >= 2 adapters")
        keys = list(adapters[0].keys())
        key_set = set(keys)
        for i, ad in enumerate(adapters[1:], start=1):
            if set(ad.keys()) != key_set:
                raise ValueError(f"adapter {i} key set differs")
        n = len(adapters)
        w = weights if weights is not None else [1.0 / n] * n
        wsum = float(sum(w))
        merged: StateDict = {}
        for key in keys:
            stacked = torch.stack([ad[key].float() for ad in adapters])  # (N, ...)
            wt = torch.tensor(w, dtype=torch.float32).view(n, *([1] * (stacked.dim() - 1)))
            merged[key] = ((wt * stacked).sum(0) / wsum).to(adapters[0][key].dtype)
        return merged


register(OFTNaiveMerge())
