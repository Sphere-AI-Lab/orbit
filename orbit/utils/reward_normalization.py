from __future__ import annotations

from collections.abc import Sequence

import torch


def normalize_grouped_rewards(
    raw_rewards: Sequence[float],
    group_indices: Sequence[int | None],
    *,
    normalize_by_std: bool,
    epsilon: float = 1e-6,
) -> list[float]:
    """Normalize scalar rewards within each prompt group.

    Rewards are centered per `group_index` and optionally scaled by the group's
    sample standard deviation, matching GRPO's group-wise semantics even when
    samples are interleaved.
    """

    if len(raw_rewards) != len(group_indices):
        raise ValueError("raw_rewards and group_indices must have the same length")

    grouped_positions: dict[int, list[int]] = {}
    for position, group_index in enumerate(group_indices):
        if group_index is None:
            raise ValueError("group_index must be set when normalizing grouped rewards")
        grouped_positions.setdefault(group_index, []).append(position)

    normalized = [0.0] * len(raw_rewards)
    for positions in grouped_positions.values():
        rewards = torch.tensor([raw_rewards[pos] for pos in positions], dtype=torch.float)
        rewards = rewards - rewards.mean()

        if normalize_by_std and rewards.numel() > 1:
            rewards = rewards / (rewards.std() + epsilon)

        for position, value in zip(positions, rewards.tolist(), strict=True):
            normalized[position] = value

    return normalized
