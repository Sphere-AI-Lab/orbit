"""Held-out negative log-likelihood evaluation for SFT runs.

Orbit's built-in eval generates completions and grades them. Every SFT figure
in the LoRA-without-regret study is held-out *test NLL*, so this adds a
forward-only eval that reuses the actor's existing ``compute_log_prob``
primitive rather than introducing a second forward path.

Everything in this module is pure Python/torch: no megatron, no CUDA, no
``torch.distributed``. That is deliberate. The two defects this eval can
plausibly carry -- dropping rows, and reducing wrongly across ranks -- both
live in logic that is testable on CPU, and they are tested there
(``tests/fast/utils/test_eval_nll.py``). The megatron-side wiring lives in
``MegatronTrainRayActor.compute_eval_nll``.

Three invariants this module exists to enforce:

1. **Every row is scored.** ``get_data_iterator`` computes
   ``num_local_samples // num_local_gbs`` -- floor division -- so a 100-row
   held-out file at ``global_batch_size=32`` would silently score 96 rows and
   drop 4, and the reported NLL would become a function of the training batch
   size. :func:`plan_eval_nll_microbatches` keeps the short final group, and
   the caller asserts that the number of scored samples equals the number of
   rows read.

2. **The reduction is token-weighted**, not sample-weighted: total negative
   log-prob over all *scored* tokens divided by the number of scored tokens.
   That is what HF's ``Trainer`` reports, and gate G2 compares the two numbers
   directly. A sample-weighted mean would disagree by a factor that varies with
   the length distribution.

3. **Only scored tokens count.** A response span can contain unscored tokens
   (interleaved user turns in a multi-turn conversation), exactly the tokens HF
   marks ``label == -100``. :func:`accumulate_nll` weights by the loss mask, so
   train and eval agree token-for-token.

Because a token-weighted mean is *not* the mean of per-shard token-weighted
means, :func:`accumulate_nll` returns accumulators (:class:`NllStats`) instead
of a pre-divided float. Sum them across data-parallel ranks, divide once.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "EvalNllRow",
    "NllStats",
    "accumulate_nll",
    "build_eval_nll_batch",
    "build_eval_nll_rows",
    "EVAL_NLL_METRIC_KEY",
    "build_eval_nll_metrics",
    "is_eval_nll_reporting_rank",
    "load_eval_nll_rows",
    "plan_eval_nll_microbatches",
    "plan_eval_nll_shards",
    "reduce_nll",
    "reject_eval_nll_on_unsupported_entrypoint",
    "select_eval_nll_result",
]

#: The metric the study reports. Task 10's results ledger keys on this exact
#: string, so it is pinned here (and by a test) rather than spelled inline.
EVAL_NLL_METRIC_KEY = "eval/test_nll"
#: The same measurement taken before any optimizer step, under its own key so
#: gate G4 can read the base model's NLL back unambiguously.
EVAL_NLL_BEFORE_TRAIN_METRIC_KEY = "eval/test_nll_before_train"

#: Keys we will look for in a held-out JSONL row, in order, when the caller has
#: not named one explicitly. ``prompt`` is what Orbit's SFT launcher uses
#: (``--input-key prompt``); ``messages`` is the common HF chat convention.
_CONVERSATION_KEYS = ("prompt", "messages", "conversation")


@dataclass(frozen=True)
class EvalNllRow:
    """One held-out example: a chat conversation and its optional tool schema."""

    messages: list[dict]
    tools: list[dict] | None = None


@dataclass(frozen=True)
class NllStats:
    """Additive accumulators for a token-weighted NLL.

    Kept un-divided so they can be summed across data-parallel shards (which
    hold different token counts) before the single final division.

    Attributes:
        sum_neg_logprob: Sum of ``-log p(token)`` over every scored token.
        num_tokens: Number of scored tokens (``loss_mask == 1``).
        num_samples: Number of samples scored, padding excluded. The caller
            asserts this equals the number of rows read from the held-out file.
        sum_sample_mean_nll: Sum over samples of that sample's own token-mean
            NLL. Reported as a diagnostic only -- the primary metric is
            token-weighted (see module docstring).
        num_scored_samples: Number of samples with at least one scored token,
            i.e. the denominator for ``sample_mean_nll``.
    """

    sum_neg_logprob: float = 0.0
    num_tokens: int = 0
    num_samples: int = 0
    sum_sample_mean_nll: float = 0.0
    num_scored_samples: int = 0

    @classmethod
    def zero(cls) -> NllStats:
        return cls()

    def __add__(self, other: NllStats) -> NllStats:
        if not isinstance(other, NllStats):
            return NotImplemented
        return NllStats(
            sum_neg_logprob=self.sum_neg_logprob + other.sum_neg_logprob,
            num_tokens=self.num_tokens + other.num_tokens,
            num_samples=self.num_samples + other.num_samples,
            sum_sample_mean_nll=self.sum_sample_mean_nll + other.sum_sample_mean_nll,
            num_scored_samples=self.num_scored_samples + other.num_scored_samples,
        )

    def to_values(self) -> list[float]:
        """Flatten for an all-reduce. Order is part of the contract."""
        return [
            float(self.sum_neg_logprob),
            float(self.num_tokens),
            float(self.num_samples),
            float(self.sum_sample_mean_nll),
            float(self.num_scored_samples),
        ]

    @classmethod
    def from_values(cls, values: Sequence[float]) -> NllStats:
        if len(values) != 5:
            raise ValueError(f"expected 5 values, got {len(values)}")
        return cls(
            sum_neg_logprob=float(values[0]),
            num_tokens=int(round(float(values[1]))),
            num_samples=int(round(float(values[2]))),
            sum_sample_mean_nll=float(values[3]),
            num_scored_samples=int(round(float(values[4]))),
        )

    @property
    def mean_nll(self) -> float:
        """Token-weighted mean NLL in nats; NaN when nothing was scored."""
        if self.num_tokens == 0:
            return math.nan
        return self.sum_neg_logprob / self.num_tokens

    @property
    def sample_mean_nll(self) -> float:
        """Mean over samples of each sample's token-mean NLL (diagnostic)."""
        if self.num_scored_samples == 0:
            return math.nan
        return self.sum_sample_mean_nll / self.num_scored_samples

    def as_dict(self) -> dict[str, float | int]:
        return {
            "nll": self.mean_nll,
            "sample_mean_nll": self.sample_mean_nll,
            "sum_neg_logprob": self.sum_neg_logprob,
            "num_tokens": self.num_tokens,
            "num_samples": self.num_samples,
            "num_scored_samples": self.num_scored_samples,
        }


def accumulate_nll(
    log_probs: Sequence[torch.Tensor],
    loss_masks: Sequence[torch.Tensor | Sequence[int]] | None = None,
    *,
    is_padding: Sequence[bool] | None = None,
) -> NllStats:
    """Accumulate a token-weighted NLL over one rank's samples.

    Args:
        log_probs: One 1-D tensor of per-token log-probabilities per sample,
            aligned with the sample's response span (this is exactly what
            ``get_log_probs_and_entropy`` returns: ``[R]`` per sample).
        loss_masks: One 1-D 0/1 mask per sample, the response-aligned suffix
            stored by ``sft_rollout.generate_rollout`` (``loss_mask[-R:]``).
            ``None`` means every response token is scored -- correct only for
            single-turn data, so callers evaluating real conversations must
            pass the masks.
        is_padding: Per-sample flag. Padding rows (duplicates inserted only to
            equalise data-parallel shard sizes) are dropped from the numerator,
            the token count, AND the sample count.

    Returns:
        Un-divided :class:`NllStats`.

    Raises:
        ValueError: on any per-sample length disagreement, or if the sequences
            themselves have different lengths.
    """
    if loss_masks is None:
        loss_masks = [None] * len(log_probs)
    if len(log_probs) != len(loss_masks):
        raise ValueError(f"length mismatch: {len(log_probs)} log-prob tensors vs {len(loss_masks)} loss masks")
    if is_padding is None:
        is_padding = [False] * len(log_probs)
    if len(is_padding) != len(log_probs):
        raise ValueError(f"length mismatch: {len(log_probs)} log-prob tensors vs {len(is_padding)} padding flags")

    stats = NllStats.zero()
    for log_prob, loss_mask, padded in zip(log_probs, loss_masks, is_padding, strict=True):
        if padded:
            continue
        # float64 throughout: the target table spans 0.009 nats, and a float32
        # sum over ~10^5 tokens does not have the digits to resolve that.
        values = torch.as_tensor(log_prob).detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        if loss_mask is None:
            mask = torch.ones_like(values)
        else:
            mask = torch.as_tensor(loss_mask).detach().to(dtype=torch.float64, device="cpu").reshape(-1)
            if mask.numel() != values.numel():
                raise ValueError(
                    f"length mismatch: log-prob tensor has {values.numel()} elements, "
                    f"loss mask has {mask.numel()}"
                )

        sum_neg = float(-(values * mask).sum())
        num_tokens = int(mask.sum())
        stats = stats + NllStats(
            sum_neg_logprob=sum_neg,
            num_tokens=num_tokens,
            num_samples=1,
            sum_sample_mean_nll=(sum_neg / num_tokens) if num_tokens else 0.0,
            num_scored_samples=1 if num_tokens else 0,
        )
    return stats


def reduce_nll(
    log_probs: Sequence[torch.Tensor],
    response_lengths: Sequence[int],
    loss_masks: Sequence[torch.Tensor | Sequence[int]] | None = None,
) -> float:
    """Token-weighted mean negative log-likelihood for a single rank.

    A convenience wrapper over :func:`accumulate_nll` that additionally checks
    each tensor against its declared response length. Prefer
    :func:`accumulate_nll` whenever the result has to cross a rank boundary:
    a token-weighted mean is not the mean of per-rank token-weighted means.

    Args:
        log_probs: Per-sample 1-D log-probability tensors.
        response_lengths: Declared number of response tokens per sample.
        loss_masks: Optional per-sample 0/1 masks. When omitted, every response
            token is treated as scored.

    Returns:
        Mean NLL in nats, or NaN when there is nothing to score.

    Raises:
        ValueError: if any declared length disagrees with the tensors.
    """
    if len(log_probs) != len(response_lengths):
        raise ValueError(
            f"length mismatch: {len(log_probs)} tensors vs {len(response_lengths)} response lengths"
        )
    for tensor, length in zip(log_probs, response_lengths, strict=True):
        numel = torch.as_tensor(tensor).numel()
        if numel != length:
            raise ValueError(f"length mismatch: tensor has {numel} elements, expected {length}")
    return accumulate_nll(log_probs, loss_masks).mean_nll


def is_eval_nll_reporting_rank(parallel_state) -> bool:
    """Whether this rank is the single one that reports the held-out NLL.

    ``RayTrainGroup._broadcast`` returns one value per actor across the whole
    TP x PP x DP grid. TP/PP replicas hold the *same* samples, so averaging or
    summing per-actor values would over-count by ``tp_size * pp_size``. Rather
    than hoping those factors cancel, exactly one rank reports and the rest
    return ``None``.

    The chosen rank is DP 0 / TP 0 / CP 0 on the last pipeline stage: the DP
    all-reduce leaves the global totals on every DP rank, and only the last
    pipeline stage holds logits at all (``forward_only`` returns ``{}``
    elsewhere).

    Args:
        parallel_state: Anything with ``is_pp_last_stage`` (a bool, not a
            callable -- a bound method would be truthy always and silently
            disable the pipeline half of this check) plus ``tp``/``cp``/
            ``intra_dp`` groups exposing ``.rank``.

    Returns:
        True on exactly one rank of any legal TP x PP x DP x CP grid.
    """
    is_pp_last_stage = parallel_state.is_pp_last_stage
    if callable(is_pp_last_stage):
        raise TypeError(
            "parallel_state.is_pp_last_stage must be a bool, not a callable; "
            "a bound method is always truthy and would silently disable the "
            "pipeline-stage half of the reporting-rank check."
        )
    return bool(
        is_pp_last_stage
        and parallel_state.tp.rank == 0
        and parallel_state.cp.rank == 0
        and parallel_state.intra_dp.rank == 0
    )


def select_eval_nll_result(results: Sequence[dict | None]) -> dict:
    """Pick the single reported result out of one-value-per-actor list.

    Args:
        results: What ``RayTrainGroup._broadcast`` returned -- one entry per
            actor, ``None`` from every non-reporting rank.

    Returns:
        The one reported statistics dict.

    Raises:
        AssertionError: if zero or more than one rank reported. Failing loudly
            here is the point: a silently averaged or summed set of per-actor
            values would be wrong by a factor of ``tp_size * pp_size``.
    """
    reported = [result for result in results if result is not None]
    assert len(reported) == 1, (
        f"expected exactly one actor to report eval NLL, got {len(reported)} of {len(results)}"
    )
    return reported[0]


def build_eval_nll_metrics(stats: dict, step: int, *, before_train: bool = False) -> dict:
    """Metric dict for one held-out NLL measurement.

    Lives here rather than inline in ``train.py`` because neither entrypoint is
    importable without CUDA, so key strings spelled there could not be pinned by
    a test -- and Task 10's sweep driver keys on them.

    Args:
        stats: :meth:`NllStats.as_dict` output, as returned across Ray.
        step: Value for ``rollout/step``.
        before_train: When True, also emit the measurement under
            :data:`EVAL_NLL_BEFORE_TRAIN_METRIC_KEY`.

    Returns:
        Metrics ready for ``tracking_utils.log(..., step_key="rollout/step")``.
    """
    metrics = {
        EVAL_NLL_METRIC_KEY: stats["nll"],
        "eval/test_nll_sample_mean": stats["sample_mean_nll"],
        "eval/test_nll_tokens": stats["num_tokens"],
        "eval/test_nll_samples": stats["num_samples"],
        "rollout/step": step,
    }
    if before_train:
        metrics[EVAL_NLL_BEFORE_TRAIN_METRIC_KEY] = stats["nll"]
    return metrics


def reject_eval_nll_on_unsupported_entrypoint(args, entrypoint: str) -> None:
    """Refuse ``--eval-nll-data`` on an entrypoint that does not implement it.

    The flag is registered on the shared parser, so any entrypoint parses it
    happily. Accepting a metric flag and emitting no metric is precisely the
    failure class this eval exists to prevent, so entrypoints without the hook
    must stop at startup.

    Raises:
        ValueError: (not ``AssertionError``) if ``--eval-nll-data`` is set. This
            is a user configuration error and must not vanish under ``python -O``.
    """
    if getattr(args, "eval_nll_data", None):
        raise ValueError(
            f"--eval-nll-data is not supported by {entrypoint}; held-out NLL eval is "
            "implemented only in train.py. Either drop --eval-nll-data or run with "
            "ORBIT_ENTRYPOINT pointing at train.py."
        )


def plan_eval_nll_microbatches(num_local_rows: int, micro_batch_size: int) -> list[list[int]]:
    """Contiguous micro-batch schedule covering every local row exactly once.

    Deliberately does NOT floor-divide: the final group is short whenever
    ``num_local_rows`` is not a multiple of ``micro_batch_size``. Contiguity is
    also load-bearing -- because the flattened schedule is exactly
    ``range(num_local_rows)``, the log-probs come back in row order whether or
    not ``aggregate_forward_results`` applies its ``use_dynamic_batch_size``
    reordering pass.

    Args:
        num_local_rows: Rows assigned to this rank.
        micro_batch_size: Maximum rows per micro-batch.

    Returns:
        A list of index lists, suitable for ``DataIterator(..., micro_batch_indices=...)``.

    Raises:
        ValueError: for a non-positive row count or micro-batch size.
    """
    if num_local_rows <= 0:
        raise ValueError(f"num_local_rows must be positive, got {num_local_rows}")
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
    return [
        list(range(start, min(start + micro_batch_size, num_local_rows)))
        for start in range(0, num_local_rows, micro_batch_size)
    ]


def plan_eval_nll_shards(
    num_rows: int,
    dp_size: int,
    pad_index: int = 0,
) -> list[list[tuple[int, bool]]]:
    """Assign every held-out row to exactly one data-parallel rank.

    Shards are padded to equal length by repeating ``pad_index``. Equal length
    is required, not cosmetic: the pipeline schedule is a collective, so DP
    ranks running different micro-batch counts would hang. Padding rows are
    flagged so :func:`accumulate_nll` drops them from the numerator, the token
    count, and the sample count alike.

    Args:
        num_rows: Total rows in the held-out file.
        dp_size: Data-parallel world size.
        pad_index: Row to duplicate when padding. Callers should pass the
            shortest row so the wasted forward pass is as cheap as possible.

    Returns:
        One list per DP rank of ``(row_index, is_padding)`` pairs.

    Raises:
        ValueError: for an empty file or a non-positive ``dp_size``.
    """
    if num_rows <= 0:
        raise ValueError(f"num_rows must be positive, got {num_rows}")
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if not 0 <= pad_index < num_rows:
        raise ValueError(f"pad_index {pad_index} out of range for {num_rows} rows")

    per_rank = -(-num_rows // dp_size)  # ceil
    num_padding = per_rank * dp_size - num_rows
    if num_padding:
        logger.info(
            "eval_nll: padding %d row(s) (duplicating row %d) so all %d DP shards hold %d rows; "
            "padded rows are excluded from the NLL",
            num_padding,
            pad_index,
            dp_size,
            per_rank,
        )

    flat: list[tuple[int, bool]] = [(i, False) for i in range(num_rows)]
    flat += [(pad_index, True)] * num_padding
    return [flat[rank::dp_size] for rank in range(dp_size)]


def load_eval_nll_rows(
    path: str | Path,
    input_key: str | None = None,
    tool_key: str = "tools",
    metadata_key: str = "metadata",
) -> list[EvalNllRow]:
    """Read a held-out JSONL file into conversations.

    Args:
        path: JSONL file, one chat example per line.
        input_key: Key holding the message list. When ``None`` -- or when the
            requested key is absent from the file, which happens because the
            caller passes the *training* data's ``--input-key`` -- the first of
            ``prompt``/``messages``/``conversation`` present in the first row is
            used instead, with a warning. Whichever key is chosen is then
            required of every row.
        tool_key: Top-level key holding a tool schema.
        metadata_key: Fallback location for ``tools`` (matches how
            ``sft_rollout`` reads ``sample.metadata["tools"]``).

    Returns:
        One :class:`EvalNllRow` per non-blank line, in file order.

    Raises:
        ValueError: if the file is empty, a line is not a JSON object, or no
            conversation key can be found.
    """
    path = Path(path)
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_lines:
        raise ValueError(f"held-out NLL file has no rows: {path}")

    records = []
    for lineno, line in enumerate(raw_lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object, got {type(record).__name__}")
        records.append(record)

    requested_key = input_key
    if input_key is None or input_key not in records[0]:
        for candidate in _CONVERSATION_KEYS:
            if candidate in records[0]:
                input_key = candidate
                break
        else:
            raise ValueError(
                f"{path}: no conversation key found; looked for "
                f"{[requested_key, *_CONVERSATION_KEYS] if requested_key else list(_CONVERSATION_KEYS)} "
                f"in a row with keys {sorted(records[0])}"
            )
        if requested_key is not None:
            logger.warning(
                "eval_nll: %s has no %r column; falling back to %r",
                path,
                requested_key,
                input_key,
            )

    rows = []
    for lineno, record in enumerate(records, start=1):
        if input_key not in record:
            raise ValueError(f"{path}:{lineno}: missing conversation key {input_key!r}")
        messages = record[input_key]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}:{lineno}: {input_key!r} must be a non-empty list of chat messages")
        tools = record.get(tool_key)
        if tools is None:
            metadata = record.get(metadata_key)
            if isinstance(metadata, dict):
                tools = metadata.get("tools")
        rows.append(EvalNllRow(messages=messages, tools=tools))

    if len(rows) != len(raw_lines):
        raise ValueError(f"{path}: read {len(rows)} rows from {len(raw_lines)} lines")
    return rows


def build_eval_nll_rows(rows: Sequence[EvalNllRow], mask_generator: Any) -> dict[str, Any]:
    """Tokenize held-out rows into rollout-batch-shaped CPU lists.

    Mirrors ``sft_rollout.generate_rollout`` exactly -- same
    ``MultiTurnLossMaskGenerator``, same ``loss_mask[-response_length:]``
    suffix -- so training and eval score the identical tokens. Gate G3 pins
    that masking against an HF oracle; nothing here re-derives it.

    Args:
        rows: Conversations from :func:`load_eval_nll_rows`.
        mask_generator: Anything exposing ``get_loss_mask(messages, tools=...)``
            and ``get_response_lengths([mask])``.

    Returns:
        Dict with ``tokens``, ``loss_masks`` (response-aligned suffixes),
        ``response_lengths``, ``total_lengths``, and ``shortest_row_index``.

    Raises:
        ValueError: if a row has no scored tokens, which would mean the mask
            type does not match the data and every NLL would be meaningless.
    """
    tokens: list[list[int]] = []
    loss_masks: list[list[int]] = []
    response_lengths: list[int] = []
    total_lengths: list[int] = []

    for index, row in enumerate(rows):
        token_ids, loss_mask = mask_generator.get_loss_mask(row.messages, tools=row.tools)
        response_length = mask_generator.get_response_lengths([loss_mask])[0]
        if response_length <= 0 or sum(loss_mask) == 0:
            raise ValueError(
                f"held-out row {index} has no scored tokens; check --loss-mask-type against the data"
            )
        tokens.append(token_ids)
        loss_masks.append(loss_mask[-response_length:])
        response_lengths.append(response_length)
        total_lengths.append(len(token_ids))

    assert len(tokens) == len(rows), f"tokenized {len(tokens)} rows from {len(rows)} inputs"

    return {
        "tokens": tokens,
        "loss_masks": loss_masks,
        "response_lengths": response_lengths,
        "total_lengths": total_lengths,
        "shortest_row_index": min(range(len(total_lengths)), key=total_lengths.__getitem__),
    }


def build_eval_nll_batch(args, tokenizer=None) -> dict[str, Any]:
    """Load and tokenize ``args.eval_nll_data`` the way SFT training does.

    Args:
        args: Parsed Orbit arguments. Uses ``eval_nll_data``, ``loss_mask_type``,
            ``input_key``/``tool_key``/``metadata_key``, and (when ``tokenizer``
            is not supplied) ``hf_checkpoint``/``chat_template_path``.
        tokenizer: Pre-loaded tokenizer. The actor passes its own so eval and
            training cannot drift apart.

    Returns:
        The dict described in :func:`build_eval_nll_rows`.
    """
    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator

    if tokenizer is None:
        from orbit.utils.processing_utils import load_tokenizer

        tokenizer = load_tokenizer(
            args.hf_checkpoint,
            chat_template_path=args.chat_template_path,
            trust_remote_code=True,
        )

    rows = load_eval_nll_rows(
        args.eval_nll_data,
        input_key=getattr(args, "input_key", None),
        tool_key=getattr(args, "tool_key", "tools"),
        metadata_key=getattr(args, "metadata_key", "metadata"),
    )
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)
    batch = build_eval_nll_rows(rows, mask_generator)
    logger.info(
        "eval_nll: loaded %d held-out rows from %s (%d scored tokens total, loss_mask_type=%s)",
        len(rows),
        args.eval_nll_data,
        sum(sum(mask) for mask in batch["loss_masks"]),
        args.loss_mask_type,
    )
    return batch
