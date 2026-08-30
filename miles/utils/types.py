# ORBIT-SEAM: math import backs Sample.validate_teacher_topk's math.isfinite check on teacher logprobs
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy
import torch


@dataclass(frozen=True)
class AdapterRef:
    """Which LoRA adapter a sample is bound to (training slot routing, inference lora_path); ``None`` = no adapter."""

    name: str
    slot: int


@dataclass(frozen=True)
class RewardSpec:
    """Per-sample spec of how the response is scored; intentionally decoupled from adapter routing."""

    rm_type: str | None = None
    custom_rm_path: str | None = None


# ORBIT-SEAM: new helper - resolves the policy version tag from meta_info, preferring
# adapter_version (PEFT) over weight_version and asserting the v1 invariant that they agree when
# both are present; used by Sample.update_from_meta_info below instead of base's plain weight_version read
def _extract_policy_version(meta_info: dict) -> str | None:
    adapter_version = meta_info.get("adapter_version")
    weight_version = meta_info.get("weight_version")
    if adapter_version is not None and weight_version is not None and str(adapter_version) != str(weight_version):
        raise ValueError(
            f"adapter_version ({adapter_version!r}) and weight_version ({weight_version!r}) "
            "disagree in meta_info; expected v1 invariant adapter_version == weight_version"
        )
    if adapter_version is not None:
        return str(adapter_version)
    if weight_version is not None:
        return str(weight_version)
    return None


@dataclass
class Sample:
    """The sample generated"""

    group_index: int | None = None
    index: int | None = None
    # Rollout execution id; None falls back to ``index``. Compact / subagent
    # siblings must share it so the rollout is counted once.
    rollout_id: int | None = None
    # prompt
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    multimodal_inputs: dict[str, Any] = None  # raw multimodal data, e.g. images, videos, etc.
    multimodal_train_inputs: dict[str, Any] = None  # processed multimodal data, e.g. pixel_values, etc.
    # response
    response: str = ""
    response_length: int = 0
    label: str | None = None
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    weight_versions: list[str] = field(default_factory=list)
    rollout_log_probs: list[float] | None = None  # Log probabilities from rollout engine
    # ORBIT-SEAM: OPD teacher-scoring fields on the Sample type - teacher logprobs/hidden-states,
    # reverse-KL, and retained direct top-k transport, mirrored through strip/reset/validate below
    teacher_log_probs: list[float] | None = None  # per-response-token teacher logprobs (OPD)
    # Teacher's last-layer hidden state per response position, shape (response_length, hidden);
    # full-vocab OPD (--teacher-score-mode full_vocab) sets this instead of teacher_log_probs.
    teacher_hidden_states: numpy.ndarray | None = None
    opd_reverse_kl: list[float] | None = None  # Precomputed per-token OPD reverse-KL estimate
    teacher_topk_ids: list[list[int]] | None = None  # Per-position teacher top-k token ids (--loss-type opd_topk_loss)
    teacher_topk_logprobs: list[list[float]] | None = None  # Per-position teacher top-k logprobs (opd_topk_loss)
    rollout_routed_experts: numpy.ndarray | None = (
        None  # Routed experts from rollout engine. shape: (num_tokens-1, num_layers, moe_router_topk), dtype=int32
    )
    rollout_indexer_topk: numpy.ndarray | None = (
        None  # Indexer topk from rollout engine. shape: (num_tokens-1, num_indexer_layers, index_topk), dtype=int32
    )
    remove_sample: bool = False
    # ORBIT-SEAM: upstream re-declares teacher_log_probs / opd_reverse_kl here; the orbit OPD field
    # block above already declares them (plus teacher_hidden_states and the top-k transport), so the
    # duplicate dataclass fields are dropped

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        # Indicates a recoverable or non-critical failure during generation (e.g., tool call failure,
        # external API error, parsing error). Unlike ABORTED, FAILED samples may still contain partial
        # valid output and can be retried or handled gracefully.
        FAILED = "failed"

    status: Status = Status.PENDING

    metadata: dict = field(default_factory=dict)
    generate_function_path: str | None = None
    # metadata used during training, e.g., what loss to use for this sample.
    train_metadata: dict | None = None

    # MultiLoRA: which adapter this sample trains/infers with
    adapter: AdapterRef | None = None
    # Per-sample reward dispatch override (e.g., per-adapter RM in multi-LoRA)
    reward_spec: RewardSpec | None = None

    # Per-sample routing key for the router's consistent_hashing policy (sent as X-SMG-Routing-Key)
    routing_key: str | None = None

    non_generation_time: float = 0.0  # time spent in non-generation steps

    @dataclass
    class SpecInfo:
        spec_accept_token_num: int = 0
        spec_draft_token_num: int = 0
        spec_verify_ct: int = 0
        completion_token_num: int = 0

        @property
        def spec_accept_rate(self) -> float:
            return self.spec_accept_token_num / self.spec_draft_token_num if self.spec_draft_token_num > 0 else 0.0

        @property
        def spec_accept_length(self) -> float:
            return self.completion_token_num / self.spec_verify_ct if self.spec_verify_ct > 0 else 0.0

        def add(self, meta_info: dict):
            self.spec_accept_token_num += meta_info.get("spec_accept_token_num", 0)
            self.spec_draft_token_num += meta_info.get("spec_draft_token_num", 0)
            self.spec_verify_ct += meta_info.get("spec_verify_ct", 0)
            self.completion_token_num += meta_info.get("completion_tokens", 0)

        def to_dict(self):
            return {
                "spec_accept_token_num": self.spec_accept_token_num,
                "spec_draft_token_num": self.spec_draft_token_num,
                "spec_verify_ct": self.spec_verify_ct,
                "completion_token_num": self.completion_token_num,
            }

        @staticmethod
        def from_dict(data: dict):
            info = Sample.SpecInfo()
            info.spec_accept_token_num = data.get("spec_accept_token_num", 0)
            info.spec_draft_token_num = data.get("spec_draft_token_num", 0)
            info.spec_verify_ct = data.get("spec_verify_ct", 0)
            info.completion_token_num = data.get("completion_token_num", 0)
            return info

    spec_info: SpecInfo = field(default_factory=SpecInfo)

    @dataclass
    class PrefixCacheInfo:
        cached_tokens: int = 0
        total_prompt_tokens: int = 0

        @property
        def prefix_cache_hit_rate(self) -> float:
            return self.cached_tokens / self.total_prompt_tokens if self.total_prompt_tokens > 0 else 0.0

        def add(self, meta_info: dict):
            self.cached_tokens += meta_info.get("cached_tokens", 0)
            # new_tokens = input_tokens - cached_tokens
            self.total_prompt_tokens += meta_info.get("prompt_tokens", 0)

        def to_dict(self):
            return {
                "cached_tokens": self.cached_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
            }

        @staticmethod
        def from_dict(data: dict):
            info = Sample.PrefixCacheInfo()
            info.cached_tokens = data.get("cached_tokens", 0)
            info.total_prompt_tokens = data.get("total_prompt_tokens", 0)
            return info

    prefix_cache_info: PrefixCacheInfo = field(default_factory=PrefixCacheInfo)

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        value["spec_info"] = self.spec_info.to_dict()
        value["prefix_cache_info"] = self.prefix_cache_info.to_dict()
        return value

    @staticmethod
    def from_dict(data: dict):
        data = dict(data)
        data["status"] = Sample.Status(data["status"])
        data["spec_info"] = Sample.SpecInfo.from_dict(data.get("spec_info", {}))
        data["prefix_cache_info"] = Sample.PrefixCacheInfo.from_dict(data.get("prefix_cache_info", {}))

        field_names = set(Sample.__dataclass_fields__.keys())
        init_data = {k: v for k, v in data.items() if k in field_names}
        sample = Sample(**init_data)

        for key, value in data.items():
            if key not in field_names:
                setattr(sample, key, value)

        return sample

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]

    @property
    def effective_response_length(self):
        return sum(self.loss_mask) if self.loss_mask is not None else self.response_length

    # ORBIT-SEAM: new method - validates the retained direct-OPD top-k transport (teacher_topk_ids
    # / teacher_topk_logprobs): row-count/response_length agreement, uniform row width, padding
    # convention (logprob == -1e4 <=> token id 0), and duplicate-id detection per row
    def validate_teacher_topk(self, expected_top_k: int | None = None) -> int | None:
        """Validate the retained direct-OPD top-k pair and return its row width.

        Empty responses carry ``([], [])`` and therefore cannot encode their row
        width.  In that case ``expected_top_k`` is returned when the caller knows
        it from configuration; otherwise the result is ``None``.
        """
        ids = self.teacher_topk_ids
        logprobs = self.teacher_topk_logprobs
        if (ids is None) != (logprobs is None):
            raise ValueError(
                "teacher_topk_ids and teacher_topk_logprobs must be present together; "
                f"got ids={ids is not None}, logprobs={logprobs is not None}"
            )
        if ids is None:
            return None

        if not isinstance(ids, (list, tuple)) or not isinstance(logprobs, (list, tuple)):
            raise ValueError("teacher_topk_ids and teacher_topk_logprobs must be lists or tuples of rows")
        if len(ids) != self.response_length:
            raise ValueError(f"teacher_topk_ids row count ({len(ids)}) != response_length ({self.response_length})")
        if len(logprobs) != self.response_length:
            raise ValueError(
                f"teacher_topk_logprobs row count ({len(logprobs)}) != response_length ({self.response_length})"
            )

        if expected_top_k is not None:
            if type(expected_top_k) is not int or expected_top_k <= 0:
                raise ValueError(f"expected_top_k must be a positive exact integer, got {expected_top_k!r}")

        if self.response_length == 0:
            return expected_top_k

        row_width = None
        for row_idx, (ids_row, logprobs_row) in enumerate(zip(ids, logprobs, strict=True)):
            if not isinstance(ids_row, (list, tuple)):
                raise ValueError(
                    f"teacher_topk_ids row {row_idx} must be a list or tuple, got {type(ids_row).__name__}"
                )
            if not isinstance(logprobs_row, (list, tuple)):
                raise ValueError(
                    f"teacher_topk_logprobs row {row_idx} must be a list or tuple, "
                    f"got {type(logprobs_row).__name__}"
                )
            if row_width is None:
                row_width = len(ids_row)
                if row_width <= 0:
                    raise ValueError("teacher top-k rows must have positive width")
            if len(ids_row) != row_width:
                raise ValueError(
                    f"teacher_topk_ids is ragged: row {row_idx} has width {len(ids_row)}, " f"expected {row_width}"
                )
            if len(logprobs_row) != row_width:
                raise ValueError(
                    f"teacher_topk_logprobs row {row_idx} has width {len(logprobs_row)}, "
                    f"expected {row_width} to match teacher_topk_ids"
                )

            observed_ids = set()
            for col_idx, (token_id, logprob) in enumerate(zip(ids_row, logprobs_row, strict=True)):
                if type(token_id) is not int or token_id < 0:
                    raise ValueError(
                        f"teacher_topk_ids[{row_idx}][{col_idx}] must be a nonnegative exact integer, "
                        f"got {token_id!r}"
                    )
                if type(logprob) not in (int, float) or not math.isfinite(logprob) or logprob > 0:
                    raise ValueError(
                        f"teacher_topk_logprobs[{row_idx}][{col_idx}] must be finite and <= 0, " f"got {logprob!r}"
                    )
                is_padding = logprob == -1e4
                if is_padding and token_id != 0:
                    raise ValueError(f"teacher top-k padding at row {row_idx}, column {col_idx} must use token id 0")
                if not is_padding:
                    if token_id in observed_ids:
                        raise ValueError(f"teacher_topk_ids row {row_idx} contains duplicate token id {token_id}")
                    observed_ids.add(token_id)

        if row_width is None:
            raise ValueError("teacher top-k row width could not be determined")
        if expected_top_k is not None:
            if row_width != expected_top_k:
                raise ValueError(f"teacher top-k row width ({row_width}) != configured top-k ({expected_top_k})")
        return row_width

    def validate(self):
        assert self.response_length >= 0, f"response_length must be >= 0, got {self.response_length}"
        assert (
            len(self.tokens) >= self.response_length
        ), f"tokens length ({len(self.tokens)}) must be >= response_length ({self.response_length})"
        if self.loss_mask is not None:
            assert (
                len(self.loss_mask) == self.response_length
            ), f"loss_mask length ({len(self.loss_mask)}) != response_length ({self.response_length})"
        if self.rollout_log_probs is not None:
            assert (
                len(self.rollout_log_probs) == self.response_length
            ), f"rollout_log_probs length ({len(self.rollout_log_probs)}) != response_length ({self.response_length})"
        # ORBIT-SEAM: length-parity asserts for the OPD teacher fields above, plus a
        # validate_teacher_topk() call for the retained direct top-k transport
        if self.teacher_log_probs is not None:
            assert (
                len(self.teacher_log_probs) == self.response_length
            ), f"teacher_log_probs length ({len(self.teacher_log_probs)}) != response_length ({self.response_length})"
        if self.teacher_hidden_states is not None:
            assert len(self.teacher_hidden_states) == self.response_length, (
                f"teacher_hidden_states length ({len(self.teacher_hidden_states)}) != "
                f"response_length ({self.response_length})"
            )
        if self.opd_reverse_kl is not None:
            assert (
                len(self.opd_reverse_kl) == self.response_length
            ), f"opd_reverse_kl length ({len(self.opd_reverse_kl)}) != response_length ({self.response_length})"
        self.validate_teacher_topk()
        if self.rollout_routed_experts is not None:
            actual = len(self.rollout_routed_experts)
            expect = len(self.tokens) - 1
            mm = self.multimodal_train_inputs or {}
            extra = sum(
                int(c) - 1 for key in ("mm_vision_num_patches", "mm_audio_num_tokens") for c in list(mm.get(key) or [])
            )
            assert actual in (expect, expect + extra), (
                f"rollout_routed_experts length ({actual}) != len(tokens) - 1 ({expect})"
                f" or media-expanded ({expect + extra})"
            )
        if self.rollout_indexer_topk is not None:
            actual = len(self.rollout_indexer_topk)
            expect = len(self.tokens) - 1
            assert actual == expect, f"rollout_indexer_topk length ({actual}) != len(tokens) - 1 ({expect})"

    def strip_last_output_tokens(self, n: int, tokenizer) -> None:
        """Remove the last *n* output tokens and all associated per-token info."""
        if n <= 0:
            return
        assert (
            n <= self.response_length
        ), f"cannot strip {n} tokens: only {self.response_length} output tokens available"
        self.tokens = self.tokens[:-n]
        self.response_length -= n
        if self.rollout_log_probs is not None:
            self.rollout_log_probs = self.rollout_log_probs[:-n]
        # ORBIT-SEAM: strip the OPD teacher fields (and the OPD student top-logprobs metadata key)
        # in lockstep with the base fields above, so a stripped sample's per-token arrays stay aligned
        if self.teacher_log_probs is not None:
            self.teacher_log_probs = self.teacher_log_probs[:-n]
        if self.teacher_hidden_states is not None:
            self.teacher_hidden_states = self.teacher_hidden_states[:-n]
        if self.opd_reverse_kl is not None:
            self.opd_reverse_kl = self.opd_reverse_kl[:-n]
        if self.teacher_topk_ids is not None:
            self.teacher_topk_ids = self.teacher_topk_ids[:-n]
        if self.teacher_topk_logprobs is not None:
            self.teacher_topk_logprobs = self.teacher_topk_logprobs[:-n]
        if self.metadata and "opd_student_top_logprobs" in self.metadata:
            self.metadata["opd_student_top_logprobs"] = self.metadata["opd_student_top_logprobs"][:-n]
        if self.loss_mask is not None:
            self.loss_mask = self.loss_mask[:-n]
        self.response = tokenizer.decode(self.tokens[-self.response_length :]) if self.response_length > 0 else ""
        if self.rollout_routed_experts is not None:
            self.rollout_routed_experts = self.rollout_routed_experts[:-n]
        if self.rollout_indexer_topk is not None:
            self.rollout_indexer_topk = self.rollout_indexer_topk[:-n]

    def reset_for_retry(self) -> None:
        """Reset generated outputs so the original prompt can be re-sampled.

        Keeps identity / prompt fields (group_index, index, prompt, label,
        multimodal_inputs, metadata, generate_function_path, routing_key) and
        restores everything else to dataclass defaults.
        """
        self.tokens = []
        self.multimodal_train_inputs = None
        self.response = ""
        self.response_length = 0
        self.reward = None
        self.loss_mask = None
        self.weight_versions = []
        self.rollout_log_probs = None
        # ORBIT-SEAM: reset the OPD teacher fields (and pop OPD scoring artifacts from metadata,
        # which is otherwise kept across retries) alongside base's reset-to-defaults above
        self.teacher_log_probs = None
        self.teacher_hidden_states = None
        self.opd_reverse_kl = None
        self.teacher_topk_ids = None
        self.teacher_topk_logprobs = None
        if self.metadata:
            # metadata is kept across retries, but OPD scoring artifacts belong to
            # the discarded generation and would poison the retried sample.
            self.metadata.pop("opd_student_top_logprobs", None)
            self.metadata.pop("opd_teacher_response", None)
        self.rollout_routed_experts = None
        self.rollout_indexer_topk = None
        self.status = Sample.Status.ABORTED
        self.non_generation_time = 0.0
        self.spec_info = Sample.SpecInfo()
        self.prefix_cache_info = Sample.PrefixCacheInfo()
        self.remove_sample = False
        self.train_metadata = None

    @property
    def oldest_weight_version(self) -> int | None:
        """Minimum weight version across all turns (generation calls) for this trajectory."""
        numeric = [int(v) for v in self.weight_versions if str(v).isdigit()]
        return min(numeric) if numeric else None

    def update_from_meta_info(self, args, meta_info: dict):
        """
        Update the sample with new information from meta_info returned by the rollout engine.
        And extract
        """
        if args.sglang_speculative_algorithm:
            # cannot directly use spec info from sglang because of partial rollout.
            self.spec_info.add(meta_info=meta_info)

        # Collect prefix cache statistics
        self.prefix_cache_info.add(meta_info=meta_info)

        # ORBIT-SEAM: delegates to _extract_policy_version above (adapter_version-or-weight_version,
        # with the v1 agreement check) instead of base's plain meta_info["weight_version"] read
        version = _extract_policy_version(meta_info)
        if version is not None:
            self.weight_versions.append(version)

        match meta_info["finish_reason"]["type"]:
            case "length":
                self.status = Sample.Status.TRUNCATED
            case "abort":
                self.status = Sample.Status.ABORTED
            case "stop":
                self.status = Sample.Status.COMPLETED


# ORBIT-SEAM: new function - validates and gathers a batch's retained direct-OPD top-k rows
# (requires every sample carry the fields once any sample does, and a consistent top-k width)
def collect_teacher_topk_data(samples: list[Sample], expected_top_k: int | None) -> dict[str, list] | None:
    """Validate and collect a batch of retained direct-OPD top-k rows."""
    if not any(sample.teacher_topk_ids is not None or sample.teacher_topk_logprobs is not None for sample in samples):
        return None

    observed_top_k = None
    for sample_idx, sample in enumerate(samples):
        if sample.teacher_topk_ids is None and sample.teacher_topk_logprobs is None:
            raise ValueError(
                f"teacher top-k fields are missing on sample {sample_idx}/{len(samples)}; "
                "the direct top-k OPD scorer must score every sample in the batch."
            )
        try:
            sample_top_k = sample.validate_teacher_topk(expected_top_k=expected_top_k)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid teacher top-k transport on sample {sample_idx}: {exc}") from exc

        if sample_top_k is None:
            continue
        if observed_top_k is None:
            observed_top_k = sample_top_k
        elif sample_top_k != observed_top_k:
            raise ValueError(
                f"teacher top-k width differs across samples: sample {sample_idx} has K={sample_top_k}, "
                f"expected K={observed_top_k}."
            )

    return {
        "teacher_topk_ids": [sample.teacher_topk_ids for sample in samples],
        "teacher_topk_logprobs": [sample.teacher_topk_logprobs for sample in samples],
    }


@dataclass(frozen=True)
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int


# A dict-based batch produced along the rollout -> training path
# In Megatron backend, several fields are converted to torch.Tensor lists on GPU
# before being consumed by data iterators (see megatron_utils.actor._get_rollout_data).
RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]


@dataclass
class MultimodalType:
    name: str  # Type identifier used in message content (e.g., "image")
    placeholder: str  # Placeholder token in conversation messages (e.g., "<image>")


class MultimodalTypes:
    IMAGE = MultimodalType(name="image", placeholder="<image>")
    VIDEO = MultimodalType(name="video", placeholder="<video>")
    AUDIO = MultimodalType(name="audio", placeholder="<audio>")

    @classmethod
    def all(cls) -> list[MultimodalType]:
        return [cls.IMAGE, cls.VIDEO, cls.AUDIO]

    @classmethod
    def get(cls, name: str) -> MultimodalType | None:
        return next((m for m in cls.all() if m.name == name), None)
