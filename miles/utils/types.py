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


@dataclass
class Sample:
    """The sample generated"""

    @dataclass
    class RetryMultimodalInputsSnapshot:
        inputs: dict[str, Any] | None

    group_index: int | None = None
    index: int | None = None
    # prompt
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    multimodal_inputs: dict[str, Any] = None  # raw multimodal data, e.g. images, videos, etc.
    multimodal_train_inputs: dict[str, Any] = None  # processed multimodal data, e.g. pixel_values, etc.
    # response
    response: str = ""
    response_length: int = 0
    retry_multimodal_inputs_snapshot: RetryMultimodalInputsSnapshot | None = field(
        default=None,
        kw_only=True,
        repr=False,
    )
    label: str | None = None
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    weight_versions: list[str] = field(default_factory=list)
    rollout_log_probs: list[float] | None = None  # Log probabilities from rollout engine
    rollout_routed_experts: numpy.ndarray | None = (
        None  # Routed experts from rollout engine. shape: (num_tokens-1, num_layers, moe_router_topk), dtype=int32
    )
    rollout_indexer_topk: numpy.ndarray | None = (
        None  # Indexer topk from rollout engine. shape: (num_tokens-1, num_indexer_layers, index_topk), dtype=int32
    )
    remove_sample: bool = False
    teacher_log_probs: list[float] | None = None  # Log probabilities from teacher model for OPD
    opd_reverse_kl: list[float] | None = None  # Precomputed per-token OPD reverse-KL estimate
    # Native per-position sparse targets from the teacher. All three tensors have shape
    # (response_length, top_k) and stay on CPU while the sample crosses Ray.
    teacher_topk_token_ids: torch.Tensor | None = None
    teacher_topk_log_probs: torch.Tensor | None = None
    teacher_topk_valid_mask: torch.Tensor | None = None

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
        value.pop("retry_multimodal_inputs_snapshot", None)
        value["status"] = self.status.value
        value["spec_info"] = self.spec_info.to_dict()
        value["prefix_cache_info"] = self.prefix_cache_info.to_dict()
        return value

    @staticmethod
    def from_dict(data: dict):
        data = dict(data)
        data.pop("retry_multimodal_inputs_snapshot", None)
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
        if self.teacher_log_probs is not None:
            assert (
                len(self.teacher_log_probs) == self.response_length
            ), f"teacher_log_probs length ({len(self.teacher_log_probs)}) != response_length ({self.response_length})"
        if self.opd_reverse_kl is not None:
            assert (
                len(self.opd_reverse_kl) == self.response_length
            ), f"opd_reverse_kl length ({len(self.opd_reverse_kl)}) != response_length ({self.response_length})"
        topk_presence = (
            self.teacher_topk_token_ids is not None,
            self.teacher_topk_log_probs is not None,
            self.teacher_topk_valid_mask is not None,
        )
        assert all(topk_presence) or not any(topk_presence), (
            "teacher_topk_token_ids, teacher_topk_log_probs, and teacher_topk_valid_mask "
            "must either all be present or all be None"
        )
        if all(topk_presence):
            assert isinstance(self.teacher_topk_token_ids, torch.Tensor), "teacher_topk_token_ids must be a tensor"
            assert isinstance(self.teacher_topk_log_probs, torch.Tensor), "teacher_topk_log_probs must be a tensor"
            assert isinstance(self.teacher_topk_valid_mask, torch.Tensor), "teacher_topk_valid_mask must be a tensor"
            assert (
                self.teacher_topk_token_ids.ndim == 2
            ), f"teacher_topk_token_ids must have shape [T, K], got {tuple(self.teacher_topk_token_ids.shape)}"
            assert (
                self.teacher_topk_log_probs.ndim == 2
            ), f"teacher_topk_log_probs must have shape [T, K], got {tuple(self.teacher_topk_log_probs.shape)}"
            assert (
                self.teacher_topk_valid_mask.ndim == 2
            ), f"teacher_topk_valid_mask must have shape [T, K], got {tuple(self.teacher_topk_valid_mask.shape)}"
            assert (
                self.teacher_topk_token_ids.shape
                == self.teacher_topk_log_probs.shape
                == self.teacher_topk_valid_mask.shape
            ), (
                "teacher top-k tensor shape mismatch: "
                f"ids={tuple(self.teacher_topk_token_ids.shape)}, "
                f"log_probs={tuple(self.teacher_topk_log_probs.shape)}, "
                f"valid_mask={tuple(self.teacher_topk_valid_mask.shape)}"
            )
            assert self.teacher_topk_token_ids.shape[0] == self.response_length, (
                f"teacher top-k response dimension ({self.teacher_topk_token_ids.shape[0]}) "
                f"!= response_length ({self.response_length})"
            )
            assert self.teacher_topk_token_ids.shape[1] > 0, "teacher top-k tensors must have K > 0"
            assert (
                self.teacher_topk_token_ids.dtype == torch.long
            ), f"teacher_topk_token_ids must use torch.long, got {self.teacher_topk_token_ids.dtype}"
            assert torch.is_floating_point(
                self.teacher_topk_log_probs
            ), f"teacher_topk_log_probs must be floating point, got {self.teacher_topk_log_probs.dtype}"
            assert (
                self.teacher_topk_valid_mask.dtype == torch.bool
            ), f"teacher_topk_valid_mask must use torch.bool, got {self.teacher_topk_valid_mask.dtype}"

            valid_log_probs = self.teacher_topk_log_probs[self.teacher_topk_valid_mask]
            invalid_log_probs = self.teacher_topk_log_probs[~self.teacher_topk_valid_mask]
            invalid_token_ids = self.teacher_topk_token_ids[~self.teacher_topk_valid_mask]
            assert torch.isfinite(valid_log_probs).all(), "valid teacher_topk_log_probs must be finite"
            assert torch.isneginf(
                invalid_log_probs
            ).all(), "invalid teacher_topk_log_probs must use -inf sentinel values"
            assert (invalid_token_ids == 0).all(), "invalid teacher_topk_token_ids must use 0 sentinel values"
        if self.rollout_routed_experts is not None:
            actual = len(self.rollout_routed_experts)
            expect = len(self.tokens) - 1
            assert actual == expect, f"rollout_routed_experts length ({actual}) != len(tokens) - 1 ({expect})"
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
        if self.teacher_log_probs is not None:
            self.teacher_log_probs = self.teacher_log_probs[:-n]
        if self.opd_reverse_kl is not None:
            self.opd_reverse_kl = self.opd_reverse_kl[:-n]
        if self.teacher_topk_token_ids is not None:
            self.teacher_topk_token_ids = self.teacher_topk_token_ids[:-n]
            self.teacher_topk_log_probs = self.teacher_topk_log_probs[:-n]
            self.teacher_topk_valid_mask = self.teacher_topk_valid_mask[:-n]
        if self.metadata and "opd_student_top_logprobs" in self.metadata:
            self.metadata["opd_student_top_logprobs"] = self.metadata["opd_student_top_logprobs"][:-n]
        if self.loss_mask is not None:
            self.loss_mask = self.loss_mask[:-n]
        self.response = tokenizer.decode(self.tokens[-self.response_length :]) if self.response_length > 0 else ""
        if self.rollout_routed_experts is not None:
            self.rollout_routed_experts = self.rollout_routed_experts[:-n]
        if self.rollout_indexer_topk is not None:
            self.rollout_indexer_topk = self.rollout_indexer_topk[:-n]

    @staticmethod
    def _copy_multimodal_input_containers(
        value: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return {key: list(item) if isinstance(item, list) else item for key, item in value.items()}

    def capture_multimodal_inputs_for_retry(self) -> None:
        if self.retry_multimodal_inputs_snapshot is None:
            self.retry_multimodal_inputs_snapshot = self.RetryMultimodalInputsSnapshot(
                self._copy_multimodal_input_containers(self.multimodal_inputs)
            )

    def reset_for_retry(self) -> None:
        """Reset generated outputs so the original prompt can be re-sampled.

        Keeps identity / prompt fields (group_index, index, prompt, label,
        multimodal_inputs, metadata, generate_function_path, routing_key) and
        restores everything else to dataclass defaults.

        Multi-turn rollouts mutate ``multimodal_inputs`` in place as the
        episode grows, so "keeping" it would carry the aborted attempt's
        observations into the retry. When such a rollout has called
        ``capture_multimodal_inputs_for_retry``, the pristine containers are
        restored from that snapshot instead.
        """
        if self.retry_multimodal_inputs_snapshot is not None:
            self.multimodal_inputs = self._copy_multimodal_input_containers(
                self.retry_multimodal_inputs_snapshot.inputs
            )
        self.tokens = []
        self.multimodal_train_inputs = None
        self.response = ""
        self.response_length = 0
        self.reward = None
        self.loss_mask = None
        self.weight_versions = []
        self.rollout_log_probs = None
        self.teacher_log_probs = None
        self.opd_reverse_kl = None
        self.rollout_routed_experts = None
        self.rollout_indexer_topk = None
        self.teacher_topk_token_ids = None
        self.teacher_topk_log_probs = None
        self.teacher_topk_valid_mask = None
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

        if "weight_version" in meta_info:
            self.weight_versions.append(meta_info["weight_version"])

        match meta_info["finish_reason"]["type"]:
            case "length":
                self.status = Sample.Status.TRUNCATED
            case "abort":
                self.status = Sample.Status.ABORTED
            case "stop":
                self.status = Sample.Status.COMPLETED


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
