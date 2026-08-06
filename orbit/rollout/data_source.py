import abc
import copy
import logging
import os
import re
from pathlib import Path

import torch

from orbit.utils.data import Dataset
from orbit.utils.misc import load_function
from orbit.utils.processing_utils import load_processor, load_tokenizer
from orbit.utils.types import Sample

logger = logging.getLogger(__name__)


_ITERATION_DIRECTORY_RE = re.compile(r"iter_([0-9]+)")
_MAX_ROLLOUT_COUNTER = 2**63 - 1


def _canonical_orbit_peft_iteration(adapter_path: Path) -> int | None:
    """Return the iteration encoded by an Orbit-owned PEFT adapter path.

    Orbit writes adapters below ``<actor_root>/iter_%07d/adapter``.  Keep this
    check lexical and exact: resolving symlinks or searching parent directories
    could associate an arbitrary weights-only adapter with unrelated rollout
    state.
    """
    if adapter_path.name != "adapter":
        return None

    match = _ITERATION_DIRECTORY_RE.fullmatch(adapter_path.parent.name)
    if match is None:
        return None

    iteration_text = match.group(1)
    if len(iteration_text) > 19:
        raise ValueError(f"PEFT adapter checkpoint iteration is out of range: {adapter_path}")
    iteration = int(iteration_text)
    if iteration > _MAX_ROLLOUT_COUNTER:
        raise ValueError(f"PEFT adapter checkpoint iteration is out of range: {adapter_path}")
    if adapter_path.parent.name != f"iter_{iteration:07d}":
        return None
    return iteration


def _resolve_rollout_dataset_state_location(args, rollout_id: int | None) -> tuple[Path | None, bool]:
    """Select the checkpoint root and whether its rollout state is required.

    ``args.load`` remains the source for full checkpoints and ordinary
    weights-only warm starts.  A resumed Orbit PEFT checkpoint is different:
    its adapter lives below a per-iteration directory while rollout state lives
    at the actor checkpoint root.  Only derive that root when the canonical
    layout binds it to the exact requested rollout id.
    """
    load_path = getattr(args, "load", None)
    default_root = Path(load_path) if load_path is not None else None

    # Rollout -1 is requested for a fresh start.  In particular, a canonical-
    # looking but weights-only adapter must remain a warm start rather than
    # being mistaken for a training resume.
    if rollout_id is None:
        return default_root, False
    if type(rollout_id) is not int or not -1 <= rollout_id <= _MAX_ROLLOUT_COUNTER:
        raise ValueError(f"invalid rollout dataset checkpoint id: {rollout_id!r}")
    if rollout_id < 0:
        return default_root, False

    adapter_path_value = (
        getattr(args, "peft_adapter_path", None)
        or getattr(args, "lora_adapter_path", None)
        or getattr(args, "oft_adapter_path", None)
    )
    if adapter_path_value is None:
        return default_root, False

    adapter_path = Path(adapter_path_value)
    adapter_iteration = _canonical_orbit_peft_iteration(adapter_path)
    if adapter_iteration is None:
        # Arbitrary HF/weights-only adapter exports have no reliable association
        # with an Orbit actor root.  Preserve the historical args.load behavior.
        return default_root, False
    if adapter_iteration != rollout_id:
        raise ValueError(
            "PEFT adapter checkpoint iteration does not match requested rollout dataset state: "
            f"adapter iteration {adapter_iteration}, rollout id {rollout_id}"
        )
    return adapter_path.parent.parent, True


def _validate_rollout_dataset_state(state_dict, *, dataset_size: int) -> dict:
    if type(state_dict) is not dict:
        raise RuntimeError("rollout dataset checkpoint must contain a dictionary")

    validated = {}
    for name in ("sample_offset", "epoch_id", "sample_group_index", "sample_index"):
        value = state_dict.get(name, 0)
        if type(value) is not int or not 0 <= value <= _MAX_ROLLOUT_COUNTER:
            raise RuntimeError(f"rollout dataset checkpoint has invalid {name}: {value!r}")
        validated[name] = value

    if validated["sample_offset"] > dataset_size:
        raise RuntimeError(
            "rollout dataset checkpoint sample_offset exceeds the current dataset size: "
            f"{validated['sample_offset']} > {dataset_size}"
        )

    metadata = state_dict.get("metadata", {})
    if type(metadata) is not dict:
        raise RuntimeError("rollout dataset checkpoint metadata must be a dictionary")
    validated["metadata"] = metadata
    return validated


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """


# Follow-up may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # Follow-up remove this
        self.metadata = {}

        if args.rollout_global_dataset:
            tokenizer = load_tokenizer(
                args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
            )
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # Follow-up move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    def get_samples(self, num_samples):
        # Follow-up further improve code
        if self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        state_root, state_required = _resolve_rollout_dataset_state_location(self.args, rollout_id)
        if state_root is None:
            return

        path = state_root / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
        if not path.exists():
            if state_required:
                raise FileNotFoundError(f"required PEFT rollout dataset checkpoint does not exist: {path}")
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load metadata from {path}")
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        state = _validate_rollout_dataset_state(state_dict, dataset_size=len(self.dataset))

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(state["epoch_id"])

        self.sample_offset = state["sample_offset"]
        self.epoch_id = state["epoch_id"]
        self.sample_group_index = state["sample_group_index"]
        self.sample_index = state["sample_index"]
        self.metadata = state["metadata"]
        logger.info(f"load metadata: {self.metadata}")


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # Follow-up remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # Follow-up remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
