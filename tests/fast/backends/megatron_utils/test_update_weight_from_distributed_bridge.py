"""Behavior tests for the bridge-mode path in UpdateWeightFromDistributed.

Covers the contracts a refactor must not regress on:
- Bridge mode rejects ``is_lora=True`` (the iterator filters LoRA out of
  base chunks, so silently no-syncing would be worse than failing).
- Bridge mode broadcasts only from the global source rank (DP=TP=PP=0).
- Bridge ``update_weights`` drains the iterator on every rank but only the
  source rank invokes the broadcast implementation, and ``convert_to_hf``
  (the legacy per-param dispatch with no VL support) is never reached.
- Raw mode still delegates to the mixin's bucketed gather pipeline.
- ``engine_gpu_counts`` flows from ``connect_rollout_engines`` into the
  NCCL init helper (otherwise heterogeneous engines get a wrong
  world_size/rank cursor).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")


from argparse import Namespace
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


_BC_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"
_MX_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin"


def _make_args(mode: str) -> Namespace:
    return Namespace(
        megatron_to_hf_mode=mode,
        hf_checkpoint="/fake/path",
        update_weight_buffer_size=1,
        pause_generation_mode="default",
        # the synced mixin asserts PP==1 for LoRA sync before building the config
        pipeline_model_parallel_size=1,
    )


def _make_parallel_state(pp_rank: int = 0, tp_rank: int = 0, dp_rank: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        pp=SimpleNamespace(rank=pp_rank, size=1, group=MagicMock()),
        tp=SimpleNamespace(rank=tp_rank, size=1, group=MagicMock()),
        intra_dp_cp=SimpleNamespace(rank=dp_rank, size=1, group=MagicMock()),
        ep=SimpleNamespace(rank=0, size=1, group=MagicMock()),
        etp=SimpleNamespace(rank=0, size=1, group=MagicMock()),
    )


@contextmanager
def _noop_timer(_name):  # matches miles.utils.timer.timer's context-manager form
    yield


class TestLoRASyncModeContract:
    """The 2026-08 sync flipped the LoRA weight-sync contract: upstream's
    multi-LoRA series implements LoRA sync ON the bridge path (via
    ``build_lora_sync_config``) and asserts bridge mode at construction, so
    raw+LoRA is now the rejected combination. Behavior of the sync itself is
    covered by upstream's ``test_lora_update_weight.py`` suite."""

    @patch("miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin.HfWeightIteratorBase")
    @patch(f"{_BC_MODULE}.HfWeightIteratorBase")
    @patch("miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin.build_lora_sync_config")
    def test_bridge_plus_lora_constructs_ok(self, mock_lora_cfg, mock_iter_base, mock_iter_base_mixin):
        from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
            UpdateWeightFromDistributed,
        )

        updater = UpdateWeightFromDistributed(
            args=_make_args("bridge"),
            model=[MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen3vlconfig",
            quantization_config=None,
            is_lora=True,
        )
        assert updater.is_lora is True
        assert updater._bridge_mode is True
        mock_lora_cfg.assert_called_once()
        assert mock_iter_base.create.called or mock_iter_base_mixin.create.called

    @patch(f"{_BC_MODULE}.HfWeightIteratorBase")
    def test_raw_plus_lora_rejected(self, mock_iter_base):
        from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
            UpdateWeightFromDistributed,
        )

        with pytest.raises(AssertionError, match="requires\\s+--megatron-to-hf-mode bridge"):
            UpdateWeightFromDistributed(
                args=_make_args("raw"),
                model=[MagicMock()],
                weights_getter=lambda: {},
                model_name="qwen3",
                quantization_config=None,
                is_lora=True,
            )
        mock_iter_base.create.assert_not_called()


class TestIsSourceGating:
    @patch(f"{_BC_MODULE}.get_parallel_state")
    def test_bridge_source_requires_pp_zero(self, mock_get_parallel_state):
        from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
            UpdateWeightFromDistributed,
        )

        args = _make_args("bridge")
        with patch(f"{_BC_MODULE}.HfWeightIteratorBase"):
            updater = UpdateWeightFromDistributed(
                args=args,
                model=[MagicMock()],
                weights_getter=lambda: {},
                model_name="qwen3vlconfig",
                quantization_config=None,
            )

        mock_get_parallel_state.return_value = _make_parallel_state(pp_rank=0, tp_rank=0, dp_rank=0)
        assert updater._is_source is True

        mock_get_parallel_state.return_value = _make_parallel_state(pp_rank=1, tp_rank=0, dp_rank=0)
        assert updater._is_source is False

    @patch(f"{_BC_MODULE}.get_parallel_state")
    def test_raw_source_allows_any_pp_rank(self, mock_get_parallel_state):
        from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
            UpdateWeightFromDistributed,
        )

        args = _make_args("raw")
        updater = UpdateWeightFromDistributed(
            args=args,
            model=[MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen3",
            quantization_config=None,
        )

        for pp_rank in (0, 1, 2):
            mock_get_parallel_state.return_value = _make_parallel_state(pp_rank=pp_rank)
            assert updater._is_source is True, f"raw should not gate on pp_rank={pp_rank}"


class TestConnectRolloutEnginesEngineGpuCounts:
    """``engine_gpu_counts`` must reach ``connect_rollout_engines_from_distributed``;
    otherwise heterogeneous engines (e.g. prefill TP=2 + decode TP=4) get a
    homogeneous fallback and the NCCL world_size/rank_cursor is wrong."""

    @patch(f"{_BC_MODULE}.disconnect_rollout_engines_from_distributed")
    @patch(f"{_BC_MODULE}.connect_rollout_engines_from_distributed")
    @patch(f"{_BC_MODULE}.get_parallel_state")
    def test_bridge_passes_engine_gpu_counts_through(self, mock_get_parallel_state, mock_connect, mock_disconnect):
        from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
            UpdateWeightFromDistributed,
        )

        args = _make_args("bridge")
        with patch(f"{_BC_MODULE}.HfWeightIteratorBase"):
            updater = UpdateWeightFromDistributed(
                args=args,
                model=[MagicMock()],
                weights_getter=lambda: {},
                model_name="qwen3vlconfig",
                quantization_config=None,
            )

        mock_get_parallel_state.return_value = _make_parallel_state(pp_rank=0)
        mock_connect.return_value = MagicMock(name="model_update_groups")

        engine_gpu_counts = [2, 4]  # heterogeneous: prefill TP=2, decode TP=4
        updater.connect_rollout_engines(
            rollout_engines=[MagicMock(), MagicMock()],
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=engine_gpu_counts,
        )

        mock_connect.assert_called_once()
        assert mock_connect.call_args.kwargs.get("engine_gpu_counts") == engine_gpu_counts

    @patch(f"{_BC_MODULE}.disconnect_rollout_engines_from_distributed")
    @patch(f"{_BC_MODULE}.connect_rollout_engines_from_distributed")
    @patch(f"{_BC_MODULE}.get_parallel_state")
    def test_none_engine_gpu_counts_passes_none(self, mock_get_parallel_state, mock_connect, mock_disconnect):
        """When the caller doesn't know per-engine counts, ``None`` must
        propagate so the helper applies its own ``rollout_num_gpus_per_engine``
        fallback (rather than silently substituting something else)."""
        from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
            UpdateWeightFromDistributed,
        )

        args = _make_args("bridge")
        with patch(f"{_BC_MODULE}.HfWeightIteratorBase"):
            updater = UpdateWeightFromDistributed(
                args=args,
                model=[MagicMock()],
                weights_getter=lambda: {},
                model_name="qwen3vlconfig",
                quantization_config=None,
            )

        mock_get_parallel_state.return_value = _make_parallel_state(pp_rank=0)
        mock_connect.return_value = MagicMock(name="model_update_groups")

        updater.connect_rollout_engines(
            rollout_engines=[MagicMock()],
            rollout_engine_lock=MagicMock(),
        )
        assert mock_connect.call_args.kwargs.get("engine_gpu_counts") is None


class TestBridgeUpdateWeightsDoesNotCallConvertToHf:
    """End-to-end mock of bridge-mode ``update_weights``: assert the bridge
    iterator is drained and ``convert_to_hf`` is never invoked."""

    @patch(f"{_MX_MODULE}.convert_to_hf")
    @patch(f"{_BC_MODULE}.timer", new=_noop_timer)
    @patch(f"{_BC_MODULE}.dist")
    @patch(f"{_BC_MODULE}.get_gloo_group")
    @patch(f"{_BC_MODULE}.get_parallel_state")
    @patch(f"{_BC_MODULE}.HfWeightIteratorBase")
    def test_bridge_path_skips_convert_to_hf(
        self,
        mock_iter_base,
        mock_get_parallel_state,
        mock_get_gloo_group,
        mock_dist,
        mock_convert_to_hf,
    ):
        from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
            UpdateWeightFromDistributed,
        )

        iterator = MagicMock(name="hf_weight_iterator")
        chunk_a = [("model.embed_tokens.weight", torch.zeros(2))]
        chunk_b = [("vision_model.patch_embed.proj.weight", torch.zeros(2))]
        iterator.get_hf_weight_chunks.return_value = iter([chunk_a, chunk_b])
        mock_iter_base.create.return_value = iterator

        mock_get_parallel_state.return_value = _make_parallel_state(pp_rank=0, tp_rank=0, dp_rank=0)
        mock_get_gloo_group.return_value = MagicMock()
        mock_dist.barrier.return_value = None

        args = _make_args("bridge")
        updater = UpdateWeightFromDistributed(
            args=args,
            model=[MagicMock()],
            weights_getter=lambda: {"actor.weight": torch.zeros(1)},
            model_name="qwen3vlconfig",
            quantization_config=None,
        )

        updater._pause_and_prepare_engines = MagicMock(name="pause")
        updater._finalize_and_resume_engines = MagicMock(name="finalize")
        broadcast_calls = []
        updater._update_weight_implementation = MagicMock(
            side_effect=lambda chunk, pbar=None: broadcast_calls.append(list(chunk))
        )
        updater._group_name = "miles-bridge"

        updater.update_weights()

        iterator.get_hf_weight_chunks.assert_called_once()
        kwargs = iterator.get_hf_weight_chunks.call_args.kwargs
        assert kwargs.get("weight_type") == "base"

        # All produced chunks were forwarded to the broadcast implementation,
        # and the legacy text-only converter was never touched.
        assert [names for names in broadcast_calls] == [chunk_a, chunk_b]
        mock_convert_to_hf.assert_not_called()
        updater._pause_and_prepare_engines.assert_called_once()
        updater._finalize_and_resume_engines.assert_called_once()

    @patch(f"{_MX_MODULE}.convert_to_hf")
    @patch(f"{_BC_MODULE}.timer", new=_noop_timer)
    @patch(f"{_BC_MODULE}.dist")
    @patch(f"{_BC_MODULE}.get_gloo_group")
    @patch(f"{_BC_MODULE}.get_parallel_state")
    @patch(f"{_BC_MODULE}.HfWeightIteratorBase")
    def test_bridge_non_source_rank_does_not_broadcast(
        self,
        mock_iter_base,
        mock_get_parallel_state,
        mock_get_gloo_group,
        mock_dist,
        mock_convert_to_hf,
    ):
        """Non-source ranks must still drain the iterator (so bridge collectives
        complete) but must not invoke ``_update_weight_implementation``."""
        from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
            UpdateWeightFromDistributed,
        )

        iterator = MagicMock(name="hf_weight_iterator")
        chunk = [("model.embed_tokens.weight", torch.zeros(2))]
        iterator.get_hf_weight_chunks.return_value = iter([chunk])
        mock_iter_base.create.return_value = iterator

        # Non-source: pp_rank != 0 in bridge mode disqualifies a rank.
        mock_get_parallel_state.return_value = _make_parallel_state(pp_rank=1)
        mock_get_gloo_group.return_value = MagicMock()

        args = _make_args("bridge")
        updater = UpdateWeightFromDistributed(
            args=args,
            model=[MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen3vlconfig",
            quantization_config=None,
        )
        updater._pause_and_prepare_engines = MagicMock()
        updater._finalize_and_resume_engines = MagicMock()
        updater._update_weight_implementation = MagicMock()

        updater.update_weights()

        iterator.get_hf_weight_chunks.assert_called_once()
        updater._update_weight_implementation.assert_not_called()
        mock_convert_to_hf.assert_not_called()


class TestRawModeStillUsesLegacyPath:
    """Raw mode must continue to call the mixin's bucketed gather + ``convert_to_hf``
    pipeline and must NOT instantiate the bridge weight iterator."""

    @patch(f"{_BC_MODULE}.HfWeightIteratorBase")
    def test_raw_mode_update_weights_delegates_to_mixin(self, mock_iter_base):
        from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
            UpdateWeightFromDistributed,
        )
        from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin import (
            DistBucketedWeightUpdateMixin,
        )

        args = _make_args("raw")
        updater = UpdateWeightFromDistributed(
            args=args,
            model=[MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen3",
            quantization_config=None,
        )

        mock_iter_base.create.assert_not_called()

        with patch.object(DistBucketedWeightUpdateMixin, "update_weights") as mock_super:
            updater.update_weights()
            mock_super.assert_called_once()
