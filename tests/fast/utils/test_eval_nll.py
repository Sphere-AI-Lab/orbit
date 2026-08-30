"""Pure logic behind the held-out NLL eval hook.

Everything exercised here runs on CPU with no megatron and no GPU: the
token-weighted reduction, the DP shard plan, and the micro-batch schedule that
guarantees every row of the held-out file is scored exactly once.

The pieces that are NOT covered here (and cannot be, without a GPU) are the
actual forward pass, the ``dist.all_reduce`` over the DP group, and the
alignment between megatron's returned log-probs and the loss masks. Those are
gate G4's job.
"""

import json
import math

import pytest
import torch

from miles.orbit.utils.eval_nll import (
    NllStats,
    accumulate_nll,
    build_eval_nll_metrics,
    build_eval_nll_rows,
    is_eval_nll_reporting_rank,
    load_eval_nll_rows,
    plan_eval_nll_microbatches,
    plan_eval_nll_shards,
    reduce_nll,
    select_eval_nll_result,
)


# --------------------------------------------------------------------------
# reduce_nll: token-weighted mean negative log-likelihood
# --------------------------------------------------------------------------


def test_single_sample_mean_of_negatives():
    lp = [torch.tensor([-1.0, -2.0, -3.0])]
    assert reduce_nll(lp, [3]) == pytest.approx(2.0)


def test_token_weighted_not_sample_weighted():
    # Sample A: 1 token at -10. Sample B: 9 tokens at 0.
    # Token-weighted -> 10/10 = 1.0. Sample-weighted would be (10 + 0)/2 = 5.0.
    lp = [torch.tensor([-10.0]), torch.zeros(9)]
    assert reduce_nll(lp, [1, 9]) == pytest.approx(1.0)


def test_matches_naive_concatenation():
    lp = [torch.tensor([-0.5, -1.5]), torch.tensor([-2.5])]
    expected = -(torch.cat(lp).sum().item()) / 3
    assert reduce_nll(lp, [2, 1]) == pytest.approx(expected)


def test_empty_input_returns_nan():
    assert math.isnan(reduce_nll([], []))


def test_zero_total_length_returns_nan():
    assert math.isnan(reduce_nll([torch.tensor([])], [0]))


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        reduce_nll([torch.tensor([-1.0, -2.0])], [3])


def test_count_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        reduce_nll([torch.tensor([-1.0])], [1, 1])


# --------------------------------------------------------------------------
# The loss mask is load-bearing: multi-turn rows have unscored tokens INSIDE
# the response span, and HF's Trainer ignores them (label == -100). Averaging
# over the whole response span instead would be a different number.
# --------------------------------------------------------------------------


def test_loss_mask_excludes_unscored_tokens_from_both_numerator_and_denominator():
    # 4 response tokens; the middle two are an interleaved user turn.
    lp = [torch.tensor([-1.0, -100.0, -100.0, -3.0])]
    masks = [torch.tensor([1, 0, 0, 1])]
    assert reduce_nll(lp, [4], masks) == pytest.approx(2.0)


def test_loss_mask_changes_the_answer_versus_unmasked():
    lp = [torch.tensor([-1.0, -9.0])]
    masks = [torch.tensor([1, 0])]
    assert reduce_nll(lp, [2]) == pytest.approx(5.0)
    assert reduce_nll(lp, [2], masks) == pytest.approx(1.0)


def test_all_zero_mask_contributes_nothing():
    lp = [torch.tensor([-1.0, -2.0]), torch.tensor([-4.0])]
    masks = [torch.tensor([0, 0]), torch.tensor([1])]
    assert reduce_nll(lp, [2, 1], masks) == pytest.approx(4.0)


def test_loss_mask_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        reduce_nll([torch.tensor([-1.0, -2.0])], [2], [torch.tensor([1])])


# --------------------------------------------------------------------------
# accumulate_nll returns accumulators, not a pre-divided float, so they can be
# summed across DP ranks before the single final division.
# --------------------------------------------------------------------------


def test_accumulators_are_additive_across_shards():
    """The whole point of returning (sum, n_tokens): a token-weighted mean is
    NOT the mean of per-shard token-weighted means when shards hold different
    token counts."""
    shard_a = accumulate_nll([torch.tensor([-10.0])], [torch.tensor([1])])
    shard_b = accumulate_nll([torch.zeros(9)], [torch.ones(9, dtype=torch.int)])

    combined = shard_a + shard_b
    assert combined.num_tokens == 10
    assert combined.num_samples == 2
    assert combined.mean_nll == pytest.approx(1.0)

    naive_mean_of_means = (shard_a.mean_nll + shard_b.mean_nll) / 2
    assert naive_mean_of_means == pytest.approx(5.0)
    assert combined.mean_nll != pytest.approx(naive_mean_of_means)


def test_zero_stats_is_additive_identity():
    stats = accumulate_nll([torch.tensor([-2.0, -4.0])], [torch.ones(2, dtype=torch.int)])
    assert (NllStats.zero() + stats) == stats
    assert (stats + NllStats.zero()) == stats


def test_stats_roundtrip_through_flat_values():
    stats = accumulate_nll(
        [torch.tensor([-1.0, -3.0]), torch.tensor([-5.0])],
        [torch.tensor([1, 1]), torch.tensor([1])],
    )
    assert NllStats.from_values(stats.to_values()) == stats


def test_padding_rows_are_dropped_entirely():
    lp = [torch.tensor([-2.0]), torch.tensor([-2.0])]
    masks = [torch.tensor([1]), torch.tensor([1])]
    stats = accumulate_nll(lp, masks, is_padding=[False, True])
    assert stats.num_samples == 1
    assert stats.num_tokens == 1
    assert stats.mean_nll == pytest.approx(2.0)


def test_sample_mean_is_reported_alongside_token_mean():
    stats = accumulate_nll(
        [torch.tensor([-10.0]), torch.zeros(9)],
        [torch.tensor([1]), torch.ones(9, dtype=torch.int)],
    )
    assert stats.mean_nll == pytest.approx(1.0)
    assert stats.sample_mean_nll == pytest.approx(5.0)


def test_empty_stats_report_nan():
    stats = NllStats.zero()
    assert math.isnan(stats.mean_nll)
    assert math.isnan(stats.sample_mean_nll)


def test_accumulation_is_float64():
    """Summing float32 log-probs in float32 loses digits the study cannot
    spare -- the whole target table spans 0.009 nats. The accumulator must
    upcast BEFORE summing, not after."""
    n = 200_000
    generator = torch.Generator().manual_seed(0)
    values = -torch.rand(n, generator=generator, dtype=torch.float32)
    masks = [torch.ones(n, dtype=torch.int)]

    exact = float(-values.to(torch.float64).sum()) / n
    naive_float32 = float(-values.sum()) / n
    assert naive_float32 != pytest.approx(exact, rel=1e-12), (
        "test is not discriminating: float32 summation happened to be exact here"
    )

    stats = accumulate_nll([values], masks)
    assert stats.mean_nll == pytest.approx(exact, rel=1e-12)


# --------------------------------------------------------------------------
# Coverage: every row must be scored. This is the defect the study cannot
# absorb -- get_data_iterator's floor division silently drops the remainder.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_rows", [1, 2, 7, 31, 32, 33, 100, 101])
@pytest.mark.parametrize("micro_batch_size", [1, 3, 32])
def test_microbatch_schedule_covers_every_row_exactly_once(num_rows, micro_batch_size):
    schedule = plan_eval_nll_microbatches(num_rows, micro_batch_size)
    flat = [i for mb in schedule for i in mb]
    assert flat == list(range(num_rows)), "schedule must cover every row, in order, exactly once"
    assert all(mb for mb in schedule), "no empty micro-batch (get_batch cannot concatenate zero samples)"
    assert all(len(mb) <= micro_batch_size for mb in schedule)


def test_microbatch_schedule_keeps_the_short_final_group():
    """100 rows at batch 32 is the plan's actual SFT configuration. Megatron's
    get_data_iterator would floor-divide to 3 steps of 32 and silently drop 4
    rows; the eval schedule must keep them."""
    schedule = plan_eval_nll_microbatches(100, 32)
    assert [len(mb) for mb in schedule] == [32, 32, 32, 4]
    assert sum(len(mb) for mb in schedule) == 100


def test_microbatch_schedule_rejects_zero_rows():
    with pytest.raises(ValueError):
        plan_eval_nll_microbatches(0, 8)


def test_microbatch_schedule_rejects_nonpositive_batch():
    with pytest.raises(ValueError):
        plan_eval_nll_microbatches(10, 0)


# --------------------------------------------------------------------------
# DP sharding: every row lands on exactly one rank, shards are equal length so
# every rank runs an identical micro-batch schedule (the pipeline schedule is
# collective; a rank-dependent micro-batch count would hang).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_rows", [1, 5, 100, 101])
@pytest.mark.parametrize("dp_size", [1, 2, 3, 8])
def test_shards_cover_every_row_exactly_once(num_rows, dp_size):
    shards = plan_eval_nll_shards(num_rows, dp_size)
    assert len(shards) == dp_size

    real = sorted(idx for shard in shards for idx, is_pad in shard if not is_pad)
    assert real == list(range(num_rows)), "every row scored exactly once across DP ranks"


@pytest.mark.parametrize("num_rows", [1, 5, 100, 101])
@pytest.mark.parametrize("dp_size", [1, 2, 3, 8])
def test_shards_are_all_the_same_length(num_rows, dp_size):
    shards = plan_eval_nll_shards(num_rows, dp_size)
    sizes = {len(shard) for shard in shards}
    assert len(sizes) == 1, f"unequal shards {sizes} would give DP ranks different micro-batch counts"


def test_shards_are_identity_when_dp_is_one():
    shards = plan_eval_nll_shards(100, 1)
    assert shards == [[(i, False) for i in range(100)]]


def test_padding_uses_the_requested_row_and_is_flagged():
    shards = plan_eval_nll_shards(5, 2, pad_index=3)
    padded = [(idx, is_pad) for shard in shards for idx, is_pad in shard if is_pad]
    assert padded == [(3, True)], "one padding row, taken from the requested index, flagged as padding"


def test_shards_reject_empty_input():
    with pytest.raises(ValueError):
        plan_eval_nll_shards(0, 1)


def test_end_to_end_coverage_across_dp_and_microbatches():
    """The composition is what matters: shard, then schedule, then reduce.
    101 rows over 3 DP ranks at micro-batch 8 is deliberately coprime with
    everything."""
    num_rows, dp_size, mbs = 101, 3, 8
    shards = plan_eval_nll_shards(num_rows, dp_size)

    schedules = [plan_eval_nll_microbatches(len(shard), mbs) for shard in shards]
    assert len({len(s) for s in schedules}) == 1, "all DP ranks must run the same number of micro-batches"

    total = NllStats.zero()
    for shard, schedule in zip(shards, schedules, strict=True):
        visited = [i for mb in schedule for i in mb]
        assert visited == list(range(len(shard)))
        log_probs = [torch.tensor([-float(shard[i][0] + 1)]) for i in visited]
        masks = [torch.ones(1, dtype=torch.int) for _ in visited]
        total = total + accumulate_nll(log_probs, masks, is_padding=[shard[i][1] for i in visited])

    assert total.num_samples == num_rows, "scored-sample count must equal the number of rows read"
    assert total.num_tokens == num_rows
    assert total.mean_nll == pytest.approx(sum(range(1, num_rows + 1)) / num_rows)


# --------------------------------------------------------------------------
# Row loading and tokenization, with a stub mask generator (no tokenizer, so
# this runs anywhere).
# --------------------------------------------------------------------------


class _StubMaskGenerator:
    """Scores the second half of every conversation, deterministically."""

    def __init__(self):
        self.calls = []

    def get_loss_mask(self, messages, tools=None):
        self.calls.append((messages, tools))
        n = 2 * len(messages)
        token_ids = list(range(100, 100 + n))
        loss_mask = [0] * (n // 2) + [1] * (n - n // 2)
        return token_ids, loss_mask

    def get_response_lengths(self, loss_masks):
        return [len(m[m.index(1) :]) if 1 in m else 0 for m in loss_masks]


def _write_jsonl(tmp_path, rows, name="eval.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_load_rows_reads_every_line(tmp_path):
    rows = [{"prompt": [{"role": "user", "content": f"q{i}"}]} for i in range(7)]
    path = _write_jsonl(tmp_path, rows)
    loaded = load_eval_nll_rows(path)
    assert len(loaded) == 7
    assert loaded[3].messages == rows[3]["prompt"]


def test_load_rows_skips_blank_lines_but_keeps_the_count_honest(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"prompt": [{"role": "user", "content": "a"}]}\n'
        "\n"
        "   \n"
        '{"prompt": [{"role": "user", "content": "b"}]}\n',
        encoding="utf-8",
    )
    assert len(load_eval_nll_rows(path)) == 2


def test_load_rows_accepts_messages_key(tmp_path):
    path = _write_jsonl(tmp_path, [{"messages": [{"role": "user", "content": "a"}]}])
    assert load_eval_nll_rows(path)[0].messages == [{"role": "user", "content": "a"}]


def test_load_rows_honours_an_explicit_input_key(tmp_path):
    path = _write_jsonl(tmp_path, [{"conversation": [{"role": "user", "content": "a"}]}])
    assert load_eval_nll_rows(path, input_key="conversation")[0].messages[0]["content"] == "a"


def test_load_rows_rejects_a_file_with_no_recognisable_key(tmp_path):
    path = _write_jsonl(tmp_path, [{"text": "hello"}])
    with pytest.raises(ValueError, match="no conversation key"):
        load_eval_nll_rows(path)


def test_load_rows_rejects_an_empty_file(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        load_eval_nll_rows(path)


def test_load_rows_picks_up_tools(tmp_path):
    tools = [{"name": "calc"}]
    path = _write_jsonl(tmp_path, [{"prompt": [{"role": "user", "content": "a"}], "tools": tools}])
    assert load_eval_nll_rows(path)[0].tools == tools


def test_build_batch_shapes_align_with_what_the_forward_pass_expects(tmp_path):
    rows = [{"prompt": [{"role": "user", "content": "a"}] * (i + 1)} for i in range(4)]
    path = _write_jsonl(tmp_path, rows)
    gen = _StubMaskGenerator()

    batch = build_eval_nll_rows(load_eval_nll_rows(path), gen)

    n = len(batch["total_lengths"])
    assert n == 4, "one entry per row of the held-out file"
    for tokens, mask, total_length, response_length in zip(
        batch["tokens"], batch["loss_masks"], batch["total_lengths"], batch["response_lengths"], strict=True
    ):
        assert len(tokens) == total_length
        assert len(mask) == response_length, "loss mask is the response-aligned suffix, matching log_probs"
        assert response_length <= total_length


def test_build_batch_matches_the_sft_rollout_masking_contract(tmp_path):
    """sft_rollout stores loss_mask[-response_length:]; eval must store the
    identical suffix or the reduction silently misaligns."""
    path = _write_jsonl(tmp_path, [{"prompt": [{"role": "user", "content": "a"}] * 3}])
    gen = _StubMaskGenerator()

    batch = build_eval_nll_rows(load_eval_nll_rows(path), gen)

    token_ids, loss_mask = _StubMaskGenerator().get_loss_mask(
        [{"role": "user", "content": "a"}] * 3
    )
    response_length = _StubMaskGenerator().get_response_lengths([loss_mask])[0]
    assert batch["tokens"][0] == token_ids
    assert batch["loss_masks"][0] == loss_mask[-response_length:]
    assert batch["response_lengths"][0] == response_length
    assert batch["total_lengths"][0] == len(token_ids)


def test_build_batch_forwards_tools_to_the_mask_generator(tmp_path):
    tools = [{"name": "calc"}]
    path = _write_jsonl(tmp_path, [{"prompt": [{"role": "user", "content": "a"}], "tools": tools}])
    gen = _StubMaskGenerator()
    build_eval_nll_rows(load_eval_nll_rows(path), gen)
    assert gen.calls[0][1] == tools


def test_build_batch_rejects_a_row_with_nothing_to_score(tmp_path):
    class _NoScore(_StubMaskGenerator):
        def get_loss_mask(self, messages, tools=None):
            return [1, 2, 3], [0, 0, 0]

    path = _write_jsonl(tmp_path, [{"prompt": [{"role": "user", "content": "a"}]}])
    with pytest.raises(ValueError, match="no scored tokens"):
        build_eval_nll_rows(load_eval_nll_rows(path), _NoScore())


def test_shortest_row_index_is_reported_for_cheap_padding(tmp_path):
    rows = [{"prompt": [{"role": "user", "content": "a"}] * n} for n in (5, 1, 3)]
    path = _write_jsonl(tmp_path, rows)
    batch = build_eval_nll_rows(load_eval_nll_rows(path), _StubMaskGenerator())
    assert batch["shortest_row_index"] == 1


# --------------------------------------------------------------------------
# CLI registration. Confirms the flags are reachable from the main parser,
# not just present in the source file.
# --------------------------------------------------------------------------


def _parse(extra_argv):
    import argparse
    import sys
    from unittest.mock import patch

    from miles.utils.arguments import get_orbit_extra_args_provider

    required = ["--rollout-batch-size", "64"]
    with patch.object(sys, "argv", ["test"] + required):
        parser = argparse.ArgumentParser()
        get_orbit_extra_args_provider()(parser)
    args, _ = parser.parse_known_args(required + extra_argv)
    return args


def test_eval_nll_flags_default_to_disabled():
    args = _parse([])
    assert args.eval_nll_data is None
    assert args.eval_nll_interval == 0
    assert args.eval_nll_micro_batch_size is None


def test_eval_nll_flags_parse():
    args = _parse(
        ["--eval-nll-data", "/tmp/x.jsonl", "--eval-nll-interval", "5", "--eval-nll-micro-batch-size", "8"]
    )
    assert args.eval_nll_data == "/tmp/x.jsonl"
    assert args.eval_nll_interval == 5
    assert args.eval_nll_micro_batch_size == 8


def test_dp_reduction_equals_the_single_rank_answer():
    """The claim the whole design rests on: shard the held-out set across DP
    ranks, sum the accumulators (as the all_reduce does), divide once -- and get
    exactly the number a single rank would have computed over all rows.

    Uses variable-length rows with interleaved unscored tokens, and a row count
    coprime with the DP size so padding is exercised.
    """
    generator = torch.Generator().manual_seed(7)
    num_rows, dp_size, mbs = 37, 4, 5

    log_probs, masks = [], []
    for i in range(num_rows):
        length = 1 + (i * 7) % 23
        log_probs.append(-torch.rand(length, generator=generator))
        mask = torch.ones(length, dtype=torch.int)
        mask[::3] = 0  # unscored tokens inside the response span
        mask[-1] = 1  # every row must score something
        masks.append(mask)

    ground_truth = accumulate_nll(log_probs, masks)

    shards = plan_eval_nll_shards(num_rows, dp_size, pad_index=0)
    assert any(padded for shard in shards for _, padded in shard), "padding not exercised"

    total = NllStats.zero()
    for shard in shards:
        schedule = plan_eval_nll_microbatches(len(shard), mbs)
        visited = [i for mb in schedule for i in mb]
        rank_stats = accumulate_nll(
            [log_probs[shard[i][0]] for i in visited],
            [masks[shard[i][0]] for i in visited],
            is_padding=[shard[i][1] for i in visited],
        )
        # Round-trip through the flat float vector the all_reduce carries.
        total = total + NllStats.from_values(rank_stats.to_values())

    assert total.num_samples == ground_truth.num_samples == num_rows
    assert total.num_tokens == ground_truth.num_tokens
    assert total.mean_nll == pytest.approx(ground_truth.mean_nll, rel=1e-12)
    assert total.sample_mean_nll == pytest.approx(ground_truth.sample_mean_nll, rel=1e-12)


def test_load_rows_falls_back_when_the_requested_key_is_absent(caplog):
    """The actor passes the TRAINING data's --input-key (megatron's default is
    "input"), which a held-out file keyed on "prompt" will not have. Fall back
    rather than fail, but say so."""
    import logging
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eval.jsonl"
        path.write_text('{"prompt": [{"role": "user", "content": "a"}]}\n', encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            rows = load_eval_nll_rows(path, input_key="input")
    assert rows[0].messages[0]["content"] == "a"
    assert "falling back" in caplog.text


def test_load_rows_still_rejects_when_no_key_matches_at_all(tmp_path):
    path = _write_jsonl(tmp_path, [{"text": "hello"}])
    with pytest.raises(ValueError, match="no conversation key"):
        load_eval_nll_rows(path, input_key="input")


# --------------------------------------------------------------------------
# Rank dedup. This is the second of the two hazards the controller notes
# flagged: _broadcast returns one value per actor across the whole TP x PP x DP
# grid, TP/PP replicas hold the SAME samples, and averaging or summing them
# would over-count by tp_size * pp_size.
# --------------------------------------------------------------------------


def _parallel_state(*, tp_rank=0, tp_size=1, cp_rank=0, cp_size=1, dp_rank=0, dp_size=1, is_pp_last_stage=True):
    """A real ParallelState with no process groups.

    GroupInfo.__post_init__ short-circuits when group is None, so the real
    dataclass can be built on CPU with no torch.distributed initialisation.
    """
    from miles.backends.training_utils.parallel import GroupInfo, ParallelState

    return ParallelState(
        intra_dp=GroupInfo(rank=dp_rank, size=dp_size, group=None),
        intra_dp_cp=GroupInfo(rank=dp_rank, size=dp_size, group=None),
        cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        tp=GroupInfo(rank=tp_rank, size=tp_size, group=None),
        # upstream's ParallelState gained required pp/ep/etp/indep_dp groups. eval_nll
        # only reads is_pp_last_stage off the PP axis, so trivial groups suffice.
        pp=GroupInfo(rank=0, size=1, group=None),
        ep=GroupInfo(rank=0, size=1, group=None),
        etp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=GroupInfo(rank=0, size=1, group=None),
        is_pp_last_stage=is_pp_last_stage,
    )


@pytest.mark.parametrize(
    ("tp_size", "pp_size", "dp_size", "cp_size"),
    [
        (1, 1, 1, 1),
        (2, 1, 1, 1),
        (1, 2, 1, 1),
        (1, 1, 2, 1),
        (2, 2, 2, 1),
        (4, 1, 2, 1),
        (2, 1, 1, 2),
        (2, 2, 2, 2),
    ],
)
def test_exactly_one_rank_reports(tp_size, pp_size, dp_size, cp_size):
    """Enumerate the whole grid and count reporters. Anything other than one
    means the reported NLL is off by an integer factor."""
    reporters = 0
    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            for cp_rank in range(cp_size):
                for dp_rank in range(dp_size):
                    state = _parallel_state(
                        tp_rank=tp_rank,
                        tp_size=tp_size,
                        cp_rank=cp_rank,
                        cp_size=cp_size,
                        dp_rank=dp_rank,
                        dp_size=dp_size,
                        is_pp_last_stage=(pp_rank == pp_size - 1),
                    )
                    reporters += is_eval_nll_reporting_rank(state)
    assert reporters == 1, f"tp={tp_size} pp={pp_size} dp={dp_size} cp={cp_size} gave {reporters} reporters"


def test_pp_last_stage_is_actually_consulted():
    """Regression guard: is_pp_last_stage must be a bool FIELD. If it ever
    becomes a method, a bound method is truthy always and the pipeline half of
    the check silently disappears."""
    assert is_eval_nll_reporting_rank(_parallel_state(is_pp_last_stage=True))
    assert not is_eval_nll_reporting_rank(_parallel_state(is_pp_last_stage=False))

    from miles.backends.training_utils.parallel import ParallelState

    # A declared dataclass field, not a method/property. If it ever became a
    # method, the CLASS attribute would be a function -- and `and`-ing a bound
    # method into the predicate is always truthy, silently dropping the check.
    assert "is_pp_last_stage" in ParallelState.__dataclass_fields__
    assert not callable(getattr(ParallelState, "is_pp_last_stage", None))
    assert not isinstance(
        getattr(ParallelState, "is_pp_last_stage", None), property
    ), "a property would still work here, but is_eval_nll_reporting_rank's callable() guard would not see it"

    state = _parallel_state()
    assert isinstance(state.is_pp_last_stage, bool)
    assert not callable(state.is_pp_last_stage)


def test_reporting_rank_rejects_a_callable_pp_flag():
    class _Bad:
        is_pp_last_stage = lambda self: True  # noqa: E731
        tp = cp = intra_dp = type("G", (), {"rank": 0})()

    with pytest.raises(TypeError, match="must be a bool"):
        is_eval_nll_reporting_rank(_Bad())


@pytest.mark.parametrize(
    "field", ["tp_rank", "cp_rank", "dp_rank"]
)
def test_nonzero_rank_on_any_axis_does_not_report(field):
    assert not is_eval_nll_reporting_rank(_parallel_state(**{field: 1}))


def test_select_result_returns_the_single_reported_value():
    stats = {"nll": 1.5}
    assert select_eval_nll_result([None, stats, None, None]) is stats


def test_select_result_rejects_zero_reporters():
    with pytest.raises(AssertionError, match="got 0 of 4"):
        select_eval_nll_result([None, None, None, None])


def test_select_result_rejects_multiple_reporters():
    """Two reporters is what a TP/PP dedup bug looks like from the driver."""
    with pytest.raises(AssertionError, match="got 2 of 4"):
        select_eval_nll_result([{"nll": 1.0}, None, {"nll": 1.0}, None])


def test_ray_train_group_compute_eval_nll_dedupes():
    """The real RayTrainGroup method, with _broadcast stubbed. Constructed via
    a subclass so no Ray actors are allocated."""
    import asyncio

    from miles.ray.actor_group import RayTrainGroup

    class _StubGroup(RayTrainGroup):
        def __init__(self, results):
            self._results = results
            self.calls = []

        async def _broadcast(self, method_name, *args, **kwargs):
            self.calls.append((method_name, args, kwargs))
            return self._results

    stats = {"nll": 1.8457, "num_samples": 100}
    group = _StubGroup([None, stats, None, None])
    assert asyncio.run(group.compute_eval_nll(7)) is stats
    assert group.calls == [("compute_eval_nll", (7,), {})], "must forward rollout_id to the actors"

    with pytest.raises(AssertionError, match="exactly one actor"):
        asyncio.run(_StubGroup([None, None]).compute_eval_nll(0))
    with pytest.raises(AssertionError, match="exactly one actor"):
        asyncio.run(_StubGroup([stats, stats]).compute_eval_nll(0))


def test_ray_train_group_returns_its_result_unlike_train():
    """RayTrainGroup.train discards _broadcast's return; compute_eval_nll must
    not -- the number is the entire point."""
    import inspect

    from miles.ray.actor_group import RayTrainGroup

    assert "return" in inspect.getsource(RayTrainGroup.compute_eval_nll)


# --------------------------------------------------------------------------
# Metric keys. Task 10's results ledger keys on these strings.
# --------------------------------------------------------------------------


def _stats(nll=1.8457):
    return {
        "nll": nll,
        "sample_mean_nll": 1.9,
        "num_tokens": 41253,
        "num_samples": 100,
        "num_scored_samples": 100,
        "sum_neg_logprob": nll * 41253,
    }


def test_metric_keys_are_pinned():
    metrics = build_eval_nll_metrics(_stats(), step=3)
    assert metrics["eval/test_nll"] == pytest.approx(1.8457)
    assert set(metrics) == {
        "eval/test_nll",
        "eval/test_nll_sample_mean",
        "eval/test_nll_tokens",
        "eval/test_nll_samples",
        "rollout/step",
    }
    assert metrics["rollout/step"] == 3


def test_before_train_adds_its_own_key_without_dropping_the_primary():
    metrics = build_eval_nll_metrics(_stats(), step=0, before_train=True)
    assert metrics["eval/test_nll_before_train"] == pytest.approx(1.8457)
    assert metrics["eval/test_nll"] == pytest.approx(1.8457)


def test_metric_key_constants_match_the_emitted_strings():
    from miles.orbit.utils.eval_nll import EVAL_NLL_BEFORE_TRAIN_METRIC_KEY, EVAL_NLL_METRIC_KEY

    metrics = build_eval_nll_metrics(_stats(), step=0, before_train=True)
    assert EVAL_NLL_METRIC_KEY == "eval/test_nll"
    assert EVAL_NLL_BEFORE_TRAIN_METRIC_KEY == "eval/test_nll_before_train"
    assert EVAL_NLL_METRIC_KEY in metrics
    assert EVAL_NLL_BEFORE_TRAIN_METRIC_KEY in metrics


def test_step_key_is_present_for_tracking_utils():
    """tracking_utils.log(..., step_key="rollout/step") indexes the dict by
    that key; a missing entry is a KeyError at the first measurement."""
    assert "rollout/step" in build_eval_nll_metrics(_stats(), step=11)


# --------------------------------------------------------------------------
# Unsupported entrypoints must refuse, not silently emit nothing.
# --------------------------------------------------------------------------


def test_unsupported_entrypoint_refuses_when_flag_is_set():
    from argparse import Namespace

    from miles.orbit.utils.eval_nll import reject_eval_nll_on_unsupported_entrypoint

    with pytest.raises(ValueError, match="not supported by train_async.py"):
        reject_eval_nll_on_unsupported_entrypoint(
            Namespace(eval_nll_data="/tmp/x.jsonl"), "train_async.py"
        )


def test_unsupported_entrypoint_names_the_supported_one():
    from argparse import Namespace

    from miles.orbit.utils.eval_nll import reject_eval_nll_on_unsupported_entrypoint

    with pytest.raises(ValueError, match="train.py"):
        reject_eval_nll_on_unsupported_entrypoint(Namespace(eval_nll_data="/tmp/x.jsonl"), "other.py")


@pytest.mark.parametrize("value", [None, ""])
def test_unsupported_entrypoint_is_a_noop_when_flag_is_unset(value):
    from argparse import Namespace

    from miles.orbit.utils.eval_nll import reject_eval_nll_on_unsupported_entrypoint

    reject_eval_nll_on_unsupported_entrypoint(Namespace(eval_nll_data=value), "train_async.py")
    reject_eval_nll_on_unsupported_entrypoint(Namespace(), "train_async.py")


def test_train_async_calls_the_refusal():
    """Pin the wiring: train_async.py is not importable without CUDA, so read
    the source instead of the module."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[3] / "train_async.py").read_text(encoding="utf-8")
    assert "reject_eval_nll_on_unsupported_entrypoint(args, \"train_async.py\")" in source
    assert "from miles.orbit.utils.eval_nll import reject_eval_nll_on_unsupported_entrypoint" in source


# --------------------------------------------------------------------------
# Collective ordering vs offload. sleep() calls destroy_process_groups(), which
# sets ReloadableProcessGroup.group = None; the monkeypatched dist.all_reduce
# then unwraps group=<the DP group> to group=None, which torch reads as the
# default WORLD group -- silently, with no exception. Any collective outside the
# wake_up()..sleep() window therefore reduces over the wrong communicator.
#
# The actor is not importable without CUDA (megatron_utils/__init__.py imports
# deep_ep), so this is pinned by source order. Weaker than executing it, but it
# guards the exact regression, which nothing else does.
#
# compute_eval_nll moved out of miles/backends/megatron_utils/actor.py into the
# orbit home mixin (Phase 3 isolation, slice 3g); MegatronTrainRayActor still
# gets it as a base. Only the file this reads changed -- the assertions below
# are untouched.
# --------------------------------------------------------------------------


def _compute_eval_nll_source() -> str:
    import inspect
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[3] / "miles/orbit/megatron/actor_ext.py"
    ).read_text(encoding="utf-8")
    start = source.index("    def compute_eval_nll(")
    end = source.index("\n    def ", start + 1)
    _ = inspect, re
    return source[start:end]


def test_both_dp_collectives_sit_inside_the_wake_sleep_window():
    body = _compute_eval_nll_source()

    wake = body.index("self.wake_up()")
    sleep = body.index("self.sleep()")
    reduces = [
        m for m in range(len(body)) if body.startswith("dist.all_reduce(", m)
    ]
    assert len(reduces) == 2, f"expected exactly 2 DP collectives, found {len(reduces)}"

    for position in reduces:
        assert wake < position < sleep, (
            "a dist.all_reduce sits outside the wake_up()..sleep() window; after "
            "destroy_process_groups() the DP group unwraps to None and torch "
            "silently reduces over WORLD instead"
        )


def test_sleep_is_still_in_a_finally_so_the_model_goes_back_on_the_error_path():
    body = _compute_eval_nll_source()
    finally_at = body.index("finally:")
    assert body.index("self.sleep()") > finally_at
    assert "if woke_here:" in body[finally_at:]


def test_wake_and_sleep_are_paired_on_the_same_flag():
    """Waking without restoring the previous state would leave the training
    model resident through the next generation phase."""
    body = _compute_eval_nll_source()
    assert body.count("woke_here = True") == 1
    assert body.count("if woke_here:") == 1
    assert "getattr(self, \"_train_state_awake\", True)" in body


def test_actor_uses_the_shared_reporting_rank_helper():
    """The dedup predicate must be the tested one, not re-spelled inline."""
    body = _compute_eval_nll_source()
    assert "is_eval_nll_reporting_rank(parallel_state)" in body
