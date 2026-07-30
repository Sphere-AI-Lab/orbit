"""Read the sweep ledgers into the campaign's claims.

Every difference this module prints is in units of sigma, measured by E1-0 --
never off absolute loss values. The constant Orbit-vs-HF precision offset
(0.0032 nats) cancels in every ratio, ordering and curve-shape claim the
campaign makes, and cancels in nothing else.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

# (method, size, target_modules). `size` is the rank for LoRA, the block size
# for OFT, and None for full fine-tuning.
#
# target_modules is part of the key and must NOT be dropped: E3 runs
# `lora r256 attention-only` and `lora r256 all-modules` in the same matrix, so
# a (method, rank) key would silently collapse two different arms into one and
# report whichever happened to score better as "the r256 argmin". That is the
# exact class of bug the seed-0 filter exists to prevent, one axis over.
ArmKey = tuple[str, int | None, str]


def load_records(
    paths,
    *,
    seed: int | None = 0,
    require_ok: bool = True,
    metric: str = "nll",
) -> list[dict]:
    """Ledger records worth analysing, from files or globs.

    `seed=0` is the default and is not cosmetic: E1-0's replicates live in the
    same ledger directory at seeds 1 and 2 and are *not* grid points. Measured
    on a synthetic ledger, dropping this filter let a replicate at LR 9.95e-4
    win r256's argmin away from the real 2.5e-4 purely because that one run
    happened to score better. Pass `seed=None` to read replicates, which is
    what `sigma` wants and nothing else does.

    Arms whose trace was inconsistent are dropped: a held-out set that changed
    size mid-run makes that arm's NLL incomparable to the others.
    """
    records: list[dict] = []
    for entry in paths:
        matches = sorted(glob.glob(str(entry))) or [str(entry)]
        for match in matches:
            path = Path(match)
            if not path.exists():
                raise FileNotFoundError(f"no ledger at {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated final line from an interrupted write
                if require_ok and record.get("status") != "ok":
                    continue
                if seed is not None and record.get("seed") != seed:
                    continue
                if record.get("metric", "nll") != metric:
                    continue
                if record.get("trace_consistent") is False:
                    continue
                records.append(record)
    return records


def arm_key(record: dict) -> ArmKey:
    size = record.get("oft_block_size") if record["method"] == "oft" else record.get("rank")
    return (record["method"], size, record.get("target_modules") or "")


def score(record: dict, metric: str = "nll") -> float:
    return record["accuracy"] if metric == "accuracy" else record["test_nll"]


def sigma(records: list[dict]) -> float:
    """Seed-to-seed standard deviation, from E1-0's replicates.

    Load with `seed=None`: the replicates are seeds 0, 1 and 2 of one
    configuration, and the default seed-0 filter would leave one point.
    """
    values = [score(r) for r in records]
    if len(values) < 3:
        raise ValueError(
            f"sigma needs at least 3 replicates, got {len(values)}. "
            "Run E1-0 (runbook section 7) and load with seed=None."
        )
    return statistics.stdev(values)


def lr_grids(records: list[dict]) -> dict[ArmKey, list[float]]:
    """The learning rates actually run, per arm, sorted."""
    grids: dict[ArmKey, set[float]] = {}
    for record in records:
        grids.setdefault(arm_key(record), set()).add(record["lr"])
    return {key: sorted(values) for key, values in grids.items()}


def argmins(records: list[dict], metric: str = "nll") -> dict[ArmKey, dict]:
    """The best-scoring record per arm.

    Lower is better for NLL, higher for accuracy -- the direction is chosen by
    `metric` rather than assumed, because E4's ledgers score by accuracy.
    """
    better = (lambda a, b: a > b) if metric == "accuracy" else (lambda a, b: a < b)
    best: dict[ArmKey, dict] = {}
    for record in records:
        key = arm_key(record)
        if key not in best or better(score(record, metric), score(best[key], metric)):
            best[key] = record
    return best


def edge_of_grid(records: list[dict], metric: str = "nll") -> dict[ArmKey, str]:
    """Arms whose argmin sits on a boundary of the grid that was run.

    An argmin at either end means the true optimum may lie outside the grid, so
    any ratio quoted from it is a lower bound on an unknown. The runbook's rule
    is to **re-centre, not extend**: extending keeps the old points at the wrong
    spacing and leaves the grid asymmetric about the new estimate.

    A one-point grid is flagged, because a single LR is simultaneously the
    lowest and the highest that was tried.
    """
    grids = lr_grids(records)
    flagged: dict[ArmKey, str] = {}
    for key, best in argmins(records, metric).items():
        grid = grids[key]
        if best["lr"] in (grid[0], grid[-1]):
            flagged[key] = (
                f"argmin LR {best['lr']:g} is on the edge of the grid "
                f"[{grid[0]:g} .. {grid[-1]:g}] ({len(grid)} points); "
                "re-centre the grid on it and re-run before quoting a ratio"
            )
    return flagged


def departure_steps(
    traces: dict[str, list],
    sigma_value: float,
    *,
    threshold_sigma: float = 2.0,
    consecutive: int = 3,
) -> dict[str, int | None]:
    """Per arm, the step at which it leaves the shared envelope -- C1's number.

    The envelope is the pointwise minimum NLL across all arms at each step. An
    arm departs at the first step of the first run of `consecutive` steps where
    it sits more than `threshold_sigma` sigma above that envelope. Requiring a
    run of three is what keeps a single noisy eval from reading as a departure.

    `None` means "no departure within this arm's trace", which is NOT the same
    as "does not depart" -- the caller must print the step budget alongside.
    """
    envelope: dict[int, float] = {}
    for points in traces.values():
        for point in points:
            step = point.step
            if step not in envelope or point.nll < envelope[step]:
                envelope[step] = point.nll

    limit = threshold_sigma * sigma_value
    departures: dict[str, int | None] = {}
    for name, points in traces.items():
        run_start: int | None = None
        run_length = 0
        departures[name] = None
        for point in sorted(points, key=lambda p: p.step):
            if point.nll - envelope[point.step] > limit:
                if run_start is None:
                    run_start = point.step
                run_length += 1
                if run_length >= consecutive:
                    departures[name] = run_start
                    break
            else:
                run_start, run_length = None, 0
    return departures


def lr_band(
    records: list[dict],
    sigma_value: float,
    metric: str = "nll",
    *,
    threshold_sigma: float = 2.0,
) -> dict[ArmKey, tuple[float, float]]:
    """Per arm, the lowest and highest LR scoring within `threshold_sigma` of its best.

    C5's second half is about the *width* of the performant band, which is a
    separate checkable statement from peak parity: LoRA's band being wider is a
    claim that survives even if the peaks tie.
    """
    best = argmins(records, metric)
    bands: dict[ArmKey, tuple[float, float]] = {}
    for key, top in best.items():
        top_score = score(top, metric)
        within = [
            r["lr"]
            for r in records
            if arm_key(r) == key
            and abs(score(r, metric) - top_score) <= threshold_sigma * sigma_value
        ]
        bands[key] = (min(within), max(within))
    return bands


def batch_gaps(
    records: list[dict],
    sigma_value: float,
) -> dict[tuple[int | None, ArmKey], float]:
    """C3: `best_LoRA(batch) - best_FullFT(batch)` at each batch size, in sigma.

    The claim is a gap that *grows* with batch -- a gap absent at 32 and present
    at 512 is the signature, a constant offset at all three is not -- so the
    comparison has to be made within each batch size, never pooled. A batch with
    no FullFT arm is skipped rather than compared against another batch's
    baseline: that would attribute a batch-size effect to a placement it never
    had.
    """
    by_batch: dict[int | None, list[dict]] = {}
    for record in records:
        by_batch.setdefault(record.get("global_batch_size"), []).append(record)
    gaps: dict[tuple[int | None, ArmKey], float] = {}
    for batch, group in by_batch.items():
        best = argmins(group)
        baseline = next((v for k, v in best.items() if k[0] == "full"), None)
        if baseline is None:
            continue
        for key, record in best.items():
            if key[0] == "full":
                continue
            gaps[(batch, key)] = (record["test_nll"] - baseline["test_nll"]) / sigma_value
    return gaps


def _targeting(best: dict[ArmKey, dict], modules) -> dict[ArmKey, dict]:
    """Arms whose target set is exactly `modules`.

    Exact set equality, not "contains linear_qkv": E3 runs `attn r256` and
    `all r256` in one matrix, and a containment test would report the
    attention-only arm as the all-modules one.
    """
    wanted = set(modules)
    return {
        key: record
        for key, record in best.items()
        if {name for name in key[2].split(",") if name} == wanted
    }


def _pairwise_deltas(
    left: dict[ArmKey, dict],
    right: dict[ArmKey, dict],
    sigma_value: float,
    left_label: str,
    right_label: str,
) -> dict[str, float]:
    """Every left-against-right difference, in sigma, labelled by both ranks."""
    return {
        f"{left_label}(r{lk[1]}) - {right_label}(r{rk[1]})": (
            lr["test_nll"] - rr["test_nll"]
        ) / sigma_value
        for lk, lr in left.items()
        for rk, rr in right.items()
    }


def placement_deltas(records: list[dict], sigma_value: float) -> dict[str, float]:
    """C4: `NLL(attention) - NLL(MLP)` at matched parameters, in sigma.

    Pairs each attention-only arm with each MLP-only arm, labelled by both
    ranks, because the matched pair is `attention r256` against `MLP r92` and
    the post's own pair (`r256`/`r128`) is deliberately in the same matrix -- if
    the two disagree, the disagreement is parameter accounting rather than
    physics, and collapsing them to one number would hide exactly that.
    """
    from orbit.utils.peft_param_match import ATTENTION_MODULES, MLP_MODULES

    best = argmins(records)
    return _pairwise_deltas(
        _targeting(best, ATTENTION_MODULES),
        _targeting(best, MLP_MODULES),
        sigma_value,
        "attn",
        "mlp",
    )


def all_modules_deltas(records: list[dict], sigma_value: float) -> dict[str, float]:
    """C4's second half: `NLL(all-modules) - NLL(MLP-only)`, in sigma.

    The post claims two things about placement, and they are separately
    falsifiable: attention-only *underperforms* MLP-only, and all-modules *adds
    nothing on top of* MLP-only. `placement_deltas` reads the first. This reads
    the second, and note the direction it is read in -- the claim survives a
    delta near zero or positive, and is contradicted by a delta below
    -2 sigma, which is all-modules winning by more than noise.

    Not comparable at matched parameters, and deliberately so: all-modules at
    rank 256 carries ~2.8x the MLP-only r92 adapter. That is the point -- the
    claim is that the extra capacity buys nothing, so equalising it first would
    ask a different question.
    """
    from orbit.utils.peft_param_match import ATTENTION_MODULES, MLP_MODULES

    best = argmins(records)
    return _pairwise_deltas(
        _targeting(best, tuple(ATTENTION_MODULES) + tuple(MLP_MODULES)),
        _targeting(best, MLP_MODULES),
        sigma_value,
        "all",
        "mlp",
    )


_MODULE_SHORT = {
    "linear_qkv,linear_proj,linear_fc1,linear_fc2": "all",
    "linear_qkv,linear_proj": "attn",
    "linear_fc1,linear_fc2": "mlp",
}


def _fmt_key(key: ArmKey) -> str:
    method, size, modules = key
    if method == "full":
        return "full"
    label = "b" if method == "oft" else "r"
    return f"{method} {label}{size} {_MODULE_SHORT.get(modules, modules)}"


def _load_traces(records: list[dict], log_dir: Path) -> tuple[dict[str, list], dict[str, str]]:
    """Traces per arm, preferring the ledger's own field over re-reading a log.

    Reports the source per arm: a silently-empty trace and a silently-truncated
    one both read as "no departure", so which file the number came from is part
    of the answer.
    """
    from tools.lora_regret.trace import NllPoint, parse_trace_file

    traces: dict[str, list] = {}
    sources: dict[str, str] = {}
    for record in records:
        name = record["arm"]
        if record.get("nll_trace"):
            traces[name] = [NllPoint(**point) for point in record["nll_trace"]]
            sources[name] = "ledger"
            continue
        log_path = log_dir / f"{name}.log"
        if log_path.exists():
            traces[name] = parse_trace_file(log_path)
            sources[name] = str(log_path)
        else:
            traces[name] = []
            sources[name] = "MISSING -- no nll_trace field and no log"
    return traces, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["sigma", "argmins", "c1", "c2", "c3", "c4", "c5", "c6", "all"],
    )
    parser.add_argument("--ledgers", nargs="+", required=True, help="paths or globs")
    parser.add_argument(
        "--sigma-ledger",
        nargs="+",
        default=None,
        help="E1-0's replicate ledger. Required by every claim but 'sigma' itself, "
        "unless --sigma is given.",
    )
    parser.add_argument("--sigma", type=float, default=None, help="override the measured sigma")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/lora_regret"))
    parser.add_argument(
        "--allow-edge-argmin",
        action="store_true",
        help="quote claims that depend on an argmin sitting on a grid edge. Off by "
        "default: the runbook's rule is to re-centre and re-run first.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON document on stdout and nothing else -- the handoff to "
        "plotting. Exit codes are unchanged: an edge-of-grid argmin still exits 3, "
        "with the reason inside the payload rather than only on stderr.",
    )
    args = parser.parse_args()

    payload: dict = {"command": args.command}
    # Every human-readable line goes through this, so --json cannot be broken by
    # a stray print: one unparseable line makes the whole document useless, and
    # that failure only shows up in the consumer.
    def say(line: str = "") -> None:
        if not args.json:
            print(line)

    def emit(code: int) -> int:
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return code

    # Both metrics, loaded separately. An E4 ledger carries metric="accuracy"
    # and test_nll=null, so loading only the nll view and bailing on empty would
    # make `analyze c5 --ledgers results/e4_*.jsonl` exit before it ran.
    records = load_records(args.ledgers)
    acc_records = load_records(args.ledgers, metric="accuracy")
    if not records and not acc_records:
        print("no usable records in the given ledgers", file=sys.stderr)
        return 1

    if args.command == "sigma":
        replicates = load_records(args.ledgers, seed=None)
        value = sigma(replicates)
        payload["sigma"] = value
        payload["n"] = len(replicates)
        say(f"sigma = {value:.6f} nats  (n={len(replicates)})")
        return emit(0)

    sigma_value = args.sigma
    if sigma_value is None and args.sigma_ledger:
        sigma_value = sigma(load_records(args.sigma_ledger, seed=None))
    if sigma_value is None and args.command != "argmins":
        print(
            "no sigma: pass --sigma-ledger results/e1_0_sigma.jsonl or --sigma VALUE. "
            "Every difference this campaign claims is quoted in units of sigma, and "
            "the Qwen3-era 0.000992 does not transfer to Llama-3.1-8B / Tulu3.",
            file=sys.stderr,
        )
        return 2
    payload["sigma"] = sigma_value

    flagged = edge_of_grid(records)
    best = argmins(records)
    grids = lr_grids(records)
    order = lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2])  # noqa: E731
    payload["edge_of_grid"] = {_fmt_key(key): why for key, why in flagged.items()}

    if args.command in ("argmins", "all"):
        payload["argmins"] = [
            {
                "arm": _fmt_key(key),
                "method": key[0],
                "size": key[1],
                "target_modules": key[2],
                "lr": record["lr"],
                "test_nll": record["test_nll"],
                "adapter_params": record.get("adapter_params"),
                "lr_grid": grids[key],
                "edge_of_grid": key in flagged,
            }
            for key, record in sorted(best.items(), key=order)
        ]
        say(f"{'arm':22} {'argmin_lr':<11} {'nll':<10} {'adapter_params':>15}  grid")
        for key, record in sorted(best.items(), key=order):
            grid = grids[key]
            params = record.get("adapter_params")
            say(
                f"{_fmt_key(key):22} {record['lr']:<11g} {record['test_nll']:<10.6f} "
                f"{params if params is not None else '-':>15}  "
                f"[{grid[0]:g} .. {grid[-1]:g}]"
                + ("   EDGE OF GRID" if key in flagged else "")
            )
    if flagged and not args.allow_edge_argmin:
        print("\nedge-of-grid arms -- re-centre and re-run before quoting:", file=sys.stderr)
        for key, why in flagged.items():
            print(f"  {_fmt_key(key)}: {why}", file=sys.stderr)
        if args.command != "argmins":
            # Fail closed before any claim is computed, so no refused number can
            # reach the payload and be plotted as if it had been quoted.
            return emit(3)

    all_modules = "linear_qkv,linear_proj,linear_fc1,linear_fc2"
    if args.command in ("c2", "all"):
        lora = best.get(("lora", 256, all_modules))
        full = best.get(("full", None, ""))
        if lora and full:
            ratio = lora["lr"] / full["lr"]
            payload["c2"] = {
                "lora_r256_argmin_lr": lora["lr"],
                "fullft_argmin_lr": full["lr"],
                "ratio": ratio,
                "predicted_ratio": 9.8,
            }
            say(f"\nC2: argmin_LR(LoRA r256) / argmin_LR(FullFT) = {ratio:.2f}")
            say("    the post predicts 9.8, rising toward 15 for runs under ~100 steps")
            edges = [best.get(("lora", r, all_modules)) for r in (4, 512)]
            if all(edges):
                lrs = [record["lr"] for record in edges]
                spread = max(lrs) / min(lrs)
                payload["c2"]["rank4_vs_rank512_spread"] = spread
                say(
                    f"    rank 4 vs 512 argmin spread = {spread:.2f}x "
                    "(the tighter claim is < 2x)"
                )

    if args.command in ("c1", "all"):
        traces, sources = _load_traces(records, args.log_dir)
        departures = departure_steps(traces, sigma_value)
        payload["c1"] = [
            {
                "arm": name,
                "departure_step": departures[name],
                "step_budget": max((p.step for p in traces[name]), default=0),
                "trace_source": sources[name],
            }
            for name in sorted(departures)
        ]
        say("\nC1: departure from the envelope (>2 sigma for 3 consecutive evals)")
        for row in payload["c1"]:
            where = row["departure_step"]
            verdict = (
                f"step {where}" if where is not None
                else f"no departure within {row['step_budget']} steps"
            )
            say(f"    {row['arm']:34} {verdict:38} [trace: {row['trace_source']}]")

    if args.command in ("c3", "all"):
        gaps = batch_gaps(records, sigma_value)
        if gaps:
            ordered = sorted(gaps.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1]))
            payload["c3"] = [
                {"global_batch_size": batch, "arm": _fmt_key(key), "delta_sigma": delta}
                for (batch, key), delta in ordered
            ]
            say(f"\nC3: best_LoRA - best_FullFT per batch (sigma = {sigma_value:.6f})")
            say("    the claim is a gap that GROWS with batch; a constant offset is not it")
            for (batch, key), delta in ordered:
                say(f"    batch {str(batch):>4}  {_fmt_key(key):22} {delta:+8.2f} sigma")

    if args.command in ("c4", "all"):
        deltas = placement_deltas(records, sigma_value)
        extra = all_modules_deltas(records, sigma_value)
        if deltas or extra:
            payload["c4"] = {"attn_minus_mlp": deltas, "all_minus_mlp": extra}
            say(f"\nC4: placement at matched parameters (sigma = {sigma_value:.6f})")
            for label, delta in sorted(deltas.items()):
                say(f"    {label:28} {delta:+8.2f} sigma")
            if extra:
                # The second half of the claim, and it is read in the opposite
                # direction: all-modules is supposed to add NOTHING, so a delta
                # near zero or positive upholds it and one below -2 sigma is
                # all-modules winning by more than noise.
                say("    all-modules on top of MLP-only -- the claim is that it adds nothing,")
                say("    so a delta below -2 sigma contradicts it:")
                for label, delta in sorted(extra.items()):
                    verdict = "CONTRADICTS" if delta < -2.0 else "consistent"
                    say(f"    {label:28} {delta:+8.2f} sigma   {verdict}")

    if args.command in ("c6", "all"):
        oft = {k: v for k, v in best.items() if k[0] == "oft"}
        if oft:
            payload["c6"] = [
                {
                    "arm": _fmt_key(key),
                    "test_nll": record["test_nll"],
                    "adapter_params": record.get("adapter_params"),
                    "matched_ratio": record.get("matched_ratio"),
                }
                for key, record in sorted(oft.items(), key=order)
            ]
            say(f"\nC6: OFT against LoRA at matched parameters (sigma = {sigma_value:.6f})")
            for key, record in sorted(oft.items(), key=order):
                ratio = record.get("matched_ratio")
                params = record.get("adapter_params")
                # Mind the direction: an OFT arm carrying slightly FEWER
                # parameters that still keeps up strengthens the finding, while
                # one carrying fewer and losing is confounded, not informative.
                suffix = f"  matched_ratio={ratio:.3f}" if ratio is not None else "  matched_ratio=?"
                say(
                    f"    {_fmt_key(key):22} nll={record['test_nll']:.6f} "
                    f"params={params}{suffix}"
                )

    if args.command in ("c5", "all"):
        acc = acc_records
        if acc:
            peaks = argmins(acc, metric="accuracy")
            bands = lr_band(acc, sigma_value, metric="accuracy")
            payload["c5"] = [
                {
                    "arm": _fmt_key(key),
                    "peak_accuracy": peaks[key]["accuracy"],
                    "band_low": bands[key][0],
                    "band_high": bands[key][1],
                    "sigma_measured": False,
                }
                for key in sorted(peaks, key=lambda k: (k[0], k[1] or 0, k[2]))
            ]
            say("\nC5: peak accuracy and performant-LR band")
            for key in sorted(peaks, key=lambda k: (k[0], k[1] or 0, k[2])):
                low, high = bands[key]
                say(
                    f"    {_fmt_key(key):22} peak={peaks[key]['accuracy']:.4f} "
                    f"band=[{low:g} .. {high:g}] ({high / low:.0f}x wide)"
                )
            say(
                "    NOTE: sigma for accuracy has never been measured. These deltas "
                "are raw and none of them is resolved. Measuring it means an E1-0 "
                "for accuracy: 3 seeds of one E4 arm."
            )
    return emit(0)


if __name__ == "__main__":
    raise SystemExit(main())
