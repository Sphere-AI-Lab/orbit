"""The seven per-learning-rate launch scripts must partition E4 exactly.

`run_e4_lr{1..7}_8gpu.sh` each select one column of the sweep: FullFT at the
i-th point of its grid, plus LoRA r1/r16/r256 at the i-th point of theirs. Two
things can go wrong silently, and both cost a node before anyone notices.

A **gap** -- a learning rate no script selects -- leaves a hole in one of C5's
four curves. Every script that does run still succeeds, `analyze` still reports
an argmin, and the missing point is invisible unless someone counts.

An **overlap** -- an arm two scripts both select -- is worse than wasted compute
if the two land in different ledgers: `analyze` globs them together and the arm
appears twice, so a duplicated point silently gets double weight in whatever the
claim is read off.

`campaign.sh` already refuses a selection that is not `EXPECT_ARMS` long, which
catches a script that selects 0 or 8. It cannot catch two scripts selecting the
same 4, or seven scripts of 4 that miss an arm between them. That is what this
file is for.
"""

import re
from pathlib import Path

from tools.lora_regret.arms import e4_arms

SCRIPTS = sorted(
    (Path(__file__).resolve().parents[3] / "scripts" / "lora_regret").glob("run_e4_lr*_8gpu.sh")
)


def _method_re(path: Path) -> str:
    match = re.search(r"METHOD_RE='([^']+)'", path.read_text(encoding="utf-8"))
    assert match, f"{path.name} has no single-quoted METHOD_RE"
    return match.group(1)


def _arms():
    return e4_arms()


def _pattern(path: Path):
    return re.compile(_method_re(path))


def _selected(path: Path) -> list[str]:
    """What `sweep.py --only` would select: re.search against the arm name."""
    return [a.name for a in _arms() if _pattern(path).search(a.name)]


def test_there_is_one_script_per_grid_point():
    assert len(SCRIPTS) == 7
    assert {p.name for p in SCRIPTS} == {f"run_e4_lr{i}_8gpu.sh" for i in range(1, 8)}


def test_each_script_covers_both_panels():
    """A column is one point of each of Figure 6's two panels, so a script
    carries both datasets. They stay separable downstream because each arm
    names its own dataset and is evaluated on it alone."""
    from tools.lora_regret.arms import RL_DATASETS

    for path in SCRIPTS:
        selected = {a.name: a for a in _arms() if _pattern(path).search(a.name)}
        assert {a.dataset for a in selected.values()} == set(RL_DATASETS), path.name


def test_each_script_selects_one_fullft_arm_and_three_lora_ranks_per_panel():
    """One point on each of C5's four curves, on each of the two panels."""
    for path in SCRIPTS:
        names = _selected(path)
        assert len(names) == 8, (path.name, names)
        assert sum(n.startswith("full-") for n in names) == 2, (path.name, names)
        assert {n.split("-")[1] for n in names if n.startswith("lora-")} == {"r1", "r16", "r256"}, (
            path.name,
            names,
        )


def test_the_lora_arms_in_a_column_share_one_learning_rate():
    """A column is a vertical slice of the figure. Three ranks at three
    different LRs would not be one."""
    for path in SCRIPTS:
        # `.+`, not `[^-]+`: the learning rate itself carries a hyphen in
        # exponent form, so `lr5e-06-s0` splits wrong on a negated class.
        lora_lrs = {
            re.search(r"-lr(.+)-s\d+$", n).group(1) for n in _selected(path) if n.startswith("lora-")
        }
        assert len(lora_lrs) == 1, (path.name, lora_lrs)


def test_no_script_selects_an_oft_arm():
    """`analyze` reads a ledger as one comparable set, and an `oftscout` row
    carries a learning rate from a different search entirely."""
    for path in SCRIPTS:
        assert not any(n.startswith("oftscout") for n in _selected(path)), path.name


def test_the_seven_scripts_partition_e4_exactly():  # noqa: D401
    """The property neither `EXPECT_ARMS` nor a per-script check can see: no
    arm selected twice, and no non-OFT arm left unselected."""
    selected = [name for path in SCRIPTS for name in _selected(path)]
    expected = {a.name for a in e4_arms() if a.method != "oft"}
    assert len(selected) == len(set(selected)), "an arm is selected by two scripts"
    assert set(selected) == expected
    assert len(selected) == 56


def test_each_script_writes_its_own_ledger():
    """Seven nodes appending to one file would interleave partial rows. The
    ledgers are globbed back together at analysis time instead."""
    ledgers = {
        re.search(r"RESULTS=(\S+)", path.read_text(encoding="utf-8")).group(1) for path in SCRIPTS
    }
    assert len(ledgers) == 7
    assert all(led.startswith("results/e4_lr") for led in ledgers)


def test_each_script_asserts_its_own_arm_count():
    for path in SCRIPTS:
        assert "EXPECT_ARMS=8" in path.read_text(encoding="utf-8"), path.name
