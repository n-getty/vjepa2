"""Anchor selection in `scripts/tail_arm_compare.py` under A/B/B/A.

WHY THIS EXISTS
---------------
The reader's anchor check is the thing standing between this study and the
[[ab-window-truncation-trap]] class of overclaim, so a defect in the *check* is
worse than a defect in the numbers: it fails toward a confident verdict.

It had one. The check grouped arms by base name, treated every group with more
than one member as an anchor, and vetoed the whole comparison if ANY of them
spread. That is right for A/B/C/A, where only the baseline repeats. It is wrong
for A/B/B/A -- the layout the prefetch sweep uses precisely so the treatment's
own repeat spread is measured -- because there BOTH groups repeat, so:

  * a noisy treatment pair vetoed a study whose baseline reproduced to 1%, and
  * `min(reps.values(), key=lambda v: -len(v))` picks by group size alone, so
    on the resulting 2-vs-2 tie the TREATMENT could become the baseline and
    every reported delta would silently flip sign.

Neither failure is visible in the output: the first prints a plausible-sounding
refusal, the second prints a plausible-sounding ranking. Only synthetic arms
with known-true deltas catch them, which is what this file is.

The fix, and the three properties pinned here:
  1. the anchor is the group that BRACKETS the run (first and last rung), not
     the largest group -- bracketing is what makes it a drift measurement;
  2. only the anchor's spread can VETO; a treatment that does not reproduce
     widens the floor instead, because that is a statement about what the study
     can resolve, not about whether the arms are comparable;
  3. the floor is the max over repeated groups, so a delta is called MEASURED
     only if it clears the worst-behaved pair in the allocation.
"""

import os
import random
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "scripts", "tail_arm_compare.py")

COL_ITER, COL_DLOAD, NCOLS = 3, 5, 19


def _write_arm(root, name, dl_scale, iter_base, mtime, seed, iters=100, ranks=4):
    """One rung dir of synthetic 19-column rank CSVs.

    Zero-inflated dataload (80% exactly 0.00 s) because that is what nw>0
    actually produces -- the reader's `excess` collapses to the plain mean
    under prefetch, and a test built on a non-zero body would exercise a
    statistic the real arms never hit.
    """
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    rng = random.Random(seed)
    for r in range(ranks):
        p = os.path.join(d, f"log_r{r}.csv")
        with open(p, "w") as f:
            for it in range(iters):
                dl = 0.0 if rng.random() < 0.8 else rng.uniform(1, 10) * dl_scale
                cols = ["0"] * NCOLS
                cols[1] = str(it)
                cols[COL_ITER] = f"{(iter_base + dl) * 1000:.1f}"
                cols[COL_DLOAD] = f"{dl * 1000:.1f}"
                f.write(",".join(cols) + "\n")
    # Run order is read from the CSV mtimes: rungs are serial, and the dir's own
    # mtime is not usable (a later checkpoint write touches it).
    for r in range(ranks):
        os.utime(os.path.join(d, f"log_r{r}.csv"), (mtime, mtime))


def _run(root):
    p = subprocess.run([sys.executable, TOOL, "--ladder", root, "--lo", "10"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout


def _abba(tmp_path, order):
    """Build A/B/B/A with a real 30% tail reduction on B, in `order`."""
    root = str(tmp_path)
    spec = {
        "n2_nw2":          (1.0, 3.1, 7),
        "n2_nw2_rep2":     (1.0, 3.1, 8),
        "n2_nw2_pf8":      (0.7, 3.1, 9),
        "n2_nw2_pf8_rep2": (0.7, 3.1, 10),
    }
    for i, name in enumerate(order):
        dl, ib, seed = spec[name]
        _write_arm(root, name, dl, ib, 1_000_000_000 + i * 600, seed)
    return root


def test_bracketing_group_is_the_anchor_not_the_largest():
    """On a 2-vs-2 tie the baseline must be the arm that opened and closed.

    Picking by group size alone is a coin flip between the two, and choosing
    the treatment inverts every delta in the table while printing an equally
    confident ranking.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = _abba(tmp, ["n2_nw2", "n2_nw2_pf8", "n2_nw2_pf8_rep2", "n2_nw2_rep2"])
        out = _run(root)
        assert "ANCHOR n2_nw2:" in out, out
        assert "repeat n2_nw2_pf8:" in out, out
        assert "baseline = n2_nw2 " in out, out
        # The true effect is a REDUCTION, so every reported delta must be
        # negative. A flipped baseline shows up here as +40%.
        #
        # Select on "tail" rather than on the arm name: the summary table at the
        # top of the output also has rows starting with the arm name, and
        # matching those would assert against a line that has no delta in it at
        # all -- a test that fails for the wrong reason and gets "fixed" by
        # loosening it.
        ranked = [l for l in out.splitlines()
                  if l.strip().startswith("n2_nw2_pf8") and "tail " in l]
        assert len(ranked) == 2, out
        for ln in ranked:
            pct = float(ln.split("tail")[1].split("%")[0])
            assert pct < 0, ln


def test_noisy_treatment_pair_does_not_veto_a_clean_anchor():
    """A/B/B/A with a reproducing baseline must still produce a ranking.

    The pre-fix check vetoed here, discarding a whole allocation because the
    treatment -- the thing under test -- varied.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = str(tmp)
        _write_arm(root, "n2_nw2", 1.0, 3.1, 1_000_000_000, seed=7)
        # Deliberately mismatched seeds/scales: the two B arms disagree a lot.
        _write_arm(root, "n2_nw2_pf8", 0.9, 3.1, 1_000_000_600, seed=3)
        _write_arm(root, "n2_nw2_pf8_rep2", 0.4, 3.1, 1_000_001_200, seed=4)
        _write_arm(root, "n2_nw2_rep2", 1.0, 3.1, 1_000_001_800, seed=8)
        out = _run(root)
        assert "did not reproduce" not in out, out
        assert "baseline = n2_nw2 " in out, out


def test_floor_widens_to_the_worst_repeated_group():
    """The quoted floor must not be the anchor's when a treatment is noisier.

    Otherwise a 10% delta is called MEASURED against a 4% baseline floor while
    the treatment's own two runs differ by 20% -- an unresolvable effect
    reported as a result.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = str(tmp)
        _write_arm(root, "n2_nw2", 1.0, 3.1, 1_000_000_000, seed=7)
        _write_arm(root, "n2_nw2_pf8", 0.9, 3.1, 1_000_000_600, seed=3)
        _write_arm(root, "n2_nw2_pf8_rep2", 0.4, 3.1, 1_000_001_200, seed=4)
        _write_arm(root, "n2_nw2_rep2", 1.0, 3.1, 1_000_001_800, seed=8)
        out = _run(root)
        assert "WIDENED to the" in out, out
        floor = [l for l in out.splitlines() if l.startswith("NOISE FLOOR:")]
        anch = [l for l in out.splitlines() if "anchor's own repeats give" in l]
        assert floor and anch, out
        f_tail = float(floor[0].split("tail +/-")[1].split("%")[0])
        a_tail = float(anch[0].split("tail +/-")[1].split("%")[0])
        assert f_tail > a_tail, (floor[0], anch[0])


def test_unbracketed_layout_says_so_instead_of_pretending():
    """A/B/A/B has repeats but no bracket, so it measures no drift.

    Falling back silently would let a sweep that cannot see start-to-end drift
    be read as one that can.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = _abba(tmp, ["n2_nw2", "n2_nw2_pf8", "n2_nw2_rep2", "n2_nw2_pf8_rep2"])
        out = _run(root)
        assert "no arm brackets the allocation" in out, out


def test_no_repeats_still_refuses_to_rank():
    """The pre-existing guard must survive the anchor rework.

    An unbracketed sweep has no noise floor at all, and this refusal is the
    only thing that stops its deltas being quoted.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = str(tmp)
        _write_arm(root, "n2_nw2", 1.0, 3.1, 1_000_000_000, seed=7)
        _write_arm(root, "n2_nw2_pf8", 0.7, 3.1, 1_000_000_600, seed=9)
        out = _run(root)
        assert "NO REPEATED ARM" in out, out
        assert "baseline =" not in out, out
