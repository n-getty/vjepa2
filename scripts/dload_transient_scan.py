#!/usr/bin/env python3
"""Does the dataload tail DECAY to a floor, or is it steady state?

WHY THIS EXISTS
---------------
Job 8741663 (2 nodes, 24 ranks, 93-iteration arms) showed the dataload
order statistic collapsing *within* an arm:

    arm            dload max-over-ranks, first quarter -> last quarter
    n2_nw2         2.47 -> 0.04
    n2_nw2_omp8    2.04 -> 0.29
    n2_nw2_omp4    0.99 -> 0.00

That is not a small trend, it is the whole effect disappearing. If the tail is a
warmup transient rather than a steady state, then every efficiency number in
docs/THROUGHPUT_RECIPE_AURORA.md computed over <=100 iterations is measuring
warmup, and every A/B run as short arms was comparing positions on a decay curve
rather than configurations. It already produced one false result: the "anchor did
not reproduce, 1.75x" verdict on 8741663 was mostly a 93-iteration mean being
compared against a 45-iteration mean of the same decaying series (1.45x on the
matched window, and 1.17x on dataload alone).

But 2 nodes is not production, and at 16n the decay does NOT visibly complete
inside 60 iterations -- it falls 17.82 -> 2.72, REBOUNDS to 11.64, then settles
~3.7-5.0 with the run ending still elevated. Sixty iterations cannot distinguish
"slow decay" from "decay to a nonzero floor" from "decay plus an unrelated
episode". Longer runs can, and production has already paid for thousands of
iterations of them.

This scans those runs for the shape, at zero node-hours. It is the cheap
first step of task #26; a dedicated long rung is the expensive second step and
should only be spent if this is inconclusive.

WHAT IT REPORTS, AND WHY THESE STATISTICS
------------------------------------------
Per run, per SEGMENT (see below), dataload split into deciles of the segment:
the decile means, plus a floor estimate (the minimum decile mean) and a decay
ratio (first decile / floor). A run that reaches a floor early has a large ratio
and a flat tail of deciles; a run in steady state has a ratio near 1.

⚠️ DECILES RESCALE EACH SEGMENT TO ITS OWN LENGTH, AND THAT MISLED ONCE
-----------------------------------------------------------------------
Segments here span 300-1407 iterations, so decile 9 of the shortest and decile 9
of the longest are ~1100 iterations apart -- yet a median "across segments by
decile" averages them as if they were the same point in a run. Reading the
pooled decile profile as a time course produced a wrong shape:

    reported (deciles):  warmup 1.99 -> 1.29, then a monotone rise to 2.17
                         over "the remaining 90% of the run" (+68%)
    actual (absolute iteration axis, fixed cohort of 39 segments >= 600 iters):
       iters   0-25  25-50  50-100  100-200  200-300  300-400  400-500  500-600
       dload   2.65   1.33    1.24     1.18     1.20     1.21     1.40     1.56

i.e. fall -> FLAT from ~50 to ~400 -> late rise. The rise is real but starts
near iteration 400, not after warmup. It is entirely length-gated: median
d9/d1 is 0.92 for segments of 300-400 iterations (no rise at all), 1.68 for
400-600, and 2.13 for 600+. Short segments simply end inside the flat region.

So: use the decile view to compare WITHIN one segment, and an absolute
iteration axis with a FIXED COHORT to describe a time course across segments.
The fixed cohort matters separately -- late bins otherwise contain only the long
segments, so the sample composition changes as the x-axis advances.

What this did NOT break: the allocation-vs-epoch boundary dissociation in
`dload_rise_boundaries.py` was re-tested on the absolute axis (long segments
only, epoch pairs after iteration 400) and came back SHARPER -- allocation
2.60 -> 1.27 s, ratio 0.41 (37/45 reset); epoch 1.37 -> 1.95 s, ratio 1.38
(only 54/428 reset). Only the timing of the rise was wrong.

Rank 0 only, and here that is a REAL limitation, not a defensible shortcut as it
was in backward_drift_scan.py. Dataload is an order statistic -- rank 0's own
dataload is not the max over ranks, and the max is what the synchronous step
pays. So the ABSOLUTE numbers here understate the tail badly (0.087 s per-rank
mean vs 5.91 s mean-of-max at 16n, a factor of 68). What survives rank-0-only is
the SHAPE: whether a rank's own dataload cost decays, and on what timescale.
Treat a floor found here as evidence about timing, not magnitude, and confirm any
decision on a full max-over-ranks read of a single run.

SEGMENTS, NOT WHOLE FILES
-------------------------
A resumed run's log_r0.csv concatenates many PBS jobs, and each one restarts the
loader -- so each is its own transient. Averaging across them would smear
exactly the structure being looked for. Segment on the repeated CSV header, per
the lesson in backward-degradation-is-the-loader.md: `itr` cycles every ipe and
marks epochs, not allocations.
"""

import argparse
import os
import re
import statistics as st

COL_ITER, COL_DLOAD = 3, 5
WRAP_MS = 343597.0  # XPU event counter period; a negative delta is one wrap


def read_segments(path, min_len):
    """-> [[(iter_ms, dload_ms), ...], ...], one list per allocation."""
    segs, cur = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("epoch,"):
                if len(cur) >= min_len:
                    segs.append(cur)
                cur = []
                continue
            p = line.rstrip("\n").split(",")
            if len(p) <= COL_DLOAD:
                continue
            try:
                it, dl = float(p[COL_ITER]), float(p[COL_DLOAD])
            except ValueError:
                continue
            # Unwrap rather than drop: a wrapped row is a SLOW row, so dropping
            # biases against exactly the expensive iterations that matter.
            if it < 0:
                it += WRAP_MS
            if dl < 0:
                dl += WRAP_MS
            cur.append((it, dl))
    if len(cur) >= min_len:
        segs.append(cur)
    return segs


def deciles(seg):
    n = len(seg) // 10
    return [st.mean(d for _, d in seg[i * n:(i + 1) * n]) / 1000.0
            for i in range(10)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--min-iters", type=int, default=300,
                    help="Minimum iterations per SEGMENT. Below ~300 a decay "
                         "and a floor are not separable -- 16n needed more "
                         "than 60 and the question is how many.")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    rows = []
    for dirpath, _, files in os.walk(a.root):
        if "log_r0.csv" not in files:
            continue
        try:
            segs = read_segments(os.path.join(dirpath, "log_r0.csv"),
                                 a.min_iters)
        except OSError:
            continue
        nranks = len([f for f in files if re.match(r"^log_r\d+\.csv$", f)])
        for si, seg in enumerate(segs):
            d = deciles(seg)
            floor = min(d)
            # Ratio of the opening decile to the floor. Large => the cost the
            # run starts with is not the cost it settles at.
            ratio = d[0] / floor if floor > 1e-6 else float("inf")
            rows.append((os.path.relpath(dirpath, a.root), si, len(seg),
                         nranks, d, floor, ratio))

    if not rows:
        print(f"no segments with >= {a.min_iters} iterations under {a.root}")
        return

    rows.sort(key=lambda r: -r[6])
    print(f"{len(rows)} segments >= {a.min_iters} iters, sorted by decay ratio")
    print("dataload seconds, RANK 0 ONLY -- shape is meaningful, magnitude is not\n")
    hdr = f"{'run':44s} {'seg':>3s} {'its':>5s} {'rk':>4s} " + \
          " ".join(f"d{i}" for i in range(10)) + f" {'floor':>7s} {'d0/fl':>7s}"
    print(hdr)
    for run, si, n, nr, d, floor, ratio in rows[:a.top]:
        ds = " ".join(f"{x:4.1f}" for x in d)
        r = "inf" if ratio == float("inf") else f"{ratio:7.1f}"
        print(f"{run[:44]:44s} {si:3d} {n:5d} {nr:4d} {ds} {floor:7.2f} {r:>7s}")

    # The headline: does the LAST decile look like the first, or like the floor?
    settled = [r for r in rows if r[4][-1] <= 1.5 * r[5]]
    print(f"\n{len(settled)}/{len(rows)} segments END at or near their floor "
          f"(last decile <= 1.5x floor).")
    print("A segment that ends at its floor has CONVERGED -- its steady state is "
          "measurable.\nOne that does not is still in transient at the end of "
          "the run, and any\nefficiency number taken from it is a lower bound.")


if __name__ == "__main__":
    main()
