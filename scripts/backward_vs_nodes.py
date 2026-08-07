#!/usr/bin/env python3
"""Is the HSDP inter-node gradient all-reduce BANDWIDTH- or LATENCY-bound?

WHY THIS EXISTS
---------------
64n runs at 63% per-tile efficiency and the loss splits into two roughly equal
halves: a dataload straggler tail, and ~1.15 s of growth in `backward` that
every rank pays (median-over-ranks, so it is real per-rank work, not skew).

The second half was measured only as 1n -> 64n, which cannot distinguish two
situations that imply OPPOSITE priorities:

  * one-time  -- at 1n there is no replicate mesh dim at all, so *any* inter-node
    collective appearing costs something. If that is the whole story, 256n
    inherits the same floor and the tail is the entire remaining job.
  * grows     -- if the term scales with node count it dominates at 3072 ranks
    and the tail is secondary.

16n is the discriminating point, which is why the ladder runs 1/16/64 in ONE
allocation on identical config.

THE MODEL
---------
CCL_ALLREDUCE=ring, and a ring all-reduce over N participants costs

    T(N) ~= 2(N-1)/N * S/B     (bandwidth term -- SATURATES)
          + 2(N-1) * alpha     (latency term   -- LINEAR in N)

Those two behave completely differently between 16 and 64 nodes, which is what
makes one intermediate rung decisive:

    2(N-1)/N : 1.875 at 16n vs 1.969 at 64n  -> only +5%, essentially flat
    2(N-1)   :   30   at 16n vs   126  at 64n -> 4.2x

So under a pure-bandwidth story the 16n backward should already sit at ~the 64n
value; under a pure-latency story it should sit near a quarter of the way up.
Reporting the mixture explicitly (what fraction of the inter-node term is
already paid at 16n) is more honest than forcing a binary label.

  NOTE: an earlier draft of this analysis predicted a "log in nodes" midpoint.
  That is the wrong model for a RING allreduce (log is tree / recursive
  halving) and is retracted. Do not score against it.

WHAT IT READS, AND WHY THAT REDUCTION
-------------------------------------
median-OVER-RANKS backward, then median over the window. NOT max-over-ranks.
Max is right for "what does the synchronous step wait for" and wrong for "what
does a rank pay" -- and on this exact data the two disagree about a headline:
max-over-ranks fwd-context rises 1.07 -> 1.83 s at 64n while median-over-ranks
is flat 1.03 -> 1.01, because forward's first FSDP all-gather absorbs upstream
dataload skew. That is the long-standing "64n forward blowup", and it is not
real. A comms claim needs growth in the MEDIAN.

1n IS A FLOOR, NOT A TERM IN THE GROWTH
---------------------------------------
Under _HYBRID_SHARD_ZERO2 with top-level wrap (app/vjepa_2_1/hsdp.py) the
gradient path is a ReduceScatter inside the 12-tile shard group, then an
all-reduce across the replicate group. At 1 node the replicate group has size 1,
so only the intra-node leg runs. 1n therefore anchors "no inter-node collective"
and the inter-node SCALING must be read from 16n vs 64n against each other.

Usage:
    python scripts/backward_vs_nodes.py --ladder <outroot> [--lo 7] [--hi 50]
"""
import argparse
import os
import re
import statistics as st

# Discard iteration 0 by default: it carries loader spin-up, and at 256n it has
# logged a negative XPU event delta (32-bit counter wrap) so it is never valid.
COL_ITER, COL_DLOAD, COL_BWD = 3, 5, 8
WRAP_MS = 343597.0  # XPU event counter period; a negative delta is one wrap


def read_run(d):
    """-> {(epoch, itr): {col: [value per rank]}}, n_rank_files"""
    per, n = {}, 0
    for fn in sorted(os.listdir(d)):
        if not re.match(r"^log_r\d+\.csv$", fn):
            continue
        n += 1
        with open(os.path.join(d, fn)) as fh:
            next(fh, None)
            for line in fh:
                p = line.rstrip("\n").split(",")
                if len(p) < 9:
                    continue
                try:
                    key = (int(p[0]), int(p[1]))
                    for c in (COL_ITER, COL_DLOAD, COL_BWD):
                        v = float(p[c])
                        # Unwrap rather than drop: those rows carry the slow
                        # tail, so dropping them biases the table low.
                        if v < 0:
                            v += WRAP_MS
                        per.setdefault(key, {}).setdefault(c, []).append(v)
                except ValueError:
                    continue
    return per, n


def summarize(d, ranks, lo, hi):
    per, nfiles = read_run(d)
    # Full coverage only. Sampling a fixed NUMBER of ranks is what once made a
    # 256n run look superlinear; requiring every rank is the guard.
    keys = sorted(k for k in per
                  if lo <= k[1] < hi and len(per[k].get(COL_BWD, [])) >= ranks)
    if not keys:
        return None
    series = [st.median(per[k][COL_BWD]) / 1000.0 for k in keys]
    bwd_med = st.median(series)
    bwd_max = st.median([max(per[k][COL_BWD]) for k in keys]) / 1000.0
    it_max = [max(per[k][COL_ITER]) / 1000.0 for k in keys]

    # Is that median a steady-state value or the midpoint of a ramp? Job
    # 8741386's 16n rung read 2.46 s over the window while actually going 1.64 ->
    # 7.43 s across it: the median described where the run was cut off, not what
    # a step costs. Both the ring model below and every efficiency number assume
    # a STATIONARY per-step cost, so quoting one over a drifting series fits the
    # drift. Compare first vs last quarter and refuse the verdict if they differ.
    q = max(1, len(series) // 4)
    first_q, last_q = st.median(series[:q]), st.median(series[-q:])
    drift = last_q / first_q if first_q > 0 else 1.0

    # The min ACROSS RANKS separates "every rank got slower" from "ranks are
    # waiting on each other". If min stays at the clean floor while the median
    # climbs, the extra time is block-in-collective, not extra work -- that is
    # what 16n showed (min 1.5 s throughout, median to 12.8 s).
    min_last = st.median([min(per[k][COL_BWD]) / 1000.0 for k in keys[-q:]])
    return dict(n=len(keys), csv=nfiles, bwd_med=bwd_med, bwd_max=bwd_max,
                iter_med=st.median(it_max), iter_mean=st.mean(it_max),
                first_q=first_q, last_q=last_q, drift=drift, min_last=min_last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--lo", type=int, default=7)
    ap.add_argument("--hi", type=int, default=50)
    a = ap.parse_args()

    rows = []
    for name in sorted(os.listdir(a.ladder)):
        m = re.match(r"^n(\d+)_nw(\d+)$", name)
        if not m:
            continue
        nodes = int(m.group(1))
        s = summarize(os.path.join(a.ladder, name), nodes * 12, a.lo, a.hi)
        if s:
            rows.append((name, nodes, s))
    rows.sort(key=lambda r: r[1])
    if not rows:
        print("no usable rungs")
        return

    print(f"window itr {a.lo}-{a.hi - 1}, full rank coverage only\n")
    hdr = (f"{'rung':14s} {'nodes':>5s} {'csv':>5s} {'n':>3s} "
           f"{'bwd med-o-r':>11s} {'bwd max-o-r':>11s} {'iter med':>9s} {'iter mean':>9s} "
           f"{'Q1->Q4':>14s} {'drift':>6s} {'min@Q4':>7s}")
    print(hdr)
    print("-" * len(hdr))
    drifting = []
    for name, nodes, s in rows:
        print(f"{name:14s} {nodes:5d} {s['csv']:5d} {s['n']:3d} "
              f"{s['bwd_med']:11.2f} {s['bwd_max']:11.2f} "
              f"{s['iter_med']:9.2f} {s['iter_mean']:9.2f} "
              f"{s['first_q']:6.2f}->{s['last_q']:6.2f} {s['drift']:6.2f} "
              f"{s['min_last']:7.2f}")
        if s["drift"] > 1.25:
            drifting.append((name, s))

    if drifting:
        print("\n*** NOT STEADY STATE -- the window median is not a per-step cost ***")
        for name, s in drifting:
            shape = ("min-over-ranks stayed at the floor => ranks are WAITING in "
                     "the collective, not doing more work"
                     if s["min_last"] < 1.25 * s["first_q"]
                     else "min-over-ranks rose too => every rank really is slower")
            print(f"  {name}: backward {s['first_q']:.2f} -> {s['last_q']:.2f} s "
                  f"({s['drift']:.2f}x) across the window; min@Q4 {s['min_last']:.2f} s")
            print(f"      {shape}")
        print("  A drifting series has no single value to feed the ring model, so the")
        print("  bandwidth-vs-latency verdict below is WITHHELD. Characterize the drift")
        print("  first (does onset track iteration index or elapsed seconds?).")
        return

    by_n = {nodes: s for _, nodes, s in rows}
    if not {1, 16, 64} <= by_n.keys():
        print("\nneed rungs at 1, 16 and 64 nodes to run the verdict")
        return

    b1, b16, b64 = (by_n[k]["bwd_med"] for k in (1, 16, 64))
    inter64 = b64 - b1          # the whole inter-node term at 64n
    inter16 = b16 - b1
    if inter64 <= 0:
        print("\nno inter-node growth at 64n -- the premise does not hold")
        return
    frac = inter16 / inter64    # how much of it is already paid at 16n

    # Predicted fraction-of-64n-term under each pure regime.
    f_bw = (2 * 15 / 16) / (2 * 63 / 64)   # 0.952
    f_lat = 15.0 / 63.0                    # 0.238

    print(f"\ninter-node term (median-over-ranks backward, 1n = floor {b1:.2f} s):")
    print(f"  at 16n: {inter16:+.2f} s     at 64n: {inter64:+.2f} s")
    print(f"  fraction of the 64n term already paid at 16n: {frac:.2f}")
    print(f"    pure BANDWIDTH predicts {f_bw:.2f}  (term saturates; ~one-time)")
    print(f"    pure LATENCY   predicts {f_lat:.2f}  (term linear in N)")

    # Interpolate a mixture rather than forcing a binary label.
    if f_bw - f_lat > 1e-6:
        w_bw = max(0.0, min(1.0, (frac - f_lat) / (f_bw - f_lat)))
        print(f"  => implied split: {w_bw * 100:.0f}% bandwidth / "
              f"{(1 - w_bw) * 100:.0f}% latency")
        # Extrapolate to 256n under the fitted mixture.
        S = inter64 / (w_bw * (2 * 63 / 64) + (1 - w_bw) * 2 * 63)
        p256 = S * (w_bw * (2 * 255 / 256) + (1 - w_bw) * 2 * 255)
        print(f"  => extrapolated inter-node term at 256n: {p256:+.2f} s "
              f"(backward ~{b1 + p256:.2f} s)")
        print("     Extrapolation, not a measurement -- one rung beyond the data,")
        print("     and it assumes the ring topology and per-step byte count hold.")

    print("\nRead this against the pre-registered predictions in task #20. If frac is")
    print("near 0.95 the cost is effectively ONE-TIME: 256n inherits ~the same floor")
    print("and the dataload tail is the whole remaining job. If near 0.24 it is")
    print("LATENCY-bound and grows; the only untried lever is cutting collective")
    print("FREQUENCY (grad accumulation), and note bs=2 already amortizes one")
    print("collective per 2 clips, so accum on top only helps by doubling global")
    print("batch -- a schedule cost, not a free win.")


if __name__ == "__main__":
    main()
