#!/usr/bin/env python3
"""Is the dataload tail INDEPENDENT PER-RANK DRAWS, or SHARED CONTENTION?

WHY THIS IS THE DECIDING TEST
------------------------------
Two hypotheses survive for the dataload tail (17.2% of wall clock, the top
measured lever, task #14) after shard-opening was refuted:

  A. payload -- a rank occasionally draws a heavy clip. The corpus spreads
     13.1x in MB per sample, so this is a real effect and it does explain the
     BODY: r(mean MB/sample, dload median) = +0.716 across 19 runs.
  B. contention -- ranks periodically wait on a shared resource.

They make OPPOSITE predictions about structure, and structure is free to
measure from logs already on disk:

  A predicts INDEPENDENCE. Each rank draws from the mixture on its own, with
    nothing coupling the draws. Spikes should be uncorrelated in time within a
    rank, and simultaneous spikes across ranks should occur at chance rate.
  B predicts COUPLING. A congested resource stalls whoever is using it, so
    spikes cluster in time and coincide across the ranks that share it.

This distinction matters because it changes the fix completely. Under A the
remedy is corpus-side (normalize bytes per sample at reshard time). Under B
resharding buys nothing and the remedy is concurrency-side (how many ranks hit
the resource at once).

THE THREE TESTS
---------------
1. Within-rank autocorrelation of the spike indicator at lags 1,2,3,5.
   Independent draws => ~0.
2. Cross-rank overdispersion. Count how many ranks spike on the same
   iteration; compare its variance to the Poisson-binomial variance implied by
   each rank's own rate. Ratio 1.0 = independent. >1 = coupled.
3. Same-node vs different-node pairwise lift, P(both) / (P(a)P(b)). This
   LOCATES the shared resource: node-local coupling (CPU oversubscription,
   page cache, node NIC, the node's DAOS agent) shows a higher same-node lift,
   while a cluster-wide resource (DAOS servers, fabric) lifts both equally.

RESULT ON surg_2_1_vitG384_fixedshape (24 ranks x 720 iters, 16 nodes)
-----------------------------------------------------------------------
    within-rank autocorr  : +0.065 / +0.071 / +0.078 / +0.061 (lags 1,2,3,5)
    cross-rank overdisp   : 3.67x, and 24 of 24 sampled ranks spiked together
                            on at least one iteration
    same-node lift 5.31   vs   diff-node lift 2.99

Independent per-rank draws cannot make 24 of 24 ranks stall on the same
iteration. The tail is COUPLED, with a node-local component on top of a
cluster-wide one. Together with the absolute-measure result in
`sample_bytes_correlation.py` -- bytes predict the dload median (+0.716) but
NOT excess seconds (-0.119), p99 (-0.078), or rate over 5 s (-0.391) -- the
division is: payload owns the body, contention owns the tail.

THE CAVEAT THAT MUST TRAVEL WITH TEST 2
----------------------------------------
Ranks are already coupled by the synchronous step: HSDP collectives make every
rank's iteration boundary common, so a stall anywhere delays everyone's NEXT
dataload measurement. That mechanism alone produces cross-rank coincidence
without any shared I/O resource. Test 2 therefore cannot by itself prove
contention -- but tests 1 and 3 are not explained by it: barrier coupling is
uniform across ranks and would NOT produce a same-node lift 1.8x the
cross-node lift. The node structure is the part that carries the argument.

Per [[no-lazy-cause-labels]]: "coupled, with node-local structure" is what the
data shows. WHICH node resource is not identified here, and this script does
not name one.

Usage:
    python scripts/tail_clustering.py --run /flare/.../vitG384_n16g12_weak
    python scripts/tail_clustering.py --run ... --ppn 12 --stride 8
"""
import argparse
import os
import statistics as st

COL_DLOAD = 5
WRAP_MS = 343597.0  # XPU event counter period; a negative delta is one wrap


def rank_series(path, col=COL_DLOAD):
    """Last allocation segment of one rank, in seconds.

    Segment on the repeated CSV HEADER, not on itr decreasing -- itr cycles
    every ipe, so it marks epochs, not allocations.
    """
    seg = []
    for line in open(path):
        if line.startswith("epoch,"):
            seg = []
            continue
        q = line.rstrip("\n").split(",")
        if len(q) <= col:
            continue
        try:
            v = float(q[col])
        except ValueError:
            continue
        if v < 0:
            v += WRAP_MS  # a wrapped row is a SLOW row -- unwrap, never drop
        seg.append(v / 1000.0)
    return seg


def load(run, stride, min_iters, max_ranks):
    out = []
    r = 0
    while len(out) < max_ranks:
        f = os.path.join(run, f"log_r{r}.csv")
        if not os.path.exists(f):
            if r > 4096:
                break
            r += stride
            continue
        s = rank_series(f)
        if len(s) >= min_iters:
            out.append((r, s))
        r += stride
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir containing log_r*.csv")
    ap.add_argument("--factor", type=float, default=3.0)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--ppn", type=int, default=12, help="ranks per node")
    ap.add_argument("--min-iters", type=int, default=300)
    ap.add_argument("--max-ranks", type=int, default=24)
    a = ap.parse_args()

    rs = load(a.run, a.stride, a.min_iters, a.max_ranks)
    if len(rs) < 6:
        print(f"only {len(rs)} usable ranks -- need >=6")
        return
    n = min(len(s) for _, s in rs)
    # Per-rank median, never a global one: ranks differ systematically in their
    # draw, and a global threshold would import between-rank spread into the
    # within-rank spike count.
    B = {}
    for r, s in rs:
        m = st.median(s[:n])
        if m <= 0:
            continue
        B[r] = [1 if v > a.factor * m else 0 for v in s[:n]]
    print(f"{len(B)} ranks x {n} iterations from {os.path.basename(a.run)}")
    print(f"spike = dataload > {a.factor}x that rank's own median\n")

    # --- TEST 1: within-rank temporal clustering -------------------------
    print("TEST 1  within-rank autocorrelation of the spike indicator")
    for lag in (1, 2, 3, 5):
        acs = []
        for b in B.values():
            p = st.mean(b)
            if p <= 0 or p >= 1:
                continue
            cov = sum((b[i] - p) * (b[i + lag] - p)
                      for i in range(len(b) - lag)) / (len(b) - lag)
            acs.append(cov / (p * (1 - p)))
        if acs:
            print(f"  lag {lag}: {st.mean(acs):+.4f}   (0.0 = independent draws)")

    # --- TEST 2: cross-rank coincidence ----------------------------------
    print("\nTEST 2  cross-rank coincidence (shared-resource signature)")
    ks = sorted(B)
    cnt = [sum(B[k][i] for k in ks) for i in range(n)]
    ps = [st.mean(B[k]) for k in ks]
    expvar = sum(p * (1 - p) for p in ps)  # Poisson-binomial, if independent
    obsvar = st.variance(cnt)
    print(f"  mean ranks spiking together : {st.mean(cnt):.3f} of {len(ks)}")
    print(f"  variance observed / independent : {obsvar:.3f} / {expvar:.3f}")
    ratio = obsvar / expvar if expvar > 0 else float("nan")
    print(f"  overdispersion : {ratio:.2f}x   (1.0 = independent)")
    print(f"  max ranks spiking in ONE iteration : {max(cnt)} of {len(ks)}")
    print("  caveat: HSDP already couples iteration boundaries, so this test")
    print("  alone cannot prove contention. Test 3 is the one that does.")

    # --- TEST 3: node-local vs global ------------------------------------
    print("\nTEST 3  same-node vs different-node pairwise lift")
    same, diff = [], []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            x, y = B[ks[i]], B[ks[j]]
            px, py = st.mean(x), st.mean(y)
            if px <= 0 or py <= 0:
                continue
            both = sum(1 for u, v in zip(x, y) if u and v) / len(x)
            lift = both / (px * py)
            (same if ks[i] // a.ppn == ks[j] // a.ppn else diff).append(lift)
    if not same or not diff:
        print(f"  need pairs of both kinds (same={len(same)} diff={len(diff)});")
        print(f"  lower --stride so multiple ranks land on one node")
        return
    print(f"  lift = P(both spike) / (P(a)P(b));  1.0 = independent")
    print(f"  SAME node: n={len(same):3d}  mean {st.mean(same):5.2f}  "
          f"median {st.median(same):5.2f}")
    print(f"  DIFF node: n={len(diff):3d}  mean {st.mean(diff):5.2f}  "
          f"median {st.median(diff):5.2f}")

    print()
    ms, md = st.mean(same), st.mean(diff)
    if ms > 1.5 * md and md > 1.3:
        print("VERDICT: coupled, with BOTH a node-local and a cluster-wide")
        print("component. Independent per-rank payload draws are excluded.")
        print("The node-local excess is not explained by barrier coupling,")
        print("which is uniform across ranks. WHICH node resource is not")
        print("identified here ([[no-lazy-cause-labels]]).")
    elif ms > 1.5 * md:
        print("VERDICT: coupling is NODE-LOCAL. Points at a per-node resource")
        print("(CPU oversubscription, page cache, node NIC, node DAOS agent).")
    elif md > 1.3:
        print("VERDICT: coupling is GLOBAL with no node structure. Consistent")
        print("with a cluster-wide resource -- but also with barrier coupling")
        print("alone, which this cannot separate.")
    else:
        print("VERDICT: no strong coupling. Consistent with independent")
        print("per-rank draws, i.e. the payload hypothesis.")


if __name__ == "__main__":
    main()
