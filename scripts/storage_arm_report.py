#!/usr/bin/env python3
"""Read the four-arm staged-vs-DAOS sweep (task #25) without fooling yourself.

THE SWEEP

    arm 1  n2_nw2                daos-full      production path, OPENING anchor
    arm 2  n2_nw2_staged_cap24   staged-capped  local tmpfs; DAOS agent + NIC out
    arm 3  n2_nw2_cap24          daos-capped    THE CONTROL: arm 2's shard window
    arm 4  n2_nw2_rep2           daos-full      CLOSING anchor = the noise floor

Two contrasts, and neither is arm-2-vs-arm-1:

    arm2 vs arm3   STORAGE PATH at a fixed working set  -> DAOS agent / NIC
    arm3 vs arm1   WORKING SET at a fixed path          -> page cache

Arm 2 against arm 1 moves both at once and cannot separate "DAOS is slow" from
"the working set now fits in cache". That is why arm 3 exists.

WHY THIS SCRIPT AND NOT scaling_efficiency.py

Same phase columns, but the question is different: this compares ARMS AT ONE
NODE COUNT, so the thing that matters is whether an arm's delta clears the
anchor spread. scaling_efficiency.py compares rungs ACROSS node counts and has
no notion of a noise floor.

⚠️ THE THREE WAYS THIS READ GOES WRONG

1. READING THE MEDIAN. The tail is an order statistic; the median prices it
   ~18% cheap and can be flat while the mean doubles
   ([[dataload-tail-survives-at-1-node]]). Both are printed; the wall-clock
   claim is the MEAN.

2. COMPARING ARMS THAT HAVE NOT CONVERGED. This is the big one, and it is the
   defect in the 100-iteration sweep this script was written for. Warmup runs
   ~50-100 iterations (20-iter bin means: 11 -> 6.7 -> 5 -> 4.3 -> 3.0 s), so a
   100-iteration arm never reaches plateau and its mean records where it was
   stopped. Both 2n arms in the archive are still descending at their last
   quartile. `converged` gates the verdict on this.

   Corollary, and the reason the earlier "the mean is irreproducible" reading
   was wrong: once BOTH arms converge and the windows match, a same-node pair
   reproduces to 0.8% in the mean (8741810, 2x280 iters). The 20% and 90%
   spreads previously attributed to intrinsic noise were unequal-window
   artifacts -- one arm's warmup against the other's plateau
   ([[ab-window-truncation-trap]]). Cross-node is a different matter: two long
   1n runs differ 16.5% on a matched post-warmup window.

3. READING A TRUNCATED SWEEP. If the soft-deadline guard dropped arm 4, the
   noise floor is gone and NOTHING is interpretable
   ([[wallclock-kill-deletes-the-closing-anchor]]). This script refuses to
   print a verdict in that case rather than quietly comparing to arm 1 alone.

CONFOUND TO REPORT EITHER WAY: capped arms read a fixed 24-shard window of every
source, so their sampling diversity is not production's. Irrelevant to s/iter
(ladder loss is meaningless anyway) but it belongs in any write-up.
"""

import argparse
import collections
import csv
import glob
import os
import random
import statistics as st

WRAP_MS = 343597.0          # 32-bit XPU event counter, 2**32 * 80 ns
EVENT_COLS = {4, 6, 7, 8, 9, 10}   # only these wrap; 3/5/16 are wall, 17/18 MiB
PHASES = [("dload", 5), ("fwdt", 6), ("fwdc", 7), ("bwd", 8),
          ("opt", 9), ("ema", 10), ("barrier", 16)]
TAIL_S = 1.0                # a dataload above this is the tail firing, not decode

# WARMUP IS ~50-100 ITERATIONS, NOT 30%. Measured across every archived rung as
# 20-iteration bin means (s/iter):
#
#   8741810/n1_nw2       10.03 6.03 4.87 4.31 3.02 2.97 3.12 2.98 3.02 ...
#   8741810/n1_nw2_rep2  10.65 7.39 4.41 3.79 3.55 3.03 3.04 2.99 2.98 ...
#   8741855/n2_nw2       10.97 6.68 5.51 4.56 3.82  <- 100 iters, STILL FALLING
#
# Two long 1n runs reach within 10% of plateau at iteration 49 and 53. A
# 100-iteration arm therefore spends most of its life warming up, and its
# quartile profile is still descending at the last bin.
#
# So this constant is a floor in ITERATIONS, not a fraction: dropping 30% of a
# 100-iteration arm leaves 70 iterations that are still 25% above plateau, and
# an arm-vs-arm comparison then measures which arm got further down the warmup
# curve. Fractional warmup also silently penalises the SHORTER arm, which is
# exactly backwards.
WARMUP_ITERS = 100
WARMUP_FRAC_MIN = 0.30      # ... but never keep more than 70% of a long run

# ⚠️ RETRACTED: "the 2n mean is worth +/-90%".
#
# That came from job 8741663's n2_nw2 (100 iters) vs n2_nw2_rep2 (52 iters),
# read over each rung's own post-warmup window. rep2 was killed mid-warmup --
# its last-20 mean is 7.54 s against its twin's 3.40 s -- so the "noise floor"
# was one arm's warmup compared against the other's plateau, i.e. the window
# truncation trap ([[ab-window-truncation-trap]]) rather than run-to-run noise.
#
# The honest same-node estimate comes from 8741810, where BOTH rungs ran the
# full 280 iterations on one node in one allocation:
#
#   p10 0.2%   median 0.1%   mean 0.8%   backward 0.5%   fwd-context 3.9%
#
# The mean reproduces to UNDER ONE PERCENT once the windows match and both
# arms have converged. The earlier 20% and 90% figures were both artifacts of
# reading unequal windows.
#
# Between DIFFERENT nodes it is larger: the two long 1n runs differ 16.5% over
# a matched itr>=100 window (3.63 vs 3.03 s), which is the fwd-context episode
# phenomenon in 8741769 and remains genuinely unexplained.
PRIOR_FLOOR_SAME_NODE = 0.008
PRIOR_FLOOR_CROSS_NODE = 0.165

ARMS = [
    ("n2_nw2", "daos-full", "opening anchor"),
    ("n2_nw2_staged_cap24", "staged-cap24", "tmpfs; agent+NIC out"),
    ("n2_nw2_cap24", "daos-cap24", "THE CONTROL"),
    ("n2_nw2_rep2", "daos-full", "closing anchor"),
]


def unwrap(v, col):
    """A negative event delta is a counter wrap. It is a SLOW row -- unwrap it.

    Dropping these biased earlier phase tables low, because the wrapped rows are
    exactly the expensive ones.
    """
    return v + WRAP_MS if (col in EVENT_COLS and v < 0) else v


def read_arm(d):
    """-> {(epoch, itr): {rank: {phase: seconds}}} over the LAST segment only.

    A rung dir can hold several allocations' CSVs appended together; only the
    final segment is one continuous run, and `itr` cycles every ipe so it cannot
    be used to find the boundary. Key on (epoch, itr) and take the last run of
    monotonically non-decreasing keys.
    """
    per = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(d, "*log_r*.csv"))):
        base = os.path.basename(f)
        try:
            rank = int(base.split("log_r")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        rows = []
        with open(f) as fh:
            for r in csv.reader(fh):
                if not r or not r[0].strip().lstrip("-").isdigit():
                    continue        # repeated header = a new allocation
                if len(r) <= 16:
                    continue
                rows.append(r)
        for r in rows:
            try:
                key = (int(r[0]), int(r[1]))
                rec = {"iter": unwrap(float(r[3]), 3) / 1000.0}
                for name, col in PHASES:
                    rec[name] = unwrap(float(r[col]), col) / 1000.0
                rec["avail"] = float(r[17]) / 1024.0 if len(r) > 17 else 0.0
            except (ValueError, IndexError):
                continue
            per[key][rank] = rec
    return per


def boot_ci(xs, stat=st.mean, iters=2000, lo=2.5, hi=97.5, seed=12345):
    """Percentile bootstrap CI. Deterministic seed so a re-run reproduces.

    The mean of a heavy-tailed sample over ~70 iterations carries far more
    uncertainty than a point estimate suggests: two identical back-to-back 2n
    rungs differed 89.7% in the mean. Resampling iterations prices that
    directly, so an arm comparison can be judged on overlap instead of on the
    difference of two noisy points.

    Caveat this does NOT cover: resampling iterations treats them as
    exchangeable, but the tail is CLUSTERED in time. That makes this CI an
    UNDER-estimate of the true run-to-run spread -- it is a lower bound on the
    uncertainty, which is why the empirical anchor pair is still reported.
    """
    n = len(xs)
    if n < 8:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    vals = sorted(stat([xs[rnd.randrange(n)] for _ in range(n)])
                  for _ in range(iters))
    return (vals[int(lo / 100 * iters)], vals[min(int(hi / 100 * iters), iters - 1)])


def summarize(per, label):
    if not per:
        return None
    nr = max(len(v) for v in per.values())
    keys = sorted(k for k, v in per.items() if len(v) == nr)
    if len(keys) < 20:
        return dict(label=label, ranks=nr, n=len(keys), partial=True)
    # Drop WARMUP_ITERS absolutely, but always keep >=20 iterations so a short
    # arm still reports something (flagged un-converged rather than dropped).
    cut = max(int(WARMUP_FRAC_MIN * len(keys)),
              min(WARMUP_ITERS, len(keys) - 20))
    post = keys[cut:]

    # max-over-ranks: the iteration waits for the slowest rank, so this is the
    # only statistic that prices what the step actually cost.
    it = sorted(max(r["iter"] for r in per[k].values()) for k in post)
    out = dict(label=label, ranks=nr, n=len(post), partial=False,
               p10=it[len(it) // 10], med=st.median(it), mean=st.mean(it))
    out["mean_ci"] = boot_ci(it)
    for name, _ in PHASES:
        series = [max(r[name] for r in per[k].values()) for k in post]
        out[name] = st.mean(series)
        out[name + "_rk"] = st.mean(
            [st.median([r[name] for r in per[k].values()]) for k in post])
    # Tail FRACTION, not tail magnitude. How often the tail fires is a Bernoulli
    # rate with a well-behaved CI over ~70 iterations, whereas its magnitude is
    # heavy-tailed and barely estimable at this n. If a storage change removes
    # the tail, this moves and it moves measurably.
    dl = [max(r["dload"] for r in per[k].values()) for k in post]
    out["dl_hit"] = sum(1 for v in dl if v > TAIL_S) / len(dl)
    out["dl_hit_ci"] = boot_ci(dl, stat=lambda s: sum(1 for v in s if v > TAIL_S) / len(s))

    # CONVERGENCE, in time order -- the thing that actually decides whether this
    # arm's mean means anything. If the last quarter is still materially below
    # the previous one, the arm is on the warmup curve and its mean is a
    # property of where it was stopped, not of the condition under test.
    ordered = [max(r["iter"] for r in per[k].values()) for k in post]
    q = max(1, len(ordered) // 4)
    out["q3"], out["q4"] = st.mean(ordered[-2 * q:-q]), st.mean(ordered[-q:])
    out["converged"] = out["q4"] >= 0.95 * out["q3"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="job dir, e.g. .../scaling_ladder/8741855")
    a = ap.parse_args()

    res = {}
    for d, kind, note in ARMS:
        p = os.path.join(a.root, d)
        if not os.path.isdir(p):
            print(f"  MISSING arm: {d} ({kind}, {note})")
            continue
        res[d] = summarize(read_arm(p), d)

    print(f"\n{a.root}")
    print(f"{'arm':22s}{'kind':14s}{'rk':>4s}{'n':>5s}"
          f"{'p10':>7s}{'med':>7s}{'mean':>7s}{'mean 95% CI':>16s}"
          f"{'dload':>8s}{'tail%':>7s}")
    for d, kind, _ in ARMS:
        r = res.get(d)
        if r is None:
            continue
        if r.get("partial"):
            print(f"{d:22s}{kind:14s}{r['ranks']:4d}{r['n']:5d}"
                  f"   -- too few fully-covered iters to summarize --")
            continue
        ci = r["mean_ci"]
        print(f"{d:22s}{kind:14s}{r['ranks']:4d}{r['n']:5d}"
              f"{r['p10']:7.2f}{r['med']:7.2f}{r['mean']:7.2f}"
              f"{'[%.2f,%.2f]' % ci:>16s}"
              f"{r['dload']:8.2f}{100*r['dl_hit']:7.1f}")

    ok = {d: r for d, r in res.items() if r and not r.get("partial")}

    # ---- the noise floor, before any comparison
    a1, a4 = ok.get("n2_nw2"), ok.get("n2_nw2_rep2")
    if not (a1 and a4):
        print("\n  NO CLOSING ANCHOR -> NO VERDICT.")
        print("  Both daos-full arms are required. The closing anchor is what")
        print("  separates a real arm effect from allocation drift, and it is also")
        print("  the only check that the arms in between were converged and")
        print("  comparable at all -- arm 4 runs last, so if it lands on arm 1 the")
        print("  whole sweep held still. Comparing to a single anchor cannot.")
        return

    floor = abs(a4["mean"] - a1["mean"]) / a1["mean"]
    print(f"\n  NOISE FLOOR |arm1 - arm4| = {100*floor:.1f}% of mean "
          f"({a1['mean']:.2f} vs {a4['mean']:.2f} s)")
    print(f"  Reference: a converged same-node pair (8741810, 2x280 iters) "
          f"reproduces to {100*PRIOR_FLOOR_SAME_NODE:.1f}%;")
    print(f"  two long 1n runs on DIFFERENT nodes differ "
          f"{100*PRIOR_FLOOR_CROSS_NODE:.1f}% on a matched window.")
    print("  All four arms here share one allocation, so the same-node figure is")
    print("  the relevant one -- but only for arms that CONVERGED.")

    # ---- the two contrasts that actually separate the hypotheses
    a2, a3 = ok.get("n2_nw2_staged_cap24"), ok.get("n2_nw2_cap24")
    anchor = (a1["mean"] + a4["mean"]) / 2
    bar = max(floor, PRIOR_FLOOR_SAME_NODE)

    unconv = [d for d, r in ok.items() if not r["converged"]]
    if unconv:
        print(f"\n  ⚠️  NOT CONVERGED: {', '.join(unconv)}")
        print("  Warmup runs ~50-100 iterations (11 -> 6.7 -> 5 -> 4.3 -> 3.0 s in")
        print("  20-iter bins). An arm still descending at its last quartile has a")
        print("  mean set by where it stopped, not by the condition under test, and")
        print("  comparing two such arms measures which got further down the curve.")
        print("  Treat every number below as an UPPER BOUND on that arm's cost.")

    def verdict(x, y, lo, hi, owner):
        if not (x and y):
            print(f"\n  {lo} vs {hi}: arm missing -> not evaluable")
            return
        d = (y["mean"] - x["mean"]) / x["mean"]
        overlap = not (x["mean_ci"][1] < y["mean_ci"][0]
                       or y["mean_ci"][1] < x["mean_ci"][0])
        if not (x["converged"] and y["converged"]):
            tag = "UNREADABLE (an arm is still on the warmup curve)"
        elif abs(d) <= bar or overlap:
            why = "CI overlap" if overlap else "inside the anchor spread"
            tag = f"NULL ({why})"
        else:
            tag = f"SIGNAL -> {owner}"
        print(f"\n  {lo} vs {hi}: {x['mean']:.2f} -> {y['mean']:.2f} s "
              f"({100*d:+.1f}%, bar {100*bar:.1f}%)   {tag}")
        print(f"      q3->q4   {x['q3']:.2f}->{x['q4']:.2f}"
              f"  vs  {y['q3']:.2f}->{y['q4']:.2f}"
              f"   (converged: {x['converged']}, {y['converged']})")
        print(f"      mean CI  {x['mean_ci'][0]:.2f}-{x['mean_ci'][1]:.2f}"
              f"  vs  {y['mean_ci'][0]:.2f}-{y['mean_ci'][1]:.2f}")
        print(f"      dload    {x['dload']:.2f} -> {y['dload']:.2f} s "
              f"(per-rank median {x['dload_rk']:.2f} -> {y['dload_rk']:.2f})")
        # The rate is the better-powered statistic: a Bernoulli over ~70 iters
        # beats the mean of a heavy tail. If the tail rate moves and the mean
        # does not, believe the rate and say the mean lacks the power.
        print(f"      tail rate {100*x['dl_hit']:.1f}% "
              f"[{100*x['dl_hit_ci'][0]:.0f}-{100*x['dl_hit_ci'][1]:.0f}]"
              f" -> {100*y['dl_hit']:.1f}% "
              f"[{100*y['dl_hit_ci'][0]:.0f}-{100*y['dl_hit_ci'][1]:.0f}]"
              f"   (dload > {TAIL_S:.0f}s)")

    verdict(a3, a2, "arm3 daos-capped", "arm2 staged-capped",
            "the storage path: DAOS agent / NIC")
    verdict(a1, a3, "arm1 daos-full", "arm3 daos-capped",
            "the working set: page cache")

    print(f"\n  anchor mean {anchor:.2f} s")
    print("  Report the MEAN for cost, the TAIL RATE for whether the tail moved.")
    print("  The median prices the tail ~18% cheap; the per-rank median in the")
    print("  dload line separates a real cost (every rank slower) from an order")
    print("  statistic (only the max).")
    print("\n  CONFOUND: capped arms read a fixed 24-shard window per source, so")
    print("  their sampling diversity is not production's. State it either way.")
    print("  POWER: a converged same-node pair reproduces to <1%, so a converged")
    print("  arm pair here can resolve a small effect. An UN-converged pair can")
    print("  resolve nothing. A null bounds the effect; it does not exclude one.")


if __name__ == "__main__":
    main()
