#!/usr/bin/env python3
"""Compare the dataload TAIL across matched arms in one allocation.

WHY A SEPARATE READER
---------------------
`scaling_efficiency.py` answers "how fast is this rung" -- median max-over-ranks
iteration time and a phase breakdown. That is the right tool for a scaling
ladder and the wrong one for a tail study, because the tail is a property of the
*distribution*, and the median deliberately prices it at zero.

This script answers the one question task #24 asks: does knob X move the tail?
It reports the three ABSOLUTE tail measures and nothing relative, for the reason
recorded in [[dataload-body-vs-tail-owners]]:

    A per-run spike rate defined as "> 3x that run's own median" is contaminated
    by construction. Across 19 production runs r(dload_median, spike_rate) was
    -0.762: a run with a heavier median clears its own threshold LESS often for
    the same absolute stall, so the statistic partly measures 1/median -- the
    threshold moving, not the physics. Any arm-to-arm tail claim here must rest
    on measures whose definition does not move between arms.

The three that do not move:

    excess   mean seconds per iteration spent above that rank's own median,
             counting only iterations above it. This is the wall-clock cost of
             the tail, in seconds. (The per-rank median is a CENTERING choice,
             not a threshold: ranks differ systematically in their draw, and
             centering per rank keeps between-rank spread out of the number.
             The quantity summed is absolute seconds either way.)
    p99      99th percentile of dataload, in seconds.
    rate>T   fraction of iterations over a fixed threshold shared by ALL arms.

WITH nw>0 THE BODY COLUMN GOES TO ZERO, AND `excess` CHANGES MEANING
---------------------------------------------------------------------
Prefetch makes the dataload distribution zero-inflated: most iterations record
exactly 0.00 s because the next batch was already resident, and the rest record
a real stall. The per-rank median is then 0, so `BODY` reads 0.00 on every nw=2
arm and carries no information, and `excess` -- seconds above the median --
collapses to the plain MEAN dataload.

That is still an absolute, arm-comparable measure of what the loader costs, and
it is the right one under prefetch: when the body is free, the whole cost IS
the tail. But it is not the same statistic as `excess` on an nw=0 run, so do not
put the two in one table. Compare nw=2 arms to nw=2 arms.

BODY IS REPORTED TOO, AND SEPARATELY
------------------------------------
The dataload distribution has two owners with different fixes: payload size owns
the body (r = +0.716 vs mean MB/sample across 19 runs), coupled node-local
contention owns the tail (payload predicts excess at -0.119). An arm that moves
the median and not the excess has moved the body, which is a real win for a
corpus change and NOT evidence about the tail. Keeping the two columns adjacent
is what stops one from being read as the other.

READ max-OVER-RANKS FOR WALL CLOCK, per-rank FOR COST
------------------------------------------------------
The synchronous step waits for the slowest rank, so `dl_max_mean` -- the mean
over iterations of the max over ranks -- is what the schedule actually pays. The
per-rank columns say whether a typical rank got faster or whether only the
order statistic moved. An arm can improve one and not the other, and which one
moved determines whether the knob is worth promoting.

ANCHOR DISCIPLINE
-----------------
Arms are matched only if the study repeated its baseline. If two arms share a
base name (the second gets `_rep2` from scaling_ladder.sh), this prints their
spread FIRST and refuses to rank the middle arms when the bracketing pair
disagrees by more than the effect being claimed. That check exists because two
identical 1n rungs once differed 2x ([[1n-anchor-does-not-reproduce]]).

Usage:
    python scripts/tail_arm_compare.py --ladder /flare/.../scaling_ladder/8741645
    python scripts/tail_arm_compare.py --ladder ... --lo 10 --thresh 5.0
"""
import argparse
import glob
import os
import re
import statistics as st

COL_ITER, COL_DLOAD = 3, 5
WRAP_MS = 2**32 * 80e-9 * 1000.0  # XPU 32-bit event counter period, 343597 ms


def read_rung(d, lo, hi):
    """-> (per_rank, dmax, imax, n_files, last_covered_iter).

    per_rank is {rank: [dload seconds]}; dmax/imax are max-over-ranks series.

    Windowed on iteration index, and restricted to iterations where EVERY rank
    logged -- a max over a partial rank set understates the max, which is
    precisely how an earlier scaling table came to be wrong.

    `hi` alone does NOT give a common window: it bounds every arm from above but
    a truncated arm still ends early, so arms of unequal length are still
    compared over unequal spans. That matters because dataload is not
    stationary within an arm (job 8741663: 2.47 s -> 0.04 s across one arm), so
    the caller needs the last fully-covered iteration to truncate everyone to
    the shortest. It is returned rather than inferred from len(dmax), which
    counts covered iterations, not the span they cover.
    """
    per_rank, by_iter = {}, {}
    files = sorted(glob.glob(os.path.join(d, "log_r*.csv")))
    for f in files:
        m = re.search(r"log_r(\d+)\.csv$", f)
        if not m:
            continue
        r = int(m.group(1))
        try:
            lines = open(f).read().splitlines()
        except OSError:
            continue
        for ln in lines:
            if not ln or ln.startswith("epoch,"):
                continue
            p = ln.split(",")
            try:
                it = int(p[1])
                itr_s = float(p[COL_ITER]) / 1000.0
                dl = float(p[COL_DLOAD])
            except (ValueError, IndexError):
                continue
            if not (lo <= it < hi) or itr_s <= 0:
                continue
            # A negative delta is one counter wrap, i.e. a SLOW row. Unwrap it;
            # dropping it would silently delete exactly the tail being measured.
            if dl < 0:
                dl += WRAP_MS
            per_rank.setdefault(r, []).append(dl / 1000.0)
            slot = by_iter.setdefault(it, ([], []))
            slot[0].append(dl / 1000.0)
            slot[1].append(itr_s)
    n = len(files)
    covered = [k for k in sorted(by_iter) if len(by_iter[k][0]) >= n]
    dmax = [max(by_iter[k][0]) for k in covered]
    imax = [max(by_iter[k][1]) for k in covered]
    return per_rank, dmax, imax, n, (covered[-1] if covered else None)


def stats(per_rank, thresh):
    """Absolute tail measures, averaged over ranks."""
    med, exc, p99, rate = [], [], [], []
    for s in per_rank.values():
        if len(s) < 8:
            continue
        m = st.median(s)
        med.append(m)
        exc.append(sum(v - m for v in s if v > m) / len(s))
        srt = sorted(s)
        p99.append(srt[min(len(srt) - 1, int(0.99 * len(srt)))])
        rate.append(sum(1 for v in s if v > thresh) / len(s))
    if not med:
        return None
    return dict(n_ranks=len(med), med=st.mean(med), excess=st.mean(exc),
                p99=st.mean(p99), rate=st.mean(rate))


def base_name(name):
    return re.sub(r"_rep\d+$", "", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True, help="scaling_ladder.sh output root")
    ap.add_argument("--lo", type=int, default=10, help="first iteration (drop warmup)")
    ap.add_argument("--hi", type=int, default=10**6)
    ap.add_argument("--thresh", type=float, default=5.0,
                    help="fixed absolute threshold, SHARED by all arms")
    ap.add_argument("--no-common-window", action="store_true",
                    help="compare arms over their full lengths even when those "
                         "differ. Off by default -- see below.")
    a = ap.parse_args()

    # --- COMMON WINDOW ----------------------------------------------------
    # Truncate every arm to the shortest arm's last iteration. Dataload is not
    # stationary within an arm: at 2n it decays from 2.47 s to 0.04 s
    # (max-over-ranks, first quarter to last, job 8741663), so a mean over 93
    # iterations and a mean over 45 iterations of the SAME arm are different
    # numbers. Comparing arms of unequal length therefore reports a difference
    # in window as a difference in configuration.
    #
    # This is not hypothetical -- it is why 8741663's closing anchor read as
    # "did not reproduce, 1.75x". A wall-clock kill truncated it to 45 of 100
    # iterations; on the common window the same pair is 1.45x, and on dataload
    # alone 1.17x. The floor was mostly an artifact of this reader.
    hi = a.hi
    if not a.no_common_window:
        last = []
        for name in sorted(os.listdir(a.ladder)):
            d = os.path.join(a.ladder, name)
            if not os.path.isdir(d) or not glob.glob(os.path.join(d, "log_r*.csv")):
                continue
            _, _, _, _, mx = read_rung(d, a.lo, a.hi)
            if mx is not None:
                last.append(mx)
        if last and min(last) + 1 < hi:
            hi = min(last) + 1

    arms = []
    for name in sorted(os.listdir(a.ladder)):
        d = os.path.join(a.ladder, name)
        if not os.path.isdir(d) or not glob.glob(os.path.join(d, "log_r*.csv")):
            continue
        pr, dmax, imax, nf, _ = read_rung(d, a.lo, hi)
        s = stats(pr, a.thresh)
        if not s:
            continue
        # Run order, for the bracketing check below. Rungs are serial within an
        # allocation, so the first rank CSV's mtime orders them; the directory's
        # own mtime does not, since a later checkpoint write touches it.
        csvs = glob.glob(os.path.join(d, "log_r*.csv"))
        t0 = min((os.path.getmtime(c) for c in csvs), default=0.0)
        s.update(name=name, files=nf, iters=len(dmax), t0=t0,
                 dl_max_mean=st.mean(dmax) if dmax else float("nan"),
                 iter_mean=st.mean(imax) if imax else float("nan"),
                 iter_med=st.median(imax) if imax else float("nan"))
        arms.append(s)

    if not arms:
        print(f"no usable rung dirs under {a.ladder}")
        return

    if hi != a.hi:
        print(f"COMMON WINDOW APPLIED: truncated to [{a.lo}, {hi}) because the "
              f"shortest arm ends there.")
        print("  Arms are compared over equal spans. Dataload is not stationary "
              "within an arm,\n  so unequal spans report a window difference as "
              "a configuration difference.\n  Pass --no-common-window to see the "
              "full-length (non-comparable) numbers.")
    print(f"window: iterations [{a.lo}, {hi}), full rank coverage only")
    print(f"fixed threshold for rate: {a.thresh:.1f} s\n")
    print(f"{'arm':<34} {'rks':>4} {'its':>4} | {'iter_mean':>9} {'iter_med':>8} "
          f"{'gap':>6} | {'dl_max':>7} {'gap/dl':>7} | {'BODY':>6} {'excess':>7} "
          f"{'p99':>6} {'rate':>6}")
    print("-" * 132)
    for s in arms:
        gap = s["iter_mean"] - s["iter_med"]
        gd = gap / s["dl_max_mean"] if s["dl_max_mean"] else float("nan")
        print(f"{s['name']:<34} {s['n_ranks']:>4} {s['iters']:>4} | "
              f"{s['iter_mean']:>9.2f} {s['iter_med']:>8.2f} {gap:>6.2f} | "
              f"{s['dl_max_mean']:>7.2f} {gd:>7.2f} | "
              f"{s['med']:>6.2f} {s['excess']:>7.3f} {s['p99']:>6.2f} "
              f"{s['rate']:>6.3f}")
    print()
    print("gap = iter_mean - iter_med, the part of wall-clock the median hides.")
    print("gap/dl near 1.0 means the dataload order statistic accounts for ALL")
    print("of it -- no other phase contributes a tail. Note this is close to an")
    print("identity when dataload is zero-inflated (median iter ~ a clean step,")
    print("mean iter ~ clean step + mean dl_max), so it is not a discovery. It")
    print("is a BOUND, and that is why it is here: removing the dataload tail")
    print("entirely would move iter_mean down to iter_med and no further.")

    # --- anchor check, before any ranking ---------------------------------
    arms_by_time = sorted(arms, key=lambda x: x["t0"])
    groups = {}
    for s in arms:
        groups.setdefault(base_name(s["name"]), []).append(s)
    reps = {k: v for k, v in groups.items() if len(v) > 1}
    print()
    if not reps:
        print("NO REPEATED ARM. This study cannot detect drift within the")
        print("allocation, so any delta below is unattributable in principle.")
        print("Bracket the sweep with its baseline (A B C A) and re-run.")
        return

    # WHICH repeated group is the anchor. Under A/B/B/A -- the layout this
    # study uses so the treatment's own repeat spread is measured too -- BOTH
    # groups have two members, so "the most-repeated group" is a coin flip and
    # a treatment pair could silently become the baseline. The anchor is the
    # group that BRACKETS the allocation: it holds both the first and the last
    # arm in run order. That is what makes it a drift measurement.
    first, last = arms_by_time[0], arms_by_time[-1]
    anchor = None
    if base_name(first["name"]) == base_name(last["name"]):
        anchor = reps.get(base_name(first["name"]))
    if anchor is None:
        anchor = max(reps.values(), key=len)
        print("NOTE: no arm brackets the allocation (first and last rungs "
              "differ),\n  so the anchor below is the most-repeated group and "
              "measures repeat spread\n  rather than start-to-end drift. "
              "Bracket the sweep A/B/../A to get both.")
    a_nm = base_name(anchor[0]["name"])

    # The anchor's spread is a VETO; every other repeated group's spread is
    # information that widens the floor. Vetoing on a treatment pair (which the
    # first version of this check did) throws away a study whose baseline
    # reproduced perfectly: under A/B/B/A a noisy B says the *effect* is not
    # resolvable, which is a floor statement, not a comparability statement.
    spread_ok = True
    extra_ex, extra_it = [], []
    for k, v in reps.items():
        ex = [x["excess"] for x in v]
        it = [x["iter_mean"] for x in v]
        r_ex = max(ex) / min(ex) if min(ex) > 0 else float("inf")
        r_it = max(it) / min(it) if min(it) > 0 else float("inf")
        role = "ANCHOR" if k == a_nm else "repeat"
        print(f"{role} {k}: {len(v)} repeats  excess {min(ex):.3f}-{max(ex):.3f} "
              f"({r_ex:.2f}x)  iter_mean {min(it):.2f}-{max(it):.2f} ({r_it:.2f}x)")
        if k == a_nm:
            if r_ex > 1.5 or r_it > 1.15:
                spread_ok = False
        else:
            m_ex, m_it = st.mean(ex), st.mean(it)
            if m_ex:
                extra_ex.append((max(ex) - min(ex)) / m_ex * 100)
            if m_it:
                extra_it.append((max(it) - min(it)) / m_it * 100)

    print()
    if not spread_ok:
        print("VERDICT: the anchor did not reproduce within the allocation.")
        print("The arms are not comparable and no delta below the anchor's own")
        print("spread means anything. Report the spread, not a ranking.")
        return

    # --- ranking, only once the anchor has earned it ----------------------
    a_ex = st.mean([x["excess"] for x in anchor])
    a_it = st.mean([x["iter_mean"] for x in anchor])
    # The anchor's own spread is this study's noise floor. A delta smaller than
    # it is not a small effect -- it is an unmeasured one, and the sign of such
    # a delta is not even reliable. Gating on the spread being "tight enough"
    # is NOT sufficient: a 1.09x anchor passes any reasonable gate and still
    # swamps a 5% effect. The comparison that decides is delta vs spread.
    ex_all = [x["excess"] for x in anchor]
    it_all = [x["iter_mean"] for x in anchor]
    ex_floor = (max(ex_all) - min(ex_all)) / a_ex * 100 if a_ex else float("inf")
    it_floor = (max(it_all) - min(it_all)) / a_it * 100 if a_it else float("inf")
    # A repeated TREATMENT arm is a second, independent estimate of the same
    # noise, and taking the max is the conservative read: if the treatment pair
    # disagrees by 20% while the anchor pair agrees to 4%, a 10% treatment-vs-
    # anchor delta is not resolvable, and calling it MEASURED off the anchor's
    # narrower floor would be exactly the overclaim this reader exists to stop.
    a_ex_floor, a_it_floor = ex_floor, it_floor
    if extra_ex:
        ex_floor = max(ex_floor, max(extra_ex))
    if extra_it:
        it_floor = max(it_floor, max(extra_it))
    print(f"baseline = {a_nm} (mean of {len(anchor)}): "
          f"excess {a_ex:.3f} s/iter, iter_mean {a_it:.2f} s")
    print(f"NOISE FLOOR: tail +/-{ex_floor:.1f}%, wall +/-{it_floor:.1f}%")
    if extra_ex or extra_it:
        print(f"  anchor's own repeats give tail +/-{a_ex_floor:.1f}%, "
              f"wall +/-{a_it_floor:.1f}%; the floor above is WIDENED to the")
        print("  worst repeated group, because a treatment that does not "
              "reproduce bounds\n  what this study can resolve just as hard as "
              "a baseline that does not.")
    print("  (with 2 repeats this is a range, not a std -- it UNDERSTATES the")
    print("   floor, so a delta near it is even weaker than it looks.)")
    print()
    for s in arms:
        if base_name(s["name"]) == a_nm:
            continue
        de = (s["excess"] - a_ex) / a_ex * 100 if a_ex else float("nan")
        di = (s["iter_mean"] - a_it) / a_it * 100 if a_it else float("nan")
        vt = "MEASURED" if abs(de) > ex_floor else "below floor"
        vw = "MEASURED" if abs(di) > it_floor else "below floor"
        print(f"  {s['name']:<32} tail {de:+6.1f}% ({vt:11s}) "
              f"wall {di:+6.1f}% ({vw})")
    print()
    print("Read tail and wall together. A knob that cuts the tail without")
    print("cutting wall time moved a column, not the cost -- and one that cuts")
    print("wall time without the tail did it some other way. And a delta marked")
    print("'below floor' supports NO claim, in either direction: report it as a")
    print("null with the floor attached, never as a small win.")


if __name__ == "__main__":
    main()
