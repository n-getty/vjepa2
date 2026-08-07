#!/usr/bin/env python3
"""Node-synchronous compute EPISODES: a second cost with a different owner.

WHY THIS EXISTS
---------------
Job 8741769 was submitted to ask whether host memory tracks the within-run
dataload rise ([[dataload-rises-over-a-run]]). It answered that question NO --
and in doing so surfaced a cost that is not the dataload tail at all.

The 1-node, 12-tile, nw=2, 350-iteration rung has a **rock-solid floor**: the
p10 of max-over-ranks iteration time is 2.98 s in every 25-iteration bin from
iteration 84 to 349, and the p50 barely moves (2.99-3.41 s). The mean is 3.74 s.
The whole gap is EPISODES -- 20.3% of post-warmup wall spent above a floor that
never degrades. Half of that excess sits in one column, `fwd-context-ms`.

Episodes are not stragglers, and not the dataload tail:

    dataload on every episode iteration              0.00 s
    max/min of fwd-context ACROSS the 12 tiles       1.026 on episodes
                                                     1.057 on normal iterations
    fwd-target / fwd-context / backward ratio        x1.19 / x2.79 / x1.45

The tiles are TIGHTER together during an episode than outside one, so nothing is
waiting on a laggard -- the whole node slows at once, and it slows the context
encoder + predictor + loss far more than the target encoder. Whatever this is,
it is node-scoped and compute-side.

⚠️ THE MISTAKE THIS SCRIPT EXISTS TO PREVENT
--------------------------------------------
Read at iteration 283 (8 bins), the episode rate correlated with falling host
MemAvailable at **rho = -0.762**, which is exactly the memory-pressure story the
job was submitted to look for. With the full 11 bins the same correlation is
**rho = -0.160**, and the fwd-context MEDIAN moves the other way (+0.192). The
episode rate rises 0.00 -> 0.36 through the middle of the run and then falls
back to 0.00 in the last bin, while MemAvail keeps falling monotonically. It is
not monotone, so it cannot be a function of a monotone variable.

Same failure class as [[ab-window-truncation-trap]]: a verdict taken from a
window that ended where one arm happened to be. Here the "arm" was the run's own
middle. Do not correlate anything against MemAvail on a partial run -- MemAvail
is monotone by construction, so ANY quantity with a mid-run bulge will fit it
over a truncated prefix.

⚠️ AT 16 NODES THE SAME DETECTOR FIRES ON A DIFFERENT THING
-----------------------------------------------------------
Run this on the two 192-rank production runs and it reports 55-67% of wall in
"episodes" -- but the phase attribution and the spread both say it is not the
same phenomenon:

                              1n (8741769)      16n (obj_tempmask)
    fwd-context ratio            x2.79               x1.26
    backward ratio               x1.45               x8.33
    dataload ratio               x0.26               x4.45
    fwdc spread ep/normal    1.026 / 1.057       3.457 / 2.108

At 16n the excess is in BACKWARD, dataload rises with it, and the across-rank
spread GROWS on episode iterations -- the straggler signature. That is the known
dataload tail laundering itself into the all-reduce
([[backward-degradation-is-the-loader]]), and it is a different owner. The
detector keys on fwd-context because that is what isolates the 1n effect; when
the ratio table says backward, read it as the tail, not as this.

So the 1n episode is currently a ONE-RUN observation. It needs a second 1n long
rung before it is a finding.

WHAT IT REPORTS
---------------
Per run: the floor (p10 of max-over-ranks iter), the episode cost as a fraction
of wall, the phase attribution of the excess, and the across-rank spread on
episode vs normal iterations -- the statistic that separates "one slow rank" from
"the whole node slowed".

Usage:
    python scripts/fwdc_episode_scan.py <RUN_DIR> [...]
    python scripts/fwdc_episode_scan.py --scan <CKPT_ROOT>
"""

import argparse
import collections
import glob
import os
import statistics as st

# 0=epoch 1=itr 2=loss 3=iter 4=gpu 5=dload 6=fwdtgt 7=fwdctx 8=bwd 9=opt
# 10=ema ... 16=barrier 17=host-avail-mib 18=rss-mib
COLS = {"iter": 3, "gpu": 4, "dload": 5, "fwdt": 6, "fwdc": 7, "bwd": 8,
        "opt": 9, "ema": 10}
COL_AVAIL = 17
# Only XPU-event columns wrap; 3/5/16 are wall-clock and 17/18 are MiB.
EVENT_COLS = {4, 6, 7, 8, 9, 10}
WRAP_MS = 343597.0  # 2**32 * 80 ns

WARMUP_FRAC = 0.24   # 8741769 floors by ~iter 84 of 350; see the memory note
SPIKE_MULT = 1.5     # of the run's own per-rank fwd-context median
NODE_FRAC = 0.9      # "node-synchronous" = this fraction of a node's tiles
PPN = 12             # tiles per Aurora node


def read_run(d):
    """-> {(epoch, itr): {rank: {phase: seconds}}}, last segment only.

    Last segment only because a resumed run concatenates allocations and each
    one restarts the loader and the transient; mixing them smears the shape
    ([[dataload-rises-over-a-run]]).
    """
    per = collections.defaultdict(dict)
    seg_of = {}
    for f in sorted(glob.glob(os.path.join(d, "log_r*.csv"))):
        base = os.path.basename(f)
        try:
            rk = int(base[len("log_r"):-len(".csv")])
        except ValueError:
            continue
        seg = -1
        for ln in open(f):
            if ln.startswith("epoch,"):
                seg += 1
                continue
            p = ln.rstrip("\n").split(",")
            if len(p) <= max(COLS.values()):
                continue
            try:
                ep, itr = int(p[0]), int(p[1])
                row = {}
                for k, c in COLS.items():
                    v = float(p[c])
                    # A wrapped row is a SLOW row. Unwrap it -- dropping it
                    # would bias away exactly the episodes being counted.
                    if v < 0 and c in EVENT_COLS:
                        v += WRAP_MS
                    row[k] = v / 1000.0
            except ValueError:
                continue
            row["avail"] = (float(p[COL_AVAIL]) / 1024.0
                            if len(p) > COL_AVAIL and _num(p[COL_AVAIL])
                            else None)
            per[(ep, itr)][rk] = row
            seg_of[(ep, itr)] = max(seg, seg_of.get((ep, itr), seg))
    if not per:
        return {}
    last = max(seg_of.values())
    return {k: v for k, v in per.items() if seg_of[k] == last}


def _num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def pctl(sorted_vals, f):
    return sorted_vals[min(len(sorted_vals) - 1, int(f * len(sorted_vals)))]


def analyze(d, verbose=True):
    per = read_run(d)
    if not per:
        return None
    nr = max(len(v) for v in per.values())
    keys = sorted(k for k, v in per.items() if len(v) == nr)
    if len(keys) < 60:
        return None
    post = keys[int(WARMUP_FRAC * len(keys)):]

    fc_all = [r["fwdc"] for k in post for r in per[k].values()]
    fc_med = st.median(fc_all)
    thr = SPIKE_MULT * fc_med

    # Node-synchronous: at least NODE_FRAC of SOME node's tiles over threshold.
    def is_ep(k):
        c = collections.Counter(rk // PPN for rk, r in per[k].items()
                                if r["fwdc"] > thr)
        return bool(c) and max(c.values()) >= NODE_FRAC * PPN

    eps = [k for k in post if is_ep(k)]
    nrm = [k for k in post if k not in set(eps)]

    it = sorted(max(r["iter"] for r in per[k].values()) for k in post)
    floor = pctl(it, 0.10)
    spent = sum(it)
    excess = spent - floor * len(it)

    if verbose:
        print(f"\n{d}")
        print(f"  {nr} ranks, {len(keys)} full-coverage iters, "
              f"analysing the last {len(post)} (warmup {WARMUP_FRAC:.0%} dropped)")
        print(f"  iter max-over-ranks   p10 {floor:5.2f}  p50 {pctl(it,.5):5.2f}  "
              f"p90 {pctl(it,.9):5.2f}  mean {st.mean(it):5.2f} s")
        print(f"  EPISODE COST          {excess:6.0f} s of {spent:.0f} s = "
              f"{100*excess/spent:.1f}% of wall, in {len(eps)} of {len(post)} iters")
        if not eps:
            print("  no node-synchronous episodes at this threshold")
            return dict(dir=d, floor=floor, excess=excess / spent, n_ep=0)

        print(f"\n  {'phase':10s} {'normal':>8s} {'episode':>8s} {'ratio':>7s}")
        for k in ["dload", "fwdt", "fwdc", "bwd", "opt", "ema"]:
            a = st.mean(st.mean(r[k] for r in per[t].values()) for t in nrm)
            b = st.mean(st.mean(r[k] for r in per[t].values()) for t in eps)
            print(f"  {k:10s} {a:8.3f} {b:8.3f} "
                  f"{(b/a if a > 1e-6 else float('inf')):7.2f}")

        # The statistic that separates "one slow rank made 11 wait inside the
        # FSDP all-gather" from "the whole node slowed". A laggard gives a LARGE
        # spread; a node-wide slowdown gives a small one.
        def spread(ks):
            out = []
            for t in ks:
                v = [r["fwdc"] for r in per[t].values()]
                if min(v) > 1e-6:
                    out.append(max(v) / min(v))
            return st.median(out) if out else float("nan")
        s_ep, s_nr = spread(eps), spread(nrm)
        print(f"\n  fwd-context max/min ACROSS RANKS: "
              f"episode {s_ep:.3f}, normal {s_nr:.3f}")
        if s_ep <= s_nr * 1.1:
            print("  -> episode spread <= normal: nothing is waiting on a "
                  "laggard;\n     the node slowed as a unit.")
        else:
            print("  -> episode spread EXCEEDS normal: this is a STRAGGLER "
                  "pattern,\n     not a node-synchronous slowdown. Different "
                  "owner -- do not\n     merge it with the 1n fwd-context "
                  "episodes.")

        # Monotone-confound guard: report the episode rate over BOTH a prefix
        # and the whole run, because that difference is the whole lesson.
        av = [r["avail"] for t in post for r in per[t].values()
              if r["avail"] is not None]
        if av:
            _rate_vs_avail(per, post, eps)
    return dict(dir=d, floor=floor, excess=excess / spent, n_ep=len(eps))


def _rho(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda k: v[k])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[o[k]] = a
            i = j + 1
        return r
    X, Y = rk(x), rk(y)
    mx, my = st.mean(X), st.mean(Y)
    num = sum((a - mx) * (b - my) for a, b in zip(X, Y))
    den = (sum((a - mx) ** 2 for a in X) * sum((b - my) ** 2 for b in Y)) ** 0.5
    return num / den if den else 0.0


def _rate_vs_avail(per, post, eps, bin_n=25):
    epset = set(eps)
    bins = []
    for i in range(0, len(post), bin_n):
        ch = post[i:i + bin_n]
        if len(ch) < bin_n // 2:
            continue
        av = [r["avail"] for t in ch for r in per[t].values()
              if r["avail"] is not None]
        if not av:
            continue
        bins.append((st.mean(av), sum(1 for t in ch if t in epset) / len(ch)))
    if len(bins) < 5:
        return
    print(f"\n  {'MemAvail GiB':>13s} {'episode rate':>13s}")
    for a, r in bins:
        print(f"  {a:13.1f} {r:13.2f}")
    full = _rho([b[0] for b in bins], [b[1] for b in bins])
    pre = _rho([b[0] for b in bins[:-3]], [b[1] for b in bins[:-3]])
    print(f"\n  rho(MemAvail, episode rate)  whole run {full:+.3f}   "
          f"first {len(bins)-3} bins {pre:+.3f}")
    if abs(pre) - abs(full) > 0.25:
        print("  ⚠️  THE PREFIX LIES. MemAvail is monotone by construction, so any")
        print("      quantity with a mid-run bulge fits it over a truncated prefix.")
        print("      Only the whole-run number is interpretable, and it is the")
        print("      smaller one. Do not stop a memory-correlation run early.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--scan", help="walk a checkpoint root instead")
    a = ap.parse_args()

    dirs = list(a.dirs)
    if a.scan:
        for dp, _, fs in os.walk(a.scan):
            if "log_r0.csv" in fs:
                dirs.append(dp)
    if not dirs:
        ap.error("give run dirs or --scan ROOT")

    got = 0
    for d in dirs:
        if analyze(d, verbose=True):
            got += 1
    if a.scan:
        print(f"\n{got} of {len(dirs)} candidate dirs had enough coverage")


if __name__ == "__main__":
    main()
