#!/usr/bin/env python3
"""How much of the dataload wall-clock cost is an ORDER STATISTIC, and does
clustering make it worse or better?

THE TWO NUMBERS THAT MOTIVATE THIS
-----------------------------------
On a 16n nw=2 rung (job 8741594, 192 ranks, iters 7-60):

    per-rank mean dataload    0.087 s/iter
    mean of max-over-ranks    5.91  s/iter        -- 68x larger

The synchronous step waits for the max, so 5.91 s is what the schedule pays.
Nothing is wrong with either number: a typical rank really is nearly free, and
the step really does wait almost six seconds. The factor of 68 is what taking a
maximum over 192 draws does to a heavy-tailed variable.

That reframes the tail problem. The question is not only "why does a rank
stall" but "how much of the max is unavoidable given ANY per-rank distribution
with this shape", because the second question decides whether the tail is
fixable at 3072 ranks at all.

WHAT THIS MEASURES
------------------
Two curves of mean-of-max against the number of ranks k, both from the same
run, both exact over the observed range -- no distributional assumption, no
bootstrap from thin marginals:

  OBSERVED  for each iteration, take the max over a random k-subset of ranks;
            average over iterations and over subsets. This is the real order
            statistic, WITH whatever cross-rank coupling the run has.
  SHUFFLED  independently permute each rank's series across iterations first.
            This destroys cross-rank alignment while preserving every rank's
            marginal distribution EXACTLY, then runs the identical curve.

The gap between them isolates the effect of coupling on the order statistic,
holding the marginals fixed. That is the only clean way to separate "ranks are
individually slow sometimes" from "ranks are slow together".

THE PREDICTION THAT MAKES THIS WORTH RUNNING
---------------------------------------------
Coupling is expected to make the max BETTER, not worse, which is the opposite
of the intuition that clustering is bad:

  * independent -- each rank spikes on its own iterations, so with 192 ranks
    almost EVERY iteration catches somebody's spike. Mean-of-max is high and
    keeps climbing with k, roughly as the (1 - 1/k) quantile of the marginal.
  * coupled -- ranks spike on the SAME iterations. Those iterations are
    expensive, but the rest are clean, and adding more ranks to an already-
    spiking iteration cannot make it spike harder. The curve SATURATES.

So if the tail is coupled (measured: same-node pairwise lift 4.65-12.05 vs
cross-node 2.84-7.57, `tail_clustering.py`), the observed curve should flatten
while the shuffled one keeps rising.

That has a consequence worth stating before looking: 64n and 256n production
runs measured 23.19 and 22.92 s/iter -- a 4x jump in node count at zero
marginal cost. A saturating order statistic PREDICTS exactly that flatness.
This script does not prove that connection (different runs, different windows),
but it does say whether the mechanism is present in a run we control.

HOW TO READ THE RESULT
----------------------
  observed saturates, shuffled climbs
      the tail is coupled and the max is near its ceiling. More ranks cost
      little extra, which is good news for 3072 -- and it means the lever is
      per-rank stall MAGNITUDE, not stall probability. Halving how often a
      rank stalls barely moves a max that is already saturated.
  both climb together
      the tail is effectively independent at this scale. Then stall
      PROBABILITY is the lever and the cost grows with rank count, which is
      the pessimistic case for scale-out.

Per [[no-lazy-cause-labels]]: this identifies the SHAPE of the cost, not its
cause. It says nothing about which resource stalls a rank.

Usage:
    python scripts/order_statistic_curve.py --run /flare/.../n16_nw2
    python scripts/order_statistic_curve.py --run ... --col 3   # iter-time
"""
import argparse
import glob
import os
import re
import statistics as st

WRAP_MS = 2**32 * 80e-9 * 1000.0


def read(run, col, lo, hi):
    """-> {iteration: {rank: seconds}} over the window."""
    by_it = {}
    for f in sorted(glob.glob(os.path.join(run, "log_r*.csv"))):
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
                it, v = int(p[1]), float(p[col])
            except (ValueError, IndexError):
                continue
            if not (lo <= it < hi):
                continue
            if v < 0:  # one counter wrap -- a wrapped row is a SLOW row
                v += WRAP_MS
            by_it.setdefault(it, {})[r] = v / 1000.0
    return by_it


def to_matrix(by_it):
    """-> (ranks, [[value per rank] per iteration]) on fully-covered iterations.

    Full coverage only: a max over a partial rank set understates the max, and
    the whole point here is the max.
    """
    if not by_it:
        return [], []
    ranks = sorted(set.intersection(*(set(d) for d in by_it.values())))
    its = [i for i in sorted(by_it) if len(by_it[i]) >= len(ranks)]
    return ranks, [[by_it[i][r] for r in ranks] for i in its]


def lcg(seed):
    """Deterministic PRNG. Reproducibility matters more than randomness quality
    here -- two people reading the same run must get the same curve."""
    s = seed

    def nxt(n):
        nonlocal s
        s = (s * 6364136223846793005 + 1442695040888963407) % (2**64)
        return (s >> 33) % n
    return nxt


def curve(mat, ks, reps, seed):
    """Mean over iterations of the max over a random k-subset of ranks."""
    nxt = lcg(seed)
    n = len(mat[0])
    out = []
    for k in ks:
        if k > n:
            continue
        tot, cnt = 0.0, 0
        for _ in range(reps):
            idx, pool = [], list(range(n))
            for j in range(k):  # partial Fisher-Yates, sampling without replacement
                t = j + nxt(len(pool) - j)
                pool[j], pool[t] = pool[t], pool[j]
                idx.append(pool[j])
            for row in mat:
                tot += max(row[i] for i in idx)
                cnt += 1
        out.append((k, tot / cnt))
    return out


def shuffle_cols(mat, seed):
    """Permute each rank's series independently across iterations.

    Preserves every rank's marginal distribution exactly and destroys all
    cross-rank alignment. This is the null the observed curve is judged against.
    """
    nxt = lcg(seed)
    T, n = len(mat), len(mat[0])
    out = [row[:] for row in mat]
    for c in range(n):
        for i in range(T - 1, 0, -1):
            j = nxt(i + 1)
            out[i][c], out[j][c] = out[j][c], out[i][c]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--col", type=int, default=5, help="5=dataload, 3=iter-time")
    ap.add_argument("--lo", type=int, default=7)
    ap.add_argument("--hi", type=int, default=10**6)
    ap.add_argument("--reps", type=int, default=40, help="subsets per k")
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    ranks, mat = to_matrix(read(a.run, a.col, a.lo, a.hi))
    if not mat or len(ranks) < 8:
        print(f"need >=8 ranks and >=1 covered iteration; got {len(ranks)} ranks")
        return
    T, n = len(mat), len(ranks)
    flat = [v for row in mat for v in row]
    print(f"{os.path.basename(a.run.rstrip('/'))}: {n} ranks x {T} iterations, "
          f"col {a.col}")
    print(f"per-rank mean {st.mean(flat):.3f} s   median {st.median(flat):.3f} s"
          f"   max {max(flat):.2f} s\n")

    ks = [k for k in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
                      384, 512, 768) if k <= n]
    if ks[-1] != n:
        ks.append(n)
    obs = curve(mat, ks, a.reps, a.seed)
    shf = curve(shuffle_cols(mat, a.seed + 1), ks, a.reps, a.seed)

    print(f"{'k ranks':>8} {'observed':>10} {'shuffled':>10} {'obs/shuf':>9} "
          f"{'obs/k=1':>8}")
    print("-" * 50)
    base = obs[0][1]
    for (k, o), (_, s) in zip(obs, shf):
        print(f"{k:>8} {o:>10.3f} {s:>10.3f} {o/s if s else float('nan'):>9.3f} "
              f"{o/base if base else float('nan'):>8.1f}")

    # Saturation: growth over the top half of the k range, both curves.
    mid = len(obs) // 2
    if not (mid >= 1 and obs[mid][1] > 0 and shf[mid][1] > 0):
        return
    go = obs[-1][1] / obs[mid][1]
    gs = shf[-1][1] / shf[mid][1]
    kmid, kmax = obs[mid][0], obs[-1][0]
    print(f"\ngrowth from k={kmid} to k={kmax}:  "
          f"observed {go:.2f}x   shuffled {gs:.2f}x")
    print(f"(rank count itself grew {kmax/kmid:.1f}x over that range)")

    # Power-law exponent over the top half. mean-of-max ~ k^beta, so beta is
    # what any extrapolation to a larger world size turns on -- and stating it
    # as a number invites the extrapolation to be checked rather than assumed.
    import math
    bo = math.log(go) / math.log(kmax / kmid)
    bs = math.log(gs) / math.log(kmax / kmid)
    print(f"exponent beta (max ~ k^beta):  observed {bo:.3f}   shuffled {bs:.3f}")
    print(f"  EXTRAPOLATION IS NOT LICENSED BY THIS FIT. It is measured over one")
    print(f"  run's rank range; the curve may bend. Production 64n and 256n came")
    print(f"  in at 23.19 and 22.92 s/iter -- a 4x rank jump at ~zero cost, which")
    print(f"  beta>0 does not predict. Treat beta as a description of THIS range.")

    # The coupling ratio is a first-class result, not a footnote: it can fall
    # steeply while the curve still climbs, and those are separate facts.
    r_first, r_last = obs[0][1] / shf[0][1], obs[-1][1] / shf[-1][1]
    print(f"\ncoupling ratio obs/shuffled: {r_first:.3f} at k={obs[0][0]} "
          f"-> {r_last:.3f} at k={kmax}")
    print()
    if r_last < 0.8:
        print(f"COUPLING SUPPRESSES THE MAX. At {kmax} ranks the observed max is")
        print(f"{(1-r_last)*100:.0f}% BELOW what the same per-rank marginals would")
        print("give if ranks stalled independently. Ranks stall together, so")
        print("extra ranks largely join an iteration that is already paying.")
        print("This is the opposite of the usual reading of contention, and it")
        print("matters: the coupling is not the thing to remove.")
    if go > 1.3:
        print()
        print("BUT STILL CLIMBING. Suppressed or not, the max has not reached a")
        print("ceiling over this rank range, so cost still grows with scale.")
        print("Both levers remain live -- per-rank stall probability AND")
        print("magnitude -- and neither is spent.")
    elif go < 1.15:
        print()
        if gs > go * 1.15:
            print("AND SATURATED. The observed max stops growing while the")
            print("marginal-preserving shuffle keeps climbing. Stall")
            print("PROBABILITY is nearly spent as a lever; what is left is")
            print("stall MAGNITUDE and the node-local resource behind it.")
        else:
            print("AND SATURATED -- but the shuffle saturates too, so this is")
            print("the marginal's own bounded shape, not coupling.")
    else:
        print()
        print("INTERMEDIATE growth. Neither saturated nor clearly climbing over")
        print("this range; a wider rank range is needed to call it.")


if __name__ == "__main__":
    main()
