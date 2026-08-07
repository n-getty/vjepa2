#!/usr/bin/env python3
"""Is the dataload rise paced by ITERATIONS or by WALL-CLOCK SECONDS?

`dload_rise_boundaries.py` narrowed the within-run rise to accumulating
per-node state -- it resets at an allocation boundary and carries across an
epoch boundary. That leaves a CLASS of candidates, not one. This tries to split
the class along its most useful axis, at zero node-hours:

  paced by ITERATIONS  -> the accumulator fills per unit of WORK. Bytes read,
                          shards opened, samples decoded, allocations made.
                          Page cache, loader RSS growth, per-open handle state.
  paced by SECONDS     -> the accumulator fills per unit of TIME, regardless of
                          how much this job did. DAOS agent/connection aging,
                          a background daemon, another tenant.

Within one segment the two clocks are collinear, so a single run cannot answer
this. The leverage is ACROSS segments, which span 2.9-16.9 s/iter.

The test. For each segment take the ITERATION INDEX at which smoothed dataload
first reaches the midpoint of that segment's own rise (d1 -> d9). Then:

    time-paced -> a slow segment accumulates more seconds per iteration, so it
                  reaches half-rise in FEWER iterations: rho(s/iter, cross-iter)
                  strongly NEGATIVE.
    work-paced -> the crossing iteration does not care how slow the segment was:
                  rho ~ 0.

⚠️ THE CONFOUND THAT DECIDES THIS SCRIPT'S ANSWER, so it is controlled, not
mentioned. Segment length is correlated with BOTH variables:

    rho(seg length, s/iter)      = -0.634   slow segments are short segments
    rho(seg length, cross-iters) = +0.395   and a short segment CANNOT cross at
                                            a high iteration index -- it ends
                                            first. Pure censoring.

So length alone manufactures a negative rho(s/iter, cross-iters) whether or not
any clock effect exists. Measured here: raw rho = -0.296 (CI excludes 0, looks
like a clean "time-paced" result), but the length-controlled PARTIAL rho is
-0.063 with a CI straddling 0. The raw number was the confound. This script
therefore reports the partial and refuses a verdict when its CI includes zero.

⚠️ Rank 0 only, like the rest of this family. Dataload is an order statistic and
rank 0 is not the max, so the SHAPE is evidence and the magnitude is not. And
this is the RISE, not the max-over-ranks tail; do not merge the conclusions.

Usage:
    python scripts/dload_rise_clock.py [CKPT_ROOT]
"""

import os
import random
import statistics as st
import sys

COL_ITER, COL_DLOAD = 3, 5
WRAP_MS = 343597.0

MIN_SEG = 300
LIVE_DECILE_S = 0.3
MIN_RISE = 1.25  # a flat segment's crossing index is pure noise
SMOOTH = 15      # dataload is spiky; cross on a running mean
N_BOOT = 2000
SEED = 0
MIN_LEVERAGE = 2.0  # s/iter spread below this and the clocks are collinear


def read_segments(path):
    """-> [[(iter_s, dload_s), ...], ...], one list per allocation.

    Each resumed PBS job re-emits the CSV header, so the header is the
    allocation boundary and each segment is one job's worth of iterations.
    """
    segs, cur = [], []
    try:
        fh = open(path)
    except OSError:
        return segs
    with fh:
        for line in fh:
            if line.startswith("epoch,"):
                if len(cur) >= MIN_SEG:
                    segs.append(cur)
                cur = []
                continue
            p = line.rstrip("\n").split(",")
            if len(p) <= COL_DLOAD:
                continue
            try:
                it = float(p[COL_ITER])
                dl = float(p[COL_DLOAD])
            except ValueError:
                continue
            # 32-bit 80 ns counter wraps at 343.597 s. A wrapped row is a SLOW
            # row -- unwrap it; dropping it biases the tail away.
            if it < 0:
                it += WRAP_MS
            if dl < 0:
                dl += WRAP_MS
            cur.append((it / 1000.0, dl / 1000.0))
    if len(cur) >= MIN_SEG:
        segs.append(cur)
    return segs


def deciles(vals):
    n = len(vals) // 10
    return [st.mean(vals[i * n : (i + 1) * n]) for i in range(10)]


def running_mean(vals, w):
    out, acc = [], 0.0
    for i, v in enumerate(vals):
        acc += v
        if i >= w:
            acc -= vals[i - w]
        out.append(acc / min(i + 1, w))
    return out


def crossing(seg):
    """Where this segment first reaches the midpoint of its OWN rise.

    -> (iters, wall_s, comp_s, ratio) or None if the segment does not rise.
    Using each segment's own midpoint rather than an absolute threshold is what
    makes fast and slow segments comparable: they rise to different heights, and
    an absolute cut would just re-sort them by height.
    """
    dl = [d for _, d in seg]
    dec = deciles(dl)
    if max(dec) < LIVE_DECILE_S or dec[1] <= 0.05:
        return None
    ratio = dec[9] / dec[1]
    if ratio < MIN_RISE:
        return None
    target = (dec[1] + dec[9]) / 2.0

    sm = running_mean(dl, SMOOTH)
    # Start after the warmup decile so the warmup's own descent cannot satisfy
    # the threshold on the way down.
    start = len(seg) // 10
    wall = comp = 0.0
    for i, (it, d) in enumerate(seg):
        wall += it
        comp += max(it - d, 0.0)
        if i >= start and sm[i] >= target:
            return i, wall, comp, ratio
    return None


def _ranks(v):
    order = sorted(range(len(v)), key=lambda k: v[k])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def spearman(x, y):
    return _pearson(_ranks(x), _ranks(y))


def partial_spearman(x, y, z):
    """Rank correlation of x,y with z held constant."""
    X, Y, Z = _ranks(x), _ranks(y), _ranks(z)
    rxy, rxz, ryz = _pearson(X, Y), _pearson(X, Z), _pearson(Y, Z)
    den = ((1 - rxz**2) * (1 - ryz**2)) ** 0.5
    return (rxy - rxz * ryz) / den if den else 0.0


def boot_ci(fn, cols, n_boot=N_BOOT, seed=SEED):
    """Percentile CI, resampling SEGMENTS -- the unit of independence."""
    rng = random.Random(seed)
    n = len(cols[0])
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(fn(*[[c[k] for k in idx] for c in cols]))
    vals.sort()
    lo = vals[int(0.025 * n_boot)]
    hi = vals[int(0.975 * n_boot) - 1]
    return lo, hi


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/flare/ModCon/ngetty/checkpoints"

    rows = []
    for dirpath, _, files in os.walk(root):
        if "log_r0.csv" not in files:
            continue
        for seg in read_segments(os.path.join(dirpath, "log_r0.csv")):
            c = crossing(seg)
            if c is None:
                continue
            i, wall, comp, ratio = c
            rows.append({
                "dir": os.path.basename(dirpath.rstrip("/")),
                "len": len(seg),
                "spi": st.median(x for x, _ in seg),
                "iters": i,
                "wall": wall,
                "comp": comp,
                "ratio": ratio,
            })

    if len(rows) < 12:
        print(f"only {len(rows)} rising segments under {root} -- too few to "
              f"control for the length confound (need >= 12)")
        return

    spi = [r["spi"] for r in rows]
    ln = [r["len"] for r in rows]
    it = [r["iters"] for r in rows]
    lev = max(spi) / min(spi)

    print(f"{len(rows)} rising segments (>= {MIN_SEG} iters, rise >= {MIN_RISE}x)")
    print(f"s/iter spread: {min(spi):.2f} .. {max(spi):.2f}  ({lev:.2f}x leverage)")
    if lev < MIN_LEVERAGE:
        print(f"\n⚠️  Under {MIN_LEVERAGE}x leverage the two clocks are nearly "
              f"collinear.\n    No verdict is possible from this set.")
        return

    # --- the confound, measured before anything is claimed -------------------
    r_ls = spearman(ln, spi)
    r_li = spearman(ln, it)
    print("\nCONFOUND CHECK -- segment length, which is correlated with both:")
    print(f"  rho(length, s/iter)      = {r_ls:+.3f}  "
          f"{'slow segments are SHORT' if r_ls < -0.2 else ''}")
    print(f"  rho(length, cross-iters) = {r_li:+.3f}  "
          f"{'and short segments cannot cross late (censoring)' if r_li > 0.2 else ''}")
    confounded = abs(r_ls) > 0.2 and abs(r_li) > 0.2
    if confounded:
        print("  -> length alone would manufacture a negative rho below.")
        print("     The RAW number is not interpretable; the PARTIAL is.")

    # --- raw vs length-controlled -------------------------------------------
    print("\nrho(s/iter, crossing point).  time-paced predicts strongly NEGATIVE")
    print("for cross-iterations; work-paced predicts ~0.\n")
    print(f"  {'measure':18s} {'raw':>7} {'95% CI':>18}   "
          f"{'| length':>9} {'95% CI':>18}")
    results = {}
    for label, key in (("cross-iterations", "iters"),
                       ("cross-wall-s", "wall"),
                       ("cross-compute-s", "comp")):
        v = [r[key] for r in rows]
        raw = spearman(spi, v)
        rlo, rhi = boot_ci(spearman, (spi, v))
        par = partial_spearman(spi, v, ln)
        plo, phi = boot_ci(partial_spearman, (spi, v, ln))
        print(f"  {label:18s} {raw:+7.3f} [{rlo:+.3f}, {rhi:+.3f}]   "
              f"{par:+9.3f} [{plo:+.3f}, {phi:+.3f}]")
        results[key] = (raw, par, plo, phi)

    par, plo, phi = results["iters"][1:]
    print("\nVERDICT")
    if plo <= 0.0 <= phi:
        print(f"  NO SEPARATION. Length-controlled rho(s/iter, cross-iters) = "
              f"{par:+.3f},")
        print(f"  95% CI [{plo:+.3f}, {phi:+.3f}] includes zero. The archive cannot")
        print("  tell a work-paced accumulator from a time-paced one.")
        if confounded:
            raw = results["iters"][0]
            print(f"\n  Note the trap: the RAW rho is {raw:+.3f} and its CI excludes")
            print("  zero, so reading the raw number would have given a confident")
            print("  and WRONG 'time-paced' answer. Slow segments are short ones,")
            print("  and a short segment cannot cross at a high iteration index.")
        print("\n  This is a limit of the data, not a null result about the system.")
        print("  Separating the clocks needs a DESIGNED pair: two rungs of equal")
        print("  length and equal iteration count whose s/iter differs by 2x+,")
        print("  e.g. throttled vs unthrottled compute on the same node.")
    elif par < 0:
        print(f"  TIME-paced (partial rho {par:+.3f}, CI [{plo:+.3f}, {phi:+.3f}]).")
        print("  The accumulator fills per unit of elapsed time regardless of how")
        print("  much this job did -- that points OUTSIDE the loader (DAOS agent")
        print("  aging, a daemon, another tenant) and needs external tooling,")
        print("  since nothing the trainer logs would see it.")
    else:
        print(f"  WORK-paced (partial rho {par:+.3f}, CI [{plo:+.3f}, {phi:+.3f}]).")
        print("  The accumulator fills per unit of work done: bytes read, shards")
        print("  opened, allocations made. Page cache, loader growth, handle state.")

    print("\n⚠️  This is the RISE (rank 0's own cost), not the max-over-ranks tail.")
    print("    They may have different owners. Do not merge the conclusions.")


if __name__ == "__main__":
    main()
