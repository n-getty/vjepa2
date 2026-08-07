#!/usr/bin/env python3
"""Does host memory track the within-run dataload rise?

`dload_rise_boundaries.py` narrowed the rise to accumulating per-node state:
it resets when a new PBS job starts and carries straight across an epoch
boundary, which rules out shard-list position. Of the candidates that fit that
pattern, host memory was the only one with no column -- so `train.py` now logs
`host-avail-mib` (MemAvailable, node-wide) and `rss-mib` (this rank's RSS) at
columns 17/18.

This reads them against dataload over a rung and answers three questions:

  1. Does MemAvailable fall as dataload rises?
       falling together -> the node is filling up; host memory is implicated
       dataload rises, MemAvailable flat -> host memory is NOT the accumulator,
         which eliminates the last CSV-visible candidate and pushes the next
         step to DAOS agent/client state (needs external instrumentation)
  2. Does RSS rise with it?
       rss up + avail down  -> the loader itself is growing
       avail down, rss flat -> something else on the node is consuming memory
  3. Across two identical rungs in ONE allocation: does rung 2 start at rung 1's
     floor or at its elevated end? The archive cannot answer this -- there every
     process restart is also a new job -- so it separates "a fresh PROCESS
     clears it" from "a fresh ALLOCATION clears it".

⚠️ Correlation over a monotone series is weak evidence. Two quantities that both
drift over a run correlate whether or not they are related. Treat a strong
correlation as consistent-with, and a FLAT MemAvailable as the informative
result: that one actually eliminates a candidate.

⚠️ This is the RISE (rank 0's own cost), not the max-over-ranks tail. They may
have different owners. Do not merge the conclusions.

Usage:
    python scripts/host_mem_vs_dload.py <rung_dir> [<rung_dir> ...]
    python scripts/host_mem_vs_dload.py /path/to/scaling_ladder/<jobid>/*/
"""

import os
import statistics as st
import sys

COL_ITR, COL_ITER, COL_DLOAD = 1, 3, 5
COL_AVAIL, COL_RSS = 17, 18
WRAP_MS = 343597.0
N_DECILES = 10


def read_rung(path):
    """-> [(itr, iter_s, dload_s, avail_mib, rss_mib), ...] for rank 0.

    Rows from before the host-memory columns existed are returned with None in
    those fields rather than skipped, so an older rung still reports its
    dataload profile instead of looking empty.
    """
    csv = os.path.join(path, "log_r0.csv")
    rows = []
    try:
        fh = open(csv)
    except OSError:
        return rows
    with fh:
        for line in fh:
            if line.startswith("epoch,"):
                continue
            p = line.rstrip("\n").split(",")
            if len(p) <= COL_DLOAD:
                continue
            try:
                itr = int(p[COL_ITR])
                it = float(p[COL_ITER])
                dl = float(p[COL_DLOAD])
            except ValueError:
                continue
            if it < 0:
                it += WRAP_MS
            if dl < 0:
                dl += WRAP_MS
            avail = rss = None
            if len(p) > COL_RSS:
                try:
                    avail = float(p[COL_AVAIL])
                    rss = float(p[COL_RSS])
                    if avail < 0:
                        avail = None
                    if rss < 0:
                        rss = None
                except ValueError:
                    pass
            rows.append((itr, it / 1000.0, dl / 1000.0, avail, rss))
    return rows


def spearman(xs, ys):
    """Rank correlation, no scipy. Ties get average ranks."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def deciles(rows, idx):
    n = len(rows) // N_DECILES
    if n == 0:
        return []
    out = []
    for i in range(N_DECILES):
        vals = [r[idx] for r in rows[i * n : (i + 1) * n] if r[idx] is not None]
        out.append(st.mean(vals) if vals else None)
    return out


def fmt(vals, scale=1.0, prec=2):
    return "  ".join("  --  " if v is None else f"{v*scale:6.{prec}f}" for v in vals)


def main():
    dirs = [d for d in sys.argv[1:] if os.path.isdir(d)]
    if not dirs:
        print(__doc__)
        raise SystemExit(2)

    summary = []
    for d in dirs:
        rows = read_rung(d)
        if len(rows) < N_DECILES * 2:
            print(f"\n=== {os.path.basename(d.rstrip('/'))}: {len(rows)} rows, too short")
            continue
        name = os.path.basename(d.rstrip("/"))
        has_mem = any(r[3] is not None for r in rows)
        print(f"\n=== {name}   {len(rows)} iters   "
              f"{'host-mem columns present' if has_mem else 'NO host-mem columns (pre-6bc3b81 run)'}")

        d_dl = deciles(rows, 2)
        print(f"  dataload s      {fmt(d_dl)}")
        if has_mem:
            d_av = deciles(rows, 3)
            d_rs = deciles(rows, 4)
            print(f"  MemAvail GiB    {fmt(d_av, 1/1024.0)}")
            print(f"  RSS GiB         {fmt(d_rs, 1/1024.0)}")

        idx = list(range(len(rows)))
        r_dl = spearman(idx, [r[2] for r in rows])
        line = f"  rho(iter, dataload) = {r_dl:+.2f}" if r_dl is not None else ""
        if has_mem:
            r_av = spearman(idx, [r[3] for r in rows])
            r_rs = spearman(idx, [r[4] for r in rows])
            r_dv = spearman([r[2] for r in rows], [r[3] for r in rows])
            line += (f"   rho(iter, MemAvail) = {r_av:+.2f}"
                     f"   rho(iter, RSS) = {r_rs:+.2f}"
                     f"   rho(dataload, MemAvail) = {r_dv:+.2f}")
            avs = [r[3] for r in rows if r[3] is not None]
            rss = [r[4] for r in rows if r[4] is not None]
            print(line)
            print(f"  MemAvail {avs[0]/1024:.1f} -> {avs[-1]/1024:.1f} GiB "
                  f"(delta {(avs[-1]-avs[0])/1024:+.1f})    "
                  f"RSS {rss[0]/1024:.1f} -> {rss[-1]/1024:.1f} GiB "
                  f"(delta {(rss[-1]-rss[0])/1024:+.1f})")
            summary.append((name, d_dl, avs, rss, r_dv))
        else:
            print(line)

    # Cross-rung: within one allocation, does rung 2 start where rung 1 ended?
    if len(summary) >= 2:
        print("\n" + "=" * 72)
        print("PROCESS RESTART vs ALLOCATION RESTART")
        print("Rungs below ran in ONE allocation, so a reset here is a PROCESS")
        print("reset. The archive could not separate these -- there every process")
        print("restart was also a new job.")
        for (n1, dl1, av1, _, _), (n2, dl2, av2, _, _) in zip(summary, summary[1:]):
            e_dl, s_dl = dl1[-1], dl2[1] if dl2[1] is not None else dl2[0]
            print(f"\n  {n1} -> {n2}")
            print(f"    dataload  end {e_dl:.2f} s  ->  next post-warmup {s_dl:.2f} s"
                  f"   ratio {s_dl/e_dl:.2f}" if e_dl else "")
            if av1 and av2:
                print(f"    MemAvail  end {av1[-1]/1024:.1f} GiB  ->  next start "
                      f"{av2[0]/1024:.1f} GiB   recovered "
                      f"{(av2[0]-av1[-1])/1024:+.1f} GiB")
        print("\n  reset on BOTH  -> a fresh PROCESS clears it (leak-like)")
        print("  reset on NEITHER -> needs a fresh ALLOCATION (node-level state)")
        print("  MemAvail recovers but dataload does not -> memory is not the cause")

    print("\n⚠️  Correlation over two monotone series is weak evidence. A FLAT")
    print("    MemAvail is the informative outcome: it eliminates the last")
    print("    CSV-visible candidate. This is rank 0's own cost, NOT the")
    print("    max-over-ranks tail -- do not merge the conclusions.")


if __name__ == "__main__":
    main()
