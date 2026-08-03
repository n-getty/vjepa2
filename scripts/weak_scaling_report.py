#!/usr/bin/env python3
"""Weak-scaling report across the 16n -> 64n -> 256n DAOS shakeout ladder.

Weak scaling because per-rank work is held constant: bs=1 with --weak-scale, so
each rank does the same forward/backward regardless of node count and the global
batch grows with the ranks. The question is therefore "does per-iteration time
stay flat as we add nodes?" -- if it does, throughput scales linearly and the
added communication is being absorbed.

METHODOLOGY, and where it departs from MULTINODE_SETUP.md
---------------------------------------------------------
That document specifies steady-state median with the first 50 iterations
dropped. These shakeouts run ipe=30, so 50 cannot be dropped. Iteration 0 is a
large cold outlier (219 s at 64n, 49 s at 16n -- first-touch DAOS reads plus
allocator warmup) and iters 1-4 are still warming, so a median over everything
is contaminated in the other direction.

The honest analogue at this run length is a median over iters >= WARMUP
(default 5), and the window is printed with every number so a reader can judge
it rather than trust it.

WHAT THESE NUMBERS ARE NOT
--------------------------
Shakeouts on a shared machine, minutes-to-hours apart, under whatever fabric
contention existed at the time. docs/vitG_2B_HSDP_findings.md calls 16n
contention "the intrinsic tax" and documents 9 s -> 32 s drift within a single
run. Treat a spread of ~2x between topologies as noise unless it reproduces.
The per-iteration SPREAD (p10..p90) is printed for exactly this reason: if it
is wide, the median is not a summary of anything stable.

Usage:
    python scripts/weak_scaling_report.py                    # auto-discover
    python scripts/weak_scaling_report.py --warmup 5
    python scripts/weak_scaling_report.py <csv> [<csv> ...]  # explicit
"""

import argparse
import glob
import os
import re
import statistics
import sys

SHAKEOUT_ROOT = "/flare/ModCon/ngetty/checkpoints/daos_shakeout"
PER_RANK_BS = 1  # bs=1 under --weak-scale; see the module docstring


def read_rows(path):
    """Data rows after the LAST header.

    CSVLogger appends, so a file shared by several runs interleaves them. Only
    the rows after the final header belong to the most recent run. (Per-job
    directories now prevent sharing, but older files predate that.)
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if parts and parts[0] == "epoch":
                rows = []  # new run starts here
                continue
            if len(parts) > 3 and parts[1].strip().lstrip("-").isdigit():
                try:
                    rows.append({
                        "itr": int(parts[1]),
                        "iter_ms": float(parts[3]),
                        "dataload_ms": float(parts[5]) if len(parts) > 5 else float("nan"),
                        "loss": float(parts[2]),
                    })
                except ValueError:
                    continue
    return rows


def discover():
    """Find shakeout CSVs, keyed by node count parsed from the directory name."""
    found = []
    for d in sorted(glob.glob(os.path.join(SHAKEOUT_ROOT, "*_n*_*"))):
        m = re.search(r"_n(\d+)_(\w+)$", os.path.basename(d))
        csv = os.path.join(d, "log_r0.csv")
        if m and os.path.isfile(csv):
            found.append((int(m.group(1)), m.group(2), csv))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="*", help="explicit log_r0.csv paths")
    ap.add_argument("--warmup", type=int, default=5,
                    help="drop iters below this index (default 5)")
    ap.add_argument("--ppn", type=int, default=12)
    args = ap.parse_args()

    if args.csvs:
        runs = [(0, os.path.basename(os.path.dirname(c)), c) for c in args.csvs]
    else:
        runs = discover()
    if not runs:
        sys.exit(f"no shakeout CSVs under {SHAKEOUT_ROOT}")

    print(f"WEAK SCALING -- median of iters >= {args.warmup} "
          f"(per-rank bs={PER_RANK_BS}, constant across topologies)\n")
    print(f"  {'nodes':>5} {'ranks':>6} {'gbatch':>7} {'n':>3} "
          f"{'iter_s':>7} {'p10..p90':>14} {'dataload_s':>10} "
          f"{'clips/s':>9} {'eff':>6}")

    base = None
    for nodes, tag, csv in runs:
        rows = [r for r in read_rows(csv) if r["itr"] >= args.warmup]
        if not rows:
            print(f"  {nodes:>5} {'-':>6} {'-':>7} {'0':>3}   (no iters past warmup)")
            continue
        ms = sorted(r["iter_ms"] for r in rows)
        med = statistics.median(ms) / 1000.0
        p10 = ms[int(0.1 * (len(ms) - 1))] / 1000.0
        p90 = ms[int(0.9 * (len(ms) - 1))] / 1000.0
        dl = statistics.median(r["dataload_ms"] for r in rows) / 1000.0
        ranks = nodes * args.ppn
        gb = ranks * PER_RANK_BS
        thru = gb / med                      # clips/s across the whole job
        per_rank = thru / ranks              # clips/s/rank -- the scaling metric
        if base is None:
            base = per_rank
        eff = 100.0 * per_rank / base
        print(f"  {nodes:>5} {ranks:>6} {gb:>7} {len(ms):>3} "
              f"{med:>7.2f} {p10:>6.1f}..{p90:<7.1f} {dl:>10.2f} "
              f"{thru:>9.1f} {eff:>5.0f}%")

    print("\n  eff = clips/s/rank relative to the smallest topology. 100% means")
    print("  adding nodes bought proportional throughput (ideal weak scaling).")
    print("  p10..p90 is the per-iteration spread: if it straddles the medians")
    print("  being compared, the efficiency delta is noise, not a measurement.")


if __name__ == "__main__":
    main()
