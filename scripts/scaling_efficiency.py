#!/usr/bin/env python
"""Weak-scaling efficiency across node counts, from PhaseTimer per-rank CSVs.

    python scripts/scaling_efficiency.py RUNDIR [RUNDIR ...]
    python scripts/scaling_efficiency.py --preset            # the recorded campaign

WHY THIS EXISTS, AND THE THREE WAYS IT GOES WRONG
-------------------------------------------------
Comparing throughput across node counts is easy to get wrong in ways that all
produce a plausible-looking number. Every guard below is here because the naive
version of this script produced a *superlinear* 256n result.

1. KEY ON (epoch, itr), NOT itr. `itr` restarts every epoch. Pooling on `itr`
   alone silently mixes a cold epoch-1 iteration with steady-state ones from
   later epochs -- it turned a 10.3 s median into 171 s for the one run here
   long enough to have many epochs.

2. USE max-over-ranks, AND REQUIRE FULL RANK COVERAGE. A synchronous step costs
   what its SLOWEST rank costs, so the per-iteration statistic must be the max.
   That makes it sensitive to how many ranks you sampled: max over 192 of 3072
   ranks misses stragglers that max over 768 of 768 catches, which flatters the
   larger job. Sampling a fixed NUMBER of ranks (rather than a fixed fraction)
   is what made 256n look superlinear. This script therefore reads every rank
   and reports only iterations where all of them logged.

3. COMPARE MATCHED WINDOWS. A long run is mostly steady state; a 30-iteration
   shakeout is mostly warmup. Whichever run is longer wins on a whole-run
   median regardless of scale.

Wall-clock `iter-time(ms)` only -- never `backward-ms`, whose XPU event deltas
go negative on exactly the stalled collectives a scaling study cares about.
"""

import argparse
import glob
import os
import statistics as st

CKPT_ROOT = "/flare/ModCon/ngetty/checkpoints"

# (label, path relative to CKPT_ROOT, ranks, clips per rank per step)
PRESET = [
    ("16n fixedshape bs2 ckptON", "surg_2_1_vitG384_fixedshape/vitG384_n16g12_weak", 192, 2),
    ("64n lbA8 bs1 ckptoff", "daos_shakeout/vitG384_lbA8_n64_8730919", 768, 1),
    ("64n accum1 bs1 ckptoff", "accum_ab/8731439/accum1", 768, 1),
    ("64n accum2 bs1 ckptoff", "accum_ab/8731439/accum2", 768, 2),
    ("256n lbA8 bs1 ckptoff", "daos_256n/vitG384_lbA8", 3072, 1),
]


def read_run(run):
    """-> {(epoch, itr): [iter_time_ms per rank]}, n_rank_files"""
    files = sorted(glob.glob(os.path.join(run, "log_r*.csv")))
    per = {}
    for f in files:
        try:
            lines = open(f).read().splitlines()
        except OSError:
            continue
        for ln in lines:
            if not ln or ln.startswith("epoch,"):  # CSVLogger appends a header per job
                continue
            p = ln.split(",")
            try:
                ep, it, ms = int(p[0]), int(p[1]), float(p[3])
            except (ValueError, IndexError):
                continue
            if ms > 0:  # negative/zero = broken phase timer
                per.setdefault((ep, it), []).append(ms)
    return per, len(files)


def summarize(per, ranks, clips, epoch=1, lo=1, hi=4):
    """Median max-over-ranks iter time in a matched window, full coverage only."""
    keys = [k for k in per if k[0] == epoch and lo <= k[1] < hi and len(per[k]) >= ranks]
    if not keys:
        return None
    vals = sorted(max(per[k]) / 1000.0 for k in sorted(keys))
    med = st.median(vals)
    return dict(
        n=len(vals),
        med=med,
        lo=vals[len(vals) // 4],
        hi=vals[3 * len(vals) // 4],
        thru=ranks * clips / med,
        per_tile=ranks * clips / med / ranks,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--preset", action="store_true")
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--lo", type=int, default=1, help="first itr in window (0 is init-dominated)")
    ap.add_argument("--hi", type=int, default=4, help="one past last itr in window")
    a = ap.parse_args()

    rows = PRESET if a.preset else [(os.path.basename(r.rstrip("/")), r, None, 1) for r in a.runs]
    print(f"window: epoch {a.epoch}, iters {a.lo}-{a.hi - 1}, max-over-ranks, full coverage only")
    print(f"{'run':30s} {'ranks':>5s} {'n':>3s} {'med s':>7s} {'IQR s':>13s} {'clips/s':>8s} {'/tile':>7s}")
    out = []
    for label, rel, ranks, clips in rows:
        path = rel if os.path.isabs(rel) else os.path.join(CKPT_ROOT, rel)
        per, nfiles = read_run(path)
        ranks = ranks or nfiles
        if not per:
            print(f"{label:30s} {ranks:5d}   no data")
            continue
        s = summarize(per, ranks, clips, a.epoch, a.lo, a.hi)
        if not s:
            print(f"{label:30s} {ranks:5d}   no fully-covered iters in window")
            continue
        print(
            f"{label:30s} {ranks:5d} {s['n']:3d} {s['med']:7.2f} "
            f"{s['lo']:6.1f}-{s['hi']:6.1f} {s['thru']:8.1f} {s['per_tile']:7.4f}"
        )
        out.append((label, ranks, s))

    # Weak-scaling efficiency is only meaningful between IDENTICALLY-configured runs.
    ref = next((o for o in out if "64n lbA8" in o[0]), None)
    tgt = next((o for o in out if "256n lbA8" in o[0]), None)
    if ref and tgt:
        eff = tgt[2]["per_tile"] / ref[2]["per_tile"]
        print(
            f"\nweak scaling 64n -> 256n (identical lbA8 config, {tgt[1] / ref[1]:.0f}x ranks): "
            f"{eff * 100:.0f}% per-tile efficiency, {tgt[2]['thru'] / ref[2]['thru']:.2f}x aggregate"
        )


if __name__ == "__main__":
    main()
