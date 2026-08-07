#!/usr/bin/env python3
"""How often does the backward-drift episode hit a REAL run, and what does it cost?

WHY THIS EXISTS
---------------
Job 8741386 caught backward (the HSDP inter-node all-reduce) degrading from
~1.7 s to 7.4 s within a 50-iteration 16n rung. The obvious follow-up -- re-run
and characterize it -- failed: job 8741490 ran the identical config, same node
count, longer, on a different node set, and never degraded at all. The episode
is intermittent and allocation-correlated, so it cannot be summoned on demand,
and burning debug-scaling slots hoping to catch one is a poor trade.

But every production run since the PhaseTimer landed has been writing
`backward-ms` per iteration for thousands of iterations. Those runs are a free,
already-paid-for sample of the same phenomenon over hundreds of node-hours --
far more exposure than any ladder can buy. This scans them.

The question it answers is the one that decides whether to chase the cause at
all: not "what is it" but "what fraction of production iteration time does it
eat". If the answer is 2%, it is a curiosity. If it is 30%, it outranks the
dataload tail.

WHAT COUNTS AS AN EPISODE, AND WHY THIS DEFINITION
--------------------------------------------------
Per run, define the FLOOR as the 10th percentile of per-block median backward.
Not the global min (one lucky iteration), not the mean (contaminated by the very
episodes being measured). An iteration block is DEGRADED when its median exceeds
`--factor` x floor.

Blocks, not iterations, because a single slow iteration is a spike -- a
different phenomenon with a different likely cause (preemption, a stray
collective, a page fault). What 8741386 showed was a sustained regime lasting
tens of iterations. Requiring a whole block to be slow filters spikes out by
construction.

RANK 0 ONLY, AND WHY THAT IS DEFENSIBLE HERE
---------------------------------------------
Reading all 192 ranks x 6800 iterations x N runs is expensive, and rank 0 is
enough for THIS question: inside the degraded 8741386 rung all 16 nodes slowed
together (per-node medians 4.61-7.96 s, no outlier), so rank 0 saw the episode
too. That makes rank 0 a valid DETECTOR.

It is not a valid MEASURER of amplitude -- for that you need median-over-ranks,
because a single rank conflates its own work with time blocked waiting for
peers. So this script reports prevalence and rank-0 amplitude, and any run it
flags should be re-read across all ranks before quoting a number.

Caveat on the floor: runs differ in bs, activation checkpointing, and model, so
floors are not comparable ACROSS runs. Everything here is within-run relative.

THE CONFOUND YOU MUST CLOSE BEFORE BELIEVING ANY NUMBER HERE
-------------------------------------------------------------
Production runs ran with **nw=0**, where the rotating dataload straggler tail is
present. When one rank is late with its batch, EVERY OTHER RANK blocks inside
the gradient all-reduce -- and that blocked time is charged to `backward`. A
loader tail therefore MANUFACTURES a backward-degradation signature, and the
ladder rung that motivated this (nw=2, `dataload` 0.00 on every rank,
`barrier-ms` ~0) is the only place that confound was absent by construction.

And that confound is not hypothetical here -- IT EXPLAINS THE HEADLINE. Sorting
the scanned runs by date splits them cleanly in two:

    through 2026-07-01 : wall% is 0.0 on 20 of 22 runs (max 0.6, one 14.3)
    from   2026-07-31  : wall% is 5.5 - 31.9 on every run

and the thing that changed with it is the LOADER, not the fabric:

    run (era)                     bwd med  bwd p90  dload med  dload max-o-r
    cleandata      (2026-06-28)      4.09     4.38       0.00           0.00
    v3_lambdaoff   (2026-06-26)      2.67     2.88       0.00           0.00
    cooldown       (2026-07-01)      9.07    10.16       0.00           0.00
    abl_laponly    (2026-07-28)      6.57     7.67       0.73           3.59
    abl_full       (2026-07-31)     14.39    40.60       1.35          11.90
    samp_t050      (2026-08-06)      9.29    35.60       1.29           6.22

Every clean-era run reports `dataload` **identically 0.00 at max-over-ranks
across 11-14k iterations** -- a loader that never registers, i.e. the staged
`/tmp` path. Every degraded-era run has a live tail. Config is NOT the split:
abl_laponly (0.0%) and abl_full (27.6%) have identical model/bs/ckpt/workers
three days apart. So the p90 backward blowups this script finds in production
are mostly the dataload tail's downstream blocking -- one late rank stalls every
other rank inside the gradient all-reduce, and that wait is charged to
`backward`. That is [[64n-efficiency-63pct-two-halves]] and task #14, already
known, NOT new evidence for the 8741386 drift.

**So do not quote the 7% weighted figure as the cost of the drift.** It is an
upper bound dominated by a different, already-tracked phenomenon.

What survives: conditioning on a quiet loader in surg_2_1_vitG384_fixedshape
still leaves 11 degraded blocks (vs 4 with a tail) at 10.5 s against a 4.3 s
floor, with min-over-ranks up 2.00 -> 3.87. That residual is small, and it is
the only part of the production record that bears on the drift at all.

Restart/warmup contamination is the second trap: these runs resume in chunks and
the first iterations after each resume are slow for ordinary reasons. Note that
`itr` cycles every `ipe` (30) iterations, so it does NOT mark a resume -- the
repeated CSV **header line** does, because the trainer re-opens the log per PBS
job. Segmenting fixedshape that way gives 14 allocations whose backward medians
span 4.78-14.34 s (3.0x), which is the allocation-to-allocation spread from
[[backward-growth-is-drift-not-per-n-cost]] visible in production.

Usage:
    python scripts/backward_drift_scan.py --root /flare/.../checkpoints
    python scripts/backward_drift_scan.py --root ... --factor 2.0 --block 20
"""
import argparse
import os
import re
import statistics as st

COL_ITER, COL_DLOAD, COL_BWD = 3, 5, 8
WRAP_MS = 343597.0  # XPU event counter period; a negative delta is one wrap


def read_rank0(path):
    """-> [(iter_ms, bwd_ms, dload_ms)] in file order.

    File order, not sorted by itr: these runs resume across epochs and itr
    restarts, so the row sequence IS the time axis. Sorting by itr would
    interleave epochs and destroy exactly the temporal structure being looked
    for. Header lines are skipped wherever they appear, not just at the top --
    a resumed run has one per PBS job.
    """
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("epoch,"):
                continue
            p = line.rstrip("\n").split(",")
            if len(p) <= COL_BWD:
                continue
            try:
                it, bw, dl = (float(p[COL_ITER]), float(p[COL_BWD]),
                              float(p[COL_DLOAD]))
            except ValueError:
                continue
            # Unwrap rather than drop -- a wrapped row is a SLOW row, so
            # dropping them would bias precisely against the episodes sought.
            if bw < 0:
                bw += WRAP_MS
            if it < 0:
                it += WRAP_MS
            if dl < 0:
                dl += WRAP_MS
            out.append((it, bw, dl))
    return out


def scan(rows, block, factor):
    if len(rows) < block * 4:
        return None
    blocks = [rows[i:i + block] for i in range(0, len(rows) - block + 1, block)]
    med = [st.median(b for _, b, _ in blk) / 1000.0 for blk in blocks]
    itmed = [st.median(i for i, _, _ in blk) / 1000.0 for blk in blocks]
    dmed = [st.median(d for _, _, d in blk) / 1000.0 for blk in blocks]

    floor = sorted(med)[max(0, int(0.10 * len(med)) - 1)]
    if floor <= 0:
        return None
    bad = [k for k, m in enumerate(med) if m > factor * floor]

    # Contiguous runs of degraded blocks = episodes. Length matters: a lone
    # degraded block is near the spike/regime boundary, a run of six is the
    # 8741386 shape.
    episodes, cur = [], []
    for k in bad:
        if cur and k == cur[-1] + 1:
            cur.append(k)
        else:
            if cur:
                episodes.append(cur)
            cur = [k]
    if cur:
        episodes.append(cur)

    # Excess iteration-seconds attributable to degraded blocks, as a fraction of
    # the run's total. This is the number that decides whether to chase it --
    # and it uses ITER time, not backward time, because the fraction of the
    # WALL CLOCK is what a scheduling decision cares about.
    tot = sum(m * block for m in itmed)
    excess = sum((med[k] - floor) * block for k in bad)

    # Loader era, printed alongside every row so the confound is impossible to
    # miss. A run whose rank-0 dataload is flat 0.00 predates the DAOS switch:
    # it has no tail to blame, so its (rare) degraded blocks are the only ones
    # that speak to the drift. A run with a live tail is mostly measuring
    # all-reduce blocking behind a late peer -- task #14, not this.
    live_loader = st.median(dmed) > 0.05
    return dict(nblk=len(med), floor=floor, peak=max(med),
                frac_bad=len(bad) / len(med), n_ep=len(episodes),
                longest=max((len(e) for e in episodes), default=0),
                cost=excess / tot if tot > 0 else 0.0,
                iter_med=st.median(itmed), dload=st.median(dmed),
                live=live_loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--min-iters", type=int, default=200)
    a = ap.parse_args()

    found = []
    for dirpath, _, files in os.walk(a.root):
        if "log_r0.csv" not in files:
            continue
        p = os.path.join(dirpath, "log_r0.csv")
        try:
            rows = read_rank0(p)
        except OSError:
            continue
        if len(rows) < a.min_iters:
            continue
        s = scan(rows, a.block, a.factor)
        if s:
            nranks = len([f for f in files if re.match(r"^log_r\d+\.csv$", f)])
            found.append((os.path.relpath(dirpath, a.root), len(rows), nranks, s))

    if not found:
        print("no runs with enough iterations")
        return
    found.sort(key=lambda r: -r[3]["cost"])

    hdr = (f"{'run':48s} {'iters':>6s} {'rnk':>4s} {'ldr':>4s} {'floor':>6s} "
           f"{'peak':>7s} {'%blk':>5s} {'eps':>4s} {'long':>4s} {'wall%':>6s}")
    print(f"block={a.block} iters, degraded = block median > {a.factor}x the "
          f"run's 10th-pct floor\nrank 0 only -- a DETECTOR, not an amplitude "
          f"measurement (see docstring)\n")
    print(hdr)
    print("-" * len(hdr))
    for name, n, nr, s in found:
        print(f"{name[:48]:48s} {n:6d} {nr:4d} "
              f"{'LIVE' if s['live'] else 'stgd':>4s} "
              f"{s['floor']:6.2f} {s['peak']:7.2f} "
              f"{100 * s['frac_bad']:5.1f} {s['n_ep']:4d} {s['longest']:4d} "
              f"{100 * s['cost']:6.1f}")

    tot_it = sum(n for _, n, _, _ in found)
    wtd = sum(s["cost"] * n for _, n, _, s in found) / tot_it
    for lab, sel in (("LIVE loader", True), ("staged loader", False)):
        g = [(n, s) for _, n, _, s in found if s["live"] is sel]
        if not g:
            continue
        it = sum(n for n, _ in g)
        w = sum(s["cost"] * n for n, s in g) / it
        deg = sum(1 for _, s in g if s["frac_bad"] > 0.01)
        print(f"\n{lab:14s}: {len(g):2d} runs, {it:6d} iters, "
              f"weighted wall% {100 * w:5.1f}, {deg}/{len(g)} runs show episodes")
    print(f"\nall: {len(found)} runs, {tot_it} rank-0 iterations, "
          f"weighted wall% {100 * wtd:.1f}")
    print("\nRead the two loader groups SEPARATELY -- that is the whole point of")
    print("the 'ldr' column. A LIVE-loader run's degraded blocks are mostly ranks")
    print("blocked in the all-reduce behind a late peer, which is the dataload")
    print("tail (task #14) showing up in the backward column, NOT the 8741386")
    print("drift. Only staged-loader runs, which have no tail to blame, are clean")
    print("evidence -- and they are also where the episode rate collapses to ~0.")
    print("Re-read any flagged run across ALL ranks before quoting an amplitude.")


if __name__ == "__main__":
    main()
