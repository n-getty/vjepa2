#!/usr/bin/env python
"""Compare the num_workers arms of a scaling-ladder job at fixed node count.

This answers ONE question: at `num_workers=0` the decode runs inline, so its cost
shows up in `dataload-time(ms)`. Does `num_workers=2` actually buy iteration time,
or does it only move the cost out of a visible column and into the shadow of
compute? Those look identical if you only read the dataload column, and they have
opposite implications:

  - If iteration time DROPS, the stall was I/O latency and prefetch hides it.
  - If iteration time is FLAT while dataload drops, the cost was never
    overlappable -- the node is out of CPU (12 ranks x OMP_NUM_THREADS=16 = 192
    threads on 104 physical cores) and prefetch just relabels it.

Reports max-over-ranks per iteration (a synchronous step pays the slowest rank,
never the median) on the common fully-covered window, with IQRs, because two arms
whose IQRs overlap have not been separated -- see the measurement notes in
docs/THROUGHPUT_RECIPE_AURORA.md.
"""

import argparse
import glob
import os
import re
import statistics as st

WRAP_MS = 2**32 * 80e-9 * 1000.0  # XPU 32-bit counter, 80 ns tick
EVENT_COLS = {4, 6, 7, 8, 9, 10}


def unwrap(v):
    return v + WRAP_MS if -WRAP_MS < v < 0 else v


def read(run):
    per = {}
    files = sorted(glob.glob(os.path.join(run, "log_r*.csv")))
    for f in files:
        for ln in open(f).read().splitlines():
            if not ln or ln.startswith("epoch,"):
                continue
            p = ln.split(",")
            if len(p) < 11:
                continue
            try:
                key = (int(p[0]), int(p[1]))
                it = float(p[3])
            except (ValueError, IndexError):
                continue
            if it <= 0:
                continue
            d = per.setdefault(key, {})
            for c in (3, 5, 6, 7, 8):
                try:
                    v = float(p[c])
                except (ValueError, IndexError):
                    continue
                d.setdefault(c, []).append(unwrap(v) if c in EVENT_COLS else v)
    return per, len(files)


def q(vals, f):
    v = sorted(vals)
    return v[min(len(v) - 1, int(len(v) * f))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ranks", type=int, default=0, help="0 = infer from dir name n<R>_")
    args = ap.parse_args()

    arms = []
    for d in sorted(glob.glob(os.path.join(args.root, "n*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        m = re.match(r"n(\d+)_", name)
        ranks = args.ranks or (int(m.group(1)) * 12 if m else 0)
        per, nf = read(d)
        if nf < ranks:
            print(f"{name:18s} ** {nf} of {ranks} rank CSVs -- max-over-ranks "
                  f"UNDERSTATES the max; row unusable **")
            continue
        arms.append((name, per, ranks))

    if not arms:
        print("no usable arms")
        return

    # Common window: iterations every arm covered fully. Discard iteration 0
    # (its XPU event deltas are never valid) and iteration 1 (first-touch).
    common = None
    for _, per, ranks in arms:
        ok = {k for k in per if k[1] >= 2 and len(per[k].get(3, [])) >= ranks}
        common = ok if common is None else (common & ok)
    common = sorted(common)
    if not common:
        print("no commonly-covered iterations")
        return
    print(f"common window: {len(common)} iters, {common[0]} .. {common[-1]}, "
          f"max-over-ranks, full coverage only\n")

    print(f"{'arm':18s} {'iter s':>17s} {'dataload s':>17s} {'fwd s':>9s} "
          f"{'bwd s':>8s} {'iter-DL':>8s}")
    base = None
    for name, per, ranks in arms:
        it = [max(per[k][3]) / 1000 for k in common]
        dl = [max(per[k][5]) / 1000 for k in common]
        fw = [max(a + b for a, b in zip(per[k][6], per[k][7])) / 1000 for k in common]
        bw = [max(per[k][8]) / 1000 for k in common]
        gap = [i - d for i, d in zip(it, dl)]
        print(f"{name:18s} {st.median(it):6.2f} [{q(it,.25):5.2f}-{q(it,.75):5.2f}] "
              f"{st.median(dl):6.2f} [{q(dl,.25):5.2f}-{q(dl,.75):5.2f}] "
              f"{st.median(fw):9.2f} {st.median(bw):8.2f} {st.median(gap):8.2f}")
        if base is None:
            base = (name, it, dl)

    # Verdict.
    #
    # The usual rule here is "overlapping IQRs => not separated". That rule
    # assumes each arm is unimodal, and the num_workers arms are NOT: with
    # prefetch working, iter time sits on a hard floor (~3.2 s at 2n, dataload
    # exactly 0.00) on most iterations and spikes on the few where the prefetch
    # queue ran dry. A bimodal arm has a wide IQR no matter how much faster it
    # is, so the IQR test alone would report "no winner" for a 2x speedup --
    # and the MEDIAN would report 4.3x for that same 2x. Both are wrong.
    #
    # For a throughput question the honest statistic is TOTAL WALL over the
    # common window: a synchronous run pays the sum of its iterations, not their
    # median. Report all three and say which one governs.
    print()
    bname, bit, bdl = base
    for name, per, ranks in arms[1:]:
        it = [max(per[k][3]) / 1000 for k in common]
        dl = [max(per[k][5]) / 1000 for k in common]
        spd = st.median(bit) / st.median(it)
        tot = sum(bit) / sum(it)
        ov = not (q(bit, .75) < q(it, .25) or q(it, .75) < q(bit, .25))
        dl_drop = st.median(bdl) - st.median(dl)
        print(f"{name} vs {bname}: TOTAL WALL {tot:.2f}x  (median {spd:.2f}x, "
              f"dataload {dl_drop:+.2f} s)")
        print(f"  total wall over {len(common)} iters: {sum(bit):.0f}s -> {sum(it):.0f}s"
              f"   <-- this is the number to quote")
        # Bimodality: how often does each arm reach its own floor?
        floor = min(it) * 1.15
        at_floor = sum(1 for x in it if x <= floor)
        b_at_floor = sum(1 for x in bit if x <= floor)
        print(f"  iters at/near {floor:.1f}s floor: {name} {at_floor}/{len(it)}, "
              f"{bname} {b_at_floor}/{len(bit)}")
        if ov:
            print("  IQRs overlap -- but check the floor counts above before calling")
            print("  it unresolved: a bimodal arm has a wide IQR even when it wins.")
        if dl_drop > 1.0 and not ov and spd < 1.05:
            print("  dataload fell but iter time did not -> cost was RELOCATED,")
            print("  not removed. Consistent with a CPU/core limit, not I/O latency.")
        elif dl_drop > 1.0 and spd >= 1.05:
            print("  dataload fell AND iter time fell -> prefetch genuinely hides")
            print("  the stall. Consistent with I/O latency, not a core limit.")


if __name__ == "__main__":
    main()
