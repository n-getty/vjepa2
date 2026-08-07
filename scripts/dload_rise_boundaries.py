#!/usr/bin/env python3
"""Where does the within-run dataload rise reset?

`dload_transient_scan.py` established that dataload is not a warmup transient:
after a short warmup it RISES over the rest of a run. This script asks what
the rise is attached to, using the two boundaries already present in production
artifacts -- so it costs zero node-hours.

Two boundaries, two different things reset at each:

  allocation boundary (CSV header repeats)
      A new PBS job. New processes, cold host page cache, fresh DAOS agent and
      client connections, re-opened shards. Everything node-local resets.

  epoch boundary (itr wraps within one segment)
      Same processes, same caches, same agent. What resets is the shard
      iteration order -- the loader starts its shard list over.

So the pair discriminates:

  resets at BOTH          -> shard-list position (where you are in the epoch)
  resets at allocation
      but NOT at epoch    -> accumulating process/node state, independent of
                             which shard is being read
  resets at NEITHER       -> something outside the job entirely (other tenants,
                             storage-side state)

Statistics notes, both learned the hard way:

  * Filter to live-loader segments the SAME way the decile scan does -- on the
    max DECILE MEAN, not on individual rows. Filtering on any single large row
    admits staged segments that spike once, and they carry no trend.
  * Drop each segment's first epoch before comparing epochs. That is where the
    warmup lives, and counting it as "first" makes a rising series look falling.
    Comparing "first epoch" to "last epoch" without this inverts the answer.

⚠️ Rank 0 only, like the decile scan. Dataload is an order statistic and rank 0
is not the max, so the SHAPE here is evidence and the magnitude is not.

Usage:
    python scripts/dload_rise_boundaries.py [CKPT_ROOT]
"""

import os
import statistics as st
import sys

COL_EPOCH, COL_ITR, COL_DLOAD = 0, 1, 5

# The XPU event counter is 32-bit at 80 ns/tick: it wraps at 343.597 s. A
# wrapped row is a SLOW row, so unwrap it -- dropping it biases the tail away.
WRAP_MS = 343597.0

MIN_SEG = 300  # iterations; shorter segments cannot show a decile trend
LIVE_DECILE_S = 0.3  # a decile mean this high means rank 0 really is loading
MIN_EPOCH_ITERS = 30  # epochs shorter than this are resume fragments


def read_segments(path, min_len=MIN_SEG):
    """-> [[(epoch, itr, dload_s), ...], ...], one list per allocation.

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
                if len(cur) >= min_len:
                    segs.append(cur)
                cur = []
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) <= COL_DLOAD:
                continue
            try:
                ep = int(parts[COL_EPOCH])
                it = int(parts[COL_ITR])
                dl = float(parts[COL_DLOAD])
            except ValueError:
                continue
            if dl < 0:
                dl += WRAP_MS
            cur.append((ep, it, dl / 1000.0))
    if len(cur) >= min_len:
        segs.append(cur)
    return segs


def deciles(seg):
    n = len(seg) // 10
    return [st.mean(d for _, _, d in seg[i * n : (i + 1) * n]) for i in range(10)]


def is_live(dec):
    return max(dec) >= LIVE_DECILE_S


def collect(root):
    """-> {run_dir: [(segment, deciles, live), ...]} in file order."""
    runs = {}
    for dirpath, _, files in os.walk(root):
        if "log_r0.csv" not in files:
            continue
        entries = []
        for seg in read_segments(os.path.join(dirpath, "log_r0.csv")):
            dec = deciles(seg)
            entries.append((seg, dec, is_live(dec)))
        if any(live for _, _, live in entries):
            runs[dirpath] = entries
    return runs


def by_epoch(seg):
    out = {}
    for ep, _, dl in seg:
        out.setdefault(ep, []).append(dl)
    return out


def ratio_report(label, num, den, reset_below=0.8):
    rs = [b / a for a, b in zip(den, num) if a > 0.05]
    if not rs:
        print(f"  {label}: no usable pairs")
        return None
    resets = sum(1 for x in rs if x < reset_below)
    print(
        f"  {label}: ratio median {st.median(rs):.2f}   "
        f"resets(<{reset_below}) {resets}/{len(rs)}   "
        f"carries {len(rs) - resets}/{len(rs)}"
    )
    return st.median(rs)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/flare/ModCon/ngetty/checkpoints"
    runs = collect(root)
    live = [(s, d) for e in runs.values() for s, d, lv in e if lv]
    print(f"{len(live)} live segments >= {MIN_SEG} iters under {root}")
    print("dataload seconds, RANK 0 ONLY -- shape is evidence, magnitude is not\n")

    prof = [st.median(d[i] for _, d in live) for i in range(10)]
    print("median decile profile:  " + "  ".join(f"{v:.2f}" for v in prof))
    print(f"  warmup  d0 -> d1: {prof[0]:.2f} -> {prof[1]:.2f}")
    print(f"  rise    d1 -> d9: {prof[1]:.2f} -> {prof[9]:.2f}  ({prof[9]/prof[1] - 1:+.0%})\n")

    # --- boundary 1: allocation. End of segment k vs post-warmup start of k+1.
    ends, starts = [], []
    for entries in runs.values():
        for (_, da, la), (_, db, lb) in zip(entries, entries[1:]):
            if la and lb:
                ends.append(da[9])
                starts.append(db[1])
    print(f"ALLOCATION boundary  ({len(ends)} consecutive live segment pairs)")
    print(f"  end of job k      median {st.median(ends):.2f} s" if ends else "  none")
    if ends:
        print(f"  start of job k+1  median {st.median(starts):.2f} s")
        alloc = ratio_report("start(k+1) / end(k)", starts, ends)

    # --- boundary 2: epoch, within a segment. Drop each segment's first epoch:
    #     that is the warmup, and including it inverts the sign of the answer.
    e_end, e_start = [], []
    for seg, _ in live:
        ep = by_epoch(seg)
        keys = [k for k in sorted(ep) if len(ep[k]) >= MIN_EPOCH_ITERS]
        for a, b in zip(keys[1:], keys[2:]):
            if b != a + 1:
                continue  # not adjacent -- a short epoch was filtered out
            q = max(1, len(ep[a]) // 4)
            e_end.append(st.mean(ep[a][-q:]))
            e_start.append(st.mean(ep[b][:q]))
    print(f"\nEPOCH boundary  ({len(e_end)} consecutive epoch pairs, first epoch dropped)")
    if e_end:
        print(f"  end of epoch k      median {st.median(e_end):.2f} s")
        print(f"  start of epoch k+1  median {st.median(e_start):.2f} s")
        epoch = ratio_report("start(k+1) / end(k)", e_start, e_end)

    if ends and e_end:
        a_reset, e_reset = alloc < 0.8, epoch < 0.8
        print("\nVERDICT")
        print(f"  allocation boundary: {'RESETS' if a_reset else 'CARRIES'} ({alloc:.2f})")
        print(f"  epoch boundary:      {'RESETS' if e_reset else 'CARRIES'} ({epoch:.2f})")
        if a_reset and not e_reset:
            print("  -> rise is attached to accumulating process/node state, NOT to")
            print("     shard-list position: the epoch restarts the shard list and the")
            print("     rise carries straight through it.")
        elif a_reset and e_reset:
            print("  -> rise tracks shard-list position within an epoch.")
        elif not a_reset:
            print("  -> rise survives a fresh allocation: look outside the job.")


if __name__ == "__main__":
    main()
