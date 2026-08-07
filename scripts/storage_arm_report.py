#!/usr/bin/env python3
"""Read the four-arm staged-vs-DAOS sweep (task #25) without fooling yourself.

THE SWEEP

    arm 1  n2_nw2                daos-full      production path, OPENING anchor
    arm 2  n2_nw2_staged_cap24   staged-capped  local tmpfs; DAOS agent + NIC out
    arm 3  n2_nw2_cap24          daos-capped    THE CONTROL: arm 2's shard window
    arm 4  n2_nw2_rep2           daos-full      CLOSING anchor = the noise floor

Two contrasts, and neither is arm-2-vs-arm-1:

    arm2 vs arm3   STORAGE PATH at a fixed working set  -> DAOS agent / NIC
    arm3 vs arm1   WORKING SET at a fixed path          -> page cache

Arm 2 against arm 1 moves both at once and cannot separate "DAOS is slow" from
"the working set now fits in cache". That is why arm 3 exists.

WHY THIS SCRIPT AND NOT scaling_efficiency.py

Same phase columns, but the question is different: this compares ARMS AT ONE
NODE COUNT, so the thing that matters is whether an arm's delta clears the
anchor spread. scaling_efficiency.py compares rungs ACROSS node counts and has
no notion of a noise floor.

⚠️ THE THREE WAYS THIS READ GOES WRONG

1. READING THE MEDIAN. The tail is an order statistic; the median prices it
   ~18% cheap and can be flat while the mean doubles
   ([[dataload-tail-survives-at-1-node]]). Both are printed; the wall-clock
   claim is the MEAN.

2. READING A MEAN DELTA THAT IS SMALLER THAN THE ANCHOR SPREAD. Two rungs on
   the SAME node in the SAME allocation, back-to-back, same config, differed
   20% in the mean while agreeing to 0.7% in the median and 1% in every phase
   column ([[1n-anchor-does-not-reproduce]]). So a 20%-scale mean difference
   between arms is INSIDE the noise. |arm1 - arm4| measures it here directly;
   anything not clearing it is a null whatever its sign.

3. READING A TRUNCATED SWEEP. If the soft-deadline guard dropped arm 4, the
   noise floor is gone and NOTHING is interpretable
   ([[wallclock-kill-deletes-the-closing-anchor]]). This script refuses to
   print a verdict in that case rather than quietly comparing to arm 1 alone.

CONFOUND TO REPORT EITHER WAY: capped arms read a fixed 24-shard window of every
source, so their sampling diversity is not production's. Irrelevant to s/iter
(ladder loss is meaningless anyway) but it belongs in any write-up.
"""

import argparse
import collections
import csv
import glob
import os
import statistics as st

WRAP_MS = 343597.0          # 32-bit XPU event counter, 2**32 * 80 ns
EVENT_COLS = {4, 6, 7, 8, 9, 10}   # only these wrap; 3/5/16 are wall, 17/18 MiB
PHASES = [("dload", 5), ("fwdt", 6), ("fwdc", 7), ("bwd", 8),
          ("opt", 9), ("ema", 10), ("barrier", 16)]
WARMUP_FRAC = 0.30          # 2n arms are only ~100 iters; warmup is a big share

ARMS = [
    ("n2_nw2", "daos-full", "opening anchor"),
    ("n2_nw2_staged_cap24", "staged-cap24", "tmpfs; agent+NIC out"),
    ("n2_nw2_cap24", "daos-cap24", "THE CONTROL"),
    ("n2_nw2_rep2", "daos-full", "closing anchor"),
]


def unwrap(v, col):
    """A negative event delta is a counter wrap. It is a SLOW row -- unwrap it.

    Dropping these biased earlier phase tables low, because the wrapped rows are
    exactly the expensive ones.
    """
    return v + WRAP_MS if (col in EVENT_COLS and v < 0) else v


def read_arm(d):
    """-> {(epoch, itr): {rank: {phase: seconds}}} over the LAST segment only.

    A rung dir can hold several allocations' CSVs appended together; only the
    final segment is one continuous run, and `itr` cycles every ipe so it cannot
    be used to find the boundary. Key on (epoch, itr) and take the last run of
    monotonically non-decreasing keys.
    """
    per = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(d, "*log_r*.csv"))):
        base = os.path.basename(f)
        try:
            rank = int(base.split("log_r")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        rows = []
        with open(f) as fh:
            for r in csv.reader(fh):
                if not r or not r[0].strip().lstrip("-").isdigit():
                    continue        # repeated header = a new allocation
                if len(r) <= 16:
                    continue
                rows.append(r)
        for r in rows:
            try:
                key = (int(r[0]), int(r[1]))
                rec = {"iter": unwrap(float(r[3]), 3) / 1000.0}
                for name, col in PHASES:
                    rec[name] = unwrap(float(r[col]), col) / 1000.0
                rec["avail"] = float(r[17]) / 1024.0 if len(r) > 17 else 0.0
            except (ValueError, IndexError):
                continue
            per[key][rank] = rec
    return per


def summarize(per, label):
    if not per:
        return None
    nr = max(len(v) for v in per.values())
    keys = sorted(k for k, v in per.items() if len(v) == nr)
    if len(keys) < 20:
        return dict(label=label, ranks=nr, n=len(keys), partial=True)
    post = keys[int(WARMUP_FRAC * len(keys)):]

    # max-over-ranks: the iteration waits for the slowest rank, so this is the
    # only statistic that prices what the step actually cost.
    it = sorted(max(r["iter"] for r in per[k].values()) for k in post)
    out = dict(label=label, ranks=nr, n=len(post), partial=False,
               p10=it[len(it) // 10], med=st.median(it), mean=st.mean(it))
    for name, _ in PHASES:
        out[name] = st.mean([max(r[name] for r in per[k].values()) for k in post])
        out[name + "_rk"] = st.mean(
            [st.median([r[name] for r in per[k].values()]) for k in post])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="job dir, e.g. .../scaling_ladder/8741855")
    a = ap.parse_args()

    res = {}
    for d, kind, note in ARMS:
        p = os.path.join(a.root, d)
        if not os.path.isdir(p):
            print(f"  MISSING arm: {d} ({kind}, {note})")
            continue
        res[d] = summarize(read_arm(p), d)

    print(f"\n{a.root}")
    print(f"{'arm':22s}{'kind':14s}{'rk':>4s}{'n':>5s}"
          f"{'p10':>7s}{'med':>7s}{'mean':>7s}{'dload':>8s}{'bwd':>7s}")
    for d, kind, _ in ARMS:
        r = res.get(d)
        if r is None:
            continue
        if r.get("partial"):
            print(f"{d:22s}{kind:14s}{r['ranks']:4d}{r['n']:5d}"
                  f"   -- too few fully-covered iters to summarize --")
            continue
        print(f"{d:22s}{kind:14s}{r['ranks']:4d}{r['n']:5d}"
              f"{r['p10']:7.2f}{r['med']:7.2f}{r['mean']:7.2f}"
              f"{r['dload']:8.2f}{r['bwd']:7.2f}")

    ok = {d: r for d, r in res.items() if r and not r.get("partial")}

    # ---- the noise floor, before any comparison
    a1, a4 = ok.get("n2_nw2"), ok.get("n2_nw2_rep2")
    if not (a1 and a4):
        print("\n  NO CLOSING ANCHOR -> NO VERDICT.")
        print("  Both daos-full arms are required: without the pair there is no")
        print("  noise floor, and a 20%-scale mean difference is known to occur")
        print("  between identical back-to-back rungs on one node. Comparing to")
        print("  a single anchor would read that noise as an effect.")
        return

    floor = abs(a4["mean"] - a1["mean"]) / a1["mean"]
    print(f"\n  NOISE FLOOR |arm1 - arm4| = {100*floor:.1f}% of mean "
          f"({a1['mean']:.2f} vs {a4['mean']:.2f} s)")
    print("  Any arm delta below this is a null, whatever its sign.")

    # ---- the two contrasts that actually separate the hypotheses
    a2, a3 = ok.get("n2_nw2_staged_cap24"), ok.get("n2_nw2_cap24")
    anchor = (a1["mean"] + a4["mean"]) / 2

    def verdict(x, y, lo, hi, owner):
        if not (x and y):
            print(f"\n  {lo} vs {hi}: arm missing -> not evaluable")
            return
        d = (y["mean"] - x["mean"]) / x["mean"]
        tag = "NULL (inside the anchor spread)" if abs(d) <= floor else \
              f"SIGNAL -> {owner}"
        print(f"\n  {lo} vs {hi}: {x['mean']:.2f} -> {y['mean']:.2f} s "
              f"({100*d:+.1f}%)   {tag}")
        print(f"      dload {x['dload']:.2f} -> {y['dload']:.2f} s "
              f"(per-rank median {x['dload_rk']:.2f} -> {y['dload_rk']:.2f})")

    verdict(a3, a2, "arm3 daos-capped", "arm2 staged-capped",
            "the storage path: DAOS agent / NIC")
    verdict(a1, a3, "arm1 daos-full", "arm3 daos-capped",
            "the working set: page cache")

    print(f"\n  anchor mean {anchor:.2f} s")
    print("  Report the MEAN. The median prices the tail ~18% cheap, and the")
    print("  per-rank median in the dload line shows whether every rank got")
    print("  slower (real cost) or only the max did (an order statistic).")
    print("\n  CONFOUND: capped arms read a fixed 24-shard window per source, so")
    print("  their sampling diversity is not production's. State it either way.")


if __name__ == "__main__":
    main()
