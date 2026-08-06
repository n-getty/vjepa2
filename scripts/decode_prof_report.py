#!/usr/bin/env python3
"""Read the LIVE decode profiler out of a ladder rung and name the tail's owner.

    python scripts/decode_prof_report.py <rung_dir> [<rung_dir> ...]

Why a separate reader. The offline profiler (scripts/decode_per_source_profile.py)
cannot answer this question, and that is a structural limit, not a gap in its
coverage: it opens shard bytes directly and never reads through DAOS under a
running training job. The number that needs explaining is only visible live --
8.7% of 18,252 observed per-rank dataload samples exceed 10.74 s, which is twice
the slowest per-clip decode ever measured offline (lapgyn6_events, 5.37 s) and
therefore unreachable by ANY mixture of measured decode costs at bs=2. Median of
that excess population is 15.1 s and its max is 67.8 s, 6.3x the ceiling. So the
instrument has to run in the training process, and this script reads what it
emits.

WHAT THE TWO COLUMNS MEAN, AND WHEN EACH IS TRUSTWORTHY
-------------------------------------------------------
`decode` is the demux+extract call itself. Clean at any num_workers -- it is one
call in one process either way.

`gap` is the wall time since the previous sample left the same worker. Clean at
nw>0, where that process does nothing but decode in a loop, so a gap is genuinely
upstream wait (tar read / DAOS / shuffle refill). NOT clean at nw=0: decode then
runs inline in the training process, so every batch-boundary gap also contains
the whole training step (~3.4 s of compute on this model). At bs=2 that makes the
nw=0 gap bimodal BY CONSTRUCTION -- one real within-batch gap and one compute-
contaminated across-batch gap per pair.

Hence the verdicts below are asymmetric on purpose. A decode-owned tail can be
declared from the nw=0 rung alone. A storage-owned tail cannot: it needs the nw=2
rung, because at nw=0 "not in decode" and "in gap" are the same statement and
neither excludes compute.

The ceiling test is what makes this decisive rather than suggestive. Per source
we already know the slowest single clip ever decoded offline; if the live decode
p99 stays under that and the tail is elsewhere, the codec story is dead on its
own numbers.
"""
import argparse
import glob
import os
import re
import sys

# Slowest per-clip decode measured offline, seconds (docs/THROUGHPUT_RECIPE_AURORA.md
# and memory dataload-tail-is-keyframe-spacing). The worst of these sets the bs=2
# pure-decode ceiling at 2x5.37 = 10.74 s.
OFFLINE_MAX = {
    "surgtoolloc2022": 0.05, "multibypass140": 0.15, "heichole_512": 0.36,
    "grasp_noleak": 0.56, "sitl": 0.94, "lemon": 2.77, "cholec80": 3.04,
    "surgvu24_clean": 3.10, "sitl_2026": 3.55, "lapgyn6_events": 5.37,
}
CEILING_S = 2 * max(OFFLINE_MAX.values())

# "  <src> <n> <dec p50> <dec p99> <dec max>   <gap p50> <gap p99> <gap max>"
ROW = re.compile(
    r"^\s{2,}(\S+)\s+(\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)


def parse_rung(rung_dir):
    """Return {source: {...}} merged over ranks, keeping the WORST table seen.

    Each rank emits a cumulative table every N samples, so the last table from a
    rank supersedes its earlier ones; across ranks we keep the max of each
    percentile. Max, not mean: the subject is a tail that hits a median of 2 of
    12 ranks, and averaging over the 10 quiet ranks is exactly how it would be
    made to disappear.
    """
    by_src, enabled, tables = {}, False, 0
    files = sorted(glob.glob(os.path.join(rung_dir, "rank.*.out")))
    files += sorted(glob.glob(os.path.join(rung_dir, "rank.*.err")))
    for f in files:
        try:
            lines = open(f, errors="replace").read().splitlines()
        except OSError:
            continue
        for i, ln in enumerate(lines):
            if "[decode-prof] ENABLED" in ln:
                enabled = True
            if "[decode-prof] n=" not in ln:
                continue
            tables += 1
            for nxt in lines[i + 1:]:
                m = ROW.match(nxt)
                if not m:
                    break
                s = m.group(1)
                n = int(m.group(2))
                vals = [float(x) for x in m.groups()[2:]]
                cur = by_src.setdefault(
                    s, dict(n=0, dp50=0.0, dp99=0.0, dmax=0.0,
                            gp50=0.0, gp99=0.0, gmax=0.0))
                cur["n"] = max(cur["n"], n)
                for k, v in zip(("dp50", "dp99", "dmax", "gp50", "gp99", "gmax"), vals):
                    cur[k] = max(cur[k], v)
    return by_src, enabled, tables, len(files)


def report(rung_dir):
    name = os.path.basename(rung_dir.rstrip("/"))
    by_src, enabled, tables, nfiles = parse_rung(rung_dir)
    nw = 2 if "_nw2" in name else 0
    print(f"\n=== {name}   ({nfiles} rank files, {tables} tables, nw={nw}) ===")

    if not enabled and not by_src:
        # The distinction this branch draws is the reason the ENABLED banner
        # exists: an instrument that is OFF and an instrument that is ON but saw
        # nothing produce the same empty output, and only one of them is a result.
        print("  profiler never announced itself -- it was OFF, or "
              "VJEPA_DECODE_PROFILE did not reach the ranks. NOT a null result.")
        return None
    if not by_src:
        print(f"  profiler ON but emitted no table. Period "
              f"(VJEPA_DECODE_PROFILE_EVERY) is above this rung's samples/rank.")
        return None

    print(f"  {'source':<22} {'n':>6}  {'dec p50':>8} {'dec p99':>8} {'dec max':>8}"
          f"   {'gap p50':>8} {'gap p99':>8} {'gap max':>8}   offline max")
    worst_dec = worst_gap = 0.0
    for s, v in sorted(by_src.items(), key=lambda kv: -kv[1]["dmax"]):
        off = OFFLINE_MAX.get(s)
        flag = ""
        if off and v["dmax"] > 3 * off:
            flag = f"  <-- {v['dmax']/off:.1f}x its offline max"
        print(f"  {s:<22} {v['n']:>6}  {v['dp50']:>8.2f} {v['dp99']:>8.2f} "
              f"{v['dmax']:>8.2f}   {v['gp50']:>8.2f} {v['gp99']:>8.2f} "
              f"{v['gmax']:>8.2f}   {('%.2f' % off) if off else '   ?'}{flag}")
        worst_dec = max(worst_dec, v["dmax"])
        worst_gap = max(worst_gap, v["gmax"])

    print(f"\n  worst decode {worst_dec:.2f} s | worst gap {worst_gap:.2f} s "
          f"| bs=2 pure-decode ceiling {CEILING_S:.2f} s")

    # Verdicts. Deliberately asymmetric -- see the module docstring.
    if worst_dec > CEILING_S:
        print("  VERDICT: the tail is IN DECODE. Live decode exceeds the offline "
              "ceiling, so the offline benchmark under-measured the codec cost "
              "(contention, thread oversubscription, or a source it never saw). "
              "The re-encode is then the direct fix.")
    elif nw > 0 and worst_gap > CEILING_S:
        print("  VERDICT: the tail is UPSTREAM of decode, on a rung where gap is "
              "a clean storage measurement. Consistent with a per-node DAOS "
              "client stall. Still a hypothesis until traced on the DAOS side.")
    elif nw == 0 and worst_gap > CEILING_S:
        print("  INCONCLUSIVE: the tail is not in decode, but at nw=0 `gap` also "
              "contains the training step, so this cannot separate storage from "
              "compute. Read the nw=2 rung.")
    else:
        print("  NO TAIL IN THIS RUNG: nothing exceeded the ceiling. Check the "
              "rung's own dataload column before concluding -- if p(>10 s) is "
              "also ~0 here, the tail simply did not occur and the rung needs "
              "re-running longer, not re-interpreting.")
    return by_src


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rungs", nargs="+", help="ladder rung dirs (…/n1_nw0_prof)")
    a = ap.parse_args()
    for r in a.rungs:
        report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
