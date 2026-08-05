#!/usr/bin/env python
"""Verdict for a paired, matched-global-batch throughput A/B.

Standalone on purpose: the PBS script calls it at the end of the job, but the
per-rank CSVs persist, so if the wall kills the job before the verdict runs you
can still get the number:

    python scripts/ab_verdict.py /flare/.../bs_vs_accum/<jobid> --world 768

Usage:
    ab_verdict.py OUTROOT --world N [--arms A B] [--warmup K] [--clips-per-step C]

`--clips-per-step` is the per-rank clips each arm moves per optimizer step. It
must be EQUAL across arms for the median ratio to be the throughput ratio --
that is the whole design of a matched-global-batch A/B, and the reason this
tool refuses to correct for unequal work. If you need arms with different
work-per-step, use the accum_ab verdict instead, which multiplies it in.

Measurement rules encoded here (each was learned by getting it wrong):
  * WALL-CLOCK iter-time only. `backward-ms` is unusable -- XPU event deltas go
    negative on stalled collectives, i.e. exactly the iterations under test.
  * MAX over ranks, not mean: a synchronous step costs what its slowest rank
    costs.
  * Read EVERY rank. The max statistic is sensitive to sample size, so a fixed
    COUNT of ranks flatters whichever arm has more of them (this is what made a
    prior cross-scale comparison look superlinear).
  * Only iterations where every rank logged. A partially-logged iteration
    understates the max.
  * Compare a COMMON iteration window. Equal counts are not enough: iteration
    position is not exchangeable, because late iterations carry the fabric
    spikes. If the wall truncates one arm, trim the other's tail to match.
  * Never verdict on a window that ends where an arm merely HAPPENS to be.
    Pass --expect-last-itr (= ipe-1) so a still-running arm is called out. On
    job 8735877 the partial window 3..19 gave disjoint IQRs and 1.31x; the
    complete window 3..21 overlapped. The tail is a run's worst part, so
    truncating flatters whichever arm is behind.
  * Require DISJOINT IQRs before naming a winner, and judge on CSV rows, not
    exit codes. Overlap means UNRESOLVED, not equivalent -- 8735877's medians
    differed 1.30x while overlapping.
"""

import argparse
import glob
import os
import statistics as st
import sys


def load(arm_dir, warmup):
    """-> ({itr: max-over-ranks seconds}, n_rank_files, min l0-free GiB)."""
    files = glob.glob(os.path.join(arm_dir, "log_r*.csv"))
    per, mem = {}, []
    for f in files:
        try:
            lines = open(f).read().splitlines()
        except OSError:
            continue
        for ln in lines:
            if not ln or ln.startswith("epoch,"):  # CSVLogger re-emits a header
                continue
            p = ln.split(",")
            try:
                itr, ms = int(p[1]), float(p[3])
            except (ValueError, IndexError):
                continue
            if ms > 0:  # <= 0 means the phase timer was broken for that row
                per.setdefault(itr, []).append(ms)
            if itr >= warmup and len(p) > 14:
                try:
                    free_mib = float(p[14])  # l0-free-mib
                except ValueError:
                    continue
                # train.py logs -1.0 when torch.xpu.mem_get_info() raises. One
                # such row would otherwise become the reported minimum.
                if free_mib >= 0:
                    mem.append(free_mib)
    n_ranks = len(files)
    full = {
        itr: max(v) / 1000.0
        for itr, v in per.items()
        if len(v) >= n_ranks and itr >= warmup
    }
    return full, n_ranks, (min(mem) / 1024.0 if mem else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outroot")
    ap.add_argument("--world", type=int, required=True, help="total ranks")
    ap.add_argument("--arms", nargs=2, default=["bs2_accum1", "bs1_accum2"],
                    help="two subdirectory names of OUTROOT")
    ap.add_argument("--warmup", type=int, default=3,
                    help="drop iterations below this index")
    ap.add_argument("--clips-per-step", type=int, default=2,
                    help="per-rank clips per optimizer step (SAME on both arms)")
    ap.add_argument("--expect-last-itr", type=int, default=None,
                    help="final iteration index each arm should reach (ipe-1). "
                         "Warns loudly if the common window stops short, which "
                         "means an arm was still running when you ran this.")
    a = ap.parse_args()

    cps = a.clips_per_step
    print("=========== paired A/B, matched global batch ===========")
    print(f"root {a.outroot}")
    print(f"world {a.world} ranks   global batch {a.world * cps} on BOTH arms")
    print(f"window: itr >= {a.warmup}, max over ranks, "
          f"only iters where EVERY rank logged\n")

    arms = {}
    for name in a.arms:
        full, n_ranks, freemem = load(os.path.join(a.outroot, name), a.warmup)
        arms[name] = dict(full=full, n_ranks=n_ranks, mem=freemem)
        if not full:
            print(f"  {name:12s}: NO fully-covered iters past warmup "
                  f"({n_ranks} rank files)")

    common = None
    for v in arms.values():
        if v["full"]:
            s = set(v["full"])
            common = s if common is None else (common & s)
    truncated = False
    if common:
        print(f"  common iteration window: {min(common)}..{max(common)}  "
              f"(n={len(common)})")
        # A window that ends where one arm merely HAPPENS to be is not a fair
        # one. On job 8735877 the window 3..19 (arm 2 still running) gave
        # disjoint IQRs and 1.31x; the real window 3..21 overlapped. A run's
        # tail is systematically its worst part, so truncating drops the
        # slowest iterations of whichever arm is behind -- flattering it.
        if a.expect_last_itr is not None and max(common) < a.expect_last_itr:
            truncated = True
            print(f"  *** TRUNCATED: expected to reach itr "
                  f"{a.expect_last_itr}. An arm is still running, or died. "
                  f"The tail carries the fabric spikes, so this window "
                  f"flatters whichever arm is behind. Do not cite it. ***")
        print()

    res = {}
    for name, v in arms.items():
        if not v["full"] or not common:
            continue
        vals = sorted(v["full"][i] for i in common)
        r = dict(n=len(vals), med=st.median(vals),
                 p25=vals[len(vals) // 4], p75=vals[3 * len(vals) // 4],
                 lo=vals[0], hi=vals[-1], mem=v["mem"])
        res[name] = r
        mm = f"{r['mem']:.1f} GiB" if r["mem"] is not None else "n/a"
        print(f"  {name:12s}: n={r['n']:3d}  median {r['med']:6.2f}s  "
              f"IQR {r['p25']:.1f}..{r['p75']:.1f}  "
              f"range {r['lo']:.1f}..{r['hi']:.1f}  "
              f"-> {a.world * cps / r['med']:7.1f} clips/s   min l0-free {mm}")

    first, second = a.arms
    x, y = res.get(first), res.get(second)
    if x and y:
        # Equal work per step on both arms, so the throughput ratio is exactly
        # the inverse median ratio. No work-per-step correction -- that is the
        # point of matching global batch, and what the earlier accum-vs-accum
        # comparison could not do.
        print(f"\n  {first} / {second} throughput = {y['med'] / x['med']:.2f}x  "
              f"(>1 means {first} is faster)")
        if x["p75"] < y["p25"] or y["p75"] < x["p25"]:
            faster = first if x["med"] < y["med"] else second
            if truncated:
                print(f"  IQRs disjoint ({faster} faster) -- but the window is "
                      f"TRUNCATED, so this is NOT a verdict. Rerun once both "
                      f"arms have finished.")
            else:
                print(f"  IQRs DISJOINT -- resolved: {faster} is faster at "
                      f"matched global batch.")
        else:
            print("  IQRs OVERLAP -- NOT resolved. This does NOT mean the arms")
            print("  are equivalent: the medians can still differ a lot (they")
            print("  differed 1.30x on job 8735877 with overlapping IQRs). It")
            print("  means THIS SAMPLE cannot separate them -- report the median")
            print("  ratio as suggestive and decide on HEADROOM, reading the")
            print("  l0-free column above rather than assuming accumulation is")
            print("  the cheaper one.")
            print("  TRUE_ACCUM's activations are flat, but no_sync holds the")
            print("  unreduced gradient across sub-batches: measured 10.73 ->")
            print("  7.10 GiB at 64n (job 8731439), a 3.6 GiB charge matching")
            print("  train.py:1169's own ~4 GB estimate. Both levers cost")
            print("  memory; which costs LESS is what this column decides.")
    elif x or y:
        print("\n  Only ONE arm produced data. If the missing arm OOMed, that IS")
        print("  the answer. Check its rank.*.err before rerunning; a walltime")
        print("  skip is not a result.")
        return 2
    else:
        print("\n  NEITHER arm produced usable rows. Judge on the CSVs, not exit")
        print("  codes -- an rc=0 arm can have written zero iterations.")
        return 2

    print("\n  Wall-clock iter-time only. backward-ms is unusable: XPU event")
    print("  deltas go NEGATIVE on stalled collectives -- precisely the")
    print("  iterations under test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
