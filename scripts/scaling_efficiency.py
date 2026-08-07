#!/usr/bin/env python
"""Weak-scaling efficiency across node counts, from PhaseTimer per-rank CSVs.

    python scripts/scaling_efficiency.py RUNDIR [RUNDIR ...]
    python scripts/scaling_efficiency.py --preset            # the recorded campaign
    python scripts/scaling_efficiency.py --ladder ROOT       # scripts/scaling_ladder.sh output

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

   This guard earned itself again on 2026-08-06: a run labelled "1n" in a
   scaling table turned out to be a 192-rank job that had written only 12 rank
   CSVs. Max over 6% of ranks made it the fastest row in the table and every
   efficiency percentage computed against it was void. When `ranks` is not
   passed explicitly this script infers it from the file count and CANNOT catch
   that case -- so `--ladder` derives the expected rank count from the rung's
   directory name instead, and flags any shortfall loudly.

3. COMPARE MATCHED WINDOWS. A long run is mostly steady state; a 30-iteration
   shakeout is mostly warmup. Whichever run is longer wins on a whole-run
   median regardless of scale.

Headline numbers use wall-clock `iter-time(ms)`. The XPU event-timer phase
columns ARE reported -- attributing the cost is the point of the ladder -- and a
negative one is UNWRAPPED, not dropped (see WRAP_MS below).

NEGATIVE XPU PHASE TIMES ARE A COUNTER WRAP, NOT GARBAGE (verified 2026-08-06).
The XPU event counter is 32-bit at an 80 ns tick, so it rolls over every
2**32 * 80e-9 s = 343.59738 s and a delta spanning the rollover comes out
negative by exactly one period. Adding WRAP_MS reconciles it: over 176,640 rows
of the 64n and 256n runs, 6.3-6.6% of rows had at least one negative phase, and
after adding one period ZERO remained negative -- while `sum(phases) - gpu-time`
kept the same ~0.5 ms residual as the never-negative control rows (that residual
is just the "%d" integer truncation of gpu-time). A random-garbage timer does
not reconcile to half a millisecond.

Earlier versions of this script, `scripts/ccl_knob_sweep.sh`, and the measurement
notes in `docs/THROUGHPUT_RECIPE_AURORA.md` all asserted these were garbage to be
discarded; they were wrong.

HOW MUCH IT CHANGES: less than you would guess, and it is worth knowing why.
Measured on the 64n run, recovering the 901 dropped `backward-ms` samples moved
p50 12.19 -> 12.30 s, mean 12.90 -> 13.06 s, and the max-over-ranks median not at
all. Whether a phase wraps depends on where the free-running counter happens to
sit, NOT on how long the phase took, so the dropped rows are spread through the
distribution rather than concentrated in the slow tail. The reason to unwrap is
that dropping was a 4% silent coverage hole with a systematic low bias -- not
that it was hiding a big number. Do not re-tell this as "the tail was missing".

Only the XPU EVENT columns wrap (4, 6-10). `iter-time`, `dataload-time` and
`barrier-ms` are Python `time.time()` wall clock and never wrap.

CSV COLUMN LAYOUT (positional; see app/vjepa_2_1/train.py CSVLogger)
  0 epoch  1 itr  2 loss  3 iter-time(ms)  4 gpu-time(ms)  5 dataload-time(ms)
  6 fwd-target-ms  7 fwd-context-ms  8 backward-ms  9 opt-step-ms  10 ema-ms
  11 loss-pred  12 loss-context  13 lambda  14 l0-free-mib  15 l0-ext-mib
  16 barrier-ms  <- appended 2026-08-06; absent in older CSVs, treated as 0

  NOTE on columns 6/7: before 2026-08-06 both fwd marks fired on adjacent lines
  after the forward returned, so fwd-context-ms was always ~0 and fwd-target-ms
  held the ENTIRE forward. For older CSVs read (col6 + col7) as "forward".
"""

import argparse
import glob
import os
import re
import statistics as st

CKPT_ROOT = "/flare/ModCon/ngetty/checkpoints"

# Below this many fully-covered iterations, the "median" is a sample of one or
# two and the IQR is meaningless. The script still prints the row -- suppressing
# it would hide that the run exists -- but flags it, because a one-sample row
# once produced a "231% per-tile efficiency" that looked like a finding.
MIN_ITERS = 5

# (label, path relative to CKPT_ROOT, ranks, clips per rank per step)
PRESET = [
    ("16n fixedshape bs2 ckptON", "surg_2_1_vitG384_fixedshape/vitG384_n16g12_weak", 192, 2),
    ("64n lbA8 bs1 ckptoff", "daos_shakeout/vitG384_lbA8_n64_8730919", 768, 1),
    ("64n accum1 bs1 ckptoff", "accum_ab/8731439/accum1", 768, 1),
    ("64n accum2 bs1 ckptoff", "accum_ab/8731439/accum2", 768, 2),
    ("256n lbA8 bs1 ckptoff", "daos_256n/vitG384_lbA8", 3072, 1),
]

# XPU event-counter rollover period: 32-bit counter at an 80 ns tick.
# 2**32 * 80e-9 * 1000 ms. See the module docstring for the verification.
WRAP_MS = 2**32 * 80e-9 * 1000.0  # 343597.38368

# Columns produced by XPU event deltas, and therefore subject to the wrap.
# Wall-clock columns (3 iter-time, 5 dataload, 16 barrier) are NOT in this set.
EVENT_COLS = {4, 6, 7, 8, 9, 10}


def unwrap(v):
    """Undo one XPU counter rollover on an event-timer delta.

    A single period is enough: a phase would have to last 343 s to wrap twice,
    which exceeds every iteration ever observed on this model, and the watchdog
    fires long before. If a value is still negative after one period the input
    is genuinely broken -- return it unchanged so the caller can see that rather
    than silently manufacturing a plausible number.

    Note the theoretical edge: a value just below zero unwraps to ~343 s, which
    would then dominate every max-over-ranks it entered. That case is a real
    reading of the counter (the phase truly did span a rollover), not a bug, and
    it does not occur in practice -- across the 64n/256n/ladder CSVs the smallest
    unwrapped value is 23 ms and the largest 176 s, nowhere near the period. If a
    future run shows a cluster just under WRAP_MS, distrust it and check the
    tick rate rather than adding an arbitrary ceiling here.
    """
    return v + WRAP_MS if -WRAP_MS < v < 0 else v


# Phase columns to break out, in execution order. `barrier` first because when
# the scale probe is on it is the first thing the iteration does.
PHASES = [
    ("barrier", 16),
    ("dataload", 5),
    ("fwd-tgt", 6),
    ("fwd-ctx", 7),
    ("backward", 8),
    ("opt", 9),
    ("ema", 10),
]


def read_run(run):
    """-> ({(epoch, itr): {col: [per-rank value]}}, n_rank_files)

    Keeps every column the breakdown needs, not just iter-time, so the phase
    table is computed over exactly the same fully-covered iterations as the
    headline number rather than over a separately-filtered set.
    """
    files = sorted(glob.glob(os.path.join(run, "log_r*.csv")))
    wanted = [3] + [c for _, c in PHASES]
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
            if ms <= 0:  # negative/zero = broken phase timer
                continue
            slot = per.setdefault((ep, it), {})
            for c in wanted:
                try:
                    v = float(p[c])
                except (ValueError, IndexError):
                    # Column absent (pre-barrier-column CSV) or unparseable.
                    v = 0.0 if c == 16 else None
                if v is not None:
                    slot.setdefault(c, []).append(unwrap(v) if c in EVENT_COLS else v)
    return per, len(files)


def _covered(per, ranks, epoch, lo, hi):
    """Iteration keys in the window where ALL ranks logged an iter-time."""
    return sorted(
        k for k in per
        if k[0] == epoch and lo <= k[1] < hi and len(per[k].get(3, [])) >= ranks
    )


def summarize(per, ranks, clips, epoch=1, lo=1, hi=4):
    """Median max-over-ranks iter time in a matched window, full coverage only."""
    keys = _covered(per, ranks, epoch, lo, hi)
    if not keys:
        return None
    vals = sorted(max(per[k][3]) / 1000.0 for k in keys)
    med = st.median(vals)
    s = dict(
        n=len(vals),
        med=med,
        # The MEAN of the per-iteration max, not just its median. Dataload is
        # bimodal -- exactly 0.00 s on most iters and multi-second on the rest --
        # so the median prices the stall at ZERO and flatters the run: at 64n it
        # read 68% efficiency against an honest 63%. Wall-clock is the mean, and
        # a schedule is built from wall-clock. Report both; when they disagree
        # the gap IS the stall tax, and the mean is the one that pays it.
        mean=st.mean(vals),
        lo=vals[len(vals) // 4],
        hi=vals[3 * len(vals) // 4],
        thru=ranks * clips / med,
        per_tile=clips / med,
        keys=keys,
    )
    # Within-window trend: median of the first quarter vs the last quarter, in
    # ITERATION order (not sorted by value). A ratio well above 1 means the run
    # was still warming up inside the window, so the median is a LOWER BOUND on
    # steady-state throughput and must be reported as one. At 256n the
    # first10/last10 ratio was 1.96x and still descending at iteration 50.
    seq = [max(per[k][3]) / 1000.0 for k in keys]
    q = max(1, len(seq) // 4)
    s["trend"] = st.median(seq[:q]) / st.median(seq[-q:]) if len(seq) >= 4 else float("nan")
    # Phase breakdown, max-over-ranks per iteration then median over the window.
    # Max-over-ranks per PHASE, not the phases of the max-iter rank: a
    # synchronous step waits for the slowest rank in each phase independently,
    # and stragglers rotate (measured at 64n: a different argmax rank on every
    # one of 8 consecutive iterations).
    # Values were unwrapped at parse time; anything still negative here is a
    # genuinely broken sample, counted and reported rather than hidden.
    ph, bad = {}, 0
    for name, col in PHASES:
        vv = []
        for k in keys:
            vals_c = [x for x in per[k].get(col, []) if x >= 0]
            bad += len(per[k].get(col, [])) - len(vals_c)
            if vals_c:
                vv.append(max(vals_c) / 1000.0)
        ph[name] = st.median(vv) if vv else 0.0
    s["phases"] = ph
    s["bad"] = bad
    # SECOND reduction: median-OVER-RANKS per iteration, then median over the
    # window. This is not a redundant view of the table above -- the two answer
    # different questions and have already disagreed on a headline claim.
    #
    #   max-over-ranks    = what the synchronous step WAITS for (wall-clock cost)
    #   median-over-ranks = what a TYPICAL rank actually pays (per-rank cost)
    #
    # A phase can rise in the first and be flat in the second, which means no
    # rank got slower and the growth is skew ARRIVING in that phase. That is
    # exactly what the long-standing "64n forward blowup" turned out to be:
    # max-over-ranks fwd-context 1.07 -> 1.83 s, but median-over-ranks flat
    # 1.03 -> 1.01. target_encoder is HSDP-wrapped, so forward's first
    # all-gather absorbs all upstream dataload skew and bills it to forward.
    # Conversely a phase that grows in BOTH is a real per-rank cost: backward
    # 1.23 -> 2.38 s median-over-ranks is the HSDP replicate-dim collective, and
    # every rank pays it. Only that second pattern justifies a comms conclusion.
    phm = {}
    for name, col in PHASES:
        vv = []
        for k in keys:
            vals_c = [x for x in per[k].get(col, []) if x >= 0]
            if vals_c:
                vv.append(st.median(vals_c) / 1000.0)
        phm[name] = st.median(vv) if vv else 0.0
    s["phases_med"] = phm
    # Unaccounted = iter time minus the sum of phases. Large values mean the
    # instrumentation is not covering the step (or that phase maxima come from
    # different ranks, which inflates the sum instead) -- either way it is the
    # honest residual and belongs in the table.
    s["unacct"] = med - sum(ph.values())
    return s


def discover_ladder(root):
    """-> [(label, path, ranks, clips)] from scaling_ladder.sh rung dirs.

    Rung dirs are named n<NODES>_nw<WORKERS>. Deriving the expected rank count
    from the NAME rather than from the file count is deliberate: it is the only
    way this script can notice that a run wrote fewer rank CSVs than it had
    ranks, which is the failure that voided the previous scaling table.
    """
    rows = []
    if not os.path.isdir(root):
        return rows
    for name in sorted(os.listdir(root), key=lambda s: (_nodes_of(s) or 0, s)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        n = _nodes_of(name)
        if n is None:
            continue
        rows.append((name, d, n * 12, None))  # clips filled from --clips-per-rank
    return rows


def _nodes_of(name):
    m = re.match(r"^n(\d+)", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--preset", action="store_true")
    ap.add_argument("--ladder", help="root dir of scripts/scaling_ladder.sh output")
    ap.add_argument("--clips-per-rank", type=int, default=1,
                    help="clips per rank per step = batch_size x true_accum. "
                         "lbA is bs=2, so leaving this at 1 makes every derived "
                         "throughput number 2x wrong.")
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--lo", type=int, default=1,
                    help="first itr in window (0 is init-dominated and at 256n "
                         "logged gpu: -137918 ms -- never valid)")
    ap.add_argument("--hi", type=int, default=4, help="one past last itr in window")
    a = ap.parse_args()

    if a.ladder:
        rows = [(lbl, p, r, a.clips_per_rank) for lbl, p, r, _ in discover_ladder(a.ladder)]
        if not rows:
            print(f"no rung dirs (n<NODES>_nw<N>/) under {a.ladder}")
            return
    elif a.preset:
        rows = PRESET
    else:
        rows = [(os.path.basename(r.rstrip("/")), r, None, a.clips_per_rank) for r in a.runs]

    print(f"window: epoch {a.epoch}, iters {a.lo}-{a.hi - 1}, max-over-ranks, full coverage only")
    if a.ladder:
        print(f"ladder: {a.ladder}   clips/rank/step={a.clips_per_rank}")
    print()
    hdr = (f"{'run':26s} {'ranks':>5s} {'csv':>5s} {'n':>3s} {'med s':>7s} "
           f"{'mean s':>7s} {'IQR s':>13s} {'clips/s':>8s} {'/tile':>7s} {'trend':>6s}")
    print(hdr)
    print("-" * len(hdr))
    out = []
    for label, rel, ranks, clips in rows:
        path = rel if os.path.isabs(rel) else os.path.join(CKPT_ROOT, rel)
        per, nfiles = read_run(path)
        expected = ranks
        ranks = ranks or nfiles
        if not per:
            print(f"{label:26s} {ranks:5d} {nfiles:5d}   no data")
            continue
        s = summarize(per, ranks, clips, a.epoch, a.lo, a.hi)
        if not s:
            # Distinguish "window empty" from "window never fully covered": the
            # second is a coverage problem and reporting it as no-data hides it.
            any_in_window = any(k[0] == a.epoch and a.lo <= k[1] < a.hi for k in per)
            why = ("no fully-covered iters in window (partial rank coverage)"
                   if any_in_window else "no iters in window")
            print(f"{label:26s} {ranks:5d} {nfiles:5d}   {why}")
            continue
        print(
            f"{label:26s} {ranks:5d} {nfiles:5d} {s['n']:3d} {s['med']:7.2f} "
            f"{s['mean']:7.2f} {s['lo']:6.1f}-{s['hi']:6.1f} {s['thru']:8.1f} "
            f"{s['per_tile']:7.4f} {s['trend']:6.2f}"
        )
        # A mean well above the median means a few iterations carry the cost --
        # the bimodal dataload stall. Efficiency computed from med is then an
        # OVERSTATEMENT of what the run will actually deliver in wall-clock.
        if s["mean"] > 1.10 * s["med"]:
            print(f"{'':26s}   ** mean {s['mean']:.2f} s exceeds median {s['med']:.2f} s by "
                  f"{100 * (s['mean'] / s['med'] - 1):.0f}% -- stall-dominated; quote the "
                  f"MEAN for wall-clock/efficiency **")
        if s["n"] < MIN_ITERS:
            print(f"{'':26s}   ** only {s['n']} fully-covered iter(s) in the window "
                  f"(< {MIN_ITERS}); this is not a median. Widen --lo/--hi or accept "
                  f"that the run has no comparable window **")
        if expected and nfiles < expected:
            print(f"{'':26s}   ** {nfiles} of {expected} rank CSVs -- max-over-ranks "
                  f"UNDERSTATES the max; treat this row as unusable **")
        if s["trend"] > 1.15:
            print(f"{'':26s}   ** still warming inside the window (first/last quartile "
                  f"{s['trend']:.2f}x) -- med is a LOWER BOUND on steady state **")
        if s["bad"]:
            print(f"{'':26s}   ** {s['bad']} phase sample(s) still negative after "
                  f"unwrapping one counter period -- genuinely broken, excluded **")
        out.append((label, ranks, s))

    if not out:
        return

    # ---- phase breakdown
    print()
    print("phase breakdown (s, median over window of the per-iteration MAX over ranks)")
    names = [n for n, _ in PHASES]
    print(f"{'run':26s} " + " ".join(f"{n:>9s}" for n in names) + f" {'unacct':>9s} {'iter':>9s}")
    for label, ranks, s in out:
        cells = " ".join(f"{s['phases'][n]:9.2f}" for n in names)
        print(f"{label:26s} {cells} {s['unacct']:9.2f} {s['med']:9.2f}")
    print("  barrier = VJEPA_SCALE_PROBE pre-step barrier; max-over-ranks is the RANK SKEW")
    print("            (0.00 everywhere means the probe was off, not that skew was zero).")
    print("  fwd/backward/opt/ema are XPU event deltas. A negative one is a 32-bit counter")
    print("            wrap (343.597 s period) and is UNWRAPPED, not dropped -- those rows")
    print("            carry the slow tail, so dropping them biased the phase table low.")
    print("  unacct = iter - sum(phases); phase maxima can come from different ranks, so a")
    print("           negative residual is possible and means exactly that.")

    # ---- the same phases reduced by MEDIAN over ranks
    print()
    print("phase breakdown (s, median over window of the per-iteration MEDIAN over ranks)")
    print(f"{'run':26s} " + " ".join(f"{n:>9s}" for n in names))
    for label, ranks, s in out:
        cells = " ".join(f"{s['phases_med'][n]:9.2f}" for n in names)
        print(f"{label:26s} {cells}")
    print("  THIS is the table to read for a per-rank cost claim. Growth here means every")
    print("  rank got slower (real work: e.g. the HSDP replicate-dim all-reduce in backward).")
    print("  Growth ONLY in the max table above means no rank slowed and skew merely ARRIVED")
    print("  in that phase -- forward's first FSDP all-gather absorbs upstream dataload skew,")
    print("  which is how the '64n forward blowup' read as real for months. Check both before")
    print("  naming a cause.")

    # ---- efficiency ladder, always against the SMALLEST rung actually measured
    if a.ladder:
        print()
        base = min(out, key=lambda o: o[1])
        print(f"per-tile efficiency vs {base[0]} ({base[1]} ranks = 100%)")
        print(f"{'run':26s} {'ranks':>5s} {'x ranks':>8s} {'per-tile':>9s} {'eff':>6s} "
              f"{'eff(mean)':>9s} {'aggregate':>10s}")
        for label, ranks, s in out:
            eff = s["per_tile"] / base[2]["per_tile"]
            # Efficiency from the MEAN as well. Same reason as the mean column
            # above: with a bimodal stall the median-based figure is the one that
            # looks good and the mean-based one is the one that comes true. When
            # they differ, judge the >80% bar on eff(mean).
            eff_mean = base[2]["mean"] / s["mean"]
            print(f"{label:26s} {ranks:5d} {ranks / base[1]:8.0f}x {s['per_tile']:9.4f} "
                  f"{eff * 100:5.0f}% {eff_mean * 100:8.0f}% {s['thru'] / base[2]['thru']:9.2f}x")
        print("  A rung flagged above (partial coverage / still warming) is NOT a valid")
        print("  reference; re-run it rather than quoting an efficiency against it.")
        return

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
