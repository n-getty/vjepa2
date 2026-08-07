#!/usr/bin/env python3
"""Is the dataload tail HEAVY SAMPLES -- a 13x spread in bytes per clip?

WHERE THIS CAME FROM
--------------------
The shard-open hypothesis (`scripts/shard_open_correlation.py`) is REFUTED: a
tar open costs 12-15 ms measured, and across 19 production runs the measured
spike rate correlates NEGATIVELY with predicted open rate (r = -0.538, 2.2x
spread in the predictor). Opening tars is not what the tail is made of.

But the microbenchmark that killed it pointed somewhere else. Reading one
member of heichole-000000.tar took 1377 ms for 10.8 MB, while a cholec80 member
took 128 ms for 20.9 MB -- the cost tracked the READ, not the open. And bytes
per sample is wildly non-uniform across this corpus:

    grasp_noleak   85.9 MB/sample        surgenet_robotic_clean  7.0
    sitl_2026      63.3                  sitl                    6.6
    multibypass140 29.1                  gynsurg                 8.4
    cholec80       23.8                  surgvu24_clean          8.5

A 13.1x spread, and NOT a restatement of shard size (r = -0.16 against
samples-per-shard). That is the shape the tail needs: every rank draws from the
same mixture, so on any given iteration some ranks draw an 86 MB clip and
others draw a 6.6 MB clip. The heavy draw rotates because the MIXTURE rotates
it -- no node is sick, nothing is synchronized, and max-over-ranks pays the
worst draw in the world every single step. That is exactly the unexplained
signature from [[dataload-tail-survives-at-1-node]] and
[[scaling-loss-is-a-straggler-order-statistic]].

THE PREDICTION
--------------
For each run, from its OWN `params-pretrain.yaml` source mix, compute the
temperature-weighted distribution over per-sample byte sizes. Two predictors
fall out, and they are tested separately because they say different things:

  mean_mb  -- weighted mean MB per sample. Should predict the BODY of the
              dataload distribution (its median).
  p_heavy  -- weighted probability of drawing a sample more than `--heavy`
              times the weighted-median MB. Should predict the TAIL (the
              spike rate).

If p_heavy tracks measured spike rate across runs whose mixes differ, heavy
samples own the tail. If mean_mb tracks the median but p_heavy does NOT track
the spikes, then bytes explain the body and something else still owns the tail
-- which is a genuinely useful narrowing either way.

WHAT THIS CANNOT SETTLE
-----------------------
Bytes and decode cost co-vary: a big clip is big because it has more/larger
frames, which is also more to decode. This cannot separate "slow because many
bytes crossed DAOS" from "slow because many pixels were decoded". Both are
per-sample and both rotate. Distinguishing them needs a deliberate arm (same
clips, re-encoded to uniform byte size), and that is only worth buying if the
correlation here survives.

Per-run byte sizes come from ONE shard per source (`stat`, no dfuse walk --
[[polaris-login-node-fragile]] and the glob-hangs-on-dfuse gotcha). Shards
within a source were built to a byte target, so one is representative of the
source; it is NOT representative of within-source variance, which this
therefore understates.

Usage:
    python scripts/sample_bytes_correlation.py --root /flare/.../checkpoints
"""
import argparse
import glob
import json
import os
import statistics as st

import yaml

COL_DLOAD = 5
WRAP_MS = 343597.0  # XPU event counter period; a negative delta is one wrap
_BYTES_CACHE = {}


def source_bytes_per_sample(path):
    """-> MB per sample for one source dir, or None.

    Uses the first shard only. os.listdir on the source dir is bounded (a few
    thousand entries) -- deliberately not glob.glob across the corpus root,
    which hangs on dfuse.
    """
    if path in _BYTES_CACHE:
        return _BYTES_CACHE[path]
    val = None
    try:
        meta = json.load(open(os.path.join(path, "metadata.json")))
        urls = meta.get("shard_urls") or []
        sc = meta.get("sample_count")
        if urls and sc:
            first = os.path.join(path, urls[0])
            if os.path.exists(first):
                nbytes = os.path.getsize(first)
                per_shard = sc / len(urls)
                if per_shard > 0 and nbytes > 1024:
                    val = (nbytes / 1e6) / per_shard
    except (OSError, ValueError, KeyError):
        val = None
    _BYTES_CACHE[path] = val
    return val


def predict(cfg, heavy):
    """-> (mean_mb, p_heavy, n_src) from a run's own source mix."""
    d = cfg.get("data", {})
    paths = d.get("datasets") or []
    T = d.get("sampling_temperature", 0.5)

    items = []
    for p in paths:
        f = os.path.join(p, "metadata.json")
        if not os.path.exists(f):
            continue
        try:
            m = json.load(open(f))
        except (OSError, ValueError):
            continue
        sc = m.get("sample_count")
        mb = source_bytes_per_sample(p)
        if not (sc and mb):
            continue
        items.append((sc, mb))
    if len(items) < 2:
        return None, None, 0

    tot = sum(sc for sc, _ in items)
    raw = [(sc / tot) ** T for sc, _ in items]
    Z = sum(raw)
    probs = [r / Z for r in raw]

    mean_mb = sum(pr * mb for pr, (_, mb) in zip(probs, items))

    # Weighted median MB, then the weighted mass above heavy x that median.
    order = sorted(zip((mb for _, mb in items), probs))
    cum, med_mb = 0.0, order[-1][0]
    for mb, pr in order:
        cum += pr
        if cum >= 0.5:
            med_mb = mb
            break
    p_heavy = sum(pr for mb, pr in order if mb > heavy * med_mb)
    return mean_mb, p_heavy, len(items)


def rank_series(path, col=COL_DLOAD):
    """Last allocation segment of one rank, in seconds.

    Segment on the repeated CSV HEADER, not on itr decreasing -- itr cycles
    every ipe, so it marks epochs, not allocations.
    """
    seg = []
    for line in open(path):
        if line.startswith("epoch,"):
            seg = []
            continue
        q = line.rstrip("\n").split(",")
        if len(q) <= col:
            continue
        try:
            v = float(q[col])
        except ValueError:
            continue
        if v < 0:
            v += WRAP_MS  # a wrapped row is a SLOW row -- unwrap, never drop
        seg.append(v / 1000.0)
    return seg


def measure(run_dir, factor, max_ranks, min_iters, abs_thresh):
    """-> per-run dataload statistics, with THREE tail measures, not one.

    The relative measure (`spike`, dataload > factor x that rank's own median)
    is retained but must not be used alone. It is contaminated by construction:
    across these runs r(dload_median, spike_rate) = -0.762, so a run with a
    heavier median clears its own 3x threshold LESS often for the same absolute
    stall. The relative rate partly measures 1/median -- the threshold moving,
    not the physics.

    Two threshold-free companions:
      `excess` -- mean seconds per iteration spent above the rank's median,
                  counting only iterations above it. Directly the wall-clock
                  cost of the tail, in seconds, comparable across runs.
      `p99_abs`-- the 99th percentile in absolute seconds.
    plus `abs_rate`, the fraction over a fixed absolute threshold shared by all
    runs. Any tail claim must hold on the absolute measures.
    """
    files = sorted(glob.glob(os.path.join(run_dir, "log_r*.csv")))
    if not files:
        return None
    # Stride across the rank space: ranks 0..47 are four nodes, and a node-local
    # effect would masquerade as a corpus effect.
    step = max(1, len(files) // max_ranks)
    fr, meds, exc, p99, absr = [], [], [], [], []
    for f in files[::step][:max_ranks]:
        s = rank_series(f)
        if len(s) < min_iters:
            continue
        m = st.median(s)
        if m <= 0:
            continue
        fr.append(sum(1 for v in s if v > factor * m) / len(s))
        meds.append(m)
        exc.append(sum(v - m for v in s if v > m) / len(s))
        srt = sorted(s)
        p99.append(srt[min(len(srt) - 1, int(0.99 * len(srt)))])
        absr.append(sum(1 for v in s if v > abs_thresh) / len(s))
    if len(fr) < 4:
        return None
    return dict(spike=st.mean(fr), dload_med=st.median(meds),
                excess=st.mean(exc), p99=st.median(p99),
                abs_rate=st.mean(absr), nranks=len(fr))


def pearson(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--factor", type=float, default=3.0,
                    help="dataload > factor x rank's own median = a spike")
    ap.add_argument("--heavy", type=float, default=3.0,
                    help="sample > heavy x weighted-median MB = a heavy draw")
    ap.add_argument("--max-ranks", type=int, default=48)
    ap.add_argument("--min-iters", type=int, default=200)
    ap.add_argument("--abs-thresh", type=float, default=5.0,
                    help="fixed absolute seconds defining a tail iteration, "
                         "shared by all runs (the relative threshold is not)")
    a = ap.parse_args()

    rows = []
    for dirpath, _, files in os.walk(a.root):
        if "params-pretrain.yaml" not in files or "log_r0.csv" not in files:
            continue
        try:
            cfg = yaml.safe_load(open(os.path.join(dirpath, "params-pretrain.yaml")))
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if not isinstance(cfg, dict):
            continue
        mean_mb, p_heavy, nsrc = predict(cfg, a.heavy)
        if mean_mb is None:
            continue
        m = measure(dirpath, a.factor, a.max_ranks, a.min_iters, a.abs_thresh)
        if not m:
            continue
        rows.append((os.path.relpath(dirpath, a.root), nsrc, mean_mb, p_heavy, m))

    if not rows:
        print("no runs with both a config snapshot and enough rank CSVs")
        return
    rows.sort(key=lambda r: -r[2])

    hdr = (f"{'run':38s} {'src':>4s} {'meanMB':>7s} {'p_hvy':>6s} "
           f"{'dl_med':>7s} {'spike':>7s} {'excess':>7s} {'p99':>6s} "
           f"{'>{:.0f}s'.format(a.abs_thresh):>6s}")
    print(f"predictors from each run's OWN source mix (temperature-weighted)\n"
          f"  meanMB  = mean bytes per sample   -> should predict the BODY\n"
          f"  p_hvy   = P(sample > {a.heavy}x median MB) -> should predict the TAIL\n"
          f"tail measures: spike is RELATIVE (contaminated, see below); excess "
          f"(s/iter above\nown median) and p99 and >{a.abs_thresh:.0f}s are "
          f"absolute and are what any tail claim must rest on.\n")
    print(hdr)
    print("-" * len(hdr))
    MB, PH, DL, SP, EX, P9, AR = [], [], [], [], [], [], []
    for name, nsrc, mb, ph, m in rows:
        print(f"{name[:38]:38s} {nsrc:4d} {mb:7.2f} {ph:6.3f} "
              f"{m['dload_med']:7.2f} {m['spike']:7.4f} {m['excess']:7.3f} "
              f"{m['p99']:6.2f} {m['abs_rate']:6.4f}")
        MB.append(mb)
        PH.append(ph)
        DL.append(m["dload_med"])
        SP.append(m["spike"])
        EX.append(m["excess"])
        P9.append(m["p99"])
        AR.append(m["abs_rate"])

    print(f"\nn={len(MB)} runs")
    r_body = pearson(MB, DL)
    sp_mb = max(MB) / min(MB) if min(MB) > 0 else float("inf")
    print(f"  BODY : r(meanMB, dload_median) = {r_body:+.3f}   "
          f"predictor spread {sp_mb:.2f}x")

    # The relative spike rate is reported ONLY alongside the artifact test that
    # discredits it. r(dload_median, spike_rate) measures how much of "spike
    # rate" is really 1/median: a run with a heavier median clears its own 3x
    # threshold less often for the SAME absolute stall.
    r_art = pearson(DL, SP)
    print(f"  TAIL (relative, SUSPECT):")
    print(f"    r(p_heavy, spike_rate)       = {pearson(PH, SP):+.3f}")
    print(f"    r(dload_median, spike_rate)  = {r_art:+.3f}  <-- artifact test; "
          f"strongly negative means")
    print(f"                                       'spike rate' is largely "
          f"1/median, not physics")
    print(f"  TAIL (absolute, TRUSTWORTHY):")
    print(f"    r(meanMB, excess_s_per_iter) = {pearson(MB, EX):+.3f}")
    print(f"    r(meanMB, p99_seconds)       = {pearson(MB, P9):+.3f}")
    print(f"    r(meanMB, rate_over_{a.abs_thresh:.0f}s)     = {pearson(MB, AR):+.3f}")
    print(f"    r(p_heavy, excess)           = {pearson(PH, EX):+.3f}")

    print()
    if sp_mb < 1.3:
        print("VERDICT: INCONCLUSIVE -- meanMB barely varies across these runs.")
        return
    body_ok = r_body > 0.6
    # Judge the tail on the ABSOLUTE measures only. Requiring two of three to
    # agree guards against one statistic being driven by a single outlier run.
    abs_r = [pearson(MB, EX), pearson(MB, P9), pearson(MB, AR)]
    tail_pos = sum(1 for r in abs_r if r > 0.5)
    tail_neg = sum(1 for r in abs_r if r < 0.2)

    if tail_pos >= 2:
        print("VERDICT: heavy samples are CONSISTENT WITH owning the tail --")
        print("on ABSOLUTE tail measures, which the relative spike rate is not.")
        print("Bytes and decode cost co-vary and this cannot separate them;")
        print("the separating arm is same-clips-uniform-bytes.")
    elif body_ok and tail_neg >= 2:
        print("VERDICT: bytes explain the BODY of dataload but NOT the tail.")
        print("That is a real narrowing -- the tail is not simply 'sometimes a")
        print("big clip', and the remaining candidates are contention and")
        print("per-read latency variance, not payload size.")
    else:
        print("VERDICT: AMBIGUOUS on the tail. Do not promote either way")
        print("([[no-lazy-cause-labels]]).")


if __name__ == "__main__":
    main()
