#!/usr/bin/env python3
"""Is the dataload tail the cost of OPENING A SHARD?

WHY THIS EXISTS
---------------
The dataload tail is the top measured lever (17.2% of wall clock on the live
DAOS path, `scripts/backward_drift_scan.py`), and its owner is still unnamed.
Three candidates are already excluded by measurement:

  - decode cost      -- exonerated: the GOP re-encode moved the BODY of the
                        distribution, not the tail
  - read granularity -- exonerated: GOPEN_BUFFER, default already at 2 MB
  - node count       -- exonerated: the tail is fully present at 1 node, and
                        the straggler rank ROTATES every iteration, so no node
                        is sick

That last property is the clue this script follows. A rotating straggler with
no sick node is what you get when every rank independently, occasionally, pays
a cost that most iterations do not have. Opening a new tar is exactly such a
cost, and `src/datasets/webdataset.py:735` builds the stream with
`resampled=True`, so a rank reads one shard to exhaustion and then opens
another -- with replacement, at a per-rank phase offset nothing synchronizes.

THE PREDICTION, AND WHY IT IS FALSIFIABLE
------------------------------------------
Shard sizes in this corpus are small: 3.5-20 samples on 14 of 16 sources
(pe_video at 500 and lemon at 102 are the exceptions). So the open rate is not
negligible -- it is computable, per source, from metadata alone:

    opens per sample = sum_s  p(s) / samples_per_shard(s)

where p(s) is the temperature-weighted mixing probability. Multiply by
`batch_size` for opens per rank-iteration.

The test: each run mixed a DIFFERENT subset of sources (the corpus ablations
were designed for a different purpose entirely, but they vary exactly the
variable this needs), so each has a different PREDICTED rate from its own
`params-pretrain.yaml`. If shard-opening owns the tail, the MEASURED spike rate
must track the predicted rate ACROSS runs. If measured spike rate is flat while
predicted varies severalfold, the hypothesis is dead and this script says so.

That across-run design is what makes this evidence rather than a story. Any
single run can be fitted by any hypothesis with a free parameter.

WHAT THE NUMBERS DO AND DO NOT MEAN
------------------------------------
`measured` counts iterations where a rank's dataload exceeds `--factor` x its
OWN median. That is a LOWER BOUND on open events, deliberately:

  - a shard open that lands inside a prefetch window costs nothing visible
  - a cheap open (small tar, warm cache) will not clear 3x
  - two opens in one iteration count once

So expect measured < predicted. What carries the argument is the CORRELATION,
not the ratio. Per [[no-lazy-cause-labels]], a correlation across 5-6 runs is
consistent-with, not proof of, causation -- the honest verdict wording is built
into the output.

Per-rank medians, never a global median: ranks differ systematically in their
draw, and a global median would import that between-rank spread into the
within-rank spike count.

Usage:
    python scripts/shard_open_correlation.py --root /flare/.../checkpoints
    python scripts/shard_open_correlation.py --root ... --factor 3 --max-ranks 48
"""
import argparse
import glob
import json
import os
import statistics as st

import yaml

COL_DLOAD = 5
WRAP_MS = 343597.0  # XPU event counter period; a negative delta is one wrap


def predicted_open_rate(cfg):
    """-> (opens_per_rank_iter, n_sources_resolved, samples_per_shard_by_src).

    Mirrors the mixing in `src/datasets/webdataset.py`: sources are drawn with
    temperature-weighted probability, and a source contributes opens in
    inverse proportion to its shard size. Sources whose metadata is missing are
    dropped from BOTH the numerator and the normalizer, so a partial resolve
    biases the estimate toward the sources that did resolve rather than
    silently scaling it down.
    """
    d = cfg.get("data", {})
    paths = d.get("datasets") or []
    T = d.get("sampling_temperature", 0.5)
    bs = d.get("batch_size", 1)

    info = []
    for p in paths:
        f = os.path.join(p, "metadata.json")
        if not os.path.exists(f):
            continue
        try:
            m = json.load(open(f))
        except (OSError, ValueError):
            continue
        n = len(m.get("shard_urls") or [])
        sc = m.get("sample_count")
        if not (n and sc):
            continue
        info.append((os.path.basename(p), n, sc, sc / n))
    if not info:
        return None, 0, {}

    tot = sum(sc for _, _, sc, _ in info)
    raw = [(sc / tot) ** T for _, _, sc, _ in info]
    Z = sum(raw)
    rate = sum((r / Z) / ps for r, (_, _, _, ps) in zip(raw, info))
    return rate * bs, len(info), {nm: ps for nm, _, _, ps in info}


def rank_segments(path, col=COL_DLOAD):
    """-> the LAST allocation segment of one rank's series, in seconds.

    Segment on the repeated CSV HEADER, not on `itr` decreasing: `itr` cycles
    every `ipe` by design, so it marks epochs, not allocations. (Getting this
    wrong once gave 227 "chunks" instead of 14.) The last segment is the one
    furthest past warmup and inside a single PBS job.
    """
    vals, seg = [], []
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
        vals = seg
    return vals


def measure(run_dir, factor, max_ranks, min_iters):
    files = sorted(glob.glob(os.path.join(run_dir, "log_r*.csv")))
    if not files:
        return None
    # Stride across the rank space rather than taking the first N: ranks 0..47
    # are the first four nodes, and node-local effects would masquerade as a
    # corpus effect.
    step = max(1, len(files) // max_ranks)
    fr, meds = [], []
    for f in files[::step][:max_ranks]:
        s = rank_segments(f)
        if len(s) < min_iters:
            continue
        m = st.median(s)
        if m <= 0:
            continue
        fr.append(sum(1 for v in s if v > factor * m) / len(s))
        meds.append(m)
    if len(fr) < 4:
        return None
    return dict(spike=st.mean(fr), spike_med=st.median(fr),
                dload_med=st.median(meds), nranks=len(fr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--factor", type=float, default=3.0)
    ap.add_argument("--max-ranks", type=int, default=48)
    ap.add_argument("--min-iters", type=int, default=200)
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
        pred, nsrc, _ = predicted_open_rate(cfg)
        if pred is None:
            continue
        m = measure(dirpath, a.factor, a.max_ranks, a.min_iters)
        if not m:
            continue
        rows.append((os.path.relpath(dirpath, a.root), nsrc, pred, m))

    if not rows:
        print("no runs with both a config snapshot and enough rank CSVs")
        return
    rows.sort(key=lambda r: -r[2])

    hdr = (f"{'run':44s} {'src':>4s} {'rnk':>4s} {'pred/iter':>9s} "
           f"{'meas/iter':>9s} {'ratio':>6s} {'dload_med':>9s}")
    print(f"predicted = shard-opens per rank-iteration from that run's OWN "
          f"source mix\nmeasured  = fraction of iters with dataload > "
          f"{a.factor}x that rank's own median\n")
    print(hdr)
    print("-" * len(hdr))
    P, M = [], []
    for name, nsrc, pred, m in rows:
        ratio = m["spike"] / pred if pred > 0 else float("nan")
        print(f"{name[:44]:44s} {nsrc:4d} {m['nranks']:4d} {pred:9.4f} "
              f"{m['spike']:9.4f} {ratio:6.2f} {m['dload_med']:9.2f}")
        P.append(pred)
        M.append(m["spike"])

    print()
    if len(P) < 3:
        print(f"only {len(P)} runs -- too few to correlate. Need >=3 with "
              f"differing source mixes.")
        return

    # Pearson on the raw rates. Spearman would be more robust but with n<10 and
    # a near-tie cluster it mostly reports the tie-breaking noise.
    mp, mm = st.mean(P), st.mean(M)
    num = sum((p - mp) * (q - mm) for p, q in zip(P, M))
    den = (sum((p - mp) ** 2 for p in P) * sum((q - mm) ** 2 for q in M)) ** 0.5
    r = num / den if den > 0 else float("nan")
    spread = max(P) / min(P) if min(P) > 0 else float("inf")

    print(f"n={len(P)} runs   predicted spread {spread:.2f}x   Pearson r = {r:.3f}")
    print()
    if spread < 1.5:
        print("VERDICT: INCONCLUSIVE -- the predicted rate barely varies across")
        print("these runs, so there is no lever for the correlation to detect.")
        print("Need runs whose source mixes differ more (or a deliberate arm).")
    elif r > 0.7:
        print("VERDICT: CONSISTENT WITH shard-opening owning the tail. Measured")
        print("spike rate tracks a rate predicted from METADATA ALONE, across")
        print("runs that were never designed to vary it. Not proof -- corpus mix")
        print("co-varies with decode cost per sample, and this cannot separate")
        print("them. The separating test is a deliberate arm: same corpus,")
        print("re-sharded to ~500 samples/shard, nothing else changed.")
    elif r < 0.3:
        print("VERDICT: REFUTED (as a primary cause). Measured spike rate does")
        print("not track predicted open rate despite a real spread in the")
        print("predictor. Shard-opening is not what the tail is made of; look")
        print("elsewhere for task #14.")
    else:
        print("VERDICT: WEAK/AMBIGUOUS. Some association, not enough to act on.")
        print("Do not promote this to a finding ([[no-lazy-cause-labels]]).")


if __name__ == "__main__":
    main()
