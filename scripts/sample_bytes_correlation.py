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


def measure(run_dir, factor, max_ranks, min_iters):
    files = sorted(glob.glob(os.path.join(run_dir, "log_r*.csv")))
    if not files:
        return None
    # Stride across the rank space: ranks 0..47 are four nodes, and a node-local
    # effect would masquerade as a corpus effect.
    step = max(1, len(files) // max_ranks)
    fr, meds = [], []
    for f in files[::step][:max_ranks]:
        s = rank_series(f)
        if len(s) < min_iters:
            continue
        m = st.median(s)
        if m <= 0:
            continue
        fr.append(sum(1 for v in s if v > factor * m) / len(s))
        meds.append(m)
    if len(fr) < 4:
        return None
    return dict(spike=st.mean(fr), dload_med=st.median(meds), nranks=len(fr))


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
        m = measure(dirpath, a.factor, a.max_ranks, a.min_iters)
        if not m:
            continue
        rows.append((os.path.relpath(dirpath, a.root), nsrc, mean_mb, p_heavy, m))

    if not rows:
        print("no runs with both a config snapshot and enough rank CSVs")
        return
    rows.sort(key=lambda r: -r[2])

    hdr = (f"{'run':42s} {'src':>4s} {'meanMB':>7s} {'p_heavy':>8s} "
           f"{'dl_med':>7s} {'spike':>7s}")
    print(f"predictors from each run's OWN source mix (temperature-weighted)\n"
          f"  meanMB  = mean bytes per sample   -> should predict the BODY\n"
          f"  p_heavy = P(sample > {a.heavy}x median MB) -> should predict the TAIL\n")
    print(hdr)
    print("-" * len(hdr))
    MB, PH, DL, SP = [], [], [], []
    for name, nsrc, mb, ph, m in rows:
        print(f"{name[:42]:42s} {nsrc:4d} {mb:7.2f} {ph:8.4f} "
              f"{m['dload_med']:7.2f} {m['spike']:7.4f}")
        MB.append(mb)
        PH.append(ph)
        DL.append(m["dload_med"])
        SP.append(m["spike"])

    print(f"\nn={len(MB)} runs")
    r_body = pearson(MB, DL)
    r_tail = pearson(PH, SP)
    r_cross = pearson(MB, SP)
    sp_mb = max(MB) / min(MB) if min(MB) > 0 else float("inf")
    sp_ph = max(PH) / min(PH) if min(PH) > 0 else float("inf")
    print(f"  BODY : r(meanMB, dload_median) = {r_body:+.3f}   "
          f"predictor spread {sp_mb:.2f}x")
    print(f"  TAIL : r(p_heavy, spike_rate)  = {r_tail:+.3f}   "
          f"predictor spread {sp_ph:.2f}x")
    print(f"  xchk : r(meanMB, spike_rate)   = {r_cross:+.3f}")

    print()
    if sp_mb < 1.3 and sp_ph < 1.3:
        print("VERDICT: INCONCLUSIVE -- neither predictor varies enough across")
        print("these runs to be detectable. Needs a deliberate arm.")
        return
    body_ok = r_body > 0.6 and sp_mb >= 1.3
    tail_ok = r_tail > 0.6 and sp_ph >= 1.3
    if tail_ok:
        print("VERDICT: heavy samples are CONSISTENT WITH owning the tail.")
        print("Bytes-per-sample and decode cost co-vary and this cannot")
        print("separate them; the separating arm is same-clips-uniform-bytes.")
    elif body_ok:
        print("VERDICT: bytes explain the BODY of dataload but NOT the tail.")
        print("That is a real narrowing -- it means the tail is not simply")
        print("'sometimes a big clip', and the remaining candidates are")
        print("contention and per-read latency variance, not payload size.")
    else:
        print("VERDICT: REFUTED. Per-sample bytes predict neither the body nor")
        print("the tail. Do not label this a cause ([[no-lazy-cause-labels]]).")


if __name__ == "__main__":
    main()
