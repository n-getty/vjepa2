#!/usr/bin/env python3
"""Sanity-check the WebDataset loader before a real training run.

Given a training YAML config, this script simulates a single rank inside a
larger world (default ``world_size=192``, matching 16 nodes x 12 ranks) and
pulls N batches through a *cheap* pipeline that mirrors the production loader
in ``src/datasets/webdataset.py`` — same per-dataset stream + ``RandomMix``,
same ``split_by_node`` / ``split_by_worker`` splitters, same shard list — but
skips video decode so it can run on a login node in seconds.

For each batch it tracks the per-sample ``source_dataset`` and
``source_path`` recorded in the WDS ``.json`` member, then reports:

  * per-dataset sample frequency vs configured ``datasets_weights``
  * source-video diversity within each batch
  * decode latency per batch (read+parse only — not real video decode)
  * total throughput

Exits 0 on pass, 1 on any failure.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def compute_mixing_probs(sample_counts, datasets_weights=None, temperature=0.5):
    n = len(sample_counts)
    if n == 0:
        return []
    counts = [max(1.0, float(c or 0)) for c in sample_counts]
    if datasets_weights is None:
        datasets_weights = [1.0] * n
    if len(datasets_weights) != n:
        raise ValueError("datasets_weights length must match sample_counts")
    base = [w * (c ** float(temperature)) for w, c in zip(datasets_weights, counts)]
    total = float(sum(base))
    if total <= 0:
        return [1.0 / n] * n
    return [b / total for b in base]


def _fake_distributed(rank: int, world_size: int):
    """Make ``wds.split_by_node`` think we're rank/world without a real PG.

    The webdataset splitters check ``torch.distributed.is_initialized()`` and
    fall back to env vars (``RANK``, ``WORLD_SIZE``). Setting both covers
    every code path across webdataset releases.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"


def _sample_counts(data_paths):
    counts = []
    for path in data_paths:
        meta_path = Path(path) / "metadata.json"
        if meta_path.exists():
            with meta_path.open() as handle:
                counts.append(int(json.load(handle).get("sample_count", 0)))
        else:
            counts.append(0)
    return counts


def _expected_frequencies(data_paths, datasets_weights, sampling_temperature):
    probs = compute_mixing_probs(
        _sample_counts(data_paths),
        datasets_weights=datasets_weights,
        temperature=sampling_temperature,
    )
    return {Path(path).name: prob for path, prob in zip(data_paths, probs)}


def _build_pipeline(data_paths, datasets_weights, sampling_temperature, batch_size, num_workers,
                    rank, world_size, shuffle_buffer=1000):
    import webdataset as wds

    def cheap_decode(sample):
        # Pull source_dataset / source_path out of the json sidecar. Skip
        # everything else (no video read).
        for key in ("json",):
            if key in sample:
                try:
                    meta = json.loads(sample[key].decode("utf-8"))
                except Exception:
                    return None
                return {
                    "source_dataset": meta.get("source_dataset", "unknown"),
                    "source_path": meta.get("source_path", sample.get("__key__", "?")),
                    "shard": sample.get("__url__", "?"),
                }
        return None

    def is_not_none(x):
        return x is not None

    streams = []
    for path in data_paths:
        shard_files = sorted(f for f in os.listdir(path) if f.endswith(".tar"))
        urls = [os.path.join(path, f) for f in shard_files]
        if not urls:
            raise SystemExit(f"No .tar shards in {path}")
        s = wds.WebDataset(
            urls,
            resampled=True,
            shardshuffle=True,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            handler=wds.warn_and_continue,
        ).shuffle(shuffle_buffer).map(cheap_decode).select(is_not_none)
        streams.append(s)

    if len(streams) == 1:
        mixed = streams[0]
    else:
        probs = compute_mixing_probs(
            _sample_counts(data_paths),
            datasets_weights=datasets_weights,
            temperature=sampling_temperature,
        )
        mixed = wds.RandomMix(streams, probs=probs)

    loader = wds.WebLoader(
        mixed,
        batch_size=None,  # we accumulate batches ourselves to avoid collator
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return loader


def _accumulate_batches(loader, batch_size, n_batches):
    """Pull samples from the worker pool and group into batches of size
    ``batch_size``. Yields one batch at a time."""
    batch = []
    it = iter(loader)
    while len(batch) < batch_size * 0:  # no-op; just structure
        pass
    yielded = 0
    while yielded < n_batches:
        for sample in it:
            batch.append(sample)
            if len(batch) >= batch_size:
                yield batch
                batch = []
                yielded += 1
                if yielded >= n_batches:
                    return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="Training YAML config (uses data.datasets / data.datasets_weights).")
    p.add_argument("--world-size", type=int, default=192,
                   help="Simulated world size (default 192 = 16 nodes x 12 ranks).")
    p.add_argument("--rank", type=int, default=0, help="Simulated rank (default 0).")
    p.add_argument("--n-batches", type=int, default=200, help="Batches to pull (default 200).")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (default 2).")
    p.add_argument("--weight-tolerance", type=float, default=0.15,
                   help="Per-dataset weight tolerance (default 0.15 = ±15%).")
    p.add_argument("--sampling-temperature", type=float, default=None,
                   help="Override data.sampling_temperature from config.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    datasets = data_cfg["datasets"]
    weights = data_cfg.get("datasets_weights")
    batch_size = int(data_cfg.get("batch_size", 4))
    dataset_names = [Path(p).name for p in datasets]
    if weights is None:
        weights = [1.0] * len(datasets)
    if len(weights) != len(datasets):
        print(f"FAIL: datasets_weights length ({len(weights)}) != datasets length "
              f"({len(datasets)})", file=sys.stderr)
        return 1
    sampling_temperature = (
        data_cfg.get("sampling_temperature", 0.5)
        if args.sampling_temperature is None
        else args.sampling_temperature
    )
    expected_freq = _expected_frequencies(datasets, weights, sampling_temperature)

    _fake_distributed(args.rank, args.world_size)
    loader = _build_pipeline(
        data_paths=datasets,
        datasets_weights=weights,
        sampling_temperature=sampling_temperature,
        batch_size=batch_size,
        num_workers=args.num_workers,
        rank=args.rank,
        world_size=args.world_size,
    )

    per_dataset_count: Counter = Counter()
    per_batch_unique_sources = []
    batch_latencies = []
    shard_counts: Counter = Counter()
    t_start = time.time()
    n_samples_total = 0

    last_t = time.time()
    for bi, batch in enumerate(_accumulate_batches(loader, batch_size, args.n_batches)):
        now = time.time()
        batch_latencies.append(now - last_t)
        last_t = now
        sources_in_batch = set()
        for s in batch:
            per_dataset_count[s["source_dataset"]] += 1
            sources_in_batch.add(s["source_path"])
            shard_counts[s["shard"]] += 1
            n_samples_total += 1
        per_batch_unique_sources.append(len(sources_in_batch))
        if (bi + 1) % 25 == 0:
            print(f"  pulled {bi+1}/{args.n_batches} batches", file=sys.stderr)

    elapsed = time.time() - t_start
    if n_samples_total == 0:
        print("FAIL: no samples observed", file=sys.stderr)
        return 1

    print("\n=== per-dataset frequency ===")
    failed_freq = []
    for name in dataset_names:
        observed = per_dataset_count.get(name, 0) / n_samples_total
        expected = expected_freq[name]
        delta = observed - expected
        pct_off = abs(delta) / expected if expected > 0 else float("inf")
        flag = ""
        if pct_off > args.weight_tolerance:
            flag = f"  FAIL (>±{int(args.weight_tolerance*100)}%)"
            failed_freq.append(name)
        print(f"  {name:<22} expected={expected:.3f}  observed={observed:.3f}  "
              f"Δ={delta:+.3f}  ({pct_off*100:.1f}% off){flag}")

    # Catch datasets that should have appeared but didn't.
    missing = [n for n in dataset_names if per_dataset_count.get(n, 0) == 0]
    if missing:
        print(f"  MISSING (0 samples): {missing}", file=sys.stderr)

    print("\n=== per-batch source-video diversity ===")
    if per_batch_unique_sources:
        avg_div = sum(per_batch_unique_sources) / len(per_batch_unique_sources)
        min_div = min(per_batch_unique_sources)
        max_div = max(per_batch_unique_sources)
        single_source_batches = sum(1 for d in per_batch_unique_sources if d <= 1)
        print(f"  unique source videos per batch: min={min_div}  avg={avg_div:.2f}  max={max_div}")
        print(f"  batches with only 1 source video: {single_source_batches}/{len(per_batch_unique_sources)}")
    else:
        avg_div = 0
        single_source_batches = 0

    print("\n=== throughput ===")
    avg_lat = sum(batch_latencies) / max(1, len(batch_latencies))
    print(f"  {args.n_batches} batches x {batch_size} samples in {elapsed:.1f}s")
    print(f"  avg batch latency: {avg_lat*1000:.1f} ms  ({batch_size/avg_lat:.1f} samples/s)")
    print(f"  shards observed:   {len(shard_counts)}")

    pass_ok = True
    if failed_freq:
        print(f"\nFAIL: {len(failed_freq)} datasets outside ±{int(args.weight_tolerance*100)}%: {failed_freq}",
              file=sys.stderr)
        pass_ok = False
    if missing:
        print(f"FAIL: {len(missing)} datasets had 0 samples: {missing}", file=sys.stderr)
        pass_ok = False
    if per_batch_unique_sources and avg_div < 1.5 and batch_size > 1:
        print(f"WARN: avg source-video diversity per batch is low ({avg_div:.2f}); "
              f"reshard may be needed", file=sys.stderr)
    if pass_ok:
        print("\nOK: loader behavior within tolerance.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
