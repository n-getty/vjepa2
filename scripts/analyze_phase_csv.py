#!/usr/bin/env python3
"""Summarize per-rank log_r*.csv files including the new phase columns.

Usage:
    python scripts/analyze_phase_csv.py <run_folder> [<run_folder> ...]

Prints a per-run table with median iter/gpu/data + phase breakdowns,
and a comparison row showing the delta vs the first run listed (useful
for diagnosing multi-node overhead).
"""
import csv
import glob
import statistics
import sys
from pathlib import Path

NEW_COLS = ("fwd-target-ms", "fwd-context-ms", "backward-ms",
            "opt-step-ms", "ema-ms")


def medians(folder, start_row=50):
    csvs = sorted(glob.glob(str(Path(folder) / "log_r*.csv")))
    if not csvs:
        return None
    with open(csvs[0]) as f:
        header = next(csv.reader(f))
    have_phases = "backward-ms" in header
    col = {h: i for i, h in enumerate(header)}
    per_rank = []
    for c in csvs:
        with open(c) as f:
            rdr = csv.reader(f)
            next(rdr)
            rows = list(rdr)
        if len(rows) <= start_row + 5:
            continue
        m = {}
        for k in ("iter-time(ms)", "gpu-time(ms)", "dataload-time(ms)"):
            vals = [float(r[col[k]]) for r in rows[start_row:]
                    if r[col[k]] not in ("", None)]
            m[k] = statistics.median(vals) if vals else 0
        if have_phases:
            for k in NEW_COLS:
                vals = [float(r[col[k]]) for r in rows[start_row:]
                        if r[col[k]] not in ("", None)]
                m[k] = statistics.median(vals) if vals else 0
        per_rank.append((c, m))
    return per_rank, have_phases


def summarize(folder, start_row=50):
    res = medians(folder, start_row)
    if res is None:
        print(f"NO LOGS at {folder}")
        return None
    rows, have_phases = res
    nranks = len(rows)
    avg = {}
    for k in rows[0][1]:
        avg[k] = statistics.median([m[k] for _, m in rows])
    label = Path(folder).name
    base = (f"{label:38s} ranks={nranks:2d}  iter={avg['iter-time(ms)']:6.0f}  "
            f"gpu={avg['gpu-time(ms)']:6.0f}  data={avg['dataload-time(ms)']:3.0f}")
    if have_phases:
        base += ("  | fwdT={fwd-target-ms:6.1f}  fwdC={fwd-context-ms:6.1f}  "
                 "bwd={backward-ms:7.1f}  opt={opt-step-ms:5.1f}  "
                 "ema={ema-ms:5.1f}").format(**avg)
    print(base)
    return avg


if __name__ == "__main__":
    folders = sys.argv[1:]
    if not folders:
        print(__doc__)
        sys.exit(1)
    aggregates = []
    for f in folders:
        a = summarize(f)
        if a is not None:
            aggregates.append((f, a))
    if len(aggregates) > 1 and "backward-ms" in aggregates[0][1]:
        print("\nDelta vs first run (positive = slower):")
        base_label, base = aggregates[0]
        for f, a in aggregates[1:]:
            deltas = " ".join(
                f"{k.split('-ms')[0]}+{a[k] - base[k]:+.1f}"
                for k in ("iter-time(ms)", "gpu-time(ms)") + NEW_COLS
                if k in base
            )
            print(f"  {Path(f).name:38s} {deltas}")
