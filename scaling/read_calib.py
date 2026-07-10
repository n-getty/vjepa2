"""Read calibration runs (scaling/gen_calib_configs.py) -> per-cell clips/s + OOM verdict.

For each calib_<model>_t<tiles>_bs<bs> folder, read log_r0.csv, drop warmup iters, report:
  - median iter-time (ms), clips/s/tile = per_rank_bs / (iter_ms/1000)
  - status: OK (>=20 iters logged), SHORT (ran but few iters), OOM/FAIL (no/empty log)
Grouped by model so you can see whether clips/s RISES with per_rank_bs (the amortization question)
and where OOM caps the batch. This directly sets the per-size (tiles, per_rank_bs) topology in
scaling/gen_configs.py for the real sweep.

Usage:
  python -m scaling.read_calib --exp-root /flare/ModCon/ngetty/experiments/scaling_calib
"""

import argparse
import csv
import glob
import os
import re
import statistics

SLUG_RE = re.compile(r"calib_(?P<model>vit_[a-z]+)_t(?P<tiles>\d+)_bs(?P<bs>\d+)")


def read_one(folder):
    f = os.path.join(folder, "log_r0.csv")
    if not os.path.exists(f):
        return None
    its = []
    with open(f) as fh:
        for row in csv.DictReader(fh):
            try:
                its.append(float(row["iter-time(ms)"]))
            except (KeyError, ValueError):
                pass
    if not its:
        return {"n": 0, "med_ms": None}
    drop = min(10, len(its) // 3)
    steady = its[drop:] or its
    return {"n": len(its), "med_ms": statistics.median(steady)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", default="/flare/ModCon/ngetty/experiments/scaling_calib")
    args = ap.parse_args()

    rows = []
    for folder in sorted(glob.glob(os.path.join(args.exp_root, "calib_*"))):
        m = SLUG_RE.search(os.path.basename(folder))
        if not m:
            continue
        model, tiles, bs = m["model"], int(m["tiles"]), int(m["bs"])
        r = read_one(folder)
        if r is None:
            status, med, cps, n = "OOM/FAIL(no log)", None, None, 0
        elif r["n"] == 0:
            status, med, cps, n = "FAIL(empty)", None, None, 0
        else:
            n = r["n"]; med = r["med_ms"]
            cps = bs / (med / 1000.0)
            status = "OK" if n >= 20 else f"SHORT(n={n})"
        rows.append((model, tiles, bs, n, med, cps, status))

    order = {"vit_tiny": 0, "vit_small": 1, "vit_base": 2, "vit_large": 3,
             "vit_giant": 4, "vit_gigantic": 5}
    rows.sort(key=lambda x: (order.get(x[0], 9), x[1], x[2]))
    # NOTE: 'tiles' is the slug's grid value; the calib PBS actually runs 12 tiles/cell. clips/s/tile
    # = per_rank_bs / iter_time is tile-count-independent, so it's correct regardless. Node throughput
    # = clips/s/tile * 12.
    print(f"{'model':10s} {'tiles':>5} {'bs':>4} {'iters':>6} {'med_ms':>9} {'clips/s/tile':>13}  status")
    print("-" * 70)
    prev = None
    for model, tiles, bs, n, med, cps, status in rows:
        if prev and prev != model:
            print()
        prev = model
        ms = f"{med:9.1f}" if med else f"{'--':>9}"
        cs = f"{cps:13.2f}" if cps else f"{'--':>13}"
        print(f"{model:10s} {tiles:5d} {bs:4d} {n:6d} {ms} {cs}  {status}")

    print("\nInterpretation:")
    print("  * clips/s/tile RISING with bs  => ~520ms is fixed overhead; pack more clips/iter (small models).")
    print("  * clips/s/tile FLAT with bs    => already compute-bound; bs won't help.")
    print("  * OOM/FAIL at a bs             => that bs exceeds one tile's memory at 256px; cap below it.")


if __name__ == "__main__":
    main()
