#!/usr/bin/env python
"""Cross-rank straggler + external-memory analysis for the 16n HSDP gate.

Reads ALL per-rank log_r*.csv files in a run folder and answers the reviewer's
decisive fork:

  (A) Is a backward spike COHORT-WIDE (min << p50 ~= p90 ~= max) or ONE-RANK
      (min ~= p50 ~= p90, only max jumps)?  A rotating single straggler is
      NUMA/hardware; a synchronized cohort inflation is a global collective
      problem that CPU-binding / grad_accum cannot root-fix.

  (B) Does l0-free creep DOWN as the backward floor creeps UP?  free-L0
      (torch.xpu.mem_get_info) is the ONLY counter that sees CCL/OFI external
      growth — it is invisible to reserved. external = l0_used - torch_alloc.
        - free-L0 creeps down + floor rises  -> external CCL/OFI accumulation
          on the inter-node path  -> static-buffer/registration fix.
        - free-L0 flat, spikes cohort-wide   -> pure fabric contention
          -> grad_accum is the legitimate (not band-aid) mitigation.

Usage:
    python scripts/analyze_straggler.py <run_folder>
"""
import csv
import glob
import os
import sys


def _pct(a, q):
    a = sorted(a)
    if not a:
        return 0.0
    return a[min(len(a) - 1, int(q * len(a)))]


def main(folder):
    files = sorted(glob.glob(os.path.join(folder, "log_r*.csv")))
    if not files:
        print(f"no log_r*.csv in {folder}")
        return 1
    # per-iter across ranks: backward-ms, iter-time, l0-free, l0-ext, dataload
    from collections import defaultdict

    bwd = defaultdict(list)
    itert = defaultdict(list)
    l0free = defaultdict(list)
    l0ext = defaultdict(list)
    dload = defaultdict(list)
    untracked = defaultdict(list)  # iter-time - gpu-time = host-side stall
    for f in files:
        lines = open(f).read().splitlines()
        hi = [i for i, l in enumerate(lines) if l.startswith("epoch,")]
        if not hi:
            continue
        hdr = lines[hi[-1]].split(",")
        col = {c: i for i, c in enumerate(hdr)}

        def gi(name):
            return col.get(name)

        for l in lines[hi[-1] + 1:]:
            p = l.split(",")
            try:
                it = int(float(p[col["itr"]]))
            except (ValueError, KeyError, IndexError):
                continue

            def val(name, scale=1.0):
                i = gi(name)
                if i is None or i >= len(p):
                    return None
                try:
                    return float(p[i]) / scale
                except ValueError:
                    return None

            b = val("backward-ms", 1000.0)
            if b is not None and b > 0:
                bwd[it].append(b)
            t = val("iter-time(ms)", 1000.0)
            if t is not None:
                itert[it].append(t)
            fr = val("l0-free-mib")
            if fr is not None and fr >= 0:
                l0free[it].append(fr)
            ex = val("l0-ext-mib")
            if ex is not None and ex >= 0:
                l0ext[it].append(ex)
            dl = val("dataload-time(ms)", 1000.0)
            if dl is not None:
                dload[it].append(dl)
            # UNTRACKED host-side time = iter-time - gpu-time. Large values = a
            # host-side collective stall (CCL progress hang) NOT visible in any
            # GPU phase incl. backward-ms. Job 8643398 iter70: iter=365s gpu=21s
            # -> 344s untracked, synchronized across all ranks. This is the tail-
            # latency killer, distinct from in-GPU backward inflation.
            g = val("gpu-time(ms)", 1000.0)
            if t is not None and g is not None:
                untracked[it].append(max(0.0, t - g))

    iters = sorted(bwd)
    print(f"ranks={len(files)}  iters={len(iters)}\n")
    print("=== (A) cross-rank backward-ms distribution per iter ===")
    print(f"{'itr':>4} {'n':>4} {'min':>7} {'p50':>7} {'p90':>7} {'max':>7} "
          f"{'span':>6}  {'l0free':>8} {'l0ext':>7} {'untrk_p90':>9}")
    for it in iters:
        a = bwd[it]
        if len(a) < 10:
            continue
        mn, p50, p90, mx = min(a), _pct(a, .5), _pct(a, .9), max(a)
        span = "COHORT" if (p50 > 2 * mn and mx < 1.5 * p50) else (
            "1RANK" if mx > 2 * p90 else "flat")
        fr = _pct(l0free[it], .5) if l0free[it] else -1
        ex = _pct(l0ext[it], .5) if l0ext[it] else -1
        ut = _pct(untracked[it], .9) if untracked[it] else -1
        # flag host-side stalls: >5s untracked wall that backward doesn't explain
        tag = " <<HOST-STALL" if ut > 5.0 else ""
        print(f"{it:>4} {len(a):>4} {mn:>7.1f} {p50:>7.1f} {p90:>7.1f} "
              f"{mx:>7.1f} {span:>6}  {fr:>8.0f} {ex:>7.0f} {ut:>9.1f}{tag}")

    # (B) trend: floor (min backward) and free-L0 over the run
    print("\n=== (B) floor + free-L0 trend (10-iter windows) ===")
    print(f"{'window':>10} {'floor_min_bwd':>13} {'p50_bwd':>8} "
          f"{'l0free_p50':>11} {'l0ext_p50':>10}")
    W = 10
    for s in range(0, iters[-1] + 1, W):
        wi = [it for it in iters if s <= it < s + W]
        if not wi:
            continue
        floors = [min(bwd[it]) for it in wi if bwd[it]]
        p50s = [_pct(bwd[it], .5) for it in wi if bwd[it]]
        frs = [_pct(l0free[it], .5) for it in wi if l0free[it]]
        exs = [_pct(l0ext[it], .5) for it in wi if l0ext[it]]
        floor = min(floors) if floors else -1
        p50 = sum(p50s) / len(p50s) if p50s else -1
        fr = sum(frs) / len(frs) if frs else -1
        ex = sum(exs) / len(exs) if exs else -1
        print(f"{s:>4}-{s+W-1:<5} {floor:>13.1f} {p50:>8.1f} "
              f"{fr:>11.0f} {ex:>10.0f}")

    print("\nVERDICT GUIDE:")
    print("  span=COHORT + l0free FALLING + floor RISING -> external CCL/OFI "
          "accumulation (static-buffer / registration fix)")
    print("  span=COHORT + l0free FLAT                    -> fabric contention "
          "(grad_accum is the correct mitigation)")
    print("  span=1RANK  (rotating)                       -> NUMA/hardware "
          "(CPU-bind / node exclusion)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
