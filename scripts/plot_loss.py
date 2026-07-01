"""Plot phase 1 + phase 2 loss curves. Phase 1 dotted, phase 2 solid, vertical
divider at the phase boundary. Phase 2 also shows pred and ctx components when
those columns are present in the CSV.

Optionally accepts a second phase 2 path (the divergent v1 run) for overlay
comparison."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P1 = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase1_warmup_n16g12_weak/log_r0.csv"
P2 = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak/log_r0.csv"
P2_BAD = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak_v1_divergent/log_r0.csv"

# Phase 1 config: ipe=500, epochs=12 → 6000 iters total
# Phase 2 (new): ipe=100, epochs=600 → 60000 iters total
# Want a continuous global-iter axis: phase2_iters offset by 6000
PHASE1_TOTAL_ITERS = 6000


def load(path, include_components=False):
    """Returns list of (epoch, itr, loss) or (epoch, itr, loss, pred, ctx, lam)."""
    rows = []
    with open(path) as f:
        for r in csv.reader(f):
            if not r or not r[0].isdigit():
                continue
            try:
                epoch = int(r[0])
                itr = int(r[1])
                loss = float(r[2])
                if include_components and len(r) >= 14:
                    pred = float(r[11])
                    ctx = float(r[12])
                    lam = float(r[13])
                    rows.append((epoch, itr, loss, pred, ctx, lam))
                else:
                    rows.append((epoch, itr, loss))
            except Exception:
                continue
    return rows


def to_global_iter(rows, ipe, iter_offset=0):
    """Each row: (epoch_1indexed, itr_within_epoch, loss, [pred, ctx, lam]).
    Global iter = (epoch-1)*ipe + itr + iter_offset."""
    out = []
    for r in rows:
        e, i, l = r[0], r[1], r[2]
        gi = (e - 1) * ipe + i + iter_offset
        out.append((gi,) + tuple(r[2:]))
    return out


def smooth(xs, ys, window=50):
    """Rolling-window median for clarity. ys must be aligned to xs."""
    if len(ys) < window:
        return xs, ys
    out_x, out_y = [], []
    for i in range(len(ys)):
        lo = max(0, i - window // 2)
        hi = min(len(ys), i + window // 2)
        out_x.append(xs[i])
        out_y.append(float(np.median(ys[lo:hi])))
    return out_x, out_y


def main():
    import os
    p1 = load(P1, include_components=False)
    p2 = load(P2, include_components=True)
    p2_bad = load(P2_BAD, include_components=False) if os.path.exists(P2_BAD) else []
    print(f"phase 1 (warmup):        {len(p1)} iter rows")
    print(f"phase 2 v2 (fixed):      {len(p2)} iter rows  (with components: {len(p2) and len(load(P2, include_components=True)[-1]) >= 6})")
    print(f"phase 2 v1 (divergent):  {len(p2_bad)} iter rows")

    # Convert to global iters
    p1_g = to_global_iter(p1, ipe=500, iter_offset=0)
    p2_g = to_global_iter(p2, ipe=100, iter_offset=PHASE1_TOTAL_ITERS)
    p2_bad_g = to_global_iter(p2_bad, ipe=100, iter_offset=PHASE1_TOTAL_ITERS) if p2_bad else []

    # Phase 1 csv accumulated across multiple submissions; iters are NOT monotonic.
    p1_g = sorted(set(p1_g))
    p2_g = sorted(set(p2_g))
    p2_bad_g = sorted(set(p2_bad_g))

    # Phase 1: (gi, loss)
    x1 = [r[0] for r in p1_g]; y1 = [r[1] for r in p1_g]
    # Phase 2 (fixed): (gi, loss, pred, ctx, lam)
    x2 = [r[0] for r in p2_g]; y2 = [r[1] for r in p2_g]
    if p2_g and len(p2_g[0]) >= 5:
        y2_pred = [r[2] for r in p2_g]
        y2_ctx = [r[3] for r in p2_g]
        y2_lam = [r[4] for r in p2_g]
    else:
        y2_pred = y2_ctx = y2_lam = None
    # Phase 2 (divergent)
    xb = [r[0] for r in p2_bad_g]; yb = [r[1] for r in p2_bad_g]

    # Smooth
    x1s, y1s = smooth(x1, y1, window=50)
    x2s, y2s = smooth(x2, y2, window=50)
    if y2_pred:
        _, y2_pred_s = smooth(x2, y2_pred, window=50)
        _, y2_ctx_s = smooth(x2, y2_ctx, window=50)
    if yb:
        xbs, ybs = smooth(xb, yb, window=50)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Raw points light
    ax.plot(x1, y1, color="#aac", alpha=0.25, lw=0.6)
    ax.plot(x2, y2, color="#a25", alpha=0.25, lw=0.6)
    if yb:
        ax.plot(xb, yb, color="#ccc", alpha=0.25, lw=0.6)

    # Smoothed lines
    ax.plot(x1s, y1s, color="#446", ls=":", lw=2.0,
            label=f"phase 1 warmup (n={len(p1)})")
    if yb:
        ax.plot(xbs, ybs, color="#888", ls="-", lw=1.5,
                label=f"phase 2 v1 divergent — lambda_progressive=false (n={len(p2_bad)})")
    ax.plot(x2s, y2s, color="#a25", ls="-", lw=2.5,
            label=f"phase 2 v2 fixed — lambda_progressive=true (n={len(p2)})")

    # Optional component overlays
    if y2_pred:
        ax.plot(x2, y2_pred_s, color="#28a", ls="--", lw=1.0, alpha=0.7,
                label="phase 2 v2  loss_pred component")
        ax.plot(x2, y2_ctx_s, color="#2a8", ls="--", lw=1.0, alpha=0.7,
                label="phase 2 v2  loss_context (not yet contributing, λ=0)")

    # Phase divider
    ax.axvline(PHASE1_TOTAL_ITERS, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.text(PHASE1_TOTAL_ITERS + 200, 0.65,
            "phase 1 → 2\n(crop 256→384,\nbs 4→2)",
            fontsize=9, color="gray", va="top")

    # Lambda warmup window
    ax.axvspan(PHASE1_TOTAL_ITERS + 15000, PHASE1_TOTAL_ITERS + 30000,
               color="#ffa", alpha=0.2,
               label="lambda warmup 0→0.5 (when training reaches it)")

    ax.set_xlabel("global optimizer step  (phase1 0–6000, phase2 6000+)")
    ax.set_ylabel("training loss (rank 0)")
    ax.set_title("V-JEPA 2.1 surgical pretrain — phase 1 + phase 2")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # log scale if range is wide
    ymax = max(y1 + y2 + (yb or [0]))
    ymin = min([v for v in (y1 + y2 + (yb or [0])) if v > 0])
    if ymax / max(ymin, 0.01) > 3:
        ax.set_yscale("log")

    out = Path("/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/loss_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")

    # Text summary
    print()
    print(f"phase 1   min={min(y1):.4f}  last={y1[-1]:.4f}  global range [{x1[0]},{x1[-1]}]")
    print(f"phase 2 v2 min={min(y2):.4f}  last={y2[-1]:.4f}  global range [{x2[0]},{x2[-1]}]")
    if yb:
        print(f"phase 2 v1 min={min(yb):.4f}  last={yb[-1]:.4f}  global range [{xb[0]},{xb[-1]}]")


if __name__ == "__main__":
    main()
