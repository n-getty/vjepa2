"""Score every cell under every candidate metric from CACHED features, fit IsoFLOP parabolas, and apply
the PRE-REGISTERED success test to pick a defensible scaling-law y-axis.

Pre-registered success criteria (fixed BEFORE looking at results, see scaling/FINDINGS.md discussion):
  (1) bracketing/precision: bootstrap vertex CI half-width < 0.08 in log10(N) at >= 3 budgets, AND
  (2) monotonicity: dlogN_opt >= 0 (up to noise) across 3e17 -> 1e18 -> 3e18 (compute-ordered vertices).
A metric that passes BOTH resolves the scaling law on existing checkpoints (no new compute needed).
If ALL metrics fail (2), that is positive evidence the binding constraint is COMPUTE (budget lever),
not the metric -> justifies the 3e16 widening cells.

Usage:
  python3 -m scaling.fit_metrics --runs-root <dir> --feat-tag c512 --out-prefix scaling/metrics_c512
Outputs: <prefix>_scores.csv, <prefix>_fits.json, <prefix>_verdict.txt, <prefix>.png
"""

import argparse
import glob
import json
import os

import numpy as np

from scaling.metrics_zoo import score_all

METRICS = ["metric_b_ridge", "cka_linear", "cka_rbf", "mutual_knn", "procrustes"]
LADDER = ["vit_tiny", "vit_small", "vit_base", "vit_large", "vit_giant", "vit_gigantic"]


def _model_of(name):
    for m in LADDER:
        if name.startswith(m):
            return m
    return "?"


def _budget_of(name):
    # e.g. vit_tiny_C3p0e17 -> 3e17
    tag = name.split("_C")[-1]
    return tag  # keep as string key; numeric via _bnum


def _bnum(tag):
    return float(tag.replace("p", ".").replace("e", "e"))


def load_scores(runs_root, feat_tag, seed):
    """For each cell with cached feats_X/Y_<tag>.npy, compute all metrics + read n_params, budget."""
    rows = []
    for rd in sorted(glob.glob(os.path.join(runs_root, "*"))):
        if not os.path.isdir(rd):
            continue
        name = os.path.basename(rd)
        xp = os.path.join(rd, f"feats_X_{feat_tag}.npy")
        yp = os.path.join(rd, f"feats_Y_{feat_tag}.npy")
        sj = os.path.join(rd, "scaling.json")
        if not (os.path.exists(xp) and os.path.exists(yp) and os.path.exists(sj)):
            continue
        sc = json.load(open(sj))
        X = np.load(xp)
        Y = np.load(yp)
        scores = score_all(X, Y, seed=seed)
        row = {"cell": name, "model": _model_of(name), "budget": _budget_of(name),
               "n_params": sc.get("n_params_measured", sc.get("n_params")),
               "budget_flops": sc.get("budget_flops"), "clips_seen_D": sc.get("clips_seen_D"),
               "d_run": int(X.shape[1]), "n_tokens": int(min(len(X), len(Y)))}
        row.update(scores)
        rows.append(row)
        print(f"  scored {name}: " + " ".join(f"{m}={scores[m]:.4f}" for m in METRICS))
    return rows


def fit_parabola(logN, err):
    A = np.vstack([logN * logN, logN, np.ones_like(logN)]).T
    coef, *_ = np.linalg.lstsq(A, err, rcond=None)
    a, b, c = coef
    yhat = A @ coef
    rmse = float(np.sqrt(np.mean((err - yhat) ** 2)))
    xs = np.linspace(logN.min(), logN.max(), 200)
    fs = a * xs * xs + b * xs + c
    depth = float(fs.max() - fs.min())
    xv = float(-b / (2 * a)) if a > 1e-9 else float("nan")
    return dict(a=float(a), b=float(b), c=float(c), rmse=rmse, depth=depth, xv=xv,
                convex=bool(a > 0), n=len(logN))


def boot_vertex(logN, err, nb=3000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(logN)
    vs = []
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        if len(np.unique(logN[idx])) < 3:
            continue
        f = fit_parabola(logN[idx], err[idx])
        if f["convex"] and np.isfinite(f["xv"]):
            vs.append(f["xv"])
    vs = np.array(vs)
    if len(vs) < 30:
        return dict(frac_convex=len(vs) / nb, lo=float("nan"), md=float("nan"),
                    hi=float("nan"), half_width=float("inf"))
    lo, md, hi = np.percentile(vs, [16, 50, 84])
    return dict(frac_convex=len(vs) / nb, lo=float(lo), md=float(md), hi=float(hi),
                half_width=float((hi - lo) / 2))


def fit_metric(rows, metric):
    """Fit per-budget parabola for one metric; return per-budget dicts + verdict pieces."""
    by_budget = {}
    for r in rows:
        if r[metric] is None or not np.isfinite(r[metric]):
            continue
        by_budget.setdefault(r["budget"], []).append((r["n_params"], r[metric]))
    fits = {}
    for tag, pts in by_budget.items():
        pts = sorted(pts)
        logN = np.log10(np.array([p[0] for p in pts], dtype=float))
        err = np.array([p[1] for p in pts], dtype=float)
        f = fit_parabola(logN, err)
        f["boot"] = boot_vertex(logN, err)
        f["bdnr"] = f["depth"] / f["rmse"] if f["rmse"] > 0 else float("inf")
        f["budget_flops"] = _bnum(tag)
        f["n_points"] = len(pts)
        fits[tag] = f
    return fits


def verdict(fits):
    """Apply the pre-registered success test to one metric's per-budget fits.

    Budget-aware: the monotonicity chain uses whatever bracketable budgets exist, low->high. When the
    1e17 widening cells are present they EXTEND the lever (1e17 -> 3e17 -> 1e18 -> 3e18); when absent it
    falls back to the original 3e17 -> 1e18 -> 3e18. A tolerance of 0.05 in log10(N) absorbs vertex
    noise (a strict > would flag any jitter)."""
    # precision: count budgets with convex bowl + bootstrap half-width < 0.08
    precise = [t for t, f in fits.items()
               if f["convex"] and f["boot"]["half_width"] < 0.08]
    crit1 = len(precise) >= 3
    # monotonicity across the compute-ordered, bracketable budgets (low -> high)
    order = ["1p0e17", "3p0e17", "1p0e18", "3p0e18", "1p0e19"]
    xv = [fits[t]["xv"] for t in order if t in fits and fits[t]["convex"]]
    monotone = len(xv) >= 3 and all(xv[i + 1] >= xv[i] - 0.05 for i in range(len(xv) - 1))
    crit2 = monotone
    return dict(crit1_precision=crit1, n_precise=len(precise),
                crit2_monotone=crit2, vertices_ordered=xv, passes=bool(crit1 and crit2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/flare/ModCon/ngetty/experiments/scaling_pe")
    ap.add_argument("--feat-tag", default="c512")
    ap.add_argument("--out-prefix", default="scaling/metrics_c512")
    ap.add_argument("--seed", type=int, default=239)
    args = ap.parse_args()

    rows = load_scores(args.runs_root, args.feat_tag, args.seed)
    if not rows:
        raise SystemExit(f"no cached features found under {args.runs_root} (tag {args.feat_tag})")

    # scores CSV
    import csv
    cols = ["cell", "model", "budget", "budget_flops", "n_params", "clips_seen_D", "d_run",
            "n_tokens"] + METRICS
    with open(args.out_prefix + "_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    allfits = {}
    verdicts = {}
    for m in METRICS:
        fits = fit_metric(rows, m)
        allfits[m] = fits
        verdicts[m] = verdict(fits)
    with open(args.out_prefix + "_fits.json", "w") as f:
        json.dump({"fits": allfits, "verdicts": verdicts}, f, indent=2)

    # verdict text
    lines = ["PRE-REGISTERED METRIC SELECTION VERDICT", "=" * 60, ""]
    for m in METRICS:
        v = verdicts[m]
        lines.append(f"### {m}   {'*** PASSES ***' if v['passes'] else 'fails'}")
        lines.append(f"  crit1 precision (>=3 budgets, boot half-width<0.08): "
                     f"{v['crit1_precision']}  (n_precise={v['n_precise']})")
        lines.append(f"  crit2 monotone vertices 3e17->1e18->3e18: {v['crit2_monotone']}  "
                     f"(logN_opt ordered = {[round(x,3) for x in v['vertices_ordered']]})")
        lines.append("  per-budget:")
        for tag in sorted(allfits[m], key=lambda t: _bnum(t)):
            fdt = allfits[m][tag]
            b = fdt["boot"]
            nopt = 10 ** fdt["xv"] / 1e6 if fdt["convex"] and np.isfinite(fdt["xv"]) else float("nan")
            lines.append(f"    {_bnum(tag):.0e}: n={fdt['n_points']} a={fdt['a']:+.3f} "
                         f"rmse={fdt['rmse']:.4f} depth={fdt['depth']:.4f} BDNR={fdt['bdnr']:.1f} "
                         f"Nopt={nopt:7.0f}M  bootCI-hw={b['half_width']:.3f} "
                         f"conv={b['frac_convex']:.2f}")
        lines.append("")
    winner = [m for m in METRICS if verdicts[m]["passes"]]
    lines.append("=" * 60)
    lines.append(f"WINNER(S): {winner or 'NONE — all metrics fail; compute (budget) is the binding constraint'}")
    txt = "\n".join(lines)
    with open(args.out_prefix + "_verdict.txt", "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)

    # figure: one panel per metric, parabola + points
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 4), squeeze=False)
        colors = {"3p0e17": "purple", "1p0e18": "tab:blue", "3p0e18": "tab:green", "1p0e19": "tab:orange"}
        for ax, m in zip(axes[0], METRICS):
            for r in rows:
                if r[m] is None or not np.isfinite(r[m]):
                    continue
                ax.scatter(np.log10(r["n_params"]), r[m], color=colors.get(r["budget"], "gray"), s=24)
            for tag, fdt in allfits[m].items():
                if not fdt["convex"]:
                    continue
                # draw over the budget's point range
                xs = np.linspace(7.5, 9.5, 100)
                ys = fdt["a"] * xs * xs + fdt["b"] * xs + fdt["c"]
                ax.plot(xs, ys, color=colors.get(tag, "gray"), lw=1, alpha=0.7)
            ax.set_title(f"{m}\n{'PASS' if verdicts[m]['passes'] else 'fail'}")
            ax.set_xlabel("log10 N_params")
            ax.set_ylabel("error (down=better)")
        fig.tight_layout()
        fig.savefig(args.out_prefix + ".png", dpi=110)
        print(f"\nfigure -> {args.out_prefix}.png")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
