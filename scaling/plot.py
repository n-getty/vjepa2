"""Plot the JEPA IsoFLOP fit: per-budget metric-vs-N parabolas + N_opt/D_opt-vs-C power laws.

Mirrors PRISM isoflop_plot but reads our experiments.csv (scaling/collect.py) and reuses scaling/fit.py
so the picture and the numbers can never disagree. Produces a 2x2 figure:
  (1) metric vs N (log-x), points + fitted parabola + N_opt star, one curve per budget
  (2) N_opt vs C   (log-log), + fitted power law, slope = alpha
  (3) D_opt vs C   (log-log), + fitted power law, slope = beta
  (4) per-model metric spread (convergence sanity: how tight is each model's metric across budgets)

Matplotlib only; writes a PNG. Headless-safe (Agg backend).
"""

import argparse
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scaling.fit import _load, fit  # reuse the exact fit path


def _budget_groups(rows):
    g = {}
    for r in rows:
        g.setdefault(r["budget_flops"], []).append(r)
    return dict(sorted(g.items()))


def plot(csv_path, metric, maximize, all_status, out_png, bootstrap=0, seed=0):
    rows = _load(csv_path, metric, all_status, maximize)
    if not rows:
        raise SystemExit(f"no usable rows in {csv_path} for metric={metric}")
    rng = np.random.default_rng(seed)
    result = fit(rows, bootstrap, rng)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_par, ax_n, ax_d, ax_spread = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    groups = _budget_groups(rows)
    cmap = plt.get_cmap("viridis")
    colors = {b: cmap(i / max(1, len(groups) - 1)) for i, b in enumerate(groups)}
    par_by_budget = {p["budget_flops"]: p for p in result["parabolas"]}

    # (1) metric vs N + parabola + N_opt star
    for b, cells in groups.items():
        n = np.array([c["n_active_params"] for c in cells])
        y = np.array([c["y"] for c in cells])
        order = np.argsort(n)
        n, y = n[order], y[order]
        col = colors[b]
        ax_par.scatter(n, y, color=col, s=40, label=f"{b:.1e}", zorder=3)
        p = par_by_budget.get(b)
        if p and len(cells) >= 3:
            xs = np.logspace(math.log10(n.min()), math.log10(n.max()), 100)
            yy = p["a"] * np.log10(xs) ** 2 + p["b"] * np.log10(xs) + p["c"]
            ax_par.plot(xs, yy, color=col, alpha=0.6, lw=1.2)
            if math.isfinite(p["N_opt"]):
                y_opt = p["a"] * math.log10(p["N_opt"]) ** 2 + p["b"] * math.log10(p["N_opt"]) + p["c"]
                ax_par.scatter([p["N_opt"]], [y_opt], color=col, marker="*", s=220,
                               edgecolor="k", zorder=4)
    ax_par.set_xscale("log")
    ax_par.set_xlabel("N (params)")
    ax_par.set_ylabel(f"{metric}{' (neg=maximized)' if maximize else ''}")
    ax_par.set_title("IsoFLOP: metric vs N (★ = N_opt)")
    ax_par.legend(title="C (FLOPs)", fontsize=8)

    valid = [p for p in result["parabolas"] if math.isfinite(p["N_opt"]) and math.isfinite(p["D_opt"])]

    # (2) N_opt vs C
    if valid:
        C = np.array([p["budget_flops"] for p in valid])
        No = np.array([p["N_opt"] for p in valid])
        ax_n.scatter(C, No, color="tab:blue", s=50, zorder=3)
        if math.isfinite(result["alpha"]):
            xs = np.logspace(math.log10(C.min()), math.log10(C.max()), 50)
            ax_n.plot(xs, 10 ** (result["log_n0"] + result["alpha"] * np.log10(xs)),
                      "tab:blue", alpha=0.6,
                      label=f"α={result['alpha']:.3f} CI[{result['alpha_ci'][0]:.3f},{result['alpha_ci'][1]:.3f}]")
            ax_n.legend(fontsize=9)
    ax_n.set_xscale("log"); ax_n.set_yscale("log")
    ax_n.set_xlabel("C (FLOPs)"); ax_n.set_ylabel("N_opt (params)")
    ax_n.set_title("Compute-optimal N vs C")

    # (3) D_opt vs C
    if valid:
        C = np.array([p["budget_flops"] for p in valid])
        Do = np.array([p["D_opt"] for p in valid])
        ax_d.scatter(C, Do, color="tab:red", s=50, zorder=3)
        if math.isfinite(result["beta"]):
            xs = np.logspace(math.log10(C.min()), math.log10(C.max()), 50)
            ax_d.plot(xs, 10 ** (result["log_d0"] + result["beta"] * np.log10(xs)),
                      "tab:red", alpha=0.6,
                      label=f"β={result['beta']:.3f} CI[{result['beta_ci'][0]:.3f},{result['beta_ci'][1]:.3f}]")
            ax_d.legend(fontsize=9)
    ax_d.set_xscale("log"); ax_d.set_yscale("log")
    ax_d.set_xlabel("C (FLOPs)"); ax_d.set_ylabel("D_opt (clips)")
    ax_d.set_title("Compute-optimal D vs C")

    # (4) per-model metric spread
    by_model = {}
    for r in rows:
        by_model.setdefault(r["backbone"], []).append(r["y"])
    names = list(by_model)
    ax_spread.boxplot([by_model[m] for m in names], labels=names, vert=True)
    ax_spread.set_ylabel(metric)
    ax_spread.set_title("Per-model metric spread across budgets")
    ax_spread.tick_params(axis="x", rotation=45, labelsize=8)

    suptitle = f"JEPA IsoFLOP — metric={metric}"
    if math.isfinite(result["alpha"]) and math.isfinite(result["beta"]):
        suptitle += f"  |  α={result['alpha']:.3f}  β={result['beta']:.3f}  α+β={result['alpha']+result['beta']:.3f}"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")
    if result["warnings"]:
        print("warnings:")
        for w in result["warnings"]:
            print(f"  - {w}")


def main():
    ap = argparse.ArgumentParser(description="Plot JEPA IsoFLOP fit")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metric", default="loss_main")
    ap.add_argument("--maximize", action="store_true")
    ap.add_argument("--all-status", action="store_true")
    ap.add_argument("--out", default="scaling/isoflop.png")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    plot(args.csv, args.metric, args.maximize, args.all_status, args.out, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
