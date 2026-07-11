"""Fit IsoFLOP scaling exponents for the JEPA sweep.

Reuses the PRISM isoflop_fit method (Approach 2: per-budget parabola in log-N -> vertex N_opt;
cross-budget power laws N_opt ~ C^alpha, D_opt ~ C^beta; bootstrap CIs) but with two JEPA-specific
changes:

  1. The y-axis metric is CONFIGURABLE (`--metric`). Default is `loss_main` for pipeline testing, but
     the real scaling law must use a downstream/probe metric column (Metric A) or the frozen-T* loss
     (Metric B) — raw JEPA loss is NOT comparable across N (design doc §1). Lower-is-better is assumed;
     for an accuracy metric pass error = 1 - acc upstream, or use --maximize.

  2. D_opt is computed from the MEASURED per-clip FLOPs, not the LLM `C = 6*N*D` shortcut. At the
     interpolated N_opt we estimate train_flops_per_clip(N_opt) by log-log interpolation over the
     ladder's (n_params, train_flops_per_clip) points, then D_opt = C / tf_per_clip(N_opt).

Convergence gate: only rows with status == 'done' (from scaling/collect.py) enter the fit unless
--all-status is passed. This keeps collapsed/unconverged/nan cells from poisoning a parabola.
"""

import argparse
import csv
import json
import math

import numpy as np


# ----------------------------------------------------------------------------
# PRISM-derived core (parabola / power-law / bootstrap). Metric-agnostic.
# ----------------------------------------------------------------------------


def _fit_parabola(log_n, y):
    if log_n.size < 3:
        raise ValueError(f"need >= 3 points, got {log_n.size}")
    a, b, c = np.polyfit(log_n, y, 2)
    if a <= 1e-12:  # linear/degenerate or inverted (vertex is a maximum) -> no usable min
        log_n_opt = float("nan")
    else:
        log_n_opt = float(-b / (2 * a))
    return float(a), float(b), float(c), log_n_opt


def _fit_power_law(log_x, log_y):
    if log_x.size < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(log_x, log_y, 1)
    return float(slope), float(intercept)


def _bootstrap(log_c, log_n_opt, log_d_opt, n_iter, rng):
    if log_c.size < 2 or n_iter <= 0:
        return {"alpha_ci": (float("nan"),) * 2, "beta_ci": (float("nan"),) * 2}
    alphas, betas = [], []
    for _ in range(n_iter):
        idx = rng.integers(0, log_c.size, size=log_c.size)
        a, _ = _fit_power_law(log_c[idx], log_n_opt[idx])
        b, _ = _fit_power_law(log_c[idx], log_d_opt[idx])
        if not math.isnan(a):
            alphas.append(a)
        if not math.isnan(b):
            betas.append(b)
    if not alphas or not betas:
        return {"alpha_ci": (float("nan"),) * 2, "beta_ci": (float("nan"),) * 2}
    return {
        "alpha_ci": (float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))),
        "beta_ci": (float(np.percentile(betas, 2.5)), float(np.percentile(betas, 97.5))),
    }


# ----------------------------------------------------------------------------
# JEPA-specific: measured D_opt via per-clip-FLOP interpolation.
# ----------------------------------------------------------------------------


def _make_tf_interp(rows):
    """Build tf_per_clip(N) as a log-log linear interpolator over the ladder points."""
    pts = {}
    for r in rows:
        n = r["n_active_params"]
        tf = r.get("train_flops_per_clip")
        if n and tf:
            pts[n] = tf
    xs = np.array(sorted(pts), dtype=np.float64)
    ys = np.array([pts[x] for x in xs], dtype=np.float64)
    logx, logy = np.log10(xs), np.log10(ys)

    def tf_of(n):
        if n <= 0:
            return float("nan")
        return float(10 ** np.interp(math.log10(n), logx, logy))

    return tf_of if xs.size >= 2 else None


# ----------------------------------------------------------------------------
# Load + filter.
# ----------------------------------------------------------------------------


def _load(path, metric, all_status, maximize):
    rows = []
    # The `status` column is a LOSS-stability/convergence verdict (collect.py flags a cell
    # `unconverged` when stdev/mean of loss_main exceeds a threshold). That gate is IRRELEVANT when the
    # y-axis is NOT loss: e.g. a short large@small-budget cell has a wobbly loss over its few steps but
    # a perfectly valid Metric B (it completed its target steps and its frozen encoder is scored on the
    # fixed ruler). So for a non-loss metric, accept `unconverged` cells too — they were only excluded
    # for loss noise we don't care about. Still reject collapsed/nan/no_loss/incomplete.
    loss_metric = metric.startswith("loss")
    ok_status = {"done"} if loss_metric else {"done", "unconverged"}
    with open(path) as f:
        for r in csv.DictReader(f):
            if not all_status and r.get("status") not in ok_status:
                continue
            try:
                n = float(r["n_active_params"])
                c = float(r["budget_flops"])
                y = float(r[metric])
            except (ValueError, KeyError):
                continue
            tf = r.get("train_flops_per_clip", "")
            rows.append({
                "backbone": r.get("backbone", "all"),
                "n_active_params": n,
                "budget_flops": c,
                "y": (-y if maximize else y),  # parabola finds a MIN; flip for maximize
                "train_flops_per_clip": float(tf) if tf not in ("", None) else None,
            })
    return rows


def fit(rows, bootstrap, rng):
    tf_of = _make_tf_interp(rows)
    by_budget = {}
    for r in rows:
        by_budget.setdefault(r["budget_flops"], []).append(r)

    parabolas, warnings = [], []
    for budget, cells in sorted(by_budget.items()):
        if len(cells) < 3:
            warnings.append(f"budget={budget:.3e}: {len(cells)} points (<3); skipped")
            continue
        n = np.array([c["n_active_params"] for c in cells], dtype=np.float64)
        y = np.array([c["y"] for c in cells], dtype=np.float64)
        log_n = np.log10(n)
        a, b, c_, log_n_opt = _fit_parabola(log_n, y)
        if not math.isfinite(log_n_opt):
            warnings.append(
                f"budget={budget:.3e}: parabola {'inverted (a<0)' if a < 0 else 'degenerate'}; N_opt rejected")
        elif log_n_opt < log_n.min() or log_n_opt > log_n.max():
            # Edge/extrapolated minimum: vertex lies outside the sampled ladder, so N_opt and any
            # tf_per_clip(N_opt) interpolation are extrapolations (PLAN.md "no edge minima"). Reject
            # rather than report an unreliable optimum — add a ladder point that brackets it instead.
            warnings.append(
                f"budget={budget:.3e}: N_opt vertex outside ladder "
                f"[{10**log_n.min()/1e6:.0f}M, {10**log_n.max()/1e6:.0f}M] — extrapolated, rejected. "
                f"Add a capacity point that brackets the optimum for this budget.")
            log_n_opt = float("nan")
        n_opt = float(10 ** log_n_opt) if math.isfinite(log_n_opt) else float("nan")
        # D_opt via MEASURED per-clip FLOPs (not 6ND): D = C / tf_per_clip(N_opt)
        if math.isfinite(n_opt) and tf_of is not None:
            tf_at_opt = tf_of(n_opt)
            d_opt = float(budget / tf_at_opt) if tf_at_opt and math.isfinite(tf_at_opt) else float("nan")
        else:
            d_opt = float("nan")
        parabolas.append({
            "budget_flops": budget, "n_points": len(cells),
            "a": a, "b": b, "c": c_, "N_opt": n_opt, "D_opt": d_opt,
        })

    valid = [p for p in parabolas if math.isfinite(p["N_opt"]) and math.isfinite(p["D_opt"])]
    if len(valid) >= 2:
        log_c = np.array([math.log10(p["budget_flops"]) for p in valid])
        log_n_opt = np.array([math.log10(p["N_opt"]) for p in valid])
        log_d_opt = np.array([math.log10(p["D_opt"]) for p in valid])
        alpha, log_n0 = _fit_power_law(log_c, log_n_opt)
        beta, log_d0 = _fit_power_law(log_c, log_d_opt)
        ci = _bootstrap(log_c, log_n_opt, log_d_opt, bootstrap, rng)
    else:
        warnings.append(f"only {len(valid)} usable parabolas (<2); no alpha/beta fit")
        alpha = beta = log_n0 = log_d0 = float("nan")
        ci = {"alpha_ci": (float("nan"),) * 2, "beta_ci": (float("nan"),) * 2}

    return {
        "n_cells": len(rows), "parabolas": parabolas,
        "alpha": alpha, "alpha_ci": list(ci["alpha_ci"]),
        "beta": beta, "beta_ci": list(ci["beta_ci"]),
        "log_n0": log_n0, "log_d0": log_d0, "warnings": warnings,
    }


def render(result, metric, maximize):
    L = []
    L.append(f"# JEPA IsoFLOP fit  (metric={metric}{' [maximized]' if maximize else ''})")
    L.append(f"cells fit: {result['n_cells']}")
    a, aci = result["alpha"], result["alpha_ci"]
    b, bci = result["beta"], result["beta_ci"]
    L.append("")
    L.append(f"alpha (N_opt ~ C^a): {a:.4f}  95%CI [{aci[0]:.4f}, {aci[1]:.4f}]")
    L.append(f"beta  (D_opt ~ C^b): {b:.4f}  95%CI [{bci[0]:.4f}, {bci[1]:.4f}]")
    if math.isfinite(a) and math.isfinite(b):
        L.append(f"sanity: alpha+beta = {a + b:.3f} (Chinchilla expects ~1.0 if C~N*D)")
    L.append("")
    L.append("| budget | pts | N_opt (M) | D_opt (clips) |")
    L.append("|---|---|---|---|")
    for p in result["parabolas"]:
        no = f"{p['N_opt']/1e6:.1f}" if math.isfinite(p["N_opt"]) else "—"
        do = f"{p['D_opt']:.2e}" if math.isfinite(p["D_opt"]) else "—"
        L.append(f"| {p['budget_flops']:.2e} | {p['n_points']} | {no} | {do} |")
    if result["warnings"]:
        L.append("")
        L.append("## warnings")
        for w in result["warnings"]:
            L.append(f"- {w}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Fit JEPA IsoFLOP scaling exponents")
    ap.add_argument("--csv", required=True, help="experiments.csv from scaling/collect.py")
    ap.add_argument("--metric", default="loss_main",
                    help="y-axis column (loss_main for testing; a probe-error column for the real law)")
    ap.add_argument("--maximize", action="store_true",
                    help="metric is higher-is-better (e.g. accuracy); fit on -metric")
    ap.add_argument("--all-status", action="store_true",
                    help="include non-'done' cells (default: only converged)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", help="also write raw result JSON")
    args = ap.parse_args()

    rows = _load(args.csv, args.metric, args.all_status, args.maximize)
    rng = np.random.default_rng(args.seed)
    result = fit(rows, args.bootstrap, rng)
    print(render(result, args.metric, args.maximize))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
