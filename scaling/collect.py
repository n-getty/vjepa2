"""Collect JEPA IsoFLOP sweep runs into one experiments.csv for the fit stage.

For each run folder it joins two artifacts:
  - scaling.json   (rank-0 sidecar written by the trainer; run identity + measured params)
  - log_r0.csv     (the existing per-iter loss CSV; column `loss`, plus `loss-pred`/`loss-context`)

and emits ONE row per run with the columns the fitter expects (mirrors PRISM's experiments.csv
contract so scaling/fit.py can reuse PRISM's isoflop_fit math ~verbatim):

  backbone, model_name, budget_flops, n_active_params, loss_main, loss_stability, loss_source,
  clips_seen_D, global_batch, steps_done, status

Loss aggregation mirrors PRISM `_last_eval_losses`: take the MEAN of the last `--window` logged
loss rows (not the single final value — an unconverged cell's final value bounces), and report the
sample stdev as `loss_stability`. A large stdev flags a cell whose loss_main is noisy and should be
treated with suspicion in the fit (or gated out). We also flag NaN/inf and collapse (loss below a
floor) so a broken cell can't silently poison a parabola.

NOTE on loss comparability (design doc §1): loss_main here is the JEPA training loss, which is NOT
comparable across model sizes. It is collected for diagnostics / convergence gating ONLY. The actual
scaling-law y-axis is Metric A (downstream probe error) or Metric B (frozen-T* loss), collected
separately. This file exists so the pretraining sweep is auditable, not so we fit a law to raw loss.
"""

import argparse
import csv
import glob
import json
import math
import os

CSV_COLUMNS = [
    "run_id", "backbone", "model_name", "budget_flops", "n_active_params",
    "train_flops_per_clip",  # needed for D_opt = C / tf_per_clip(N_opt) in the fit
    "loss_main", "loss_stability", "loss_source",
    # the REAL scaling-law y-axes (joined from metric_A.json / metric_B.json sidecars):
    "metric_a_error_f1", "metric_a_error_acc", "metric_b_error",
    "clips_seen_D", "global_batch", "steps_done", "seed", "status",
]


def _read_loss_tail(csv_path, window, loss_col="loss"):
    """Return (mean_last_window, stdev, n_rows, last_val, any_nan) from a log_r*.csv."""
    vals = []
    any_nan = False
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if loss_col not in (reader.fieldnames or []):
            return None
        for row in reader:
            raw = row.get(loss_col, "").strip()
            if raw == "":
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            if math.isnan(v) or math.isinf(v):
                any_nan = True
                continue
            vals.append(v)
    if not vals:
        return None
    tail = vals[-window:] if window > 0 else vals
    mean = sum(tail) / len(tail)
    if len(tail) >= 2:
        stdev = (sum((x - mean) ** 2 for x in tail) / (len(tail) - 1)) ** 0.5
    else:
        stdev = 0.0
    return mean, stdev, len(vals), vals[-1], any_nan


def _classify(mean, stdev, any_nan, n_rows, expected_steps, collapse_floor, rel_stab):
    """Convergence/validity gate. Returns status string."""
    if any_nan:
        return "nan"
    if mean is None:
        return "no_loss"
    if mean <= collapse_floor:
        return "collapsed"
    if expected_steps and n_rows < 0.9 * expected_steps:
        return "incomplete"
    if mean > 0 and (stdev / mean) > rel_stab:
        return "unconverged"
    return "done"


def collect_run(run_dir, window, collapse_floor, rel_stab):
    sc_path = os.path.join(run_dir, "scaling.json")
    if not os.path.exists(sc_path):
        return None  # not a scaling run
    with open(sc_path) as f:
        sc = json.load(f)

    # prefer rank-0 loss csv
    log_csv = os.path.join(run_dir, "log_r0.csv")
    if not os.path.exists(log_csv):
        cands = sorted(glob.glob(os.path.join(run_dir, "log_r*.csv")))
        log_csv = cands[0] if cands else None

    loss_main = loss_stab = last = None
    n_rows = 0
    any_nan = False
    if log_csv:
        res = _read_loss_tail(log_csv, window)
        if res:
            loss_main, loss_stab, n_rows, last, any_nan = res

    expected_steps = None
    gb = sc.get("global_batch")
    D = sc.get("clips_seen_D")
    if gb and D:
        expected_steps = int(D // gb)

    status = _classify(loss_main, loss_stab or 0.0, any_nan, n_rows,
                       expected_steps, collapse_floor, rel_stab)

    # join metric sidecars (produced by scaling/eval_metric_a.py / eval_metric_b.py). Missing => blank
    # so fit.py --metric on that column just drops the run until its eval lands.
    def _sidecar(name, *keys):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            return {}
        try:
            d = json.load(open(p))
        except Exception:
            return {}
        return {k: d.get(k) for k in keys}

    ma = _sidecar("metric_A.json", "metric_a_error_f1", "metric_a_error_acc")
    mb = _sidecar("metric_B.json", "metric_b_error")

    def _fmt(v):
        return f"{v:.6f}" if isinstance(v, (int, float)) else ""

    n_params = sc.get("n_params_measured") or sc.get("n_params")
    return {
        "run_id": os.path.basename(os.path.normpath(run_dir)),
        "backbone": sc.get("model_name", ""),   # PRISM groups parabolas by `backbone`
        "model_name": sc.get("model_name", ""),
        "budget_flops": sc.get("budget_flops", ""),
        "n_active_params": n_params if n_params is not None else "",
        "train_flops_per_clip": sc.get("train_flops_per_clip", ""),
        "loss_main": f"{loss_main:.6f}" if loss_main is not None else "",
        "loss_stability": f"{loss_stab:.6f}" if loss_stab is not None else "",
        "loss_source": "train_running_mean",
        "metric_a_error_f1": _fmt(ma.get("metric_a_error_f1")),
        "metric_a_error_acc": _fmt(ma.get("metric_a_error_acc")),
        "metric_b_error": _fmt(mb.get("metric_b_error")),
        "clips_seen_D": D if D is not None else "",
        "global_batch": gb if gb is not None else "",
        "steps_done": n_rows,
        "seed": sc.get("seed", ""),
        "status": status,
    }


def main():
    ap = argparse.ArgumentParser(description="Collect JEPA IsoFLOP runs into experiments.csv")
    ap.add_argument("--runs", required=True,
                    help="glob of run folders, e.g. 'experiments/scaling_pilot/*'")
    ap.add_argument("--csv", required=True, help="output experiments.csv")
    ap.add_argument("--window", type=int, default=50,
                    help="avg loss over last N logged rows (stabilizes noisy tail)")
    ap.add_argument("--collapse-floor", type=float, default=1e-4,
                    help="loss_main below this => flagged 'collapsed'")
    ap.add_argument("--rel-stab", type=float, default=0.10,
                    help="stdev/mean above this => flagged 'unconverged'")
    args = ap.parse_args()

    run_dirs = sorted(d for d in glob.glob(args.runs) if os.path.isdir(d))
    rows = []
    for d in run_dirs:
        r = collect_run(d, args.window, args.collapse_floor, args.rel_stab)
        if r:
            rows.append(r)

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"collected {len(rows)} runs -> {args.csv}")
    for s, n in sorted(by_status.items()):
        print(f"  {s}: {n}")
    if rows:
        print("\nrun_id                          model         params(M)  budget      loss     stab    status")
        for r in rows:
            pm = (r["n_active_params"] / 1e6) if isinstance(r["n_active_params"], (int, float)) else float("nan")
            bf = float(r["budget_flops"]) if r["budget_flops"] != "" else float("nan")
            lm = r["loss_main"] or "-"
            st = r["loss_stability"] or "-"
            print(f"{r['run_id']:<30} {r['model_name']:<13} {pm:9.1f} {bf:.2e}  {lm:>8} {st:>7}  {r['status']}")


if __name__ == "__main__":
    main()
