#!/usr/bin/env python3
"""ESAD arm results collector -- discovers arms from disk, never hardcodes numbers.

`collect_ft.py` carried its results as a literal dict, so every new arm meant
editing the collector and every stale edit meant a silently wrong table. This
walks the runs directory instead: an arm is any set of `..._s{N}` run dirs
sharing a stem, and its numbers come from the JSON the scorers already write.

Columns are the four the ledger uses:
  AP_mean        detection AP, FULL denominator, maxpick fusion (the headline)
  presence mAP   oracle-gated `test_macro_map`
  meanIoU        oracle-gated `test_mean_iou`
  boxAP50        oracle-gated `test_map50`

Population is ASSERTED on every scored cell (gt_full=11207, frames=5903). A cell
computed on a different population is not comparable and is dropped loudly
rather than averaged in -- that mistake is what the coverage-denominator bug was.

Writes a machine-readable index next to the runs so downstream tooling (and the
next session) reads results instead of re-deriving them.

Usage:  python3 collect_esad_arms.py [--json OUT] [--baseline STEM] [filter...]
"""
import argparse
import json
import math
import os
import re
import statistics as st
import sys

RUNS = "/eagle/projects/ModCon/ngetty/esad_probe/runs"
ENS = "/eagle/projects/ModCon/ngetty/esad_probe/seed_ensemble_scores"
INDEX = "/eagle/projects/ModCon/ngetty/esad_probe/results_index.json"

GT_FULL = 11207
FRAMES = 5903
# §4b-seeded observed per-arm spreads were 0.022-0.063. Kept only as a sanity
# reference: comparing a MEAN delta against a raw per-seed SPREAD is not a test
# (it ignores that the mean's error shrinks with n), so the verdict is a p-value.
NOISE_REF = 0.063
# A t-statistic without its df is not a verdict: at n=3 the paired df is 2, and
# t=2.0 there is p=0.18, nowhere near significant. Threshold on p, not on t.
P_CLAIM = 0.05

SEED_RE = re.compile(r"^(?P<stem>esad_double_.+)_s(?P<seed>\d+)$")


def det_ap(run):
    """Full-denominator maxpick AP_mean, or None. Asserts the population."""
    for fn in ("test_detection_ap_cov_fulldenom.json",
               "test_detection_ap_masked_fulldenom.json"):
        p = os.path.join(RUNS, run, fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        b = d.get("combined_not_isolated") or d.get("single_source") or d.get("source0_only")
        if not b:
            return None
        vfd = b.get("variants_full_denominator")
        if not vfd:
            return None
        gt, nf = b.get("gt_instances_full"), b.get("num_frames_scored")
        if gt != GT_FULL or nf != FRAMES:
            print("  !! %s scored on gt=%s frames=%s (expected %d/%d) -- DROPPED"
                  % (run, gt, nf, GT_FULL, FRAMES), file=sys.stderr)
            return None
        return vfd["maxpick"]["ap_mean"]
    return None


def oracle(run):
    p = os.path.join(RUNS, run, "test_oracle_cov.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    return {"presence_map": d.get("test_macro_map"),
            "mean_iou": d.get("test_mean_iou"),
            "box_map50": d.get("test_map50")}


def best_epoch(run):
    """Peak selection-metric epoch, to flag arms still improving at the tail."""
    p = os.path.join(RUNS, run, "log_r0.csv")
    if not os.path.exists(p):
        return None, None
    import csv
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None, None
    col = next((c for c in ("val_well_map", "val_macro_map", "val_map50")
                if c in rows[0]), None)
    if col is None:
        return None, len(rows)
    b = max(rows, key=lambda r: float(r[col]))
    return int(b["epoch"]), len(rows)


def ensemble_ap(stem):
    """Cross-seed ensemble AP if the ensemble job has finished for this arm."""
    tag = stem.replace("esad_double_", "")
    for fn in os.listdir(ENS) if os.path.isdir(ENS) else []:
        if not fn.endswith("_seed_ensemble.json"):
            continue
        if fn[: -len("_seed_ensemble.json")] != tag:
            continue
        d = json.load(open(os.path.join(ENS, fn)))
        e = d.get("seed_ensemble")
        if not e:
            return None  # partial file: per-seed written, ensemble still scoring
        vfd = e.get("variants_full_denominator")
        if not vfd:
            return None
        if e.get("gt_instances_full") != GT_FULL:
            print("  !! ensemble %s wrong population -- DROPPED" % fn, file=sys.stderr)
            return None
        return vfd["maxpick"]["ap_mean"]
    return None


def _betacf(a, b, x):
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return f - 1.0


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1) / (a + b + 2):
        return math.exp(math.log(x) * a + math.log(1 - x) * b - lb) / a * _betacf(a, b, x)
    return 1.0 - math.exp(math.log(1 - x) * b + math.log(x) * a - lb) / b * _betacf(b, a, 1 - x)


def t_pvalue(t, df):
    """Two-sided p for Student-t. No scipy on these nodes."""
    if t is None or df is None or df <= 0:
        return None
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def welch_t(a, b):
    """Unequal-variance t + Welch-Satterthwaite df. (t, df) or (None, None)."""
    if len(a) < 2 or len(b) < 2:
        return None, None
    n1, n2 = len(a), len(b)
    s1, s2 = st.variance(a) / n1, st.variance(b) / n2
    if s1 + s2 <= 0:
        return None, None
    df = (s1 + s2) ** 2 / (s1 ** 2 / (n1 - 1) + s2 ** 2 / (n2 - 1))
    return (st.mean(a) - st.mean(b)) / (s1 + s2) ** 0.5, df


def paired_t(arm, base):
    """Paired t on seeds present in BOTH arms.

    Seed N means the same init + data order in every arm, so the pairing is
    real and removes the between-seed variance that dominates at n=3. Reported
    alongside Welch because the pairing is weaker for arms that change the data
    stream itself (augmentation): there, seed N no longer implies the same
    samples, only the same draw sequence.
    """
    common = sorted(set(arm) & set(base))
    d = [arm[s] - base[s] for s in common]
    if len(d) < 2:
        return None, None, d
    sd = st.stdev(d)
    if sd <= 0:
        return None, None, d
    return st.mean(d) / (sd / len(d) ** 0.5), len(d) - 1, d


def discover(filters):
    arms = {}
    for d in sorted(os.listdir(RUNS)):
        m = SEED_RE.match(d)
        if not m:
            continue
        stem = m.group("stem")
        if filters and not any(f in stem for f in filters):
            continue
        arms.setdefault(stem, []).append((int(m.group("seed")), d))
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filters", nargs="*", help="substring filters on the arm stem")
    ap.add_argument("--baseline", default="prod37m_e199_ft_last4",
                    help="arm stem substring to diff every other arm against")
    ap.add_argument("--json", default=INDEX, help="write the index here ('' to skip)")
    args = ap.parse_args()

    arms = discover(args.filters)
    table = {}
    for stem, seeds in sorted(arms.items()):
        rows = []
        for sd, run in sorted(seeds):
            a = det_ap(run)
            o = oracle(run)
            be, ne = best_epoch(run)
            rows.append({"seed": sd, "run": run, "ap_mean": a,
                         "best_epoch": be, "n_epochs": ne, **o})
        aps = [r["ap_mean"] for r in rows if r["ap_mean"] is not None]
        table[stem] = {
            "seeds": rows,
            "n_scored": len(aps),
            "ap_mean": st.mean(aps) if aps else None,
            "ap_sd": st.stdev(aps) if len(aps) > 1 else 0.0,
            "ensemble_ap_mean": ensemble_ap(stem),
        }
        for k in ("presence_map", "mean_iou", "box_map50"):
            v = [r[k] for r in rows if r.get(k) is not None]
            table[stem][k] = st.mean(v) if v else None

    base_stem = next((k for k, v in table.items()
                      if args.baseline in k and v["ap_mean"] is not None), None)
    base = table[base_stem]["ap_mean"] if base_stem else None
    base_aps = ([r["ap_mean"] for r in table[base_stem]["seeds"]
                 if r["ap_mean"] is not None] if base_stem else [])
    base_by_seed = ({r["seed"]: r["ap_mean"] for r in table[base_stem]["seeds"]
                     if r["ap_mean"] is not None} if base_stem else {})

    print("=== ESAD arms, AP_mean full denominator (gt=%d, frames=%d) ===" % (GT_FULL, FRAMES))
    print("    verdict: two-sided p vs baseline (Welch and paired-by-seed), "
          "claimable at p < %.2f" % P_CLAIM)
    print("    'seeds' = how many of the paired deltas are positive; a 2/3 win "
          "with one flat seed is not a uniform shift\n")
    hdr = ("%-44s %-11s %6s %7s %6s %6s %5s %7s %7s %7s  %s"
           % ("arm", "AP_mean", "n", "vs base", "p_w", "p_pair", "seeds",
              "presMAP", "IoU", "AP50", "ens"))
    print(hdr)
    print("-" * len(hdr))
    for stem, v in sorted(table.items(), key=lambda x: -(x[1]["ap_mean"] or -1)):
        if v["ap_mean"] is None:
            prog = "/".join(str(r["best_epoch"] or "-") for r in v["seeds"])
            print("%-44s %-11s %6d %7s   (unscored; best-ep %s)"
                  % (stem[:44], "--", len(v["seeds"]), "", prog))
            continue
        aps = [r["ap_mean"] for r in v["seeds"] if r["ap_mean"] is not None]
        d = v["ap_mean"] - base if base is not None else None
        by_seed = {r["seed"]: r["ap_mean"] for r in v["seeds"]
                   if r["ap_mean"] is not None}
        t, dfw = welch_t(aps, base_aps) if stem != base_stem else (None, None)
        tp, dfp, pd_ = (paired_t(by_seed, base_by_seed) if stem != base_stem
                        else (None, None, []))
        pw, pp = t_pvalue(t, dfw), t_pvalue(tp, dfp)
        # How many seeds moved the same way? A "win" carried by 2 of 3 seeds
        # with the third flat is a different object from a uniform shift, and
        # the mean alone hides that.
        wins = sum(1 for x in pd_ if x > 0)
        v.update({"delta_vs_base": d, "welch_t": t, "welch_p": pw,
                  "paired_t": tp, "paired_p": pp,
                  "paired_deltas": pd_, "seeds_improved": wins})
        dstr = "%+.4f" % d if d is not None else "--"
        def pf(p):
            return "  --  " if p is None else "%5.3f%s" % (p, "*" if p < P_CLAIM else " ")
        def f(x):
            return "%.4f" % x if x is not None else "  --  "
        seedw = "%d/%d" % (wins, len(pd_)) if pd_ else " -- "
        print("%-44s %.4f±%.3f %6d %7s %6s %6s %5s %7s %7s %7s  %s"
              % (stem[:44], v["ap_mean"], v["ap_sd"], v["n_scored"], dstr,
                 pf(pw), pf(pp), seedw,
                 f(v["presence_map"]), f(v["mean_iou"]), f(v["box_map50"]),
                 f(v["ensemble_ap_mean"])))
    print("\n  * = p < %.2f vs baseline; everything else is a tie at n=3." % P_CLAIM)
    print("  base = %s (%s)" % (args.baseline, "%.4f" % base if base else "not found"))

    if args.json:
        json.dump({"gt_instances_full": GT_FULL, "num_frames_scored": FRAMES,
                   "seed_spread_ref": NOISE_REF, "p_claim": P_CLAIM,
                   "baseline_stem": base_stem,
                   "arms": table}, open(args.json, "w"), indent=1)
        print("\n  index -> %s" % args.json)


if __name__ == "__main__":
    main()
