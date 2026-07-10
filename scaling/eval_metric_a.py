"""Metric A driver: score sweep checkpoints with the SSv2 frozen-probe eval.

Two subcommands:
  gen   — for each run folder (with a scaling.json + a saved checkpoint), emit a per-checkpoint SSv2
          eval YAML derived from a base eval config, overriding ONLY: checkpoint path, encoder
          model_name + resolution (matched to the run's own size), num_classes, and output folder.
          The eval itself is GPU/MPI work; run the emitted YAMLs with the existing Aurora eval
          launcher (app.main_dist_aurora / run_asformer_probe_aurora.sh pattern).
  read  — after evals finish, parse each eval's log_r0.csv for best_val_f1 / best_val_acc and write a
          metric_A.json sidecar into the RUN folder. scaling/collect.py joins that into experiments.csv,
          and scaling/fit.py --metric metric_a_error fits the scaling law on it.

Design-doc alignment (§2 Metric A): the y-axis is downstream probe ERROR with headroom. SSv2 is the
discriminating benchmark; we emit both f1 and acc and let the fit consume error = 1 - metric.
Ceiling gate is the caller's responsibility: check the largest-N point isn't saturated before trusting
the exponent.

The base eval config must be a working SSv2 config (configs/eval_2_1/*/ssv2.yaml). This driver never
touches probe hyperparameters (epochs, LR sweep, num_heads) — those stay fixed across the sweep so the
probe is a constant function of the representation.
"""

import argparse
import copy
import csv
import glob
import json
import os

import yaml

# encoder model_name used at PRETRAIN time -> model_name the eval encoder wrapper expects.
# The eval side uses the xformers RoPE variants for giant/gigantic; small sizes are unchanged.
PRETRAIN_TO_EVAL_MODEL = {
    "vit_gigantic_xformers": "vit_gigantic_xformers",
    "vit_giant_xformers": "vit_giant_xformers",
    "vit_gigantic": "vit_gigantic",
    "vit_giant": "vit_giant",
    "vit_large": "vit_large",
    "vit_base": "vit_base",
    "vit_small": "vit_small",
    "vit_tiny": "vit_tiny",
}


def _load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def _find_checkpoint(run_dir):
    """Prefer latest.pt/latest.pth.tar; else the highest-epoch checkpoint."""
    for name in ("latest.pt", "latest.pth.tar"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return p
    cands = sorted(glob.glob(os.path.join(run_dir, "*.pt")) +
                   glob.glob(os.path.join(run_dir, "*.pth.tar")))
    return cands[-1] if cands else None


def _set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def gen(base_cfg_path, runs_glob, out_dir, ssv2_train, ssv2_val, num_classes, resolution):
    base = _load_yaml(base_cfg_path)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for run_dir in sorted(d for d in glob.glob(runs_glob) if os.path.isdir(d)):
        sc_path = os.path.join(run_dir, "scaling.json")
        if not os.path.exists(sc_path):
            continue
        sc = json.load(open(sc_path))
        ckpt = _find_checkpoint(run_dir)
        if not ckpt:
            print(f"  [skip] no checkpoint in {run_dir}")
            continue
        pre_model = sc.get("model_name")
        eval_model = PRETRAIN_TO_EVAL_MODEL.get(pre_model, pre_model)

        cfg = copy.deepcopy(base)
        run_id = os.path.basename(os.path.normpath(run_dir))
        eval_folder = os.path.join(out_dir, f"eval_{run_id}")

        _set(cfg, "model_kwargs.checkpoint", ckpt)
        _set(cfg, "model_kwargs.pretrain_kwargs.encoder.checkpoint_key", "target_encoder")
        _set(cfg, "model_kwargs.pretrain_kwargs.encoder.model_name", eval_model)
        _set(cfg, "folder", eval_folder)
        _set(cfg, "experiment.data.dataset_train", ssv2_train)
        _set(cfg, "experiment.data.dataset_val", ssv2_val)
        _set(cfg, "experiment.data.num_classes", num_classes)
        if resolution:
            _set(cfg, "experiment.data.resolution", resolution)
        # stamp back-reference so `read` can find the run folder
        cfg["_scaling_run_dir"] = os.path.abspath(run_dir)

        out_yaml = os.path.join(out_dir, f"{run_id}_ssv2.yaml")
        with open(out_yaml, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        written.append((run_id, out_yaml, eval_model, ckpt))

    print(f"wrote {len(written)} SSv2 eval configs to {out_dir}/")
    for run_id, y, m, c in written:
        print(f"  {run_id}: model={m} ckpt={os.path.basename(c)}")
    if written:
        print("\nRun them with the Aurora eval launcher, e.g.:")
        print(f"  for f in {out_dir}/*_ssv2.yaml; do <launch> --fname $f; done")
    return written


def _read_eval_csv(eval_folder):
    """Find the eval's log_r0.csv (may be nested under a tag) and return best f1/acc."""
    cands = glob.glob(os.path.join(eval_folder, "**", "log_r0.csv"), recursive=True)
    if not cands:
        return None
    best_f1 = best_acc = None
    for csv_path in cands:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            for row in reader:
                for key, tgt in (("best_val_f1", "f1"), ("val_macro_f1", "f1"),
                                 ("best_val_acc", "acc"), ("val_acc", "acc")):
                    if key in cols and row.get(key, "") != "":
                        try:
                            v = float(row[key])
                        except ValueError:
                            continue
                        if tgt == "f1":
                            best_f1 = v if best_f1 is None else max(best_f1, v)
                        else:
                            best_acc = v if best_acc is None else max(best_acc, v)
    if best_f1 is None and best_acc is None:
        return None
    return {"best_val_f1": best_f1, "best_val_acc": best_acc}


def read(configs_glob):
    n = 0
    for cfg_path in sorted(glob.glob(configs_glob)):
        cfg = _load_yaml(cfg_path)
        run_dir = cfg.get("_scaling_run_dir")
        eval_folder = cfg.get("folder")
        if not run_dir or not eval_folder:
            continue
        res = _read_eval_csv(eval_folder)
        if not res:
            print(f"  [pending] no results yet: {os.path.basename(cfg_path)}")
            continue
        out = dict(res)
        # error columns (fit expects lower-is-better; probe metrics are 0..1 or 0..100)
        def _err(v):
            if v is None:
                return None
            return (1.0 - v) if v <= 1.0 else (100.0 - v)
        out["metric_a_error_f1"] = _err(res.get("best_val_f1"))
        out["metric_a_error_acc"] = _err(res.get("best_val_acc"))
        out["source"] = "ssv2_frozen_probe"
        with open(os.path.join(run_dir, "metric_A.json"), "w") as f:
            json.dump(out, f, indent=2)
        n += 1
        print(f"  {os.path.basename(run_dir)}: f1={res.get('best_val_f1')} acc={res.get('best_val_acc')}")
    print(f"wrote {n} metric_A.json sidecars")


def main():
    ap = argparse.ArgumentParser(description="Metric A (SSv2 frozen probe) driver")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="emit per-checkpoint SSv2 eval configs")
    g.add_argument("--base", required=True, help="base SSv2 eval YAML")
    g.add_argument("--runs", required=True, help="glob of run folders")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--ssv2-train", required=True, help="ssv2_train_paths.csv")
    g.add_argument("--ssv2-val", required=True, help="ssv2_val_paths.csv")
    g.add_argument("--num-classes", type=int, default=174)
    g.add_argument("--resolution", type=int, default=None,
                   help="override eval resolution (default: keep base config's)")

    r = sub.add_parser("read", help="harvest eval results into metric_A.json sidecars")
    r.add_argument("--configs", required=True, help="glob of *_ssv2.yaml emitted by gen")

    args = ap.parse_args()
    if args.cmd == "gen":
        gen(args.base, args.runs, args.out_dir, args.ssv2_train, args.ssv2_val,
            args.num_classes, args.resolution)
    else:
        read(args.configs)


if __name__ == "__main__":
    main()
