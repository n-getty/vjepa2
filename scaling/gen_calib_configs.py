"""Generate short trainer configs to CALIBRATE per-size batch/topology for the scaling sweep.

The pilot (jepa-scaling-pilot memory) showed small models are OVERHEAD-bound at 256px: fwd-target was
flat ~520ms across tiny->base despite 15x params, so clips/s barely moved (20->15.3). But the pilot ran
per_rank_bs=2 everywhere, so two things are UNKNOWN and gate the real-sweep cost:
  (1) does a LARGER per_rank_bs raise small-model clips/s (i.e. is ~520ms fixed overhead we can
      amortize by packing more clips per iter)?
  (2) what is the MAX per_rank_bs each size fits in one XPU tile at 256px before OOM?

This emits tiny configs (few iters) over a (model x per_rank_bs x tiles) grid from the pilot template.
Each runs the REAL trainer briefly and writes log_r0.csv; scaling/read_calib.py reads median iter-time
-> clips/s and the OOM/no-OOM verdict. We deliberately use the real trainer (not a synthetic microbench)
so the numbers transfer directly to the sweep.

Usage:
  python -m scaling.gen_calib_configs --template configs/scaling/pilot6/vit_small_C3p0e16.yaml \
      --out-dir configs/scaling/calib --grid small:1:2,8,32 base:1:2,8 large:1:1,2,4 \
      --iters 60
Grid entries are  MODEL:TILES:BS1,BS2,...  (tiles = ranks in this cell; keep global batch implicit —
calibration measures raw per-tile throughput + OOM, not the fixed-global-batch sweep invariant).
"""

import argparse
import copy
import os

import yaml

# Off-the-shelf 2.1 encoder embed dims (for the scaling: stamp; exact params come from the trainer).
MODEL_EMBED = {
    "vit_tiny": 192, "vit_small": 384, "vit_base": 768,
    "vit_large": 1024, "vit_giant": 1408, "vit_gigantic": 1664,
}


def gen(template_path, out_dir, grid, iters, corpus, exp_root):
    base = yaml.safe_load(open(template_path))
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for entry in grid:
        model, tiles_s, bss = entry.split(":")
        # trainer's video_vit registry keys are 'vit_tiny'... not 'tiny'; accept either in the grid.
        if not model.startswith("vit_"):
            model = f"vit_{model}"
        tiles = int(tiles_s)
        for bs in [int(x) for x in bss.split(",")]:
            cfg = copy.deepcopy(base)
            slug = f"calib_{model}_t{tiles}_bs{bs}"
            cfg["nodes"] = 1
            cfg["tasks_per_node"] = tiles
            cfg["model"]["model_name"] = model
            cfg["data"]["batch_size"] = bs
            if corpus:
                cfg["data"]["datasets"] = [corpus]
                cfg["data"]["datasets_weights"] = [1]
            # tiny run: 1 epoch of `iters` steps, no warmup shenanigans
            cfg["optimization"]["epochs"] = 1
            cfg["optimization"]["ipe"] = iters
            if "warmup" in cfg["optimization"]:
                cfg["optimization"]["warmup"] = 0
            # from scratch, no checkpoint load (calibration is throughput-only)
            cfg["meta"]["load_checkpoint"] = False
            if "pretrain_checkpoint" in cfg.get("meta", {}):
                cfg["meta"]["pretrain_checkpoint"] = None
            cfg["folder"] = os.path.join(exp_root, slug)
            # provenance stamp so collect ignores it as a sweep cell but trainer writes scaling.json
            cfg["scaling"] = {"calib": True, "model": model, "tiles": tiles, "per_rank_bs": bs}
            out = os.path.join(out_dir, f"{slug}.yaml")
            yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
            written.append((slug, model, tiles, bs, out))
    return written


def main():
    ap = argparse.ArgumentParser(description="Generate calibration configs")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--grid", nargs="+", required=True, help="entries MODEL:TILES:BS1,BS2,...")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--corpus", default="/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/kinetics400")
    ap.add_argument("--exp-root", default="/flare/ModCon/ngetty/experiments/scaling_calib")
    args = ap.parse_args()
    written = gen(args.template, args.out_dir, args.grid, args.iters, args.corpus, args.exp_root)
    print(f"wrote {len(written)} calib configs to {args.out_dir}:")
    for slug, model, tiles, bs, out in written:
        print(f"  {slug}: {model} tiles={tiles} per_rank_bs={bs}")


if __name__ == "__main__":
    main()
