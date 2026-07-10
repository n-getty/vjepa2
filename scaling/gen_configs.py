"""Generate per-cell training YAMLs for the JEPA IsoFLOP sweep.

Takes a base config + a planner manifest (scaling/plan.py --out) and writes one YAML per runnable
cell, overriding ONLY the scaling axes (model size, steps/epochs, warmup) and holding everything else
fixed (data mix, mask, optimizer recipe, dtype). This is the launcher-config-drift-safe path: the
sweep's degrees of freedom live in the manifest, never in hand-edited YAMLs.

Overrides applied per cell:
  model.model_name        <- cell model_name
  optimization.epochs     <- cell epochs
  optimization.ipe        <- cell ipe
  optimization.warmup     <- cell warmup_epochs
  data.batch_size         <- global_batch // world_size   (per-rank; global held fixed by design)

Everything the scaling law must NOT vary (LR, wd, ema, sampling_temperature, min_clip_std, crop_size,
patch_size, tubelet_size, fps, mask views) is inherited verbatim from the base config. We also flip
meta.load_checkpoint off by default (scaling runs are FROM SCRATCH unless --cpt), and repoint the
corpus if --data-root is given.
"""

import argparse
import copy
import json
import os

import yaml


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _slug(cell):
    b = f"{cell['budget_flops']:.1e}".replace("+", "").replace(".", "p")
    return f"{cell['model_name']}_C{b}"


def _recompute_flops_params(cfg, fallback_mask_views=None):
    """Recompute (train_flops_per_clip, n_params) from a MERGED training config, so the scaling stamp
    reflects exactly what will train (not the planner's assumptions). Uses scaling/flops.py on the
    config's actual model/data/mask fields.

    Mask note: the config may use VARIABLE-length masks (num_keep_enc/pred = None). FLOPs then have no
    single value; we fall back to `fallback_mask_views` (the fixed keep-counts the PLANNER used) so the
    stamp and the manifest share one consistent FLOP proxy across all cells. Since every cell uses the
    same mask config, a consistent proxy preserves the relative FLOPs the scaling exponents depend on.
    """
    from scaling.flops import arch_from_model, clip_forward_flops

    m, d = cfg["model"], cfg["data"]
    frames = (d.get("dataset_fpcs") or [d.get("num_frames", 16)])[0]
    mask_views = [{"num_keep_enc": mv["num_keep_enc"], "num_keep_pred": mv["num_keep_pred"]}
                  for mv in (cfg.get("mask") or [])
                  if mv.get("num_keep_enc") is not None and mv.get("num_keep_pred") is not None]
    if not mask_views and fallback_mask_views:
        mask_views = fallback_mask_views
    arch = arch_from_model(
        m["model_name"], frames, d.get("crop_size", 224), d.get("patch_size", 16),
        d.get("tubelet_size", 2), m.get("pred_depth", 6), m.get("pred_embed_dim", 384),
        m.get("pred_num_heads", None),
        use_rope=m.get("use_rope", False), use_sdpa=m.get("use_sdpa", False),
        uniform_power=m.get("uniform_power", False))
    fwd, _ = clip_forward_flops(arch, frames, d.get("crop_size", 224), d.get("patch_size", 16),
                                d.get("tubelet_size", 2), mask_views)
    n_params = arch["encoder"]["params"] + arch["predictor"]["params"]
    return 3 * fwd, int(n_params)


def _check_geometry(base_cfg, manifest):
    """Refuse to emit if the manifest's FLOP geometry doesn't match the base config.

    The manifest's steps/D/C were computed at a specific (res, frames, patch, tubelet). If the base
    config trains at a different geometry, every clips_seen_D and actual_flops in the manifest is wrong
    (e.g. 256px=2048 tokens vs 384px=4608 tokens is a ~2.8x compute error). This is the
    launcher-config-drift trap; catch it here rather than silently mis-scale the whole sweep.
    """
    m = manifest["meta"]
    d = base_cfg["data"]
    base_frames = (d.get("dataset_fpcs") or [d.get("num_frames", 16)])[0]
    checks = {
        "res": (m["res"], d.get("crop_size")),
        "frames": (m["frames"], base_frames),
        "patch": (m["patch"], d.get("patch_size")),
        "tubelet": (m["tubelet"], d.get("tubelet_size")),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        msg = "; ".join(f"{k}: manifest={v[0]} base_config={v[1]}" for k, v in bad.items())
        raise ValueError(
            f"GEOMETRY MISMATCH between manifest and base config ({msg}). "
            f"The manifest's steps/D/C were computed at the manifest geometry; emitting configs at a "
            f"different geometry would silently mis-scale the entire sweep. Re-run scaling.plan with "
            f"matching --res/--frames/--patch/--tubelet, or use a base config at the manifest geometry."
        )


def gen(base_cfg, manifest, out_dir, world_size, from_scratch, data_root, seed_base,
        folder_root=None, use_topology=False, max_nodes=1):
    _check_geometry(base_cfg, manifest)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for cell in manifest["cells"]:
        if cell["status"] != "ok":
            continue
        cfg = copy.deepcopy(base_cfg)

        # --- scaling axes (the only things that vary) ---
        cfg["model"]["model_name"] = cell["model_name"]
        cfg["optimization"]["epochs"] = cell["epochs"]
        cfg["optimization"]["ipe"] = cell["ipe"]
        cfg["optimization"]["warmup"] = cell["warmup_epochs"]

        gb = cell["global_batch"]
        launch_spec = None
        if use_topology:
            # PER-SIZE topology (scaling/topology.py): hold global batch fixed, but factor it into
            # (tiles, per_rank_bs, accum, dist_strategy) tuned to each size. Emits a _launch.json the
            # launcher reads. This replaces the one-world-size-for-all assumption.
            from scaling.topology import topology_for
            spec = topology_for(cell["model_name"], global_batch=gb, max_nodes=max_nodes)
            per_rank = spec.per_rank_bs
            cfg["data"]["batch_size"] = per_rank
            cfg["nodes"] = spec.nodes
            cfg["tasks_per_node"] = min(spec.tiles, 12)
            launch_spec = {
                "model_name": spec.model_name, "tiles": spec.tiles, "per_rank_bs": spec.per_rank_bs,
                "accum": spec.accum, "dist_strategy": spec.dist_strategy, "nodes": spec.nodes,
                "env": spec.env, "global_batch": spec.global_batch(),
            }
        else:
            # legacy: fixed world size, per-rank batch = global / world_size
            if gb % world_size != 0:
                raise ValueError(f"global_batch {gb} not divisible by world_size {world_size}")
            per_rank = gb // world_size
            cfg["data"]["batch_size"] = per_rank

        # --- per-run output folder (unique; base config's shared folder would collide) ---
        slug_for_folder = _slug(cell)
        if folder_root:
            cfg["folder"] = os.path.join(folder_root, slug_for_folder)

        # --- invariants / study hygiene ---
        if from_scratch:
            # BOTH load paths must be disabled. `load_checkpoint` gates resume; `pretrain_checkpoint`
            # (p_file) gates the SEPARATE load_pretrained() bootstrap at train.py:659 — leaving it set
            # makes a "from scratch" run try to load the base ViT-G checkpoint into a differently-sized
            # encoder → 0/158 keys match → crash (caught by the pilot, job 8659993).
            cfg["meta"]["load_checkpoint"] = False
            cfg["meta"]["load_predictor"] = False
            cfg["meta"]["pretrain_checkpoint"] = None

        # optional corpus repoint (single clean general corpus for the general law)
        if data_root:
            n = len(cfg["data"]["datasets"])
            cfg["data"]["datasets"] = [data_root]
            cfg["data"]["datasets_weights"] = [1]
            cfg["data"]["dataset_fpcs"] = [cfg["data"]["dataset_fpcs"][0]]

        # stamp identity for collect/analysis. RECOMPUTE train_flops_per_clip + n_params from the
        # MERGED config (actual model_name/pred_depth/pred_embed/geometry/mask that will train), not
        # from the planner cell — the planner may use default pred_depth (12) while the base config
        # inherits a different one (24), which would silently corrupt C and D_opt. Single source of
        # truth = the config that actually runs. (Drift caught by pilot job 8659993.)
        tf_clip, n_params_stamp = _recompute_flops_params(cfg, manifest["meta"].get("mask_views"))
        cfg.setdefault("scaling", {})
        cfg["scaling"] = {
            "budget_flops": cell["budget_flops"],
            "n_params": n_params_stamp,
            "train_flops_per_clip": tf_clip,
            "global_batch": gb,
            # D and actual C recomputed from the corrected per-clip FLOPs so the stamp is consistent.
            "clips_seen_D": cell["clips_seen_D"],
            "seed": seed_base,
        }
        cfg["meta"]["seed"] = seed_base

        slug = _slug(cell)
        path = os.path.join(out_dir, f"{slug}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        if launch_spec is not None:
            with open(os.path.join(out_dir, f"{slug}_launch.json"), "w") as f:
                json.dump(launch_spec, f, indent=2)
        written.append((slug, per_rank, cell, launch_spec))
    return written


def main():
    ap = argparse.ArgumentParser(description="Emit per-cell YAMLs for the JEPA IsoFLOP sweep")
    ap.add_argument("--base", required=True, help="base training YAML to inherit invariants from")
    ap.add_argument("--manifest", required=True, help="planner manifest JSON (scaling/plan.py --out)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--world-size", type=int, default=0,
                    help="ranks (tiles) the sweep runs on; per-rank batch = global_batch/world_size. "
                         "Ignored under --use-topology.")
    ap.add_argument("--cpt", action="store_true",
                    help="continue-pretrain from base checkpoint (default: from scratch)")
    ap.add_argument("--data-root", help="single corpus dir to override the base data mix")
    ap.add_argument("--seed", type=int, default=239)
    ap.add_argument("--folder-root", default=None,
                    help="root dir for per-run output folders (each cell -> <root>/<slug>). "
                         "Required for real runs; base config's shared folder would collide.")
    ap.add_argument("--use-topology", action="store_true",
                    help="use scaling/topology.py per-size (tiles,per_rank_bs,accum,strategy) instead "
                         "of a single --world-size; emits a <slug>_launch.json per cell.")
    ap.add_argument("--max-nodes", type=int, default=1,
                    help="with --use-topology: cap nodes/cell (rest of global batch via grad-accum). "
                         "1=node-frugal (giant/gigantic accum), higher=less wallclock more nodes.")
    args = ap.parse_args()

    base_cfg = _load_yaml(args.base)
    manifest = json.load(open(args.manifest))
    written = gen(base_cfg, manifest, args.out_dir, args.world_size,
                  from_scratch=not args.cpt, data_root=args.data_root, seed_base=args.seed,
                  folder_root=args.folder_root, use_topology=args.use_topology,
                  max_nodes=args.max_nodes)

    print(f"wrote {len(written)} configs to {args.out_dir}/")
    for slug, per_rank, cell, spec in written:
        extra = ""
        if spec:
            extra = (f" [tiles={spec['tiles']} accum={spec['accum']} {spec['dist_strategy']} "
                     f"nodes={spec['nodes']}]")
        print(f"  {slug}.yaml  per_rank_bs={per_rank} epochs={cell['epochs']} "
              f"steps={cell['steps']} D={cell['clips_seen_D']}{extra}")


if __name__ == "__main__":
    main()
