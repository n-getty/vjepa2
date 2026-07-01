#!/usr/bin/env python3
"""Generate fewshot-10% CACHED asformer-probe configs (export + probe pair) per
checkpoint.

Two configs per checkpoint, sharing the bs2 Leo-exact head recipe:
  <tag>_export.yaml  -> export_cache: true; runs the frozen encoder once over the
                       fs10 train+val CSVs and writes a backbone-feature cache to
                       per-checkpoint cache_root dirs (then exits).
  <tag>_probe.yaml   -> train_cache_root/val_cache_root set; reads the cache and
                       trains the asformer head fast (seconds/epoch).

Data: Leonardo's video-stratified 10% subset CSVs (already point at our Aurora
data root). Caching makes every checkpoint probed on byte-identical deterministic
features -> apples-to-apples TREND comparison vs the Meta-raw fs10 anchor.

NOTE: NOT comparable to Leo's full-data 79.38 numbers (different subset, no aug).
Re-anchor Meta-raw under this recipe.
"""
import copy
from pathlib import Path

import yaml

ROOT = Path("/lus/flare/projects/ModCon/ngetty/vjepa2")
TEMPLATE = ROOT / "configs/heads/sarrarp50/v2_probe/metaraw_leomatch.yaml"
OUT_DIR = ROOT / "configs/heads/sarrarp50/fs10_cached"
FS10 = "/lus/flare/projects/ModCon/leonardo_borgioli/probes/few_shot/csv"
TRAIN_CSV = f"{FS10}/sarrarp50_actions_4fps_asformer_ctx3_seq_train_fs10.csv"
VAL_CSV = f"{FS10}/sarrarp50_actions_4fps_asformer_ctx3_seq_val_fs10.csv"
RUNS = "/flare/ModCon/ngetty/surg_2_1_v2_final/probes/fs10_cached"
CACHE = "/flare/ModCon/ngetty/surg_2_1_v2_final/probes/fs10_cache"

V2 = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v2_final/lr75e6_wu2_decay_n16g12_weak"
V1 = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak"
# v1 ran in two phases (epoch counter does NOT continue across phases — phase2
# starts its own counter at 0). So v1 "e9" = phase2-epoch-10 = ~22 TOTAL
# surgical epochs (12 phase1 warmup + 10 phase2), trained at 384px. The true
# epoch+res match to v3 (~10 ep from raw Meta, 256px) is v1 PHASE1/e12.
V1P1 = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase1_warmup_n16g12_weak"
# v3 = lambda-off, BLACK-CLIP-FILTERED data, otherwise identical to v2 (gb384,
# lr7.5e-5 wu2, 256px). Probing v3 e9 vs v2 e9 (63.74) isolates the data fix;
# e4 vs e9 reads the WITHIN-version trend (does v3 decline like v1/v2 did?).
V3 = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v3_lambdaoff_cleandata/lr75e6_wu2_256_n16g12_weak"
META = "/flare/ModCon/ngetty/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"

# (tag, checkpoint, checkpoint_key)
SPECS = [
    ("metaraw", META, "ema_encoder"),
    ("v2_e9", f"{V2}/e9.pth.tar", "target_encoder"),
    ("v2_e19", f"{V2}/e19.pth.tar", "target_encoder"),
    ("v1_e9", f"{V1}/e9.pth.tar", "target_encoder"),
    ("v1_e29", f"{V1}/e29.pth.tar", "target_encoder"),
    # v1 epoch-match + decline-curve probes (2026-06-25):
    ("v1p1_e12", f"{V1P1}/e12.pth.tar", "target_encoder"),  # true ~epoch/res match to v3
    ("v1_e19", f"{V1}/e19.pth.tar", "target_encoder"),      # phase2 decline curve
    ("v1_e39", f"{V1}/e39.pth.tar", "target_encoder"),      # phase2 decline curve
    ("v3_e4", f"{V3}/e4.pth.tar", "target_encoder"),
    ("v3_e9", f"{V3}/e9.pth.tar", "target_encoder"),
    ("v3_e14", f"{V3}/e14.pth.tar", "target_encoder"),
    ("v3_e19", f"{V3}/e19.pth.tar", "target_encoder"),
]


def _apply_speed_safe(cfg):
    """Speed knobs that do NOT change the optimization trajectory (so F1 stays
    comparable to existing fs10 numbers — no re-anchor needed):
      - save_every_iters off: the bs2 full run wrote 72x ~2GB Lustre ckpts/epoch
        mid-training, stalling all ranks. We only need end-of-epoch latest.pt
        (resume_checkpoint still works). Pure overhead removal.
      - cache_num_workers up: parallel cache-shard reads (was 1) for the cached
        path's I/O. No effect on results.
      - num_workers up: more video-decode parallelism for the full path.
    NOTE: batch_size / head-LR / use_sdpa are NOT touched here — those change the
    result and require a benchmark + re-anchor (see scripts/bench_probe_speed.sh).
    """
    cfg["save_every_iters"] = 10**9
    cfg.setdefault("num_workers", 16)
    cfg["num_workers"] = 16
    cfg["experiment"]["data"]["cache_num_workers"] = 8
    return cfg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(open(TEMPLATE))
    for tag, ckpt, key in SPECS:
        train_cache = f"{CACHE}/{tag}/train"
        val_cache = f"{CACHE}/{tag}/val"

        common = copy.deepcopy(base)
        common["model_kwargs"]["checkpoint"] = ckpt
        common["model_kwargs"]["pretrain_kwargs"]["encoder"]["checkpoint_key"] = key
        _apply_speed_safe(common)
        # fs10 subset CSVs
        common["experiment"]["data"]["dataset_train"] = TRAIN_CSV
        common["experiment"]["data"]["dataset_val"] = VAL_CSV

        # --- export config ---
        exp = copy.deepcopy(common)
        exp["export_cache"] = True
        exp["experiment"]["data"]["train_cache_root"] = train_cache
        exp["experiment"]["data"]["val_cache_root"] = val_cache
        exp["folder"] = f"{RUNS}/{tag}_export"
        exp["tag"] = f"fs10-{tag}-export"
        ep = OUT_DIR / f"{tag}_export.yaml"
        yaml.safe_dump(exp, open(ep, "w"), sort_keys=False)

        # --- probe config (reads cache) ---
        prb = copy.deepcopy(common)
        prb["experiment"]["data"]["train_cache_root"] = train_cache
        prb["experiment"]["data"]["val_cache_root"] = val_cache
        prb["folder"] = f"{RUNS}/{tag}"
        prb["tag"] = f"fs10-{tag}"
        pp = OUT_DIR / f"{tag}_probe.yaml"
        yaml.safe_dump(prb, open(pp, "w"), sort_keys=False)

        print(f"{tag}: export={ep.name} probe={pp.name} (key={key})")


if __name__ == "__main__":
    main()
