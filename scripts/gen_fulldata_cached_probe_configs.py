#!/usr/bin/env python3
"""Generate FULL-DATA cached asformer-probe configs (export + probe pair) per
checkpoint.

Same mechanism as the fs10 cached generator, but over the FULL SAR-RARP50 split
(the one the metaraw=78.2 anchor used) instead of Leo's 10% subset. Caching is
the real lever for a frozen-backbone probe: the 300M-param encoder forward
dominates and is identical every epoch, so we run it ONCE (export) and then
train only the tiny head on stored features (~0.5 min/epoch vs ~6-16 min/epoch
re-encoding). See the benchmark finding: batch size barely helps because the
probe is encoder-FLOP-bound, not iteration-bound; caching removes the encoder
cost entirely.

Recipe choices:
- use_sdpa: TRUE everywhere. The XPU layout bug is fixed (cos=1.0, commit
  6d483b2), so flash attention is correct AND ~2x faster for the one encoder
  pass the export does. (Cached probe training never runs the encoder, so sdpa
  is moot there, but harmless.)
- batch_size: 4 (matches Leo's full-data recipe; benchmark showed throughput
  plateaus past bs4 anyway). Head LRs linearly scaled x2 from the bs2 template.
- save_every_iters off, num_workers up, cache_num_workers up (result-neutral).

Footprint: full-data cache ~353 GB/checkpoint (bf16). Orchestrator deletes per
checkpoint after its probe completes (one on disk at a time); 976T free on flare.

CAVEAT: cached features are DETERMINISTIC (no per-epoch augmentation). This is
the accepted tradeoff for the speedup -- valid for checkpoint TREND/ranking. The
augmented headline number (metaraw 78.2) requires the non-cached re-encode path.
"""
import copy
from pathlib import Path

import yaml

ROOT = Path("/lus/flare/projects/ModCon/ngetty/vjepa2")
TEMPLATE = ROOT / "configs/heads/sarrarp50/v2_probe/metaraw_leomatch.yaml"
OUT_DIR = ROOT / "configs/heads/sarrarp50/full_cached"
# Full split CSVs (the metaraw=78.2 anchor's data).
TRAIN_CSV = "/flare/ModCon/ngetty/surg_2_1_v1_probes_aurora/csv/sarrarp50_actions_4fps_asformer_ctx3_seq_train.csv"
VAL_CSV = "/flare/ModCon/ngetty/surg_2_1_v1_probes_aurora/csv/sarrarp50_actions_4fps_asformer_ctx3_seq_val.csv"
RUNS = "/flare/ModCon/ngetty/surg_2_1_v2_final/probes/full_cached"
CACHE = "/flare/ModCon/ngetty/surg_2_1_v2_final/probes/full_cache"

V2 = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v2_final/lr75e6_wu2_decay_n16g12_weak"
V1 = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak"
V1P1 = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase1_warmup_n16g12_weak"
V3 = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v3_lambdaoff_cleandata/lr75e6_wu2_256_n16g12_weak"
META = "/flare/ModCon/ngetty/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"

# (tag, checkpoint, checkpoint_key)
SPECS = [
    ("metaraw", META, "ema_encoder"),
    ("v3_e9", f"{V3}/e9.pth.tar", "target_encoder"),
    ("v1_e9", f"{V1}/e9.pth.tar", "target_encoder"),
    ("v1p1_e12", f"{V1P1}/e12.pth.tar", "target_encoder"),
    ("v2_e9", f"{V2}/e9.pth.tar", "target_encoder"),
]

BATCH_SIZE = 4
LR_SCALE = BATCH_SIZE / 2.0  # template is tuned for bs2; linear-scale the head LRs


def _set_sdpa(cfg, val):
    cfg["model_kwargs"]["pretrain_kwargs"]["encoder"]["use_sdpa"] = val


def _scale_head_lrs(cfg, scale):
    for h in cfg["experiment"]["optimization"].get("multihead_kwargs", []):
        for k in ("lr", "start_lr", "final_lr"):
            if k in h:
                h[k] = h[k] * scale


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(open(TEMPLATE))
    for tag, ckpt, key in SPECS:
        train_cache = f"{CACHE}/{tag}/train"
        val_cache = f"{CACHE}/{tag}/val"

        common = copy.deepcopy(base)
        common["model_kwargs"]["checkpoint"] = ckpt
        common["model_kwargs"]["pretrain_kwargs"]["encoder"]["checkpoint_key"] = key
        common["experiment"]["data"]["dataset_train"] = TRAIN_CSV
        common["experiment"]["data"]["dataset_val"] = VAL_CSV
        common["experiment"]["optimization"]["batch_size"] = BATCH_SIZE
        _scale_head_lrs(common, LR_SCALE)
        # result-neutral speed knobs
        common["save_every_iters"] = 10**9
        common["num_workers"] = 16
        common["experiment"]["data"]["cache_num_workers"] = 8

        # --- export config: runs encoder once, sdpa TRUE (flash attn) ---
        exp = copy.deepcopy(common)
        _set_sdpa(exp, True)
        exp["export_cache"] = True
        exp["experiment"]["data"]["train_cache_root"] = train_cache
        exp["experiment"]["data"]["val_cache_root"] = val_cache
        exp["folder"] = f"{RUNS}/{tag}_export"
        exp["tag"] = f"full-{tag}-export"
        ep = OUT_DIR / f"{tag}_export.yaml"
        yaml.safe_dump(exp, open(ep, "w"), sort_keys=False)

        # --- probe config: reads cache, encoder not run (sdpa moot) ---
        prb = copy.deepcopy(common)
        _set_sdpa(prb, True)
        prb["experiment"]["data"]["train_cache_root"] = train_cache
        prb["experiment"]["data"]["val_cache_root"] = val_cache
        prb["folder"] = f"{RUNS}/{tag}"
        prb["tag"] = f"full-{tag}"
        pp = OUT_DIR / f"{tag}_probe.yaml"
        yaml.safe_dump(prb, open(pp, "w"), sort_keys=False)

        print(f"{tag}: export={ep.name} probe={pp.name} key={key} bs={BATCH_SIZE} lr_scale={LR_SCALE}")


if __name__ == "__main__":
    main()
