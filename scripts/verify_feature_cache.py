"""Single-tile verification of the backbone-feature-cache exporter.

Checks, on a tiny slice of the fs10 train CSV:
  1. FORMAT: export writes a cache that BackboneFeatureCacheDataset loads, and
     returns (features, label, row_index, path) with the expected feature shape.
  2. EQUIVALENCE: cached features[:, view] == a fresh live encoder(clips)[view]
     (cosine ~1, max-abs within bf16 noise) -> the cache reproduces the live path.

Run on one Aurora tile:
  ZE_AFFINITY_MASK=0 python scripts/verify_feature_cache.py
"""
import os
os.environ.setdefault("ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK", "0"))
import sys
import tempfile

REPO = "/lus/flare/projects/ModCon/ngetty/vjepa2"
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.eval import (
    export_feature_cache, DEFAULT_NORMALIZATION,
)
from src.datasets.backbone_feature_cache import BackboneFeatureCacheDataset
from src.datasets.video_dataset import make_videodataset
from evals.video_classification_frozen.utils import make_transforms

DEVICE = "xpu"
META = "/flare/ModCon/ngetty/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"
TRAIN_CSV = ("/lus/flare/projects/ModCon/leonardo_borgioli/probes/few_shot/csv/"
             "sarrarp50_actions_4fps_asformer_ctx3_seq_train_fs10.csv")

# Match the fs10 probe config recipe.
RES, FPC, FSTEP, NSEG, NVIEW = 384, 16, 1, 3, 1


def make_tiny_csv(n=16):
    """Write a temp CSV with the first n rows of the fs10 train CSV."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    with open(TRAIN_CSV) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            tmp.write(line)
    tmp.close()
    return tmp.name


def build_encoder():
    return init_module(
        module_name="evals.video_classification_frozen.modelcustom.vit_encoder_multiclip_v21",
        frames_per_clip=FPC, resolution=RES, checkpoint=META,
        model_kwargs={"encoder": {
            "checkpoint_key": "ema_encoder", "model_name": "vit_large",
            "patch_size": 16, "tubelet_size": 2, "use_rope": True,
            "use_sdpa": False, "img_temporal_dim_size": 1, "uniform_power": True,
        }},
        wrapper_kwargs={"max_frames": 128, "use_pos_embed": False, "preserve_clip_dim": True},
        device=DEVICE,
    )


def main():
    assert torch.xpu.is_available(), "no XPU"
    tiny = make_tiny_csv(16)
    cache_root = tempfile.mkdtemp(prefix="featcache_")
    print(f"tiny csv={tiny}  cache_root={cache_root}")

    enc = build_encoder()

    # --- 1. EXPORT ---
    n = export_feature_cache(
        encoder=enc, cache_root=cache_root, dataset_type="VideoDataset",
        root_path=[tiny], resolution=RES, frames_per_clip=FPC, frame_step=FSTEP,
        num_segments=NSEG, num_views_per_segment=NVIEW, allow_segment_overlap=True,
        sequence_labels=True, normalization=DEFAULT_NORMALIZATION,
        batch_size=1, num_workers=2, world_size=1, rank=0, device=DEVICE,
        use_bfloat16=True, shard_size=8,
    )
    print(f"exported {n} samples")

    # --- 2. FORMAT: load via reader ---
    ds = BackboneFeatureCacheDataset(cache_root)
    feats0, label0, row0, path0 = ds[0]
    print(f"reader[0]: features={tuple(feats0.shape)} label={getattr(label0,'shape',label0)} "
          f"row={row0} path=...{path0[-40:]}")
    print(f"feature_shape_per_sample={ds.feature_shape}  len={len(ds)}")

    # --- 3. EQUIVALENCE: cached vs fresh live forward for the same clip ---
    # Build a deterministic loader identical to the exporter's, grab batch 0.
    transform = make_transforms(training=False, num_views_per_clip=NVIEW,
                                crop_size=RES, normalize=DEFAULT_NORMALIZATION)
    _d, loader, _s = make_videodataset(
        data_paths=[tiny], batch_size=1, frames_per_clip=FPC, frame_step=FSTEP,
        num_clips=NSEG, allow_clip_overlap=True, transform=transform,
        world_size=1, rank=0, drop_last=False, num_workers=2,
        sequence_labels=True, return_sample_path=True,
    )
    data = next(iter(loader))
    clips = [[dij.to(DEVICE) for dij in di] for di in data[0]]
    clip_indices = [d.to(DEVICE) for d in data[2]]
    with torch.no_grad(), torch.amp.autocast(DEVICE, dtype=torch.bfloat16, enabled=True):
        live = enc(clips, clip_indices)  # list[num_views] of [B, ...]
    live0 = live[0][0].float().cpu()         # view 0, sample 0
    cached0 = feats0[0].float()               # view 0 of cached sample 0

    cos = F.cosine_similarity(live0.flatten(), cached0.flatten(), dim=0).item()
    maxd = (live0 - cached0).abs().max().item()
    print(f"\nEQUIVALENCE (sample 0, view 0): cos={cos:.6f}  max|delta|={maxd:.4e}")
    print(f"live shape={tuple(live0.shape)} cached shape={tuple(cached0.shape)}")
    if cos > 0.999:
        print("PASS: cache reproduces the live encoder path.")
    else:
        print("FAIL: cached features diverge from live path!")


if __name__ == "__main__":
    main()
