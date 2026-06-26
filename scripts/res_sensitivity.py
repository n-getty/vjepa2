#!/usr/bin/env python3
"""Resolution-sensitivity test for the v3 regression.

Question: is the train-256 / probe-384 mismatch a driver of the monotonic
downstream regression? If so, v3 checkpoints should diverge from the Meta init
MORE at 384 (the probe res, which CPT de-adapted from) than at 256 (CPT-native).

For each checkpoint [meta, e9, e19] and resolution [256, 384], embed the SAME
surgical clips and report cos(ckpt, meta) per resolution. Also reports the
e9->e19 drop at each resolution: if the drop is larger at 384, resolution
mismatch amplifies the regression.

Pure inference (~minutes), no probe training. Run on a held node:
  ZE_AFFINITY_MASK=0 python scripts/res_sensitivity.py
"""
import sys, numpy as np, torch
sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/vjepa2")
from scripts.analyze_features import build_encoder, sample_clips

V3 = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v3_lambdaoff_cleandata/lr75e6_wu2_256_n16g12_weak"
META = "/flare/ModCon/ngetty/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"
CKPTS = [("meta", META, "ema_encoder"),
         ("e9", f"{V3}/e9.pth.tar", "target_encoder"),
         ("e19", f"{V3}/e19.pth.tar", "target_encoder")]
SURG = "/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/surgvu24"


def main():
    dev = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    assert dev.type == "xpu", "XPU not available (check ZE_AFFINITY_MASK=0)"
    print(f"device={dev}", flush=True)

    @torch.no_grad()
    def emb(enc, clip):
        x = torch.from_numpy(clip).float() / 255.0
        x = x.permute(3, 0, 1, 2).unsqueeze(0)
        m = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        s = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        x = ((x - m) / s).to(dev)
        o = enc([x]); o = o[0] if isinstance(o, (list, tuple)) else o
        if o.dim() == 3:
            o = o.mean(1)
        return o.float().cpu().numpy()[0]

    # Sample clips ONCE per resolution (same clips across checkpoints).
    clips = {}
    for res in (256, 384):
        clips[res] = sample_clips(SURG, 24, frames_per_clip=16, frame_step=1, target_hw=res)
        print(f"res={res}: {len(clips[res])} clips", flush=True)

    # embed all checkpoints at both res
    emb_by = {}  # (tag,res) -> [N,D]
    for tag, path, key in CKPTS:
        enc = build_encoder(path, key, dev)
        for res in (256, 384):
            E = np.stack([emb(enc, c) for c in clips[res]])
            emb_by[(tag, res)] = E
        del enc
        if dev.type == "xpu":
            torch.xpu.empty_cache()
        print(f"embedded {tag}", flush=True)

    def cos_to_meta(tag, res):
        a = emb_by[(tag, res)].mean(0); b = emb_by[("meta", res)].mean(0)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    print("\n=== cos(ckpt, meta) per resolution ===", flush=True)
    print(f"{'ckpt':6}{'res256':>10}{'res384':>10}{'384-256':>10}")
    for tag, _, _ in CKPTS:
        c256, c384 = cos_to_meta(tag, 256), cos_to_meta(tag, 384)
        print(f"{tag:6}{c256:10.4f}{c384:10.4f}{c384-c256:10.4f}")
    print("\n=== KEY: e9->e19 divergence-from-meta drop at each res ===")
    for res in (256, 384):
        d = cos_to_meta("e9", res) - cos_to_meta("e19", res)
        print(f"  res={res}: cos drop e9->e19 = {d:.4f}")
    print("\nIf the e9->e19 drop (and/or the e19 384-256 gap) is LARGER at 384,")
    print("resolution mismatch amplifies the regression the 384-probe sees.")


if __name__ == "__main__":
    main()
