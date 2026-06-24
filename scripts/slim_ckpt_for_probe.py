"""Slim a full V-JEPA 2.1 pretraining checkpoint for cross-system probing.

The asformer probe only reads `target_encoder`. The full checkpoint is ~5.5 GB
(encoder + predictor + opt + scaler + target_encoder); the probe needs ~1.2 GB.
Slimming keeps target_encoder + small scalars so the rsync to Polaris/Eagle is
~4.5x smaller. Output format matches the existing v1 _probe_export checkpoints.

    python scripts/slim_ckpt_for_probe.py <full.pth.tar> <slim_out.pth.tar>
"""
import sys
import torch

KEEP = ("target_encoder", "epoch", "loss", "batch_size", "world_size", "lr")


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if "target_encoder" not in ck:
        raise SystemExit(f"{src}: no target_encoder key (keys: {list(ck)})")
    slim = {k: ck[k] for k in KEEP if k in ck}
    torch.save(slim, dst)
    print(f"slimmed {src} -> {dst}  (kept: {[k for k in KEEP if k in ck]})")


if __name__ == "__main__":
    main()
