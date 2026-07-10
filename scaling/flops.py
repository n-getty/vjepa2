"""Analytic FLOP + exact parameter accounting for V-JEPA 2.1 (encoder + predictor).

This is the FLOP/param layer for the JEPA scaling-law study (docs/JEPA_SCALING_LAWS_DESIGN.md).
It deliberately does NOT use the LLM `C = 6*N*D` shortcut: at V-JEPA token counts (giant @ 384px,
16f, patch16, tubelet2 => 4608 tokens) the O(n^2) attention term is a large fraction of compute, so
the naive rule is wrong. We count matmul FLOPs term-by-term for both the encoder and the predictor,
and we take the *shape* of the model (embed_dim/depth/heads/mlp_ratio, token counts, predictor keep-
counts) from the SAME construction path the trainer uses (`init_video_model`), instantiated on the
`meta` device so it costs no memory and can never drift from the real arch.

FLOP convention: 1 multiply-accumulate = 2 FLOPs. A matmul (m,k)x(k,n) = 2*m*k*n FLOPs. We report
FORWARD FLOPs; training-step FLOPs are ~3x forward (1 fwd + 2 bwd), applied by the caller. We ignore
LayerNorm/GELU/softmax elementwise costs (<1% at these dims) and the patch-embed conv (one projection,
negligible vs the transformer stack) EXCEPT we include it for completeness since it is cheap to add.

Usage:
    python -m scaling.flops --config configs/.../vitG384_fixedshape.yaml
    python -m scaling.flops --model vit_large --frames 16 --res 224 --patch 16 --tubelet 2 \
        --pred-depth 12 --pred-embed 384 --pred-heads 16
"""

import argparse
import math

import torch
import yaml

# ---------------------------------------------------------------------------
# Matmul FLOP primitives (2 FLOPs per MAC).
# ---------------------------------------------------------------------------


def _linear_flops(n_tokens, d_in, d_out):
    """Forward FLOPs of an (n_tokens, d_in) x (d_in, d_out) projection."""
    return 2 * n_tokens * d_in * d_out


def _transformer_stack_flops(n_tokens, embed_dim, depth, num_heads, mlp_ratio):
    """Forward FLOPs of `depth` standard pre-norm transformer blocks over `n_tokens` tokens.

    Per block:
      - QKV projection:  linear d->3d              = 3 * 2 * n * d^2
      - attention scores Q@K^T: (h, n, hd)x(h, hd, n) = 2 * n^2 * d   (sum over heads = d)
      - attention*value:        (h, n, n)x(h, n, hd)  = 2 * n^2 * d
      - output projection: linear d->d             = 2 * n * d^2
      - MLP: d->hidden->d, hidden = mlp_ratio*d    = 2 * (2 * n * d * hidden)
    RoPE adds only elementwise rotation on Q,K (negligible), so it does not change this count.
    """
    d = embed_dim
    hidden = int(d * mlp_ratio)
    per_block = 0
    # attention linear projections (qkv + out)
    per_block += _linear_flops(n_tokens, d, 3 * d)      # qkv
    per_block += _linear_flops(n_tokens, d, d)          # out proj
    # attention core (two n^2*d matmuls), independent of head split since h*hd = d
    per_block += 2 * (2 * n_tokens * n_tokens * d)
    # mlp (two linears)
    per_block += _linear_flops(n_tokens, d, hidden)
    per_block += _linear_flops(n_tokens, hidden, d)
    return depth * per_block


# ---------------------------------------------------------------------------
# Token-count helpers.
# ---------------------------------------------------------------------------


def num_tokens(frames, res, patch, tubelet):
    """Number of patch tokens for a video clip (no class token in V-JEPA)."""
    return (frames // tubelet) * (res // patch) * (res // patch)


# ---------------------------------------------------------------------------
# Arch extraction from the real construction path (meta device => free).
# ---------------------------------------------------------------------------


def arch_from_model(model_name, frames, res, patch, tubelet,
                    pred_depth, pred_embed_dim, pred_num_heads,
                    use_rope=True, use_sdpa=True, uniform_power=True,
                    use_mask_tokens=True, num_mask_tokens=2):
    """Instantiate encoder+predictor on meta device via init_video_model and read back the
    architecture + exact (analytic-free) parameter counts. No memory is allocated."""
    from app.vjepa_2_1.utils import init_video_model

    encoder, predictor = init_video_model(
        device=torch.device("meta"),
        model_name=model_name,
        patch_size=patch,
        max_num_frames=frames,
        tubelet_size=tubelet,
        crop_size=res,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        pred_num_heads=pred_num_heads,
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=num_mask_tokens,
        use_sdpa=use_sdpa,
        use_rope=use_rope,
    )
    enc = encoder.backbone  # unwrap MultiSeqWrapper
    pred = predictor.backbone  # unwrap PredictorMultiSeqWrapper

    def _param_count(m):
        # meta tensors have real shapes => numel() is exact, no memory used
        return sum(p.numel() for p in m.parameters())

    def _mlp_ratio(block):
        # infer mlp_ratio from the first block's MLP hidden dim vs embed_dim
        fc1 = block.mlp.fc1
        return fc1.out_features / fc1.in_features

    arch = {
        "encoder": {
            "embed_dim": enc.embed_dim,
            "depth": len(enc.blocks),
            "num_heads": enc.num_heads,
            "mlp_ratio": _mlp_ratio(enc.blocks[0]),
            "params": _param_count(encoder),
        },
        "predictor": {
            # predictor embed_dim = the block working dim (fc1.in_features); robust to whether
            # predictor_embed is a bare Linear or a Sequential in this build.
            "embed_dim": pred.predictor_blocks[0].mlp.fc1.in_features,
            "depth": len(pred.predictor_blocks),
            "num_heads": pred.predictor_blocks[0].attn.num_heads,
            "mlp_ratio": _mlp_ratio(pred.predictor_blocks[0]),
            "params": _param_count(predictor),
        },
    }
    return arch


# ---------------------------------------------------------------------------
# Per-clip forward FLOPs.
# ---------------------------------------------------------------------------


def clip_forward_flops(arch, frames, res, patch, tubelet, mask_views):
    """Forward FLOPs to process ONE clip through target-encoder + context-encoder + predictor.

    V-JEPA per-iteration compute (see app/vjepa_2_1/train.py forward_target/forward_context):
      - target encoder: full clip, N tokens, no grad (still costs forward FLOPs)
      - context encoder: only the *kept* context tokens (num_keep_enc), per mask view
      - predictor: context tokens + predicted mask tokens (num_keep_enc + num_keep_pred), per view
    `mask_views` is a list of dicts with num_keep_enc / num_keep_pred (from the config `mask:` section).
    If empty, we fall back to the full-clip encoder pass only (upper-bound-free lower estimate).
    """
    N = num_tokens(frames, res, patch, tubelet)
    e = arch["encoder"]
    p = arch["predictor"]

    def enc_flops(n):
        return _transformer_stack_flops(n, e["embed_dim"], e["depth"], e["num_heads"], e["mlp_ratio"])

    def pred_flops(n):
        # predictor also has in-proj (enc_dim->pred_dim) and out-proj (pred_dim->enc_dim)
        core = _transformer_stack_flops(n, p["embed_dim"], p["depth"], p["num_heads"], p["mlp_ratio"])
        proj = _linear_flops(n, e["embed_dim"], p["embed_dim"]) + _linear_flops(n, p["embed_dim"], e["embed_dim"])
        return core + proj

    total = 0
    # target encoder: sees the full clip once (per view, since each view LayerNorms its own targets)
    # context encoder + predictor: per mask view on kept tokens
    if mask_views:
        for mv in mask_views:
            n_enc = mv["num_keep_enc"]
            n_pred = mv["num_keep_enc"] + mv["num_keep_pred"]
            total += enc_flops(N)          # target encoder on full clip
            total += enc_flops(n_enc)      # context encoder on kept tokens
            total += pred_flops(n_pred)    # predictor on context+mask tokens
    else:
        total += enc_flops(N)
    return total, N


# ---------------------------------------------------------------------------
# Config loading.
# ---------------------------------------------------------------------------


def _load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    model = cfg["model"]
    data = cfg["data"]
    fpcs = data.get("dataset_fpcs") or [data.get("num_frames", 16)]
    frames = fpcs[0] if isinstance(fpcs, list) else fpcs
    mask_views = []
    for mv in cfg.get("mask", []) or []:
        if "num_keep_enc" in mv and "num_keep_pred" in mv:
            mask_views.append({"num_keep_enc": mv["num_keep_enc"], "num_keep_pred": mv["num_keep_pred"]})
    return {
        "model_name": model["model_name"],
        "patch": data.get("patch_size", 16),
        "tubelet": data.get("tubelet_size", 2),
        "res": data.get("crop_size", 224),
        "frames": frames,
        "pred_depth": model.get("pred_depth", 6),
        "pred_embed_dim": model.get("pred_embed_dim", 384),
        "pred_num_heads": model.get("pred_num_heads", None),
        "use_rope": model.get("use_rope", False),
        "use_sdpa": model.get("use_sdpa", False),
        "uniform_power": model.get("uniform_power", False),
        "mask_views": mask_views,
    }


def report(model_name, frames, res, patch, tubelet, pred_depth, pred_embed_dim,
           pred_num_heads, use_rope, use_sdpa, uniform_power, mask_views):
    arch = arch_from_model(
        model_name, frames, res, patch, tubelet,
        pred_depth, pred_embed_dim, pred_num_heads,
        use_rope=use_rope, use_sdpa=use_sdpa, uniform_power=uniform_power,
    )
    fwd, N = clip_forward_flops(arch, frames, res, patch, tubelet, mask_views)
    train = 3 * fwd  # fwd + 2*bwd
    e, p = arch["encoder"], arch["predictor"]
    print(f"model_name         : {model_name}")
    print(f"clip               : {frames}f x {res}px, patch {patch}, tubelet {tubelet}  ->  {N} tokens (full clip)")
    print(f"encoder            : dim={e['embed_dim']} depth={e['depth']} heads={e['num_heads']} "
          f"mlp={e['mlp_ratio']:.4f}  params={e['params']/1e6:.2f}M")
    print(f"predictor          : dim={p['embed_dim']} depth={p['depth']} heads={p['num_heads']} "
          f"mlp={p['mlp_ratio']:.4f}  params={p['params']/1e6:.2f}M")
    print(f"total params (N)    : {(e['params']+p['params'])/1e6:.2f}M  "
          f"(encoder-only {e['params']/1e6:.2f}M)")
    if mask_views:
        print(f"mask views         : " + ", ".join(
            f"[enc {mv['num_keep_enc']}/pred {mv['num_keep_pred']}]" for mv in mask_views))
    else:
        print(f"mask views         : NONE (full-clip encoder pass only)")
    print(f"fwd FLOPs / clip    : {fwd:.3e}")
    print(f"train FLOPs / clip  : {train:.3e}  (3x fwd)")
    return arch, fwd, train, N


def main():
    ap = argparse.ArgumentParser(description="V-JEPA 2.1 FLOP + param accounting")
    ap.add_argument("--config", help="YAML config to read model/data/mask from")
    ap.add_argument("--model", default="vit_large")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--tubelet", type=int, default=2)
    ap.add_argument("--pred-depth", type=int, default=12)
    ap.add_argument("--pred-embed", type=int, default=384)
    ap.add_argument("--pred-heads", type=int, default=None)
    ap.add_argument("--rope", action="store_true")
    ap.add_argument("--no-mask", action="store_true", help="ignore mask views (full-clip encoder only)")
    args = ap.parse_args()

    if args.config:
        c = _load_config(args.config)
        report(c["model_name"], c["frames"], c["res"], c["patch"], c["tubelet"],
               c["pred_depth"], c["pred_embed_dim"], c["pred_num_heads"],
               c["use_rope"], c["use_sdpa"], c["uniform_power"],
               [] if args.no_mask else c["mask_views"])
    else:
        report(args.model, args.frames, args.res, args.patch, args.tubelet,
               args.pred_depth, args.pred_embed, args.pred_heads,
               args.rope, True, True, [])


if __name__ == "__main__":
    main()
