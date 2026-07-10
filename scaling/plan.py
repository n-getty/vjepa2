"""IsoFLOP planner for the JEPA scaling-law sweep (docs/JEPA_SCALING_LAWS_DESIGN.md).

Given a capacity ladder (model sizes) and a set of FLOP budgets C, compute for each (model, budget)
cell the number of optimizer steps and epochs to run so that the *training compute* lands on C.

Compute model (no 6ND shortcut):
    C(cell) = train_flops_per_clip(model) * D
    D       = clips_seen = global_batch * steps          # the data axis
    steps   = ipe * epochs
=> steps  = C / (train_flops_per_clip * global_batch)

Design invariants (see design doc §3 "Decouple D from N"):
  - global_batch is HELD FIXED across all models and budgets. That is what makes D a clean axis
    independent of N. (Under weak scaling the old runs grew batch with N, confounding D and N.)
  - warmup is a FIXED FRACTION of total steps (trainer: warmup_steps = warmup_epochs * ipe), so the
    LR schedule shape is identical across cells and can't bias the scaling fit.

The planner is pure arithmetic on top of scaling/flops.py; it prints a table and (optionally) emits a
manifest JSON that scaling/gen_configs.py turns into runnable YAMLs.
"""

import argparse
import json

# Allowed encoder depths in the 2.1 trainer (hierarchical_layers table,
# app/vjepa_2_1/models/vision_transformer.py:147-174). Off-table depths fail to construct.
ALLOWED_DEPTHS = {12, 24, 40, 48}

# Default capacity ladder: (model_name, depth) for the off-the-shelf 2.1-constructible sizes.
# vit_huge (depth 32) is intentionally EXCLUDED — not buildable without a table edit.
DEFAULT_LADDER = [
    "vit_tiny",     # d12
    "vit_small",    # d12
    "vit_base",     # d12
    "vit_large",    # d24
    "vit_giant",    # d40
    "vit_gigantic",  # d48
]


def _measure_cell(model_name, frames, res, patch, tubelet,
                  pred_depth, pred_embed_dim, pred_num_heads,
                  use_rope, use_sdpa, uniform_power, mask_views):
    """Return (train_flops_per_clip, total_params) for one model via scaling/flops.py."""
    from scaling.flops import arch_from_model, clip_forward_flops

    arch = arch_from_model(
        model_name, frames, res, patch, tubelet,
        pred_depth, pred_embed_dim, pred_num_heads,
        use_rope=use_rope, use_sdpa=use_sdpa, uniform_power=uniform_power,
    )
    fwd, _ = clip_forward_flops(arch, frames, res, patch, tubelet, mask_views)
    train_flops_per_clip = 3 * fwd
    n_params = arch["encoder"]["params"] + arch["predictor"]["params"]
    return train_flops_per_clip, n_params, arch


def plan(ladder, budgets, global_batch, ipe,
         frames, res, patch, tubelet,
         pred_depth, pred_embed_dim, pred_num_heads,
         use_rope, use_sdpa, uniform_power, mask_views,
         warmup_frac, min_steps=200, corpus_size=None, max_corpus_epochs=None):
    """Build the IsoFLOP grid. Returns a list of cell dicts."""
    import logging
    logging.disable(logging.CRITICAL)  # silence init_video_model model dumps

    cells = []
    measured = {}
    for model_name in ladder:
        tf, npar, arch = _measure_cell(
            model_name, frames, res, patch, tubelet,
            pred_depth, pred_embed_dim, pred_num_heads,
            use_rope, use_sdpa, uniform_power, mask_views)
        measured[model_name] = (tf, npar, arch)

    for C in budgets:
        for model_name in ladder:
            tf, npar, arch = measured[model_name]
            # steps = C / (train_flops_per_clip * global_batch)
            steps = C / (tf * global_batch)
            steps = int(round(steps))
            # round steps to a whole number of epochs given ipe
            epochs = max(1, int(round(steps / ipe)))
            actual_steps = epochs * ipe
            D = actual_steps * global_batch
            actual_C = tf * D
            # corpus-loop epochs = how many times D passes over the corpus. Beyond max_corpus_epochs,
            # D stops being an independent data axis and collapses into "epochs-over-the-same-data"
            # (the parallelogram trim, design doc §7b). This is a SEPARATE cap from min_steps.
            corpus_epochs = (D / corpus_size) if corpus_size else None
            if steps < min_steps:
                status = f"SKIP (steps={steps} < {min_steps}; model too big for this budget)"
            elif max_corpus_epochs and corpus_epochs and corpus_epochs > max_corpus_epochs:
                status = (f"SKIP (corpus_epochs={corpus_epochs:.1f} > {max_corpus_epochs}; "
                          f"D collapses into epochs-over-data)")
            else:
                status = "ok"
            cells.append({
                "budget_flops": float(C),
                "model_name": model_name,
                "depth": arch["encoder"]["depth"],
                "n_params": int(npar),
                "train_flops_per_clip": float(tf),
                "global_batch": int(global_batch),
                "ipe": int(ipe),
                "epochs": int(epochs),
                "steps": int(actual_steps),
                "clips_seen_D": int(D),
                "actual_flops": float(actual_C),
                "corpus_epochs": round(corpus_epochs, 2) if corpus_epochs else None,
                "warmup_epochs": round(warmup_frac * epochs, 3),
                "status": status,
            })
    return cells


def print_table(cells):
    hdr = f"{'budget':>9} {'model':>15} {'params(M)':>10} {'gbatch':>7} {'epochs':>7} {'steps':>8} {'D(clips)':>11} {'actualC':>10}  status"
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        print(f"{c['budget_flops']:9.2e} {c['model_name']:>15} {c['n_params']/1e6:10.1f} "
              f"{c['global_batch']:7d} {c['epochs']:7d} {c['steps']:8d} {c['clips_seen_D']:11d} "
              f"{c['actual_flops']:10.2e}  {c['status']}")


def _parse_budgets(s):
    """Accept '1e17,3e17,1e18' or 'logspace:1e16:1e19:5'."""
    if s.startswith("logspace:"):
        import math
        _, lo, hi, n = s.split(":")
        lo, hi, n = float(lo), float(hi), int(n)
        step = (math.log10(hi) - math.log10(lo)) / (n - 1)
        return [10 ** (math.log10(lo) + i * step) for i in range(n)]
    return [float(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description="IsoFLOP planner for JEPA scaling sweep")
    ap.add_argument("--budgets", required=True,
                    help="comma list '1e17,3e17' or 'logspace:LO:HI:N'")
    ap.add_argument("--ladder", default=",".join(DEFAULT_LADDER),
                    help="comma list of model_names")
    ap.add_argument("--global-batch", type=int, required=True,
                    help="clips per optimizer step (HELD FIXED across the whole sweep)")
    ap.add_argument("--ipe", type=int, default=500, help="iterations per epoch (checkpoint granularity)")
    ap.add_argument("--warmup-frac", type=float, default=0.05, help="warmup as fraction of total epochs")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--tubelet", type=int, default=2)
    ap.add_argument("--pred-depth", type=int, default=12)
    ap.add_argument("--pred-embed", type=int, default=384)
    ap.add_argument("--pred-heads", type=int, default=None)
    ap.add_argument("--rope", action="store_true")
    ap.add_argument("--keep-enc", type=int, default=1536, help="single-view num_keep_enc for FLOPs")
    ap.add_argument("--keep-pred", type=int, default=2560, help="single-view num_keep_pred for FLOPs")
    ap.add_argument("--min-steps", type=int, default=200)
    ap.add_argument("--corpus-size", type=int, default=None,
                    help="corpus clip count; enables the corpus-epoch (parallelogram) cap")
    ap.add_argument("--max-corpus-epochs", type=float, default=None,
                    help="drop cells whose D loops the corpus more than this (default: no cap)")
    ap.add_argument("--out", help="write manifest JSON here")
    args = ap.parse_args()

    budgets = _parse_budgets(args.budgets)
    ladder = args.ladder.split(",")
    mask_views = [{"num_keep_enc": args.keep_enc, "num_keep_pred": args.keep_pred}]

    cells = plan(
        ladder, budgets, args.global_batch, args.ipe,
        args.frames, args.res, args.patch, args.tubelet,
        args.pred_depth, args.pred_embed, args.pred_heads,
        args.rope, True, True, mask_views,
        args.warmup_frac, args.min_steps,
        corpus_size=args.corpus_size, max_corpus_epochs=args.max_corpus_epochs)

    print_table(cells)

    if args.out:
        manifest = {
            "meta": {
                "budgets": budgets, "ladder": ladder, "global_batch": args.global_batch,
                "ipe": args.ipe, "warmup_frac": args.warmup_frac,
                "frames": args.frames, "res": args.res, "patch": args.patch,
                "tubelet": args.tubelet, "pred_depth": args.pred_depth,
                "pred_embed": args.pred_embed, "pred_heads": args.pred_heads,
                "use_rope": args.rope, "mask_views": mask_views,
            },
            "cells": cells,
        }
        with open(args.out, "w") as f:
            json.dump(manifest, f, indent=2)
        n_ok = sum(1 for c in cells if c["status"] == "ok")
        print(f"\nwrote {args.out}: {len(cells)} cells ({n_ok} runnable)")


if __name__ == "__main__":
    main()
