#!/usr/bin/env python3
"""Generate the LARGE-BATCH arms that gate the 256-node prod run.

WHY THIS EXPERIMENT EXISTS
--------------------------
Scaling 16n -> 256n cannot preserve the current recipe. Per-rank bs must stay
>= 1, so at 256 nodes x 12 tiles = 3072 ranks the global batch is 3072 (bs=1) or
6144 (bs=2) -- 8x or 16x today's 384. Strong-scaling back to gb=384 would need
per-rank bs=0.125, and `prepare_runtime_config.py:scale_local_batch` raises
rather than produce it. So the batch grows whether we like it or not.

Nothing in the trainer notices. `init_opt` (app/vjepa_2_1/utils.py:425) takes no
world_size and no batch; `lr` is a raw YAML scalar; every schedule length is
expressed in optimizer STEPS via `ipe * epochs`. Three constants are therefore
silently wrong once the step count shrinks by the batch multiplier:

  1. EMA. `m` is a flat 0.99925 (train.py:706-710), a 1/(1-m) = 1333-STEP
     horizon. Holding total samples fixed, a 16x batch means 624 total steps --
     so the teacher retains e^(-624/1333) = 63% of its INITIALIZATION at the end
     of training. The JEPA target would never converge. The horizon must be held
     constant in SAMPLES, not steps: (1-m) scales with the batch multiplier.

  2. Warmup. `warmup` is in epochs, converted by `int(warmup * ipe)`
     (utils.py:493). Today 33.3 * 30 = 999 steps ~= 10% of the run. Left alone
     at 624 total steps it exceeds the entire run: LR would ramp and never reach
     ref_lr. Held here at the same ~10% fraction.

  3. Lambda. `lambda_start_iter`/`lambda_end_iter` (train.py:713-718) are raw
     iteration indices, 500/1500. In a 624-step run the context-loss ramp never
     completes. Scaled down by the same multiplier.

WHY WE CAN TEST THIS ON 16 NODES
--------------------------------
`VJEPA_TRUE_ACCUM` (train.py:167-181, 1179-1198) fetches N separate loader
batches per optimizer step and defers the collective to the last one, so the
optimizer sees an exact N x larger-batch gradient. 16n x 192 ranks x bs2 x
accum16 == gb 6144. That reproduces the 256n batch on the capacity queue (7-day
walltime) instead of burning prod time to discover the recipe is wrong.

WHAT WOULD FALSIFY THE WHOLE PLAN
---------------------------------
docs/Leo_Scaling_Results.md measured, at FIXED FLOPs, 2n loss 6.9% WORSE than
1n, and ~19% parallel efficiency at 8n. This recipe has historically disliked
larger batches, and train.py:176-177 records LR-hotness as its known collapse
mode. If both arms below sit above the 16n control at matched SAMPLES SEEN, the
answer is that 256n single-world is not the way to a better model -- and the
right move is to report that rather than spend prod hours on it.

ARMS (LR is the only variable; everything else is the derived large-batch recipe)
  lbA -- sqrt-scaled LR with a haircut. sqrt(16) x 7.5e-5 = 3.0e-4; we take
         2.0e-4 because the gradient-noise argument for sqrt assumes a
         well-conditioned objective and this one collapses when hot.
  lbB -- LR unchanged at 7.5e-5. Isolates "batch changed" from "LR changed";
         if lbA diverges and lbB does not, the recipe is salvageable by LR alone.

Usage:
    python scripts/gen_large_batch_configs.py [--batch-mult 16] [--per-rank-bs 2]
"""

import argparse
import os
import sys

import yaml

CFGD = "configs/vitg16_surg_vid_webdataset_single4"
BASE = f"{CFGD}/vitG384_fixedshape_v2.yaml"

# Baseline (16n) recipe these are derived FROM -- read from BASE, asserted below.
BASE_IPE = 30
BASE_EPOCHS = 333
BASE_EMA = 0.99925
BASE_WARMUP_EPOCHS = 33.3
BASE_LAMBDA = (500, 1500)
BASE_GLOBAL_BATCH = 384  # 16 nodes x 12 tiles x bs2

ARMS = {
    "lbA": {
        "lr": 2.0e-4,
        "role": "sqrt-scaled LR (haircut from 3.0e-4)",
        "why": (
            "sqrt(mult) is the standard rule when the gradient is a MEAN over the\n"
            "batch (it is: train.py:1076,1104,1115) and batch growth cuts gradient\n"
            "noise by sqrt(mult). Linear scaling would give 1.2e-3, far into the\n"
            "regime train.py:176-177 calls this model's known collapse mode."
        ),
    },
    "lbB": {
        "lr": 7.5e-5,
        "role": "LR-unchanged control",
        "why": (
            "Isolates the batch change from the LR change. If the large-batch\n"
            "recipe fails here too, the problem is the batch (or the EMA/warmup\n"
            "derivation), not LR tuning."
        ),
    },
}


def derive(batch_mult, per_rank_bs):
    """Every large-batch constant, derived so SAMPLES SEEN is invariant."""
    total_steps_base = BASE_IPE * BASE_EPOCHS
    total_steps = max(1, round(total_steps_base / batch_mult))

    # Keep ipe small enough that an epoch fits comfortably inside a queue slice.
    ipe = 39
    epochs = max(1, round(total_steps / ipe))

    # EMA: hold the horizon constant in SAMPLES. horizon_steps = 1/(1-m).
    ema = 1.0 - (1.0 - BASE_EMA) * batch_mult
    if ema <= 0:
        sys.exit(f"batch_mult {batch_mult} drives EMA non-positive ({ema})")

    # Warmup: hold the same fraction of the run (~10%).
    warmup_frac = (BASE_WARMUP_EPOCHS * BASE_IPE) / total_steps_base
    warmup_epochs = round(warmup_frac * ipe * epochs / ipe, 2)

    lam = (max(1, round(BASE_LAMBDA[0] / batch_mult)),
           max(2, round(BASE_LAMBDA[1] / batch_mult)))

    return {
        "global_batch": BASE_GLOBAL_BATCH * batch_mult,
        "per_rank_bs": per_rank_bs,
        "true_accum": batch_mult,
        "ipe": ipe,
        "epochs": epochs,
        "total_steps": ipe * epochs,
        "ema": round(ema, 6),
        "ema_horizon_steps": round(1.0 / (1.0 - ema), 1),
        "ema_horizon_samples": round(BASE_GLOBAL_BATCH / (1.0 - BASE_EMA)),
        "warmup": warmup_epochs,
        "warmup_steps": int(warmup_epochs * ipe),
        "lambda_start": lam[0],
        "lambda_end": lam[1],
        "samples": ipe * epochs * BASE_GLOBAL_BATCH * batch_mult,
    }


def _replay_report(cfg_path, clips_seen, fresh_rule=4.0):
    """Warn if the budget overshoots the 4x-fresh rule on the EFFECTIVE corpus.

    The raw clip count overstates the corpus badly: source-selection is
    `prob_i ~ count_i ** T` (src/datasets/webdataset.py:compute_mixing_probs), so
    at the production T=0.5 the inverse-collision effective size is only ~32% of
    nominal. A budget that reads as a safe 3.2 corpus-epochs is really ~9.9.
    Effective size is a property of the sampling DISTRIBUTION and does not grow
    with budget, so scaling compute makes the overshoot strictly worse.

    Advisory, not fatal -- T is a separate live experiment, and the right value
    is a judgement call (raising it shifts the mix toward the largest, mostly
    non-surgical sources). Best-effort: skipped if metadata is unreadable.
    """
    try:
        import json
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.datasets.webdataset import compute_mixing_probs

        cfg = yaml.safe_load(open(cfg_path))
        dsets = cfg["data"]["datasets"]
        counts = [
            json.load(open(os.path.join(d, "metadata.json")))["sample_count"]
            for d in dsets
        ]
        T = float(cfg["data"].get("sampling_temperature", 0.5))
        probs = compute_mixing_probs(counts, cfg["data"].get("datasets_weights"), T)
        eff = 1.0 / sum(p * p / n for p, n in zip(probs, counts) if n)
    except Exception as e:  # missing metadata, unmounted path, etc.
        print(f"\n  (replay check skipped: {type(e).__name__}: {e})")
        return

    total = sum(counts)
    print(f"\nReplay check (T={T}):")
    print(f"  {'raw corpus':22s} {total:,} clips")
    print(f"  {'effective corpus':22s} {eff:,.0f} ({100*eff/total:.0f}% of raw)")
    print(f"  {'nominal epochs':22s} {clips_seen/total:.2f}x")
    print(f"  {'EFFECTIVE epochs':22s} {clips_seen/eff:.1f}x  (4x-fresh rule)")
    worst = max(p * clips_seen / n for p, n in zip(probs, counts) if n)
    print(f"  {'worst per-clip replay':22s} {worst:.0f}x")
    if clips_seen / eff > fresh_rule:
        print(
            f"  WARNING: {clips_seen/eff:.1f}x exceeds the {fresh_rule}x-fresh rule on the\n"
            f"  EFFECTIVE corpus. Raising sampling_temperature (~0.75 -> ~79% effective)\n"
            f"  fits the budget, but shifts the mix toward the largest sources."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-mult", type=int, default=16,
                    help="Global-batch multiplier vs the 16n baseline (384). "
                         "16 = 256n at per-rank bs2; 8 = 256n at bs1.")
    ap.add_argument("--per-rank-bs", type=int, default=2)
    ap.add_argument("--outdir", default=CFGD)
    args = ap.parse_args()

    if not os.path.exists(BASE):
        sys.exit(f"missing base config: {BASE}")
    os.makedirs(args.outdir, exist_ok=True)
    with open(BASE) as f:
        base_lines = f.read().splitlines()

    d = derive(args.batch_mult, args.per_rank_bs)

    print(f"Derived large-batch recipe ({args.batch_mult}x):")
    for k in ("global_batch", "true_accum", "ipe", "epochs", "total_steps",
              "ema", "ema_horizon_steps", "warmup", "warmup_steps",
              "lambda_start", "lambda_end", "samples"):
        print(f"  {k:22s} {d[k]}")
    base_samples = BASE_IPE * BASE_EPOCHS * BASE_GLOBAL_BATCH
    print(f"  {'baseline samples':22s} {base_samples}  "
          f"(ratio {d['samples']/base_samples:.2f})")
    if d["warmup_steps"] >= d["total_steps"]:
        sys.exit("ABORT: warmup >= total steps; the run would never reach ref_lr")

    _replay_report(BASE, d["samples"])

    for arm, spec in ARMS.items():
        out_path = f"{args.outdir}/vitG384_{arm}.yaml"
        out = [
            f"# LARGE-BATCH arm `{arm}` ({spec['role']}) -- generated by",
            "# scripts/gen_large_batch_configs.py. Read that file first: it derives",
            "# every constant below and states what result would kill the 256n plan.",
            "#",
            f"# Emulates global batch {d['global_batch']} "
            f"({args.batch_mult}x the 16n baseline of {BASE_GLOBAL_BATCH}) on 16 nodes",
            f"# via VJEPA_TRUE_ACCUM={d['true_accum']}, so the recipe is validated on the",
            "# capacity queue before any prod time is spent.",
            "#",
        ]
        out += [f"# {ln}" for ln in spec["why"].split("\n")]
        out += [
            "#",
            "# DERIVED SCHEDULE (samples-invariant, not steps-invariant):",
            f"#   ipe x epochs   {BASE_IPE}x{BASE_EPOCHS} -> {d['ipe']}x{d['epochs']} "
            f"= {d['total_steps']} steps ({d['samples']/1e6:.2f}M clips)",
            f"#   ema            {BASE_EMA} -> {d['ema']}   "
            f"(horizon {d['ema_horizon_steps']} steps = {d['ema_horizon_samples']:,} "
            "samples, unchanged)",
            f"#   warmup         {BASE_WARMUP_EPOCHS} -> {d['warmup']} epochs "
            f"({d['warmup_steps']} steps, same ~10% of run)",
            f"#   lambda ramp    {BASE_LAMBDA[0]}/{BASE_LAMBDA[1]} -> "
            f"{d['lambda_start']}/{d['lambda_end']} iters",
            f"#   lr             {ARMS['lbB']['lr']} -> {spec['lr']}",
            "#",
            "# Corpus, mask, seed and init are identical to vitG384_fixedshape_v2.yaml.",
            "",
        ]

        in_opt = in_model = in_data = False
        for line in base_lines:
            if line.startswith("folder:"):
                out.append("folder: /flare/ModCon/ngetty/checkpoints/"
                           f"surg_2_1_vitG384_{arm}/vitG384_n16g12_{arm}")
                continue
            if line.startswith("optimization:"):
                in_opt, in_model, in_data = True, False, False
            elif line.startswith("model:"):
                in_model, in_opt, in_data = True, False, False
            elif line.startswith("data:"):
                in_data, in_opt, in_model = True, False, False
            elif line and not line.startswith((" ", "-")):
                in_opt = in_model = in_data = False

            s = line.strip()
            indent = line[: len(line) - len(line.lstrip())]

            if in_opt:
                if s.startswith("epochs:"):
                    out.append(f"{indent}epochs: {d['epochs']}"); continue
                if s.startswith("ipe:"):
                    out.append(f"{indent}ipe: {d['ipe']}"); continue
                if s.startswith("warmup:"):
                    out.append(f"{indent}warmup: {d['warmup']}"); continue
                if s.startswith("lr:"):
                    out.append(f"{indent}lr: {spec['lr']:.1e}"); continue
                if s == f"- {BASE_EMA}":
                    out.append(f"{indent}- {d['ema']}"); continue
            if in_model:
                if s.startswith("lambda_start_iter:"):
                    out.append(f"{indent}lambda_start_iter: {d['lambda_start']}")
                    continue
                if s.startswith("lambda_end_iter:"):
                    out.append(f"{indent}lambda_end_iter: {d['lambda_end']}")
                    continue
            if in_data:
                # Without this, --per-rank-bs was accepted, reported, and then
                # silently dropped -- emitting a bs=2 config while the printed
                # derivation claimed bs=1, i.e. half the intended global batch.
                if s.startswith("batch_size:"):
                    out.append(f"{indent}batch_size: {d['per_rank_bs']}")
                    continue
            out.append(line)

        with open(out_path, "w") as f:
            f.write("\n".join(out) + "\n")
        print(f"wrote {out_path}  (lr={spec['lr']:.1e}, ema={d['ema']}, "
              f"{d['ipe']}x{d['epochs']})")

    print(f"\nLaunch each arm with VJEPA_TRUE_ACCUM={d['true_accum']} at 16 nodes.")


if __name__ == "__main__":
    main()
